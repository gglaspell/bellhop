"""
shared/tiles_common.py - Shared bag-reading, world-frame merge, and
ECEF/py3dtiles export helpers used by both `tiles_3d.py` and
`color_tiles_3d.py`.

REFACTOR RATIONALE:
tiles_3d.py and color_tiles_3d.py previously duplicated near-identical
bag-reading loops, world-frame merge loops, and the entire ENU -> ECEF ->
PLY -> py3dtiles-convert -> tileset.json tail end. That duplication meant
any bugfix to the shared logic (e.g. the py3dtiles CLI flags, PLY writing,
or the odometry-nearest-timestamp merge logic) had to be applied twice by
hand, risking silent behavioral drift between the two pipelines. This
module centralizes that shared logic so both pipelines call into one
implementation.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from .reconstruction import transform_local_enu_to_ecef
from .registration import attach_view_rays_as_normals
from .ros_io import TYPESTORE, get_closest_timestamp


# ---------------------------------------------------------------------------
# Generic bag reading
# ---------------------------------------------------------------------------
def read_bag_topics(
    bag_path: Path,
    topic_handlers: dict[str, Callable[[Any, int], None]],
    desc: str = "Reading",
) -> None:
    """
    Read `bag_path` once, dispatching each deserialized message to the
    handler registered for its topic in `topic_handlers`.

    Each handler receives `(message, timestamp_ns)` and is responsible for
    storing whatever it needs (e.g. appending to a list or dict owned by
    the caller). Deserialize/handler exceptions for a single message are
    swallowed so one bad message cannot abort the whole read pass, matching
    the original per-pipeline behaviour.
    """
    topics = list(topic_handlers.keys())

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic in topics]

        for connection, timestamp, raw in tqdm(
            reader.messages(connections=connections),
            desc=desc,
        ):
            try:
                message = reader.deserialize(raw, connection.msgtype)
                handler = topic_handlers.get(connection.topic)
                if handler is not None:
                    handler(message, timestamp)
            except Exception:
                continue


# ---------------------------------------------------------------------------
# World-frame transform for a single registered frame
# ---------------------------------------------------------------------------
def transform_frame_to_world(
    pcd_raw: o3d.geometry.PointCloud,
    transform_world: np.ndarray,
    voxel_size: float,
    timestamp: int | None = None,
    odom_data: dict[int, np.ndarray] | None = None,
    odom_ts_sorted: list[int] | None = None,
    odom_max_ns: int | None = None,
) -> o3d.geometry.PointCloud:
    """
    Voxel-downsample and transform one registered frame into world frame.

    If odometry data is supplied and a close-enough timestamp match exists,
    attach unnormalised view-ray "normals" (see
    `registration.attach_view_rays_as_normals`) for later ground/obstacle
    orientation. This is the exact merge-loop logic that was previously
    duplicated between tiles_3d.py and color_tiles_3d.py.
    """
    pcd_world = pcd_raw.voxel_down_sample(voxel_size)
    pcd_world.transform(transform_world)

    if (
        timestamp is not None
        and odom_data
        and odom_ts_sorted
        and odom_max_ns is not None
    ):
        closest_ts = get_closest_timestamp(timestamp, odom_ts_sorted)
        if closest_ts is not None and abs(closest_ts - timestamp) < odom_max_ns:
            attach_view_rays_as_normals(
                pcd_world, odom_data[closest_ts][:3, 3]
            )

    return pcd_world


# ---------------------------------------------------------------------------
# py3dtiles conversion helper
# ---------------------------------------------------------------------------
def run_py3dtiles_convert(ply_path: Path, out_dir: Path, jobs: int = 1) -> None:
    """
    Invoke py3dtiles convert on a pre-georeferenced (ECEF) PLY file.

    The PLY written upstream already contains ECEF XYZ coordinates, so we
    pass --srs_in 4978 (ECEF) and --srs_out 4978 to skip any reprojection.
    py3dtiles writes tileset.json + tile content files into out_dir.
    """
    cmd = [
        "py3dtiles", "convert",
        str(ply_path),
        "--out", str(out_dir),
        "--srs_in", "4978",
        "--srs_out", "4978",
        "--jobs", str(jobs),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        sys.exit(
            f"Error: py3dtiles convert failed (exit {result.returncode}).\n"
            "Check that py3dtiles >= 4.0 is installed and the PLY file is valid."
        )


# ---------------------------------------------------------------------------
# Write ECEF point cloud as PLY (ASCII)
# ---------------------------------------------------------------------------
def write_ply_ecef(pts_ecef: np.ndarray, ply_path: Path) -> None:
    """Write an (N, 3) float64 ECEF array as a minimal PLY file."""
    n = len(pts_ecef)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "end_header\n"
    )
    with open(ply_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for x, y, z in pts_ecef:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"  Written ECEF PLY: {ply_path} ({n:,} points)")


def write_colored_ply_ecef(
    pts_ecef: np.ndarray,
    colors: np.ndarray,
    ply_path: Path,
) -> None:
    """Write an (N, 3) ECEF + (N, 3) uint8 RGB as a PLY file."""
    n = len(pts_ecef)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    rgb = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    with open(ply_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for (x, y, z), (r, g, b) in zip(pts_ecef, rgb):
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    print(f"  Written colored ECEF PLY: {ply_path} ({n:,} points)")


# ---------------------------------------------------------------------------
# Shared georeference + py3dtiles export tail
# ---------------------------------------------------------------------------
def georeference_and_export_tileset(
    pcd_clean: o3d.geometry.PointCloud,
    lat0: float,
    lon0: float,
    alt0: float,
    out_dir: Path,
    bag_stem: str,
    workers: int,
) -> tuple[Path, Path]:
    """
    Convert a cleaned, merged world-frame cloud to ECEF, write it as a
    (optionally colored) PLY, run py3dtiles convert, and save the ENU-frame
    cloud for reference.

    This is the exact tail-end logic (steps 6-9 in the module docstrings of
    both pipelines) that was previously duplicated verbatim in tiles_3d.py
    and color_tiles_3d.py. Returns (tiles_dir, enu_ply_path).
    """
    import tempfile

    pts_enu = np.asarray(pcd_clean.points, dtype=np.float64)
    pts_ecef = transform_local_enu_to_ecef(pts_enu, lat0, lon0, alt0)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        ply_path = Path(tmp.name)

    has_colors = pcd_clean.has_colors()
    if has_colors:
        colors = np.asarray(pcd_clean.colors, dtype=np.float64)
        write_colored_ply_ecef(pts_ecef, colors, ply_path)
    else:
        write_ply_ecef(pts_ecef, ply_path)

    tiles_dir = out_dir / "tileset"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_py3dtiles_convert(ply_path, tiles_dir, jobs=workers)
    finally:
        ply_path.unlink(missing_ok=True)

    enu_ply = out_dir / f"{bag_stem}_cloud_enu.ply"
    o3d.io.write_point_cloud(str(enu_ply), pcd_clean)
    print(f"  ENU cloud: {enu_ply}")

    return tiles_dir, enu_ply
