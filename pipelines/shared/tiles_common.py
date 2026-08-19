"""
shared/tiles_common.py - Shared bag-reading, world-frame merge, and
ECEF/py3dtiles export helpers used by both `tiles_3d.py` and
`color_tiles_3d.py`.

REFACTOR RATIONALE:
tiles_3d.py and color_tiles_3d.py previously duplicated near-identical
bag-reading loops, world-frame merge loops, and the entire ENU -> ECEF ->
PLY -> py3dtiles-convert -> tileset.json tail end. That duplication meant
any bugfix to the shared logic (e.g. the py3dtiles CLI flags, PLY writing,
or the odometry pose-lookup merge logic) had to be applied twice by hand,
risking silent behavioral drift between the two pipelines. This module
centralizes that shared logic so both pipelines call into one
implementation.

PATCH NOTE (odom-anchored registration + frame-awareness):
`transform_frame_to_world()` now looks up the view-ray sensor origin via
`ros_io.interpolate_odom_pose()` (linear translation + SLERP rotation
between bracketing odometry samples) instead of nearest-neighbor
timestamp snapping, for consistency with `registration.run_odom_anchored_
registration()`. Its signature/semantics are unchanged (`odom_max_ns` is
still nanoseconds); callers passing an identity `transform_world` for a
global/fixed-frame point cloud get a correct no-op transform while still
getting sensor-origin-derived view rays for normal orientation, which is
exactly the global-frame behaviour described in
pointcloud-frame-check-prompt-2.md.

FEATURE NOTE (multi-LOD tileset export):
Both pipelines previously called `georeference_and_export_tileset()` once
and produced a single `tileset/` directory (one resolution, i.e. one
"layer"). `georeference_and_export_lod_tilesets()` below generalizes this
to export the same cleaned/merged cloud multiple times at different voxel
densities (coarse/medium/fine by default), each into its own
`tileset_<name>/` subfolder, so a Cesium viewer (or any 3D Tiles client)
can offer the user a quality/performance toggle between layers. The
original single-tileset ECEF/PLY/py3dtiles-convert tail end is factored
out into `_export_pcd_as_tileset()` so both the legacy single-tileset path
and the new multi-LOD path share one implementation -- consistent with
this module's own refactor rationale above.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from .reconstruction import transform_local_enu_to_ecef
from .registration import attach_view_rays_as_normals
from .ros_io import TYPESTORE, interpolate_odom_pose

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
    Voxel-downsample and transform one frame into the world/merge frame.

    `transform_world` is applied as-is: pass the odom-anchored (optionally
    ICP-refined) pose for a local/moving-frame point cloud, or `np.eye(4)`
    for a point cloud already published in a global/fixed frame (see
    `ros_io.resolve_pc_frame_mode`) -- an identity transform is a correct
    no-op, which is exactly "skip applying any per-frame transform" per
    pointcloud-frame-check-prompt-2.md.

    If odometry data is supplied and covers `timestamp` within
    `odom_max_ns`, attach unnormalised view-ray "normals" (see
    `registration.attach_view_rays_as_normals`) using an odometry pose
    interpolated between the two bracketing samples (not nearest-neighbor
    snapping) for later ground/obstacle orientation. This is the exact
    merge-loop logic that was previously duplicated between tiles_3d.py and
    color_tiles_3d.py.
    """
    pcd_world = pcd_raw.voxel_down_sample(voxel_size)
    pcd_world.transform(transform_world)

    if (
        timestamp is not None
        and odom_data
        and odom_ts_sorted
        and odom_max_ns is not None
    ):
        interpolated_pose = interpolate_odom_pose(
            timestamp, odom_ts_sorted, odom_data, odom_max_ns
        )
        if interpolated_pose is not None:
            attach_view_rays_as_normals(pcd_world, interpolated_pose[:3, 3])

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
def _export_pcd_as_tileset(
    pcd: o3d.geometry.PointCloud,
    lat0: float,
    lon0: float,
    alt0: float,
    tiles_dir: Path,
    workers: int,
) -> None:
    """
    Convert one already-finalized world-frame cloud to ECEF, write it as a
    (optionally colored) temp PLY, and run py3dtiles convert into
    `tiles_dir`.

    This is the single-tileset ECEF/PLY/py3dtiles-convert tail end, shared
    by both the legacy `georeference_and_export_tileset()` single-layer
    path and the multi-LOD `georeference_and_export_lod_tilesets()` path
    below, so a future fix only needs to be made once.
    """
    pts_enu = np.asarray(pcd.points, dtype=np.float64)
    pts_ecef = transform_local_enu_to_ecef(pts_enu, lat0, lon0, alt0)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        ply_path = Path(tmp.name)

    try:
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float64)
            write_colored_ply_ecef(pts_ecef, colors, ply_path)
        else:
            write_ply_ecef(pts_ecef, ply_path)

        tiles_dir.mkdir(parents=True, exist_ok=True)
        run_py3dtiles_convert(ply_path, tiles_dir, jobs=workers)
    finally:
        ply_path.unlink(missing_ok=True)


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

    Legacy single-tileset entry point, kept for any other caller that only
    wants one output layer. `tiles_3d.py` and `color_tiles_3d.py` now call
    `georeference_and_export_lod_tilesets()` instead to get three LOD
    layers. Returns (tiles_dir, enu_ply_path).
    """
    tiles_dir = out_dir / "tileset"
    _export_pcd_as_tileset(pcd_clean, lat0, lon0, alt0, tiles_dir, workers)

    enu_ply = out_dir / f"{bag_stem}_cloud_enu.ply"
    o3d.io.write_point_cloud(str(enu_ply), pcd_clean)
    print(f"  ENU cloud: {enu_ply}")

    return tiles_dir, enu_ply


# Default LOD ladder: (name, voxel-size multiplier relative to the
# pipeline's own `--voxel_size`). "fine" uses a multiplier of 1.0, i.e. the
# cloud's own resolution as already produced by upstream merging/cleaning
# -- no extra downsampling -- so the fine layer is identical to what the
# single-tileset path used to produce.
DEFAULT_LOD_LEVELS: tuple[tuple[str, float], ...] = (
    ("coarse", 4.0),
    ("medium", 2.0),
    ("fine", 1.0),
)


def georeference_and_export_lod_tilesets(
    pcd_clean: o3d.geometry.PointCloud,
    lat0: float,
    lon0: float,
    alt0: float,
    out_dir: Path,
    bag_stem: str,
    workers: int,
    base_voxel_size: float,
    lod_levels: tuple[tuple[str, float], ...] = DEFAULT_LOD_LEVELS,
) -> tuple[dict[str, Path], Path]:
    """
    Export `pcd_clean` as multiple LOD (level-of-detail) 3D Tiles tilesets
    -- coarse/medium/fine by default -- instead of a single tileset, so a
    viewer can switch between quality levels.

    Each entry in `lod_levels` is a `(name, voxel_multiplier)` pair. The
    cloud for a given level is `pcd_clean` voxel-downsampled at
    `base_voxel_size * voxel_multiplier`. A multiplier of `1.0` (the
    "fine" default) reuses `pcd_clean` as-is with no extra downsampling,
    since it is already at `base_voxel_size` resolution from upstream
    merging/cleaning. Each level is written into its own
    `out_dir/tileset_<name>/` subfolder via `_export_pcd_as_tileset()`.

    Returns `({level_name: tiles_dir}, enu_ply_path)`. The ENU-frame
    reference PLY is written once, at full (base) resolution, not once per
    level.
    """
    tiles_dirs: dict[str, Path] = {}

    for name, multiplier in lod_levels:
        if multiplier > 1.0:
            lod_voxel = base_voxel_size * multiplier
            pcd_lod = pcd_clean.voxel_down_sample(lod_voxel)
        else:
            lod_voxel = base_voxel_size
            pcd_lod = pcd_clean

        print(
            f"  [LOD:{name}] voxel={lod_voxel:.3f} m -> "
            f"{len(pcd_lod.points):,} points"
        )

        tiles_dir = out_dir / f"tileset_{name}"
        _export_pcd_as_tileset(pcd_lod, lat0, lon0, alt0, tiles_dir, workers)
        tiles_dirs[name] = tiles_dir

        if pcd_lod is not pcd_clean:
            del pcd_lod

    enu_ply = out_dir / f"{bag_stem}_cloud_enu.ply"
    o3d.io.write_point_cloud(str(enu_ply), pcd_clean)
    print(f"  ENU cloud (full resolution): {enu_ply}")

    return tiles_dirs, enu_ply
