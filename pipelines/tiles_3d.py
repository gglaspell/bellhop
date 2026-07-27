#!/usr/bin/env python3
"""
tiles_3d.py – ROS 2 bag -> georeferenced registered point cloud -> Cesium 3D Tiles.

Pipeline:
1. Pre-flight: verify required topics exist.
2. Average GPS fixes to establish ENU origin (lat0, lon0, alt0).
3. Read PointCloud2 + optional Odometry messages.
4. ICP + pose-graph registration (shared).
5. Merge transformed frames into one world-frame cloud.
6. Clean (voxel -> ROR -> SOR -> DBSCAN).
7. Convert local ENU coords to ECEF (EPSG:4978).
8. Write a temp .ply then call py3dtiles convert -> tileset.json output.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud, transform_local_enu_to_ecef
from .shared.registration import run_icp_posegraph
from .shared.ros_io import (
    TYPESTORE,
    convert_ros_pc2_to_o3d,
    get_odom_transform,
    get_closest_timestamp,
    parse_gps_fixes,
)

# ---------------------------------------------------------------------------
# py3dtiles conversion helper
# ---------------------------------------------------------------------------
def _run_py3dtiles_convert(ply_path: Path, out_dir: Path, jobs: int = 1) -> None:
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
# Write ECEF point cloud as PLY (ASCII, XYZ only)
# ---------------------------------------------------------------------------
def _write_ply_ecef(pts_ecef: np.ndarray, ply_path: Path) -> None:
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


def _write_colored_ply_ecef(pts_ecef: np.ndarray, colors: np.ndarray,
                             ply_path: Path) -> None:
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
# Main pipeline
# ---------------------------------------------------------------------------
def run(args) -> None:
    bag_path = Path(args.bagpath)
    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    required = [args.pc_topic, args.gps_topic]
    if args.odom_topic:
        required.append(args.odom_topic)
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(
            f"Error: Required topics missing from bag: {missing}\n"
            "Check topic names with: ros2 bag info "
        )

    # ── GPS origin ────────────────────────────────────────────────────────
    print(f"[1/6] Reading GPS fixes from '{args.gps_topic}'...")
    lat0, lon0, alt0 = parse_gps_fixes(bag_path, args.gps_topic)

    # ── Read bag ──────────────────────────────────────────────────────────
    topics = [args.pc_topic] + ([args.odom_topic] if args.odom_topic else [])
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}

    print(f"\n[2/6] Reading point clouds and odometry from: {bag_path}")
    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, ts, raw in tqdm(reader.messages(connections=conns), desc="Reading"):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.topic == args.pc_topic:
                    pcd = convert_ros_pc2_to_o3d(msg)
                    if pcd is not None and len(pcd.points) >= 100:
                        pointclouds.append((ts, pcd))
                elif args.odom_topic and conn.topic == args.odom_topic:
                    T = get_odom_transform(msg)
                    if T is not None:
                        odom_data[ts] = T
            except Exception:
                continue

    if not pointclouds:
        sys.exit("Error: No valid point clouds extracted from bag.")
    if args.odom_topic and not odom_data:
        print("Warning: --odom_topic set but no messages found; using identity guesses.")
    print(f"  {len(pointclouds)} frames | {len(odom_data)} odom poses")

    # ── ICP + pose graph ──────────────────────────────────────────────────
    print(f"\n[3/6] ICP registration + pose-graph optimisation...")
    posegraph, good_idx = run_icp_posegraph(pointclouds, odom_data, args)
    odom_max_ns = int(args.odom_max_latency * 1e9)
    odom_ts_sorted = sorted(odom_data.keys())

    # ── Merge world-frame cloud ───────────────────────────────────────────
    print("\n[4/6] Merging registered frames...")
    pcd_combined = o3d.geometry.PointCloud()

    for node_i, pc_i in enumerate(good_idx):
        if node_i >= len(posegraph.nodes):
            break
        T_world = np.linalg.inv(posegraph.nodes[node_i].pose)
        ts, pcd_raw = pointclouds[pc_i]
        pcd_world = pcd_raw.voxel_down_sample(args.voxel_size)
        pcd_world.transform(T_world)
        pcd_combined += pcd_world

    # ── Clean ─────────────────────────────────────────────────────────────
    print("\n[5/6] Cleaning merged cloud...")
    pcd_clean = clean_point_cloud(
        pcd_combined, args.voxel_size, do_voxel_downsample=False
    )

    if len(pcd_clean.points) == 0:
        sys.exit("Error: No points remain after cleaning.")

    # ── Georeference ENU -> ECEF ──────────────────────────────────────────
    print("\n[6/6] Georeferencing (local ENU -> ECEF) and writing 3D Tiles...")
    pts_enu = np.asarray(pcd_clean.points, dtype=np.float64)
    pts_ecef = transform_local_enu_to_ecef(pts_enu, lat0, lon0, alt0)

    # Write ECEF PLY into a temp file then convert
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        ply_path = Path(tmp.name)
    _write_ply_ecef(pts_ecef, ply_path)

    tiles_dir = out_dir / "tileset"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_py3dtiles_convert(ply_path, tiles_dir, jobs=args.workers)
    finally:
        ply_path.unlink(missing_ok=True)

    enu_ply = out_dir / f"{bag_path.stem}_cloud_enu.ply"
    o3d.io.write_point_cloud(str(enu_ply), pcd_clean)
    print(f"  ENU cloud: {enu_ply}")
    print(f"\n3D Tiles written to: {tiles_dir}")
    print("Done.")


def build_parser(sub):
    p = sub.add_parser(
        "tiles_3d",
        help="ROS 2 bag -> georeferenced point cloud -> Cesium 3D Tiles"
    )
    p.add_argument("bagpath", help="Path to the ROS 2 bag directory.")
    p.add_argument("outputdir", help="Output directory.")

    # Topics
    p.add_argument("--pc_topic", default="points",
                   help="PointCloud2 topic (default: points).")
    p.add_argument("--odom_topic", default=None,
                   help="Odometry topic (nav_msgs/Odometry). Optional.")
    p.add_argument("--gps_topic", default="/gps/fix",
                   help="NavSatFix topic for GPS origin (default: /gps/fix).")

    # Registration
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--icp_dist_thresh", type=float, default=0.2)
    p.add_argument("--icp_fitness_thresh", type=float, default=0.6)
    p.add_argument("--odom_max_latency", type=float, default=0.5)
    p.add_argument("--enable_loop_closure", action="store_true", default=False)
    p.add_argument("--loop_closure_radius", type=float, default=10.0)
    p.add_argument("--loop_closure_fitness_thresh", type=float, default=0.3)
    p.add_argument("--loop_closure_search_interval", type=int, default=10)
    p.add_argument("--frame_stride", type=int, default=0,
                   help="Process every Nth frame (0 = all frames).")
    p.add_argument("--max_registration_frames", type=int, default=0,
                   help="Cap total frames used for registration (0 = all).")
    p.add_argument("--merge_chunk_frames", type=int, default=16,
                   help="Number of frames per merge chunk.")

    # Performance
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for KDTree queries and py3dtiles convert.")

    p.set_defaults(func=run)
    return p
