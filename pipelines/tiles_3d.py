#!/usr/bin/env python3
"""
tiles_3d.py - ROS 2 bag -> georeferenced registered point cloud -> Cesium 3D Tiles.

Pipeline:
1. Pre-flight: verify required topics exist.
2. Average GPS fixes to establish ENU origin (lat0, lon0, alt0).
3. Read PointCloud2 + optional Odometry messages.
4. ICP + pose-graph registration (shared).
5. Merge transformed frames into one world-frame cloud.
6. Clean (voxel -> ROR -> SOR -> DBSCAN).
7. Convert local ENU coords to ECEF (EPSG:4978).
8. Write a temp .ply then call py3dtiles convert -> tileset.json output.

REFACTOR NOTE: Bag-reading, world-frame merge, and the ENU->ECEF->PLY->
py3dtiles tail-end logic now live in `shared/tiles_common.py` and are
shared with `color_tiles_3d.py`, so bugfixes to that shared logic no
longer need to be applied twice by hand.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud
from .shared.registration import run_icp_posegraph
from .shared.ros_io import convert_ros_pc2_to_o3d, get_odom_transform, parse_gps_fixes
from .shared.tiles_common import (
    georeference_and_export_tileset,
    read_bag_topics,
    run_py3dtiles_convert,  # re-exported for backward compatibility
    transform_frame_to_world,
    write_colored_ply_ecef,  # re-exported for backward compatibility
    write_ply_ecef,  # re-exported for backward compatibility
)

# Backward-compatible private aliases: color_tiles_3d.py (and any other
# external caller) previously imported these three names directly from
# this module. Keep them importable from here without duplicating logic.
_run_py3dtiles_convert = run_py3dtiles_convert
_write_ply_ecef = write_ply_ecef
_write_colored_ply_ecef = write_colored_ply_ecef


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
            "Check topic names with: ros2 bag info <bag>"
        )

    # -- GPS origin ----------------------------------------------------
    print(f"[1/6] Reading GPS fixes from '{args.gps_topic}'...")
    lat0, lon0, alt0 = parse_gps_fixes(bag_path, args.gps_topic)

    # -- Read bag --------------------------------------------------------
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}

    def handle_pc(message, timestamp: int) -> None:
        pcd = convert_ros_pc2_to_o3d(message)
        if pcd is not None and len(pcd.points) >= 100:
            pointclouds.append((timestamp, pcd))

    def handle_odom(message, timestamp: int) -> None:
        transform = get_odom_transform(message)
        if transform is not None:
            odom_data[timestamp] = transform

    handlers = {args.pc_topic: handle_pc}
    if args.odom_topic:
        handlers[args.odom_topic] = handle_odom

    print(f"\n[2/6] Reading point clouds and odometry from: {bag_path}")
    read_bag_topics(bag_path, handlers, desc="Reading")

    if not pointclouds:
        sys.exit("Error: No valid point clouds extracted from bag.")
    if args.odom_topic and not odom_data:
        print("Warning: --odom_topic set but no messages found; using identity guesses.")
    print(f"  {len(pointclouds)} frames | {len(odom_data)} odom poses")

    # -- ICP + pose graph --------------------------------------------------
    print("\n[3/6] ICP registration + pose-graph optimisation...")
    posegraph, good_idx = run_icp_posegraph(pointclouds, odom_data, args)

    # -- Merge world-frame cloud -------------------------------------------
    print("\n[4/6] Merging registered frames...")
    pcd_combined = o3d.geometry.PointCloud()

    for node_i, pc_i in enumerate(good_idx):
        if node_i >= len(posegraph.nodes):
            break
        T_world = np.linalg.inv(posegraph.nodes[node_i].pose)
        ts, pcd_raw = pointclouds[pc_i]
        pcd_world = transform_frame_to_world(pcd_raw, T_world, args.voxel_size)
        pcd_combined += pcd_world

    # -- Clean -------------------------------------------------------------
    print("\n[5/6] Cleaning merged cloud...")
    pcd_clean = clean_point_cloud(
        pcd_combined, args.voxel_size, do_voxel_downsample=False
    )

    if len(pcd_clean.points) == 0:
        sys.exit("Error: No points remain after cleaning.")

    # -- Georeference + export ---------------------------------------------
    print("\n[6/6] Georeferencing (local ENU -> ECEF) and writing 3D Tiles...")
    tiles_dir, _ = georeference_and_export_tileset(
        pcd_clean, lat0, lon0, alt0, out_dir, bag_path.stem, args.workers
    )

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
