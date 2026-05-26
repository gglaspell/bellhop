#!/usr/bin/env python3
"""
mesh.py – ROS 2 bag -> registered point cloud -> Poisson mesh (.ply + .obj).
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud, create_mesh, level_floor
from .shared.registration import (
    attach_view_rays_as_normals,
    estimate_geometric_normals_oriented,
    run_icp_posegraph,
)
from .shared.ros_io import (
    TYPESTORE,
    convert_ros_pc2_to_o3d,
    get_odom_transform,
)


def run(args) -> None:
    bag_path = Path(args.bagpath)
    out_dir  = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    required = [args.pc_topic] + ([args.odom_topic] if args.odom_topic else [])
    missing  = check_topics(bag_path, required)
    if missing:
        sys.exit(f"Error: Required topics missing from bag: {missing}")

    # ── Read bag ──────────────────────────────────────────────────────────
    topics = [args.pc_topic] + ([args.odom_topic] if args.odom_topic else [])
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}

    print(f"Reading: {bag_path}")
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
        sys.exit("Error: No valid point clouds extracted.")
    if args.odom_topic and not odom_data:
        print("Warning: odom topic set but no messages found; using identity guesses.")

    # ── ICP + pose graph ──────────────────────────────────────────────────
    posegraph, good_idx = run_icp_posegraph(pointclouds, odom_data, args)
    odom_max_ns = int(args.odom_max_latency * 1e9)

    # ── Merge with view-ray normals ───────────────────────────────────────
    print("Merging registered frames with view-ray normals...")
    pcd_combined = o3d.geometry.PointCloud()
    odom_ts_sorted = sorted(odom_data.keys())

    for node_i, pc_i in enumerate(good_idx):
        if node_i >= len(posegraph.nodes):
            break
        T_world = np.linalg.inv(posegraph.nodes[node_i].pose)
        ts, pcd_raw = pointclouds[pc_i]

        pcd_world = pcd_raw.voxel_down_sample(args.voxel_size)
        pcd_world.transform(T_world)

        from .shared.ros_io import get_closest_timestamp
        if odom_ts_sorted:
            cts = get_closest_timestamp(ts, odom_ts_sorted)
            if cts is not None and abs(cts - ts) < odom_max_ns:
                sensor_origin = odom_data[cts][:3, 3]
                attach_view_rays_as_normals(pcd_world, sensor_origin)

        pcd_combined += pcd_world

    # ── Clean ─────────────────────────────────────────────────────────────
    print("Cleaning merged cloud...")
    if args.level_floor:
        pcd_combined = level_floor(pcd_combined)

    pcd_clean = clean_point_cloud(pcd_combined, args.voxel_size, do_voxel_downsample=False)

    # Extract view rays before estimating geometric normals
    view_rays = (
        np.asarray(pcd_clean.normals, dtype=np.float64).copy()
        if pcd_clean.has_normals() else None
    )
    estimate_geometric_normals_oriented(pcd_clean, args.voxel_size, view_rays)

    # ── Reconstruct ───────────────────────────────────────────────────────
    print("Running Poisson reconstruction...")
    mesh = create_mesh(
        pcd_clean,
        poisson_depth=args.poisson_depth,
        min_density_percentile=args.min_density_percentile,
        max_vertex_distance=args.max_vertex_distance,
        workers=args.workers,
        decimate_target=args.decimate_target,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    stem = bag_path.stem
    ply_path = out_dir / f"{stem}_cloud.ply"
    obj_path = out_dir / f"{stem}_mesh.obj"
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    print(f"Saved cloud: {ply_path}")
    print(f"Saved mesh:  {obj_path}")
    print("Done.")


def build_parser(sub):
    p = sub.add_parser("mesh", help="ROS 2 bag -> Poisson mesh (.ply + .obj)")
    p.add_argument("bagpath",   help="Path to the ROS 2 bag.")
    p.add_argument("outputdir", help="Output directory.")
    p.add_argument("--pc_topic",   default="points")
    p.add_argument("--odom_topic", default=None)
    p.add_argument("--voxel_size",           type=float, default=0.05)
    p.add_argument("--icp_dist_thresh",      type=float, default=0.2)
    p.add_argument("--icp_fitness_thresh",   type=float, default=0.6)
    p.add_argument("--odom_max_latency",     type=float, default=0.5)
    p.add_argument("--enable_loop_closure",  action="store_true", default=False)
    p.add_argument("--loop_closure_radius",          type=float, default=10.0)
    p.add_argument("--loop_closure_fitness_thresh",  type=float, default=0.3)
    p.add_argument("--loop_closure_search_interval", type=int,   default=10)
    p.add_argument("--poisson_depth",          type=int,   default=9)
    p.add_argument("--min_density_percentile", type=float, default=1.0,
                   help="Bottom %% of Poisson vertex densities to remove (default 1.0).")
    p.add_argument("--max_vertex_distance",    type=float, default=0.15)
    p.add_argument("--decimate_target",        type=float, default=None)
    p.add_argument("--level_floor",            action="store_true", default=False)
    p.add_argument("--workers",                type=int,   default=4)
    p.set_defaults(func=run)
    return p
