#!/usr/bin/env python3
"""ROS 2 bag -> registered, memory-bounded Poisson mesh (.ply + .obj)."""

from __future__ import annotations

import argparse
import gc
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
    get_closest_timestamp,
    get_odom_transform,
)


def _read_registration_data(bag_path: Path, args):
    """Read bounded, voxelized registration frames and odometry."""
    topics = [args.pc_topic] + ([args.odom_topic] if args.odom_topic else [])
    frames: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}
    seen_clouds = 0
    stride = max(1, int(args.frame_stride))
    limit = int(args.max_registration_frames) if args.max_registration_frames else 0

    print(f"Reading registration frames: {bag_path}")

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in topics
        ]

        for connection, timestamp, raw in tqdm(
            reader.messages(connections=connections),
            desc="Reading",
        ):
            try:
                message = reader.deserialize(raw, connection.msgtype)

                if connection.topic == args.odom_topic:
                    transform = get_odom_transform(message)
                    if transform is not None:
                        odom_data[timestamp] = transform
                    continue

                if connection.topic != args.pc_topic:
                    continue

                seen_clouds += 1

                if (seen_clouds - 1) % stride:
                    continue

                cloud = convert_ros_pc2_to_o3d(message)

                if cloud is None or len(cloud.points) < args.min_frame_points:
                    continue

                cloud = cloud.voxel_down_sample(args.voxel_size)

                if len(cloud.points) < args.min_frame_points:
                    continue

                frames.append((timestamp, cloud))

                if limit and len(frames) >= limit:
                    print(f"Registration frame limit reached ({limit}).")
                    break

            except Exception:
                continue

    return frames, odom_data


def _append_chunk(
    target: o3d.geometry.PointCloud,
    chunk: list[o3d.geometry.PointCloud],
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Merge one short chunk and voxelize immediately to bound memory."""
    if not chunk:
        return target

    local = o3d.geometry.PointCloud()

    for cloud in chunk:
        local += cloud

    chunk.clear()

    if len(target.points):
        target += local
    else:
        target = local

    # Repeated reduction prevents the merged cloud from growing without bound.
    return target.voxel_down_sample(voxel_size)


def _merge_registered_frames(
    bag_path: Path,
    args,
    pose_by_timestamp: dict[int, np.ndarray],
    odom_data: dict[int, np.ndarray],
) -> o3d.geometry.PointCloud:
    """Second pass: transform one cloud at a time, then discard it."""
    odom_timestamps = sorted(odom_data)
    merged = o3d.geometry.PointCloud()
    chunk: list[o3d.geometry.PointCloud] = []
    chunk_size = max(1, int(args.merge_chunk_frames))

    print("Merging registered frames (streaming, bounded memory)...")

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == args.pc_topic
        ]

        for connection, timestamp, raw in tqdm(
            reader.messages(connections=connections),
            desc="Merging",
        ):
            transform = pose_by_timestamp.get(timestamp)

            if transform is None:
                continue

            try:
                message = reader.deserialize(raw, connection.msgtype)
                cloud = convert_ros_pc2_to_o3d(message)

                if cloud is None or len(cloud.points) < args.min_frame_points:
                    continue

                cloud = cloud.voxel_down_sample(args.voxel_size)

                if len(cloud.points) < args.min_frame_points:
                    continue

                cloud.transform(transform)

                if odom_timestamps:
                    closest = get_closest_timestamp(timestamp, odom_timestamps)

                    if (
                        closest is not None
                        and abs(closest - timestamp)
                        <= int(args.odom_max_latency * 1e9)
                    ):
                        attach_view_rays_as_normals(
                            cloud,
                            odom_data[closest][:3, 3],
                        )

                chunk.append(cloud)

                if len(chunk) >= chunk_size:
                    merged = _append_chunk(
                        merged,
                        chunk,
                        args.voxel_size,
                    )
                    gc.collect()

            except Exception:
                continue

    return _append_chunk(merged, chunk, args.voxel_size)


def run(args) -> None:
    bag_path = Path(args.bagpath)
    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    required = [args.pc_topic] + (
        [args.odom_topic] if args.odom_topic else []
    )
    missing = check_topics(bag_path, required)

    if missing:
        sys.exit(f"Error: Required topics missing from bag: {missing}")

    frames, odom_data = _read_registration_data(bag_path, args)

    if not frames:
        sys.exit("Error: No valid point clouds extracted.")

    if args.odom_topic and not odom_data:
        print(
            "Warning: odom topic set but no usable messages were found; "
            "using ICP-only guesses."
        )

    print(f"Registering {len(frames)} bounded point-cloud frames...")
    posegraph, good_indices = run_icp_posegraph(
        frames,
        odom_data,
        args,
    )

    # `good_indices` and posegraph nodes are parallel in the registration API.
    pose_by_timestamp: dict[int, np.ndarray] = {}

    for node_index, frame_index in enumerate(good_indices):
        if node_index >= len(posegraph.nodes):
            break

        if frame_index >= len(frames):
            continue

        timestamp = frames[frame_index][0]
        pose_by_timestamp[timestamp] = np.linalg.inv(
            posegraph.nodes[node_index].pose
        )

    # Release all registration geometry before the streaming merge pass.
    del frames
    del posegraph
    del good_indices
    gc.collect()

    if not pose_by_timestamp:
        sys.exit("Error: Registration produced no usable poses.")

    pcd_combined = _merge_registered_frames(
        bag_path,
        args,
        pose_by_timestamp,
        odom_data,
    )

    del pose_by_timestamp
    del odom_data
    gc.collect()

    if not len(pcd_combined.points):
        sys.exit("Error: No registered points were produced.")

    print("Cleaning merged cloud...")

    if args.level_floor:
        pcd_combined = level_floor(pcd_combined)

    pcd_clean = clean_point_cloud(
        pcd_combined,
        args.voxel_size,
        do_voxel_downsample=False,
    )

    del pcd_combined
    gc.collect()

    view_rays = (
        np.asarray(pcd_clean.normals, dtype=np.float64).copy()
        if pcd_clean.has_normals()
        else None
    )

    estimate_geometric_normals_oriented(
        pcd_clean,
        args.voxel_size,
        view_rays,
    )

    print("Running Poisson reconstruction...")

    mesh = create_mesh(
        pcd_clean,
        poisson_depth=args.poisson_depth,
        min_density_percentile=args.min_density_percentile,
        distance_multiplier=args.distance_multiplier,
        max_vertex_distance=args.max_vertex_distance,
        remesh=args.remesh,
        remesh_smooth_iterations=args.remesh_smooth_iterations,
        workers=args.workers,
        decimate_target=args.decimate_target,
        curvature_percentile=args.curvature_percentile,
        curvature_protect_rings=args.curvature_protect_rings,
    )

    stem = bag_path.stem
    ply_path = out_dir / f"{stem}_cloud.ply"
    obj_path = out_dir / f"{stem}_mesh.obj"

    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)

    print(f"Saved cloud: {ply_path}")
    print(f"Saved mesh: {obj_path}")
    print("Done.")


def build_parser(sub):
    parser = sub.add_parser(
        "mesh",
        help="ROS 2 bag -> memory-bounded Poisson mesh (.ply + .obj)",
    )

    parser.add_argument("bagpath", help="Path to the ROS 2 bag.")
    parser.add_argument("outputdir", help="Output directory.")

    parser.add_argument("--pc_topic", default="points")
    parser.add_argument("--odom_topic", default=None)

    parser.add_argument("--voxel_size", type=float, default=0.10)
    parser.add_argument("--min_frame_points", type=int, default=100)

    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Use every Nth cloud for registration and merging.",
    )

    parser.add_argument(
        "--max_registration_frames",
        type=int,
        default=500,
        help="Maximum retained registration frames; 0 means unlimited.",
    )

    parser.add_argument(
        "--merge_chunk_frames",
        type=int,
        default=16,
        help="Frames merged before each voxel reduction.",
    )

    parser.add_argument("--icp_dist_thresh", type=float, default=0.2)
    parser.add_argument("--icp_fitness_thresh", type=float, default=0.6)
    parser.add_argument("--odom_max_latency", type=float, default=0.5)

    parser.add_argument(
        "--enable_loop_closure",
        action="store_true",
        default=False,
    )

    parser.add_argument("--loop_closure_radius", type=float, default=10.0)
    parser.add_argument(
        "--loop_closure_fitness_thresh",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--loop_closure_search_interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--poisson_depth",
        type=int,
        default=None,
        help=(
            "Omit for automatic depth selection capped at 11; "
            "explicit depth 12+ is allowed."
        ),
    )

    parser.add_argument(
        "--min_density_percentile",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--distance_multiplier",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--max_vertex_distance",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--remesh",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--remesh_smooth_iterations",
        type=int,
        default=5,
    )

    parser.add_argument("--decimate_target", type=float, default=None)

    parser.add_argument(
        "--curvature_percentile",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--curvature_protect_rings",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--level_floor",
        action="store_true",
        default=False,
    )

    parser.add_argument("--workers", type=int, default=2)

    parser.set_defaults(func=run)
    return parser
