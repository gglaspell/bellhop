#!/usr/bin/env python3
"""ROS 2 bag -> memory-bounded camera-coloured Poisson mesh."""

from __future__ import annotations

import argparse
import gc
import sys
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from rosbags.highlevel import AnyReader
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
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
    convert_ros_image,
    convert_ros_pc2_to_o3d,
    get_closest_timestamp,
    get_odom_transform,
    intrinsics_from_camera_info,
)


def _color_pcd_from_image(
    pcd: o3d.geometry.PointCloud,
    image: Image.Image,
    camera_pose: np.ndarray,
    intrinsics: tuple,
    min_depth: float,
    max_depth: float | None,
) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy, width, height = intrinsics
    points = np.asarray(pcd.points)
    image_array = np.asarray(image)

    rotation = R.from_matrix(camera_pose[:3, :3])
    body = rotation.inv().apply(points - camera_pose[:3, 3])

    optical_x = -body[:, 1]
    optical_y = -body[:, 2]
    optical_z = body[:, 0]
    distance = np.linalg.norm(body, axis=1)

    valid = (optical_z > 1e-6) & (distance >= min_depth)

    if max_depth is not None:
        valid &= distance <= max_depth

    safe_z = np.where(optical_z > 1e-6, optical_z, 1e-6)
    u = fx * optical_x / safe_z + cx
    v = fy * optical_y / safe_z + cy

    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

    colors = np.full((len(points), 3), 0.5, dtype=np.float64)

    if np.any(valid):
        colors[valid] = (
            image_array[
                np.clip(v[valid].astype(np.int32), 0, height - 1),
                np.clip(u[valid].astype(np.int32), 0, width - 1),
                :3,
            ]
            / 255.0
        )

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _remove_local_gray_fill(
    pcd: o3d.geometry.PointCloud,
    radius: float,
) -> o3d.geometry.PointCloud:
    """Remove neutral placeholder colour only near genuine coloured points."""
    if radius <= 0 or not pcd.has_colors() or not len(pcd.points):
        return pcd

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    neutral = (
        (np.std(colors, axis=1) < 0.08)
        & (np.abs(np.mean(colors, axis=1) - 0.5) < 0.15)
    )

    colored_points = points[~neutral]

    if not len(colored_points) or not neutral.any():
        return pcd

    neighbors = cKDTree(colored_points).query_ball_point(
        points[neutral],
        radius,
    )

    near_color = np.array(
        [len(neighbors_for_point) > 0 for neighbors_for_point in neighbors],
        dtype=bool,
    )

    keep = np.ones(len(points), dtype=bool)
    keep[np.flatnonzero(neutral)[near_color]] = False

    return pcd.select_by_index(np.flatnonzero(keep))


def _merge_chunk(
    target: o3d.geometry.PointCloud,
    chunk: list[o3d.geometry.PointCloud],
    voxel_size: float,
    gray_filter_radius: float,
) -> o3d.geometry.PointCloud:
    """Merge a bounded chunk and voxelize immediately."""
    if not chunk:
        return target

    local = o3d.geometry.PointCloud()

    for cloud in chunk:
        local += cloud

    chunk.clear()

    local = _remove_local_gray_fill(local, gray_filter_radius)

    if len(target.points):
        target += local
    else:
        target = local

    return target.voxel_down_sample(voxel_size)


def _read_registration_data(
    bag_path: Path,
    args,
) -> tuple[list[tuple[int, o3d.geometry.PointCloud]], dict[int, np.ndarray]]:
    """First pass: retain only bounded, voxelized registration scans."""
    topics = [args.pc_topic] + (
        [args.odom_topic] if args.odom_topic else []
    )

    frames: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}

    seen_clouds = 0
    frame_stride = max(1, int(args.frame_stride))
    frame_limit = int(args.max_registration_frames)

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

                if (seen_clouds - 1) % frame_stride:
                    continue

                if frame_limit and len(frames) >= frame_limit:
                    continue

                cloud = convert_ros_pc2_to_o3d(message)

                if cloud is None or len(cloud.points) < args.min_frame_points:
                    continue

                cloud = cloud.voxel_down_sample(args.voxel_size)

                if len(cloud.points) >= args.min_frame_points:
                    frames.append((timestamp, cloud))

            except Exception:
                continue

    return frames, odom_data


def _nearest_image(
    timestamp: int,
    images: deque[tuple[int, Image.Image]],
) -> tuple[int, Image.Image] | None:
    if not images:
        return None

    return min(images, key=lambda item: abs(item[0] - timestamp))


def _stream_colored_merge(
    bag_path: Path,
    args,
    poses: dict[int, np.ndarray],
    odom_data: dict[int, np.ndarray],
    initial_intrinsics: tuple | None,
) -> o3d.geometry.PointCloud:
    """
    Second pass: stream images and selected clouds in time order.

    Only a small rolling image buffer and one merge chunk are retained.
    """
    topics = [
        args.pc_topic,
        args.camera_topic,
        args.camera_info_topic,
    ]

    max_time_delta_ns = int(args.max_time_diff * 1e9)
    odom_timestamps = sorted(odom_data)
    images: deque[tuple[int, Image.Image]] = deque()
    pending_clouds: deque[tuple[int, o3d.geometry.PointCloud]] = deque()

    intrinsics = initial_intrinsics
    merged = o3d.geometry.PointCloud()
    chunk: list[o3d.geometry.PointCloud] = []
    chunk_size = max(1, int(args.merge_chunk_frames))

    def flush(until_timestamp: int, final: bool = False) -> None:
        nonlocal merged

        while pending_clouds and (
            final
            or pending_clouds[0][0] + max_time_delta_ns <= until_timestamp
        ):
            cloud_timestamp, cloud = pending_clouds.popleft()
            image_item = _nearest_image(cloud_timestamp, images)

            if (
                image_item is not None
                and intrinsics is not None
                and abs(image_item[0] - cloud_timestamp) <= max_time_delta_ns
            ):
                camera_pose = np.eye(4)

                if odom_timestamps:
                    closest_odom = get_closest_timestamp(
                        cloud_timestamp,
                        odom_timestamps,
                    )

                    if (
                        closest_odom is not None
                        and abs(closest_odom - cloud_timestamp)
                        <= int(args.odom_max_latency * 1e9)
                    ):
                        camera_pose = odom_data[closest_odom]

                cloud = _color_pcd_from_image(
                    cloud,
                    image_item[1],
                    camera_pose,
                    intrinsics,
                    args.color_min_depth,
                    args.color_max_depth,
                )

            chunk.append(cloud)

            if len(chunk) >= chunk_size:
                merged = _merge_chunk(
                    merged,
                    chunk,
                    args.voxel_size,
                    args.gray_filter_radius,
                )
                gc.collect()

        if pending_clouds:
            oldest_needed = pending_clouds[0][0] - max_time_delta_ns

            while images and images[0][0] < oldest_needed:
                images.popleft()

    print("Merging and colouring registered frames (streaming, bounded memory)...")

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in topics
        ]

        for connection, timestamp, raw in tqdm(
            reader.messages(connections=connections),
            desc="Merging",
        ):
            try:
                message = reader.deserialize(raw, connection.msgtype)

                if connection.topic == args.camera_info_topic:
                    candidate = intrinsics_from_camera_info(message)

                    if candidate is not None:
                        intrinsics = candidate

                    continue

                if connection.topic == args.camera_topic:
                    image = convert_ros_image(message)

                    if image is not None:
                        images.append((timestamp, image))
                        flush(timestamp)

                    continue

                if connection.topic != args.pc_topic:
                    continue

                transform = poses.get(timestamp)

                if transform is None:
                    continue

                cloud = convert_ros_pc2_to_o3d(message)

                if cloud is None or len(cloud.points) < args.min_frame_points:
                    continue

                cloud = cloud.voxel_down_sample(args.voxel_size)

                if len(cloud.points) < args.min_frame_points:
                    continue

                cloud.transform(transform)

                if odom_timestamps:
                    closest_odom = get_closest_timestamp(
                        timestamp,
                        odom_timestamps,
                    )

                    if (
                        closest_odom is not None
                        and abs(closest_odom - timestamp)
                        <= int(args.odom_max_latency * 1e9)
                    ):
                        attach_view_rays_as_normals(
                            cloud,
                            odom_data[closest_odom][:3, 3],
                        )

                pending_clouds.append((timestamp, cloud))
                flush(timestamp)

            except Exception:
                continue

    flush(0, final=True)

    return _merge_chunk(
        merged,
        chunk,
        args.voxel_size,
        args.gray_filter_radius,
    )


def run(args) -> None:
    bag_path = Path(args.bagpath)
    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    required = [
        args.pc_topic,
        args.camera_topic,
        args.camera_info_topic,
    ] + ([args.odom_topic] if args.odom_topic else [])

    missing = check_topics(bag_path, required)

    if missing:
        sys.exit(f"Error: Required topics missing from bag: {missing}")

    frames, odom_data = _read_registration_data(bag_path, args)

    if not frames:
        sys.exit("Error: No valid point clouds extracted.")

    print(f"Registering {len(frames)} bounded point-cloud frames...")

    posegraph, good_indices = run_icp_posegraph(
        frames,
        odom_data,
        args,
    )

    poses: dict[int, np.ndarray] = {}

    for node_index, frame_index in enumerate(good_indices):
        if node_index >= len(posegraph.nodes):
            break

        if frame_index >= len(frames):
            continue

        timestamp = frames[frame_index][0]
        poses[timestamp] = np.linalg.inv(
            posegraph.nodes[node_index].pose
        )

    del frames
    del posegraph
    del good_indices
    gc.collect()

    if not poses:
        sys.exit("Error: Registration produced no usable poses.")

    pcd_combined = _stream_colored_merge(
        bag_path,
        args,
        poses,
        odom_data,
        None,
    )

    del poses
    del odom_data
    gc.collect()

    if not len(pcd_combined.points):
        sys.exit("Error: No registered points were produced.")

    if args.level_floor:
        pcd_combined = level_floor(pcd_combined)

    print("Cleaning merged cloud...")

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
    ply_path = out_dir / f"{stem}_colored_cloud.ply"
    obj_path = out_dir / f"{stem}_colored_mesh.obj"

    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)

    print(f"Saved cloud: {ply_path}")
    print(f"Saved mesh: {obj_path}")
    print("Done.")


def build_parser(sub):
    parser = sub.add_parser(
        "color_mesh",
        help="ROS 2 bag -> memory-bounded camera-coloured Poisson mesh",
    )

    parser.add_argument("bagpath", help="Path to the ROS 2 bag.")
    parser.add_argument("outputdir", help="Output directory.")

    parser.add_argument("--pc_topic", default="points")
    parser.add_argument("--odom_topic", default=None)

    parser.add_argument("--camera_topic", required=True)
    parser.add_argument("--camera_info_topic", required=True)

    parser.add_argument("--max_time_diff", type=float, default=0.1)
    parser.add_argument("--color_min_depth", type=float, default=0.1)
    parser.add_argument("--color_max_depth", type=float, default=None)
    parser.add_argument("--gray_filter_radius", type=float, default=0.05)

    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--min_frame_points", type=int, default=100)

    parser.add_argument(
        "--frame_stride",
        type=int,
        default=4,
        help="Use every Nth cloud for registration and colouring.",
    )

    parser.add_argument(
        "--max_registration_frames",
        type=int,
        default=0,
        help="Maximum registration frames; 0 means unlimited.",
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

    parser.add_argument("--workers", type=int, default=1)

    parser.set_defaults(func=run)
    return parser
