#!/usr/bin/env python3
"""ROS 2 bag -> registered, memory-bounded Poisson mesh.

PATCH NOTE (odom-anchored registration + frame-awareness):
See odom-anchored-registration-fix-prompt.md and
pointcloud-frame-check-prompt-2.md.

- The point cloud's frame_id is detected (or overridden via
  --pc_frame_mode) and classified as 'global' (already in a fixed frame:
  odom/map/world) or 'local' (a moving sensor/base frame).
- 'local' frames are registered via `run_odom_anchored_registration()`,
  which makes timestamped odometry the PRIMARY pose source and demotes ICP
  to an optional, strongly-gated local refinement (--enable_icp_refinement,
  default off). Frames are only ever dropped for lacking odometry coverage,
  never for failing an ICP fitness check.
- 'global' frames skip registration entirely: no ICP, no per-frame
  transform is applied, frames are streamed/filtered/downsampled/merged
  directly. Odom (if provided) is still used for view-ray normal
  orientation, since there is no separate registration transform in this
  branch to create a frame mismatch.
- CAVEAT: the frame check only detects whether the cloud is already in a
  fixed/global frame -- it does NOT correct for a real sensor-to-base_link
  extrinsic offset (lever arm). See `ros_io.classify_frame_mode` docstring.

CLEANUP NOTE: removed an unused `get_closest_timestamp` import left over
from before the view-ray sensor-origin lookup was upgraded to
`interpolate_odom_pose()`.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from .shared.mesh_utils import apply_height_colormap
from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud, create_mesh, level_floor
from .shared.registration import (
    attach_view_rays_as_normals,
    estimate_geometric_normals_oriented,
    run_odom_anchored_registration,
    select_registration_frames,
)
from .shared.ros_io import (
    TYPESTORE,
    convert_ros_pc2_to_o3d,
    get_odom_transform,
    interpolate_odom_pose,
    resolve_pc_frame_mode,
)


def read_registration_data(
    bag_path: Path, args: argparse.Namespace
) -> tuple[list[tuple[int, o3d.geometry.PointCloud]], dict[int, np.ndarray]]:
    """Read voxelized registration frames and optional odometry.

    NOTE: frame_stride is intentionally NOT applied here. Striding is
    applied exactly once, downstream, via
    `registration.select_registration_frames()`, called from `run()`
    before registration. Applying it here as well would silently compound
    with that later selection (e.g. stride=2 here + stride=2 there == an
    effective stride of 4), causing far fewer frames to survive than
    --max_registration_frames would suggest. This function only enforces
    --min_frame_points and a generous read-ahead cap based on
    --max_registration_frames so we do not read an unbounded number of
    frames from very large bags.
    """
    topics = [args.pc_topic]
    if args.odom_topic:
        topics.append(args.odom_topic)

    frames: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}

    stride = max(1, int(args.frame_stride))
    max_registration_frames = max(0, int(args.max_registration_frames))
    # Read-ahead cap: enough raw frames that, after the single stride
    # selection downstream, max_registration_frames can still be reached.
    # 0 means unlimited, so read-ahead is also unlimited in that case.
    read_ahead_limit = max_registration_frames * stride if max_registration_frames else 0

    print(f"Reading registration frames: {bag_path}")
    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for connection, timestamp, raw in tqdm(
            reader.messages(connections=connections), desc="Reading"
        ):
            try:
                message = reader.deserialize(raw, connection.msgtype)

                if args.odom_topic and connection.topic == args.odom_topic:
                    transform = get_odom_transform(message)
                    if transform is not None:
                        odom_data[timestamp] = transform
                    continue

                if connection.topic != args.pc_topic:
                    continue

                cloud = convert_ros_pc2_to_o3d(message)
                if cloud is None or len(cloud.points) < args.min_frame_points:
                    continue
                cloud = cloud.voxel_down_sample(args.voxel_size)
                if len(cloud.points) < args.min_frame_points:
                    continue

                frames.append((timestamp, cloud))
                if read_ahead_limit and len(frames) >= read_ahead_limit:
                    print(f"Registration read-ahead limit reached: {read_ahead_limit}.")
                    break
            except Exception:
                continue

    return frames, odom_data


def append_chunk(
    target: o3d.geometry.PointCloud,
    chunk: list[o3d.geometry.PointCloud],
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Merge a bounded chunk and immediately downsample it."""
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
    return target.voxel_down_sample(voxel_size)


def merge_registered_frames(
    bag_path: Path,
    args: argparse.Namespace,
    pose_by_timestamp: dict[int, np.ndarray],
    odom_data: dict[int, np.ndarray],
) -> tuple[o3d.geometry.PointCloud, int]:
    """Stream registered clouds during a second bag pass.

    Returns (merged_cloud, merged_frame_count) so callers can report exactly
    how many frames made it into the combined cloud, per the end-to-end
    coverage-visibility requirement in odom-anchored-registration-fix-prompt.md.
    """
    odom_max_latency_ns = int(args.odom_max_latency * 1e9)
    odom_timestamps = sorted(odom_data)
    merged = o3d.geometry.PointCloud()
    chunk: list[o3d.geometry.PointCloud] = []
    chunk_size = max(1, int(args.merge_chunk_frames))
    merged_frame_count = 0

    print("Merging registered frames (streaming, bounded memory)...")
    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic == args.pc_topic]
        for connection, timestamp, raw in tqdm(
            reader.messages(connections=connections), desc="Merging"
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
                    # Interpolated (not nearest-neighbor) sensor origin, for
                    # consistency with the odom-anchored pose lookup above.
                    odom_pose = interpolate_odom_pose(
                        timestamp, odom_timestamps, odom_data, odom_max_latency_ns
                    )
                    if odom_pose is not None:
                        attach_view_rays_as_normals(cloud, odom_pose[:3, 3])

                chunk.append(cloud)
                merged_frame_count += 1
                if len(chunk) >= chunk_size:
                    merged = append_chunk(merged, chunk, args.voxel_size)
                    gc.collect()
            except Exception:
                continue

    merged = append_chunk(merged, chunk, args.voxel_size)
    print(f"Merge: {merged_frame_count:,} frame(s) actually merged into the combined cloud.")
    return merged, merged_frame_count


def run(args: argparse.Namespace) -> None:
    """Run registration, reconstruction, and optional height-color export."""
    bag_path = Path(args.bag_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    frame_mode = resolve_pc_frame_mode(bag_path, args.pc_topic, args.pc_frame_mode)

    if frame_mode == "local" and not args.odom_topic:
        sys.exit(
            "Error: --odom_topic is required because the point cloud on "
            f"'{args.pc_topic}' is not already published in a global/fixed "
            "frame (classified as 'local'). Provide --odom_topic pointing at "
            "a nav_msgs/Odometry topic, or pass --pc_frame_mode global if "
            "this bag's point cloud really is already in a fixed frame."
        )

    required = [args.pc_topic]
    if args.odom_topic:
        required.append(args.odom_topic)
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(f"Error: Required topics missing from bag: {missing}")

    frames, odom_data = read_registration_data(bag_path, args)
    if not frames:
        sys.exit("Error: No valid point clouds extracted.")
    print(f"Coverage: frames read = {len(frames):,}.")

    if args.odom_topic and not odom_data:
        print(
            "Warning: odom topic was set but no usable messages were found."
        )

    selected_frames, _original_indices = select_registration_frames(
        frames, frame_stride=args.frame_stride, max_registration_frames=args.max_registration_frames
    )
    print(
        f"Coverage: frames selected = {len(selected_frames):,} / {len(frames):,} "
        f"(stride={args.frame_stride}, max={args.max_registration_frames})."
    )
    del frames
    gc.collect()

    if frame_mode == "global":
        print(
            "Point cloud already in a global/fixed frame: skipping ICP/pose-graph "
            "registration and per-frame transform application; streaming, "
            "filtering, downsampling, and merging directly."
        )
        pose_by_timestamp: dict[int, np.ndarray] = {
            timestamp: np.eye(4, dtype=np.float64) for timestamp, _ in selected_frames
        }
        print(f"Coverage: frames with valid pose = {len(pose_by_timestamp):,} (identity; no registration needed).")
    else:
        print(f"Registering {len(selected_frames)} bounded point-cloud frames (odom-anchored)...")
        pose_by_timestamp, _stats = run_odom_anchored_registration(selected_frames, odom_data, args)

    del selected_frames
    gc.collect()

    if not pose_by_timestamp:
        sys.exit("Error: Registration produced no usable poses.")

    pcd_combined, _merged_frame_count = merge_registered_frames(bag_path, args, pose_by_timestamp, odom_data)
    del pose_by_timestamp
    del odom_data
    gc.collect()

    if not len(pcd_combined.points):
        sys.exit("Error: No registered points were produced.")

    if args.level_floor:
        print("Levelling dominant floor plane...")
        pcd_combined = level_floor(pcd_combined)

    print("Cleaning merged cloud...")
    pcd_clean = clean_point_cloud(pcd_combined, args.voxel_size, do_voxel_downsample=False)
    del pcd_combined
    gc.collect()

    view_rays = (
        np.asarray(pcd_clean.normals, dtype=np.float64).copy()
        if pcd_clean.has_normals()
        else None
    )

    estimate_geometric_normals_oriented(pcd_clean, args.voxel_size, view_rays)

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
    cloud_path = out_dir / f"{stem}_cloud.ply"
    mesh_ply_path = out_dir / f"{stem}_mesh.ply"

    o3d.io.write_point_cloud(str(cloud_path), pcd_clean)
    o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh, write_vertex_normals=True)

    print(f"Saved cloud: {cloud_path}")
    print(f"Saved mesh PLY: {mesh_ply_path}")

    if args.height_colormap:
        colored_ply_path = out_dir / f"{stem}_height_{args.height_colormap}.ply"
        print(f"Applying {args.height_colormap} height false-color (per-vertex PLY)...")
        colored_ply = apply_height_colormap(
            mesh_ply_path,
            colored_ply_path,
            colormap=args.height_colormap,
        )
    print(f"Saved false-color PLY: {colored_ply}")
    
    print("Done.")


def build_parser(sub):
    """Register the mesh subcommand."""
    parser = sub.add_parser(
        "mesh", help="ROS 2 bag -> memory-bounded Poisson mesh"
    )
    parser.add_argument("bag_path", help="Path to the ROS 2 bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--pc_topic", default="points")
    parser.add_argument(
        "--odom_topic",
        default=None,
        help=(
            "Odometry topic. Required unless the point cloud is already "
            "published in a global/fixed frame (see --pc_frame_mode)."
        ),
    )
    parser.add_argument(
        "--pc_frame_mode",
        choices=["auto", "global", "local"],
        default="auto",
        help=(
            "Point-cloud frame handling. 'auto' detects the frame_id of the "
            "first message on --pc_topic and classifies odom/map/world as "
            "global (skip registration) and anything else as local (register "
            "via odom-anchored ICP-refined poses). Use 'global'/'local' to "
            "override when a bag's frame_id is missing, wrong, or empty."
        ),
    )
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
        default=0,
        help=(
            "Maximum retained registration frames (0 means unlimited). "
            "Odom-anchored pose lookup is cheap, so 0 is now the default; "
            "no artificial cap is needed to bound ICP cost."
        ),
    )
    parser.add_argument(
        "--merge_chunk_frames",
        type=int,
        default=16,
        help="Frames merged before each voxel reduction.",
    )
    parser.add_argument("--odom_max_latency", type=float, default=0.5)

    parser.add_argument(
        "--enable_icp_refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional, strongly-gated local ICP refinement of the odom-derived "
            "pose. ICP can only nudge an already-valid pose or be a no-op -- "
            "it can never remove a frame from the merge."
        ),
    )
    parser.add_argument("--icp_dist_thresh", type=float, default=0.2)
    parser.add_argument(
        "--icp_fitness_thresh",
        type=float,
        default=0.7,
        help="Fitness bar for accepting an ICP refinement (raised from 0.6: this is now a correction gate, not the primary motion estimate).",
    )
    parser.add_argument(
        "--max_icp_translation_correction",
        type=float,
        default=0.3,
        help="Max allowed ICP correction translation (meters) relative to the odom guess.",
    )
    parser.add_argument(
        "--max_icp_rotation_correction_deg",
        type=float,
        default=15.0,
        help="Max allowed ICP correction rotation (degrees) relative to the odom guess.",
    )

    parser.add_argument(
        "--enable_loop_closure", action="store_true", default=False
    )
    parser.add_argument("--loop_closure_radius", type=float, default=10.0)
    parser.add_argument(
        "--loop_closure_fitness_thresh",
        type=float,
        default=0.7,
        help="Defaults to the same bar as --icp_fitness_thresh, not a separate looser value.",
    )
    parser.add_argument(
        "--loop_closure_search_interval", type=int, default=10
    )
    parser.add_argument(
        "--loop_closure_temporal_window",
        type=int,
        default=100,
        help="Bounded number of most-recent candidate frames considered for loop closure.",
    )

    parser.add_argument(
        "--poisson_depth",
        type=int,
        default=None,
        help="Omit for automatic depth selection (capped at 11); explicit depth 12 is allowed.",
    )
    parser.add_argument("--min_density_percentile", type=float, default=1.0)
    parser.add_argument("--distance_multiplier", type=float, default=3.0)
    parser.add_argument("--max_vertex_distance", type=float, default=None)
    parser.add_argument(
        "--remesh", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--remesh_smooth_iterations", type=int, default=5)
    parser.add_argument("--decimate_target", type=float, default=None)
    parser.add_argument("--curvature_percentile", type=float, default=80.0)
    parser.add_argument("--curvature_protect_rings", type=int, default=1)
    parser.add_argument("--level_floor", action="store_true", default=False)
    parser.add_argument(
        "--height_colormap",
        choices=["jet", "hot", "cool", "gray"],
        default=None,
        help=(
            "Also export a textured false-color OBJ, MTL, and PNG bundle "
            "whose color represents mesh height."
        ),
    )

    parser.add_argument("--workers", type=int, default=2)
    parser.set_defaults(func=run)
    return parser
