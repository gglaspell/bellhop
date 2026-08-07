#!/usr/bin/env python3
"""ROS 2 bag -> memory-bounded camera-coloured Poisson mesh.

PATCH NOTE (odom-anchored registration + frame-awareness):
See odom-anchored-registration-fix-prompt.md and
pointcloud-frame-check-prompt-2.md. Mirrors the same change applied to
`mesh.py`, since both pipelines share the same registration/merge pattern:

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
  (and camera-coloured) directly. Odom (if provided) is still used to
  locate the camera origin for projecting images onto the cloud and for
  view-ray normal orientation, since there is no separate registration
  transform in this branch to create a frame mismatch.
- CAVEAT: the frame check only detects whether the cloud is already in a
  fixed/global frame -- it does NOT correct for a real sensor-to-base_link
  extrinsic offset (lever arm). See `ros_io.classify_frame_mode` docstring.

PATCH NOTE (total color loss fix, see fix_pointcloud_color_loss_prompt.md):
- Root cause 1: `_color_pcd_from_image` was only invoked inside the
  "image + intrinsics found in time" branch of `flush()`, with no `else`.
  Frames that missed that condition entered the merge with
  `pcd.has_colors() == False`. Fixed: every frame now gets an explicit
  neutral-gray fallback color when it can't be camera-colored, so it always
  has a colors array of the correct length before merging.
- Root cause 2: `_merge_chunk` combined clouds with `+=`/`+`. Open3D's
  `PointCloud.__add__`/`__iadd__` clears color on the ENTIRE result if
  either operand lacks a colors array -- so a single frame missing colors
  (root cause 1) would wipe out coloring for everything merged after it,
  which is exactly why the final mesh/cloud was totally colorless instead
  of just patchy. Fixed: merging is now done via manual numpy
  concatenation + explicit Vector3dVector assignment, which never invokes
  Open3D's add operators and is therefore immune to this behavior.
- Added a running/report count of frames colored from the camera vs.
  frames that fell back to gray, printed at the end of the merge pass, so
  a future regression is visible immediately as "N/M frames used fallback
  gray" instead of resurfacing as silent total color loss.
- Root cause 3 (checked, not applicable): `convert_ros_pc2_to_o3d` in
  ros_io.py only extracts x/y/z from PointCloud2, never a native RGB
  field. This pipeline is intentionally camera-projection-colored (there
  is no RGB field expected on the lidar topic), so this is not a bug here
  -- flagging per the prompt's instruction to check it regardless.
- Root cause 4 (flagged, not silently changed): output mesh is `.obj`.
  `write_triangle_mesh` was never passed `write_vertex_colors=True`, which
  is now added -- but standard OBJ has no native per-vertex-color field;
  Open3D writes a non-standard `v x y z r g b` extension line that many
  viewers/importers (e.g. Blender's default OBJ importer) silently ignore.
  A warning is now printed at save time. If color fidelity in common
  viewers matters, prefer exporting `.ply` for the mesh too (the point
  cloud already is), or bake colors into a UV texture instead.
"""

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
    run_odom_anchored_registration,
    select_registration_frames,
)
from .shared.ros_io import (
    TYPESTORE,
    convert_ros_image,
    convert_ros_pc2_to_o3d,
    get_odom_transform,
    interpolate_odom_pose,
    intrinsics_from_camera_info,
    resolve_pc_frame_mode,
)

# Neutral fallback color used for any point/frame that could not be
# camera-colored (no image in time tolerance, no intrinsics yet, or a
# point that didn't project into the image). Keeping this as a single
# constant means Root Cause 1's fallback and _color_pcd_from_image's
# per-point fallback always agree, which also keeps _remove_local_gray_fill
# (which detects "neutral" colors near this value) correct.
FALLBACK_GRAY = (0.5, 0.5, 0.5)


def _fill_fallback_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Guarantee `pcd` has an explicit colors array before it can be merged.

    This is the Root Cause 1 fix: any frame that skips camera projection
    (no image found in time, or intrinsics not parsed yet) must still get
    an explicit neutral-gray colors array of matching length, so it never
    reaches `_merge_chunk` with `has_colors() == False`.
    """
    colors = np.full((len(pcd.points), 3), FALLBACK_GRAY, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


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

    colors = np.full((len(points), 3), FALLBACK_GRAY, dtype=np.float64)

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


def _concat_point_clouds(clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """Merge point clouds via manual array concatenation.

    Root Cause 2 fix: Open3D's `PointCloud.__add__`/`__iadd__` clears the
    ENTIRE result's colors if either operand lacks a colors array (or the
    arrays mismatch length). Since merges here happen incrementally in a
    loop, a single frame without colors would silently destroy color for
    everything merged before/after it. This function never touches
    `+`/`+=`, so it's immune to that behavior. Root Cause 1's fallback
    coloring guarantees every incoming cloud already has a `.colors` array
    of matching length, so `np.vstack` here is always safe.
    """
    non_empty = [c for c in clouds if len(c.points)]

    if not non_empty:
        return o3d.geometry.PointCloud()

    points = np.vstack([np.asarray(c.points) for c in non_empty])
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(points)

    if all(c.has_colors() for c in non_empty):
        colors = np.vstack([np.asarray(c.colors) for c in non_empty])
        merged.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Should not happen once every frame is fallback-colored before
        # reaching this function, but guard against it defensively rather
        # than silently emitting an uncolored merged cloud.
        merged = _fill_fallback_color(merged)

    return merged


def _merge_chunk(
    target: o3d.geometry.PointCloud,
    chunk: list[o3d.geometry.PointCloud],
    voxel_size: float,
    gray_filter_radius: float,
) -> o3d.geometry.PointCloud:
    """Merge a bounded chunk and voxelize immediately."""
    if not chunk:
        return target

    local = _concat_point_clouds(chunk)
    chunk.clear()

    local = _remove_local_gray_fill(local, gray_filter_radius)

    if len(target.points):
        target = _concat_point_clouds([target, local])
    else:
        target = local

    return target.voxel_down_sample(voxel_size)


def _read_registration_data(
    bag_path: Path,
    args,
) -> tuple[list[tuple[int, o3d.geometry.PointCloud]], dict[int, np.ndarray]]:
    """First pass: retain bounded, voxelized registration scans.

    NOTE: frame_stride is intentionally NOT applied here. Striding is
    applied exactly once, downstream, via
    `registration.select_registration_frames()`, called from `run()`
    before registration. Applying it here as well would silently compound
    with that later selection (e.g. stride=4 here + stride=4 there == an
    effective stride of 16), causing far fewer frames to survive than
    --max_registration_frames would suggest. This function only enforces
    --min_frame_points and a generous read-ahead cap derived from
    --max_registration_frames so very large bags are still bounded.
    """
    topics = [args.pc_topic] + (
        [args.odom_topic] if args.odom_topic else []
    )

    frames: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}

    stride = max(1, int(args.frame_stride))
    max_registration_frames = max(0, int(args.max_registration_frames))
    read_ahead_limit = (
        max_registration_frames * stride if max_registration_frames else 0
    )

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

                cloud = convert_ros_pc2_to_o3d(message)

                if cloud is None or len(cloud.points) < args.min_frame_points:
                    continue

                cloud = cloud.voxel_down_sample(args.voxel_size)

                if len(cloud.points) < args.min_frame_points:
                    continue

                frames.append((timestamp, cloud))

                if read_ahead_limit and len(frames) >= read_ahead_limit:
                    print(
                        f"Registration read-ahead limit reached: {read_ahead_limit}."
                    )
                    break
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
) -> tuple[o3d.geometry.PointCloud, int]:
    """
    Second pass: stream images and selected clouds in time order.

    Only a small rolling image buffer and one merge chunk are retained.

    Returns (merged_cloud, merged_frame_count) so callers can report exactly
    how many frames made it into the combined cloud (end-to-end coverage
    visibility per odom-anchored-registration-fix-prompt.md).

    Every frame that reaches `chunk.append(cloud)` below now has an
    explicit `.colors` array -- either real camera-projected color, or an
    explicit fallback gray (fix_pointcloud_color_loss_prompt.md, Root
    Cause 1) -- so downstream merging can never silently strip color
    (Root Cause 2, fixed in `_concat_point_clouds`/`_merge_chunk`).
    """
    topics = [
        args.pc_topic,
        args.camera_topic,
        args.camera_info_topic,
    ]

    max_time_delta_ns = int(args.max_time_diff * 1e9)
    odom_max_latency_ns = int(args.odom_max_latency * 1e9)
    odom_timestamps = sorted(odom_data)
    images: deque[tuple[int, Image.Image]] = deque()
    pending_clouds: deque[tuple[int, o3d.geometry.PointCloud]] = deque()

    intrinsics = initial_intrinsics
    merged = o3d.geometry.PointCloud()
    chunk: list[o3d.geometry.PointCloud] = []
    chunk_size = max(1, int(args.merge_chunk_frames))
    merged_frame_count = 0
    camera_colored_count = 0
    fallback_gray_count = 0

    def flush(until_timestamp: int, final: bool = False) -> None:
        nonlocal merged, merged_frame_count, camera_colored_count, fallback_gray_count

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
                    # Interpolated (linear translation + SLERP rotation)
                    # camera origin, not nearest-neighbor snapping.
                    interpolated = interpolate_odom_pose(
                        cloud_timestamp, odom_timestamps, odom_data, odom_max_latency_ns
                    )

                    if interpolated is not None:
                        camera_pose = interpolated

                cloud = _color_pcd_from_image(
                    cloud,
                    image_item[1],
                    camera_pose,
                    intrinsics,
                    args.color_min_depth,
                    args.color_max_depth,
                )
                camera_colored_count += 1
            else:
                # Root Cause 1 fix: no image/intrinsics available in time
                # for this frame. Previously this branch simply fell
                # through with no colors assigned at all, which meant the
                # frame reached `_merge_chunk` with `has_colors() ==
                # False` and (via Root Cause 2) wiped color from the
                # entire merged cloud. Explicitly fall back to gray so the
                # frame always has a valid colors array.
                cloud = _fill_fallback_color(cloud)
                fallback_gray_count += 1

            chunk.append(cloud)
            merged_frame_count += 1

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
                    interpolated = interpolate_odom_pose(
                        timestamp, odom_timestamps, odom_data, odom_max_latency_ns
                    )

                    if interpolated is not None:
                        attach_view_rays_as_normals(cloud, interpolated[:3, 3])

                pending_clouds.append((timestamp, cloud))
                flush(timestamp)
            except Exception:
                continue

    flush(0, final=True)

    merged = _merge_chunk(
        merged,
        chunk,
        args.voxel_size,
        args.gray_filter_radius,
    )

    print(f"Merge: {merged_frame_count:,} frame(s) actually merged into the combined cloud.")
    print(
        f"Colour coverage: {camera_colored_count:,} frame(s) camera-colored, "
        f"{fallback_gray_count:,} frame(s) used fallback gray "
        f"({camera_colored_count}/{merged_frame_count} real color coverage)."
    )

    if merged_frame_count and fallback_gray_count == merged_frame_count:
        print(
            "Warning: EVERY merged frame fell back to gray -- no frame was ever "
            "camera-colored. Check --camera_topic/--camera_info_topic, "
            "--max_time_diff, and that CameraInfo messages actually precede "
            "point cloud frames in the bag."
        )

    return merged, merged_frame_count


def run(args) -> None:
    bag_path = Path(args.bagpath)
    out_dir = Path(args.outputdir)
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
            "filtering, downsampling, colouring, and merging directly."
        )
        poses: dict[int, np.ndarray] = {
            timestamp: np.eye(4, dtype=np.float64) for timestamp, _ in selected_frames
        }
        print(f"Coverage: frames with valid pose = {len(poses):,} (identity; no registration needed).")
    else:
        print(f"Registering {len(selected_frames)} bounded point-cloud frames (odom-anchored)...")
        poses, _stats = run_odom_anchored_registration(selected_frames, odom_data, args)

    del selected_frames
    gc.collect()

    if not poses:
        sys.exit("Error: Registration produced no usable poses.")

    pcd_combined, _merged_frame_count = _stream_colored_merge(
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
    mesh_ply_path = out_dir / f"{stem}_colored_mesh.ply"

    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    # PLY has a native per-vertex-color field, so vertex colors round-trip
    # cleanly here (unlike the old OBJ export, which relied on a
    # non-standard extension many viewers ignored).
    o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh, write_vertex_colors=True)

    print(f"Saved cloud: {ply_path}")
    print(f"Saved mesh PLY: {mesh_ply_path}")
    print("Done.")


def build_parser(sub):
    parser = sub.add_parser(
        "color_mesh",
        help="ROS 2 bag -> memory-bounded camera-coloured Poisson mesh",
    )
    parser.add_argument("bagpath", help="Path to the ROS 2 bag.")
    parser.add_argument("outputdir", help="Output directory.")

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
        default=1,
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
        "--enable_loop_closure",
        action="store_true",
        default=False,
    )
    parser.add_argument("--loop_closure_radius", type=float, default=10.0)
    parser.add_argument(
        "--loop_closure_fitness_thresh",
        type=float,
        default=0.7,
        help="Defaults to the same bar as --icp_fitness_thresh, not a separate looser value.",
    )
    parser.add_argument(
        "--loop_closure_search_interval",
        type=int,
        default=10,
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
