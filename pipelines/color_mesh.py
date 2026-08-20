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

PATCH NOTE (motion-gated frame selection + odometry health check):
`--frame_stride` has been removed in favor of motion-gated selection:
`--min_move_distance`/`--min_rotation_angle_deg` now keep a frame only if
it has actually moved/turned relative to the last KEPT frame's odometry
pose (OR'd together), instead of thinning purely by message count. An
odometry health check now runs automatically (unless
--disable_odom_health_check) and truncates registration at the first
detected tracking loss/teleport in the raw odometry stream, before any
frame is selected -- see `shared/registration.py` for the full design.
Both changes are handled internally by `run_odom_anchored_registration()`;
this file only had to change its argparse surface, its read-ahead cap
(now bounding RAW frames read rather than a stride-thinned count), and
this docstring.

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

REFACTOR NOTE (shared color-projection helpers):
The camera-projection coloring math, the fallback-gray guarantee, the
"is this pixel a placeholder gray" heuristic, and the Open3D
`+`/`+=`-avoiding merge helper were previously duplicated almost verbatim
in `color_tiles_3d.py`. They now live in `shared/color_projection.py` (see
that module's docstring) and are imported from there, so a future bugfix
to any of them only needs to be made once. Only the streaming/chunked
merge control flow -- which is specific to this pipeline's two-pass,
bounded-memory bag read -- remains local to this file.

PATCH NOTE (remeshing removed -- it destroyed vertex color):
`--remesh`/`--remesh_smooth_iterations` have been removed from this
pipeline entirely. `reconstruction.remesh_isotropic()` rebuilds the mesh
via PyMeshLab using only `vertex_matrix`/`face_matrix` -- it never carries
a color matrix through the remesh -- so isotropic remeshing silently
discarded every per-vertex color this pipeline exists to produce. Unlike
`mesh.py` (whose optional height false-color pass runs as a *separate*
post-process on the already-remeshed output file, so remeshing there is
harmless), this pipeline bakes color onto the point cloud before Poisson
reconstruction, so any remesh step downstream of it would destroy that
color. `create_mesh()` is now always called with `remesh=False`.

PATCH NOTE (global gray-fill cleanup pass, see gray_fill_global_pass_prompt.md
-- the "hallway problem"):
`_remove_local_gray_fill()` was previously only ever invoked from inside
`_merge_chunk()`, i.e. it only ever compared points within the same batch
of `--merge_chunk_frames` frames. Consider a robot with a front-facing
camera and a 360-degree LIDAR that drives up a hallway, turns around, and
drives back down the same hallway: on the way in the front camera faces
the direction of travel and the walls get real camera color; after the
turn-around the 360-degree LIDAR still sees the same walls, but the front
camera now faces the opposite direction, so those returns fall outside the
camera's FOV and get the neutral-gray fallback instead. The outbound and
return passes are almost always separated by many more frames than one
chunk, so the chunk-local filter can never compare the gray return-pass
points against the colored outbound-pass points describing the same
physical surface -- both a colored point and a nearby gray "ghost" point
would otherwise survive into the final cloud (visible as speckled/dull
patches on any surface seen from two different directions).

Fixed by adding a **second, global** call to the same
`_remove_local_gray_fill()` helper (no change to the helper's internal
logic -- it is scope-agnostic and only ever operates on whatever cloud
object it is given) in `run()`, run exactly once against the fully merged
cloud right after streaming/chunked merging completes and the cloud is
confirmed to have colors, but before `level_floor`/`clean_point_cloud`
run on it. The existing chunk-local call inside `_merge_chunk` is left in
place -- it's cheap and reduces how much duplicate data the global pass
has to search later. The number of points removed by the global pass is
now printed, so gray/colored duplicate cleanup is visible in the
pipeline's output logs instead of happening silently.

PATCH NOTE (defaults aligned to Bellhop GUI):
`--pc_topic` and `--loop_closure_radius` now default to the same values
the GUI's Color Mesh profile always sent (`/points` and `3.0`
respectively), so a bare CLI invocation with no flags now produces the
exact same behavior as a GUI-launched run with no changes.
"""

from __future__ import annotations

import gc
import sys
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from rosbags.highlevel import AnyReader
from scipy.spatial import cKDTree
from tqdm import tqdm

from .shared.color_projection import (
    color_pcd_from_image,
    concat_point_clouds,
    fill_fallback_color,
    remove_gray_fill_near_color,
)
from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud, create_mesh, level_floor
from .shared.registration import (
    attach_view_rays_as_normals,
    estimate_geometric_normals_oriented,
    run_odom_anchored_registration,
    select_registration_frames_by_motion,
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


def _remove_local_gray_fill(
    pcd: o3d.geometry.PointCloud,
    radius: float,
) -> o3d.geometry.PointCloud:
    """Remove neutral placeholder colour only near genuine coloured points.

    Thin adapter around the shared `remove_gray_fill_near_color()` array
    helper: this pipeline's chunks carry per-point normals (view rays)
    by the time they reach this function, so those are threaded through
    as an extra array and reattached to the filtered result.

    Scope-agnostic: this function only ever operates on whatever cloud
    object it is given, so it is called both per-chunk (inside
    `_merge_chunk`) and once globally against the fully merged cloud (in
    `run()`, see the "hallway problem" PATCH NOTE at the top of this
    file) with no change to its own logic.
    """
    if radius <= 0 or not pcd.has_colors() or not len(pcd.points):
        return pcd

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    has_normals = pcd.has_normals()
    extra = (np.asarray(pcd.normals),) if has_normals else ()

    filtered = remove_gray_fill_near_color(points, colors, radius, extra)

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(filtered[0])
    result.colors = o3d.utility.Vector3dVector(filtered[1])
    if has_normals:
        result.normals = o3d.utility.Vector3dVector(filtered[2])
    return result


def _merge_chunk(
    target: o3d.geometry.PointCloud,
    chunk: list[o3d.geometry.PointCloud],
    voxel_size: float,
    gray_filter_radius: float,
) -> o3d.geometry.PointCloud:
    """Merge a bounded chunk and voxelize immediately."""
    if not chunk:
        return target

    local = concat_point_clouds(chunk)
    chunk.clear()

    local = _remove_local_gray_fill(local, gray_filter_radius)

    if len(target.points):
        target = concat_point_clouds([target, local])
    else:
        target = local

    return target.voxel_down_sample(voxel_size)


def _read_registration_data(
    bag_path: Path,
    args,
) -> tuple[list[tuple[int, o3d.geometry.PointCloud]], dict[int, np.ndarray]]:
    """First pass: retain bounded, voxelized registration scans.

    NOTE: frame selection (motion gating + the odometry health check) is
    intentionally NOT applied here. Selection now happens exactly once,
    downstream, inside `run_odom_anchored_registration()` (for the 'local'
    branch) or via `select_registration_frames_by_motion()` directly (for
    the 'global' branch). This function only enforces --min_frame_points
    and a generous read-ahead cap derived from --max_registration_frames,
    so very large bags are still bounded. This cap now bounds RAW frames
    read, not guaranteed SELECTED frames, since selection is motion-gated
    rather than a fixed stride.
    """
    topics = [args.pc_topic] + ([args.odom_topic] if args.odom_topic else [])

    frames: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}

    read_ahead_limit = max(0, int(args.max_registration_frames)) if args.max_registration_frames else 0

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
                    print(f"Registration read-ahead limit reached: {read_ahead_limit}.")
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
    (Root Cause 2, fixed in `concat_point_clouds`/`_merge_chunk`).
    """
    topics = [args.pc_topic, args.camera_topic, args.camera_info_topic]

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

                cloud = color_pcd_from_image(
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
                cloud = fill_fallback_color(cloud)
                fallback_gray_count += 1

            chunk.append(cloud)
            merged_frame_count += 1

            if len(chunk) >= chunk_size:
                merged = _merge_chunk(
                    merged, chunk, args.voxel_size, args.gray_filter_radius
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
    merged = _merge_chunk(merged, chunk, args.voxel_size, args.gray_filter_radius)

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
        print("Warning: odom topic was set but no usable messages were found.")

    if frame_mode == "global":
        print(
            "Point cloud already in a global/fixed frame: skipping ICP/pose-graph "
            "registration and per-frame transform application; streaming, "
            "filtering, downsampling, colouring, and merging directly."
        )
        selected_frames, _original_indices = select_registration_frames_by_motion(
            frames,
            odom_data,
            min_move_distance=args.min_move_distance,
            min_rotation_angle_deg=args.min_rotation_angle_deg,
            max_registration_frames=args.max_registration_frames,
            odom_max_latency_ns=int(args.odom_max_latency * 1e9),
        )
        print(
            f"Coverage: frames selected = {len(selected_frames):,} / {len(frames):,} "
            f"(min_move_distance={args.min_move_distance}m, "
            f"min_rotation_angle_deg={args.min_rotation_angle_deg}deg, "
            f"max={args.max_registration_frames})."
        )
        poses: dict[int, np.ndarray] = {
            timestamp: np.eye(4, dtype=np.float64) for timestamp, _ in selected_frames
        }
        print(f"Coverage: frames with valid pose = {len(poses):,} (identity; no registration needed).")
    else:
        print(f"Registering up to {len(frames)} raw point-cloud frames (odom-anchored)...")
        poses, _stats = run_odom_anchored_registration(frames, odom_data, args)

    del frames
    gc.collect()

    if not poses:
        sys.exit("Error: Registration produced no usable poses.")

    pcd_combined, _merged_frame_count = _stream_colored_merge(
        bag_path, args, poses, odom_data, None
    )
    del poses
    del odom_data
    gc.collect()

    if not len(pcd_combined.points):
        sys.exit("Error: No registered points were produced.")

    # PATCH NOTE (global gray-fill cleanup pass -- the "hallway problem",
    # see gray_fill_global_pass_prompt.md and the module docstring above):
    # `_remove_local_gray_fill()` above only ever compares points within a
    # single `--merge_chunk_frames` chunk. A gray-fallback point from one
    # part of the trajectory (e.g. a hallway wall seen only from behind
    # after a turn-around, with no matching camera frame) and a genuinely
    # colored point describing the same physical surface (seen on an
    # earlier outbound pass) can easily end up many chunks apart and are
    # never compared against each other by the chunk-local pass alone.
    # Run the same scope-agnostic helper exactly once more, now against
    # the FULLY merged/voxelized cloud, before any floor-leveling or final
    # cleanup runs -- this is cheap relative to the chunk-local pass since
    # by this point the cloud is already voxel-downsampled to its final
    # resolution, and it reconciles gray/colored duplicates that span
    # arbitrarily large temporal gaps in the trajectory.
    if pcd_combined.has_colors() and args.gray_filter_radius > 0:
        points_before_global_pass = len(pcd_combined.points)
        pcd_combined = _remove_local_gray_fill(pcd_combined, args.gray_filter_radius)
        removed_by_global_pass = points_before_global_pass - len(pcd_combined.points)
        print(
            f"Global gray-fill cleanup: removed {removed_by_global_pass:,} "
            f"gray-fallback point(s) with a real-color neighbor anywhere in the "
            f"merged cloud ({points_before_global_pass:,} -> "
            f"{len(pcd_combined.points):,} points)."
        )
    elif args.gray_filter_radius > 0:
        print("Global gray-fill cleanup: skipped (merged cloud has no colors).")

    if args.level_floor:
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
        # PATCH: remeshing is intentionally never used here -- isotropic
        # remeshing (see reconstruction.remesh_isotropic) rebuilds the
        # mesh from bare vertex/face matrices with no color channel, so
        # it would silently destroy every per-vertex color this pipeline
        # exists to produce. There is no --remesh flag on this pipeline
        # anymore; this is not user-configurable.
        remesh=False,
        remesh_smooth_iterations=0,
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

    parser.add_argument("--pc_topic", default="/points")
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
    parser.add_argument(
        "--gray_filter_radius",
        type=float,
        default=0.05,
        help=(
            "Gray-fallback points with a real-colored neighbor within this "
            "radius (m) are removed. Governs BOTH the per-chunk cleanup pass "
            "(inside each --merge_chunk_frames batch) AND a second, global "
            "pass run once against the fully merged cloud (see the 'hallway "
            "problem' PATCH NOTE at the top of this file) that reconciles "
            "gray/colored duplicates separated by many chunks -- e.g. the "
            "same wall seen once from the front camera's FOV and once from "
            "outside it after a turn-around. Set to 0 to disable both passes. "
            "Should generally be set to at least --voxel_size: points closer "
            "together than one voxel are unlikely to exist after "
            "downsampling, so a smaller radius may fail to find any "
            "neighbor at all and remove nothing."
        ),
    )

    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--min_frame_points", type=int, default=100)

    parser.add_argument(
        "--min_move_distance",
        type=float,
        default=0.10,
        help="Minimum translation (m), relative to the last KEPT registration frame's "
        "odometry pose, required to keep a new frame. Replaces the old index-based "
        "--frame_stride: odom is the primary pose source, so a frame that hasn't moved "
        "(or turned) adds no new spatial coverage and is skipped instead of thinning "
        "purely by message count. A frame is kept if EITHER this OR "
        "--min_rotation_angle_deg is satisfied. Set to 0 to disable this half of the gate.",
    )
    parser.add_argument(
        "--min_rotation_angle_deg",
        type=float,
        default=5.0,
        help="Minimum rotation (deg), relative to the last KEPT registration frame's "
        "odometry pose, required to keep a new frame (OR'd with --min_move_distance). "
        "Lets a robot that spins in place without translating still accumulate new "
        "frames to cover the swept field of view. Set to 0 to disable this half of "
        "the gate.",
    )
    parser.add_argument(
        "--max_registration_frames",
        type=int,
        default=0,
        help=(
            "Maximum registration frames (0 means unlimited). Bounds BOTH raw "
            "frames read ahead of time AND the number of frames kept after the "
            "motion gate and the odometry health check, applied in temporal order."
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
        "--disable_odom_health_check",
        action="store_true",
        default=False,
        help="Disable the automatic odometry health check that truncates registration "
        "at the first detected tracking loss/teleport (implied speed or rotation rate "
        "far above the bag's own 95th-percentile baseline). Enabled by default so a "
        "lost-odom segment cannot silently skew the output. The segment after a "
        "detected loss is never auto-spliced back in.",
    )
    parser.add_argument(
        "--odom_loss_speed_multiplier",
        type=float,
        default=6.0,
        help="Sensitivity of the odometry health check: a consecutive odometry sample "
        "pair is flagged as a tracking loss/teleport if its implied linear speed OR "
        "angular rate exceeds this multiplier times the bag's own 95th-percentile "
        "baseline. Lower = more sensitive (may false-positive on genuinely fast "
        "motion); higher = less sensitive. Ignored if --disable_odom_health_check is set.",
    )

    parser.add_argument(
        "--enable_icp_refinement",
        action="store_true",
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

    parser.add_argument("--enable_loop_closure", action="store_true", default=False)
    parser.add_argument("--loop_closure_radius", type=float, default=3.0)
    parser.add_argument(
        "--loop_closure_fitness_thresh",
        type=float,
        default=0.7,
        help="Defaults to the same bar as --icp_fitness_thresh, not a separate looser value.",
    )
    parser.add_argument("--loop_closure_search_interval", type=int, default=10)
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
    parser.add_argument("--min_density_percentile", type=float, default=1.0)
    parser.add_argument("--distance_multiplier", type=float, default=3.0)
    parser.add_argument("--max_vertex_distance", type=float, default=None)

    # PATCH: --remesh / --remesh_smooth_iterations removed. Isotropic
    # remeshing discards per-vertex color (see PATCH NOTE at the top of
    # this file and reconstruction.remesh_isotropic); this pipeline's
    # entire purpose is per-vertex camera color, so remeshing is no
    # longer an option here at all -- create_mesh() is always called
    # with remesh=False.

    parser.add_argument("--decimate_target", type=float, default=None)
    parser.add_argument("--curvature_percentile", type=float, default=80.0)
    parser.add_argument("--curvature_protect_rings", type=int, default=1)
    parser.add_argument("--level_floor", action="store_true", default=False)

    parser.add_argument("--workers", type=int, default=1)

    parser.set_defaults(func=run)
    return parser
