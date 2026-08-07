#!/usr/bin/env python3
"""
color_tiles_3d.py - ROS 2 bag -> registered COLORED point cloud
-> georeferenced -> Cesium 3D Tiles.

This pipeline is the union of color_mesh coloring logic and tiles_3d
georeferencing logic. No mesh is produced; the output is a colored
ECEF point cloud converted to tileset.json via py3dtiles.

Pipeline:
1. Pre-flight: verify required topics.
2. Average GPS fixes to establish ENU origin.
3. Read PointCloud2 + optional Odometry + Camera + CameraInfo.
4. Odom-anchored registration (shared), or identity transform if the point
   cloud is already published in a global/fixed frame.
5. Project camera images onto each registered frame.
6. Merge colored frames (gray-fill filtering + voxel downsample).
7. Clean merged cloud.
8. ENU -> ECEF conversion.
9. Write colored ECEF PLY -> py3dtiles convert -> tileset.json.

REFACTOR NOTE: Bag-reading, world-frame merge, and the ENU->ECEF->PLY->
py3dtiles tail-end logic live in `shared/tiles_common.py` and are shared
with `tiles_3d.py`, so bugfixes to that shared logic no longer need to be
applied twice by hand.

PATCH NOTE (odom-anchored registration + frame-awareness):
See odom-anchored-registration-fix-prompt.md and
pointcloud-frame-check-prompt-2.md. Mirrors the same change already applied
to `mesh.py`, `color_mesh.py`, `gazebo_world.py`, and `tiles_3d.py`:

- The point cloud's frame_id is detected (or overridden via
  --pc_frame_mode) and classified as 'global' (odom/map/world) or 'local'
  (a moving sensor/base frame).
- 'local' frames are registered via `run_odom_anchored_registration()`
  (odometry primary, ICP optional/gated, loop closure gated at the same
  bar and weighted so it can only nudge the graph).
- 'global' frames use an identity transform (a correct no-op -- no ICP,
  no pose graph); the camera origin used for image projection and view-ray
  normals still comes from interpolated odometry when available, since
  there is no separate registration transform in this branch to create a
  frame mismatch.

BUGFIX (total color loss, matching fix_pointcloud_color_loss_prompt.md
already applied to color_mesh.py, but never ported here):
- Root cause 1: any frame that did not find a camera image within
  --max_time_diff previously entered the merge via `pcd_combined +=
  pcd_world` with NO colors array set at all -- `_color_pcd_from_image`
  was only ever called on frames that matched a camera image in time.
- Root cause 2: Open3D's `PointCloud.__add__`/`__iadd__` clears colors on
  the ENTIRE result if either operand lacks a colors array (or the
  arrays mismatch length). Since `pcd_combined` accumulated purely
  uncolored frames throughout the main loop (root cause 1), the single
  `pcd_combined += merged_color` after the loop -- combining an
  uncolored `pcd_combined` with a colored `merged_color` -- silently
  wiped out ALL color, even though some frames had been successfully
  camera-colored. This fired any time point-cloud and camera frame rates
  weren't perfectly aligned (i.e. essentially always), not just as an
  edge case.
Fixed exactly as in color_mesh.py: every frame that reaches the merge
now gets an explicit neutral-gray fallback color if it wasn't
camera-colored (`_fill_fallback_color`), and all merging is done via
manual numpy concatenation (`_concat_point_clouds`) instead of Open3D's
`+`/`+=` operators, which are never invoked and are therefore immune to
this behavior.

BUGFIX (camera_info_topic validation order):
`--camera_topic` requires `--camera_info_topic`, and this was already
checked -- but the check ran AFTER `required` (which unconditionally
included `args.camera_info_topic` whenever `--camera_topic` was set) was
already passed to `check_topics()`. If `--camera_info_topic` was left at
its `None` default, `required` contained a literal `None` entry, and
`check_topics()` (which just filters `t not in present`) would report a
confusing `Required topics missing from bag: [None]` and exit before the
clear, purpose-built "--camera_topic requires --camera_info_topic"
message below it was ever reached. The validation now runs first, before
`required` is built, so the intended message is the one users actually
see.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud
from .shared.registration import (
    _safe_normalize,
    run_odom_anchored_registration,
    select_registration_frames,
)
from .shared.ros_io import (
    convert_ros_image,
    convert_ros_pc2_to_o3d,
    get_closest_timestamp,
    get_odom_transform,
    interpolate_odom_pose,
    intrinsics_from_camera_info,
    parse_gps_fixes,
    resolve_pc_frame_mode,
)
from .shared.tiles_common import (
    georeference_and_export_tileset,
    read_bag_topics,
    transform_frame_to_world,
)

# Neutral fallback color used for any point/frame that could not be
# camera-colored (no image in time tolerance, no intrinsics yet, or a
# point that didn't project into the image). Matches color_mesh.py's
# FALLBACK_GRAY exactly so the two pipelines' "is this a placeholder
# color" heuristics (std/mean-based gray detection) stay consistent.
FALLBACK_GRAY = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Color projection (identical to color_mesh; kept local to avoid import cycle)
# ---------------------------------------------------------------------------
def _color_pcd_from_image(
    pcd: o3d.geometry.PointCloud,
    img: Image.Image,
    camera_pose: np.ndarray,
    intrinsics: tuple,
    color_min_depth: float = 0.1,
    color_max_depth: float | None = None,
) -> o3d.geometry.PointCloud:
    """Project camera image colours onto *pcd* in-place; returns pcd."""
    fx, fy, cx, cy, img_w, img_h = intrinsics
    pts = np.asarray(pcd.points)
    img_arr = np.asarray(img)
    cam_pos = camera_pose[:3, 3]
    cam_rot = R.from_matrix(camera_pose[:3, :3])

    body = cam_rot.inv().apply(pts - cam_pos)
    opt_x = -body[:, 1]
    opt_y = -body[:, 2]
    opt_z = body[:, 0]
    depth = np.linalg.norm(body, axis=1)

    valid = (opt_z > 1e-6) & (depth >= color_min_depth)
    if color_max_depth is not None:
        valid &= depth <= color_max_depth

    z_safe = np.where(opt_z > 1e-6, opt_z, 1e-6)
    u = fx * (opt_x / z_safe) + cx
    v = fy * (opt_y / z_safe) + cy
    valid &= (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    colors = np.full((len(pts), 3), FALLBACK_GRAY, dtype=np.float64)
    if np.any(valid):
        ui = np.clip(u[valid].astype(np.int32), 0, img_w - 1)
        vi = np.clip(v[valid].astype(np.int32), 0, img_h - 1)
        colors[valid] = img_arr[vi, ui] / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _fill_fallback_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Guarantee `pcd` has an explicit colors array before it can be merged.

    BUGFIX: any frame that skips camera projection entirely (no camera
    image within --max_time_diff, or color_mode is off) must still get
    an explicit neutral-gray colors array of matching length before it
    reaches `_concat_point_clouds`, so a mix of colored and uncolored
    frames never triggers Open3D's "either operand lacks colors -> whole
    result loses color" behavior. See the BUGFIX note in this module's
    docstring.
    """
    colors = np.full((len(pcd.points), 3), FALLBACK_GRAY, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _concat_point_clouds(clouds: list) -> o3d.geometry.PointCloud:
    """Merge point clouds via manual array concatenation.

    BUGFIX: Open3D's `PointCloud.__add__`/`__iadd__` clears the ENTIRE
    result's colors if either operand lacks a colors array (or the
    arrays mismatch length). This function never touches `+`/`+=`, so
    it's immune to that behavior. Every incoming cloud is guaranteed to
    already carry a `.colors` array of matching length (either real
    camera-projected color or `_fill_fallback_color`'s fallback), so the
    `np.vstack` calls below are always safe.
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
        # reaching this function, but guard against it defensively
        # rather than silently emitting an uncolored merged cloud.
        merged = _fill_fallback_color(merged)

    if all(c.has_normals() for c in non_empty) and non_empty:
        normals = np.vstack([np.asarray(c.normals) for c in non_empty])
        if len(normals) == len(points):
            merged.normals = o3d.utility.Vector3dVector(normals)

    return merged


def _merge_colored_pcds(
    colored_pcds: list,
    voxel_size: float,
    gray_filter_radius: float,
) -> o3d.geometry.PointCloud:
    """Concatenate per-frame colored clouds, remove gray fill near real color,
    then voxel-downsample. Returns a single merged PointCloud."""
    all_pts, all_cols, all_nors, all_gray = [], [], [], []

    for pcd in colored_pcds:
        if len(pcd.points) == 0 or not pcd.has_colors():
            continue
        pts = np.asarray(pcd.points, dtype=np.float64)
        cols = np.asarray(pcd.colors, dtype=np.float64)
        nors = (
            np.asarray(pcd.normals, dtype=np.float64)
            if pcd.has_normals()
            else np.zeros((len(pts), 3), dtype=np.float64)
        )

        std = np.std(cols, axis=1)
        mean = np.mean(cols, axis=1)
        is_gray = (std < 0.08) & (np.abs(mean - 0.5) < 0.15)
        all_pts.append(pts); all_cols.append(cols)
        all_nors.append(nors); all_gray.append(is_gray)

    if not all_pts:
        raise ValueError("No valid colored point clouds to merge.")

    pts = np.vstack(all_pts)
    cols = np.vstack(all_cols)
    nors = np.vstack(all_nors)
    is_gray = np.hstack(all_gray)

    colored_pts = pts[~is_gray]
    if len(colored_pts) > 0 and gray_filter_radius > 0:
        print(f"  Gray-fill filtering (radius={gray_filter_radius} m)...")
        tree = cKDTree(colored_pts)
        gray_idx = np.where(is_gray)[0]
        nbrs = tree.query_ball_point(pts[gray_idx], r=gray_filter_radius)
        has_col = np.array([len(n) > 0 for n in nbrs], dtype=bool)
        keep = np.ones(len(pts), dtype=bool)
        keep[gray_idx[has_col]] = False
        pts = pts[keep]; cols = cols[keep]; nors = nors[keep]
    elif len(colored_pts) == 0:
        print("  Warning: no colored points found; keeping all gray fills.")

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(pts)
    merged.colors = o3d.utility.Vector3dVector(cols)
    if np.any(np.linalg.norm(nors, axis=1) > 0):
        merged.normals = o3d.utility.Vector3dVector(_safe_normalize(nors))

    return merged.voxel_down_sample(voxel_size)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args) -> None:
    bag_path = Path(args.bagpath)
    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    # BUGFIX: this validation must run BEFORE `required` is built below --
    # otherwise a bare `--camera_topic` with no `--camera_info_topic` puts
    # a literal `None` into `required`, and check_topics() reports a
    # confusing "Required topics missing from bag: [None]" instead of this
    # clear, purpose-built message.
    if args.camera_topic and not args.camera_info_topic:
        sys.exit("Error: --camera_topic requires --camera_info_topic.")

    frame_mode = resolve_pc_frame_mode(bag_path, args.pc_topic, args.pc_frame_mode)

    if frame_mode == "local" and not args.odom_topic:
        sys.exit(
            "Error: --odom_topic is required because the point cloud on "
            f"'{args.pc_topic}' is not already published in a global/fixed "
            "frame (classified as 'local'). Provide --odom_topic pointing at "
            "a nav_msgs/Odometry topic, or pass --pc_frame_mode global if "
            "this bag's point cloud really is already in a fixed frame."
        )

    required = [args.pc_topic, args.gps_topic]
    if args.odom_topic:
        required.append(args.odom_topic)
    if args.camera_topic:
        required += [args.camera_topic, args.camera_info_topic]
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(
            f"Error: Required topics missing from bag: {missing}\n"
            "Check topic names with: ros2 bag info <bag_path>"
        )

    # -- GPS origin ----------------------------------------------------
    print(f"[1/7] Reading GPS fixes from '{args.gps_topic}'...")
    lat0, lon0, alt0 = parse_gps_fixes(bag_path, args.gps_topic)

    # -- Read bag --------------------------------------------------------
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data: dict[int, np.ndarray] = {}
    camera_images: dict[int, Image.Image] = {}
    cam_info_data: dict[int, tuple] = {}

    def handle_pc(message, timestamp: int) -> None:
        pcd = convert_ros_pc2_to_o3d(message)
        if pcd is not None and len(pcd.points) >= 100:
            pointclouds.append((timestamp, pcd))

    def handle_odom(message, timestamp: int) -> None:
        transform = get_odom_transform(message)
        if transform is not None:
            odom_data[timestamp] = transform

    def handle_camera(message, timestamp: int) -> None:
        img = convert_ros_image(message)
        if img is not None:
            camera_images[timestamp] = img

    def handle_camera_info(message, timestamp: int) -> None:
        intr = intrinsics_from_camera_info(message)
        if intr is not None:
            cam_info_data[timestamp] = intr

    handlers = {args.pc_topic: handle_pc}
    if args.odom_topic:
        handlers[args.odom_topic] = handle_odom
    if args.camera_topic:
        handlers[args.camera_topic] = handle_camera
        handlers[args.camera_info_topic] = handle_camera_info

    print(f"\n[2/7] Reading messages from: {bag_path}")
    read_bag_topics(bag_path, handlers, desc="Reading")

    if not pointclouds:
        sys.exit("Error: No valid point clouds extracted.")
    if args.odom_topic and not odom_data:
        print("Warning: --odom_topic set but no messages found.")

    color_mode = bool(args.camera_topic and camera_images and cam_info_data)
    intrinsics = None
    if color_mode:
        first_ts = min(cam_info_data.keys())
        intrinsics = cam_info_data[first_ts]
        fx, fy, cx, cy, cw, ch = intrinsics
        print(f"  Camera: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} {cw}x{ch}")
        print(f"  Images: {len(camera_images)}")
    else:
        print("  No camera data found; producing XYZ-only tiles.")

    print(f"  Coverage: frames read = {len(pointclouds):,} | odom poses = {len(odom_data):,}")

    selected_frames, _original_indices = select_registration_frames(
        pointclouds, frame_stride=args.frame_stride, max_registration_frames=args.max_registration_frames
    )
    print(
        f"  Coverage: frames selected = {len(selected_frames):,} / {len(pointclouds):,} "
        f"(stride={args.frame_stride}, max={args.max_registration_frames})."
    )
    del pointclouds

    # -- Registration --------------------------------------------------
    if frame_mode == "global":
        print(
            "\n[3/7] Point cloud already in a global/fixed frame: skipping "
            "ICP/pose-graph registration and per-frame transform application."
        )
        pose_by_timestamp: dict[int, np.ndarray] = {
            timestamp: np.eye(4, dtype=np.float64) for timestamp, _ in selected_frames
        }
        print(f"  Coverage: frames with valid pose = {len(pose_by_timestamp):,} (identity; no registration needed).")
    else:
        print("\n[3/7] Odom-anchored registration...")
        pose_by_timestamp, _stats = run_odom_anchored_registration(selected_frames, odom_data, args)

    if not pose_by_timestamp:
        sys.exit("Error: Registration produced no usable poses.")

    odom_max_ns = int(args.odom_max_latency * 1e9)
    odom_ts_sorted = sorted(odom_data.keys())
    cam_ts_sorted = sorted(camera_images.keys())

    # -- Merge + color projection -------------------------------------------
    # BUGFIX: every frame that reaches this merge -- whether it gets a real
    # camera projection or not -- is now guaranteed an explicit colors
    # array (`_fill_fallback_color` for the ones that don't project), and
    # the final combine uses `_concat_point_clouds` (manual numpy
    # concatenation) instead of Open3D's `+`/`+=`, which would otherwise
    # silently strip color from the entire result whenever an uncolored
    # frame was mixed with a colored one. See this module's docstring.
    print(f"\n[4/7] Merging registered frames{' with color projection' if color_mode else ''}...")
    uncolored_frames: list[o3d.geometry.PointCloud] = []
    colored_frames: list[o3d.geometry.PointCloud] = []
    merged_frame_count = 0
    camera_colored_count = 0
    fallback_gray_count = 0

    for timestamp, pcd_raw in selected_frames:
        transform_world = pose_by_timestamp.get(timestamp)
        if transform_world is None:
            continue

        pcd_world = transform_frame_to_world(
            pcd_raw, transform_world, args.voxel_size,
            timestamp=timestamp, odom_data=odom_data,
            odom_ts_sorted=odom_ts_sorted, odom_max_ns=odom_max_ns,
        )

        colored_this_frame = False

        if color_mode and cam_ts_sorted:
            cam_ts = get_closest_timestamp(timestamp, cam_ts_sorted)
            if cam_ts is not None and abs(cam_ts - timestamp) < int(args.max_time_diff * 1e9):
                cam_pose = np.eye(4, dtype=np.float64)
                if odom_ts_sorted:
                    # Interpolated (linear translation + SLERP rotation)
                    # camera origin, not nearest-neighbor snapping.
                    interpolated = interpolate_odom_pose(
                        timestamp, odom_ts_sorted, odom_data, odom_max_ns
                    )
                    if interpolated is not None:
                        cam_pose = interpolated
                pcd_world = _color_pcd_from_image(
                    pcd_world, camera_images[cam_ts], cam_pose,
                    intrinsics, args.color_min_depth, args.color_max_depth,
                )
                colored_frames.append(pcd_world)
                colored_this_frame = True
                camera_colored_count += 1

        if not colored_this_frame:
            # BUGFIX (root cause 1): explicitly fallback-color this frame
            # instead of letting it enter the merge with no colors array
            # at all -- see the BUGFIX note in this module's docstring.
            pcd_world = _fill_fallback_color(pcd_world)
            uncolored_frames.append(pcd_world)
            fallback_gray_count += 1

        merged_frame_count += 1

    merged_pieces: list[o3d.geometry.PointCloud] = list(uncolored_frames)

    if color_mode and colored_frames:
        print(f"  Merging {len(colored_frames)} colored frames...")
        merged_color = _merge_colored_pcds(
            colored_frames, args.voxel_size, args.gray_filter_radius
        )
        merged_pieces.append(merged_color)
    elif color_mode:
        print("  Warning: no colored frames produced; check --max_time_diff.")

    if color_mode:
        print(
            f"  Coloring: {camera_colored_count:,}/{merged_frame_count:,} frame(s) "
            f"camera-colored, {fallback_gray_count:,} used fallback gray."
        )

    # BUGFIX (root cause 2): combine via manual concatenation, never `+=`,
    # so a mix of colored and fallback-gray pieces can never silently
    # strip color from the whole result.
    pcd_combined = _concat_point_clouds(merged_pieces)

    print(f"  Merge: {merged_frame_count:,} frame(s) actually merged into the combined cloud.")
    del selected_frames, pose_by_timestamp

    # -- Clean -------------------------------------------------------------
    print("\n[5/7] Cleaning merged cloud...")
    pcd_clean = clean_point_cloud(
        pcd_combined, args.voxel_size, do_voxel_downsample=False
    )
    if len(pcd_clean.points) == 0:
        sys.exit("Error: No points remain after cleaning.")
    print(f"  Final cloud: {len(pcd_clean.points):,} points")

    # -- Georeference + export ---------------------------------------------
    print("\n[6/7] Georeferencing (local ENU -> ECEF)...")
    print("\n[7/7] Writing 3D Tiles...")
    tiles_dir, _ = georeference_and_export_tileset(
        pcd_clean, lat0, lon0, alt0, out_dir, bag_path.stem, args.workers
    )

    print(f"\nColored 3D Tiles written to: {tiles_dir}")
    print("Done.")


def build_parser(sub):
    p = sub.add_parser(
        "color_tiles_3d",
        help="ROS 2 bag -> colored georeferenced point cloud -> Cesium 3D Tiles"
    )

    p.add_argument("bagpath", help="Path to the ROS 2 bag directory.")
    p.add_argument("outputdir", help="Output directory.")

    # Topics
    p.add_argument("--pc_topic", default="points")
    p.add_argument("--odom_topic", default=None,
        help=(
            "Odometry topic. Required unless the point cloud is already "
            "published in a global/fixed frame (see --pc_frame_mode)."
        ))
    p.add_argument("--gps_topic", default="/gps/fix")
    p.add_argument("--camera_topic", default=None,
        help="sensor_msgs/Image or CompressedImage topic. Optional.")
    p.add_argument("--camera_info_topic", default=None,
        help="sensor_msgs/CameraInfo topic. Required with --camera_topic.")
    p.add_argument(
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

    # Color
    p.add_argument("--max_time_diff", type=float, default=0.1,
        help="Max timestamp diff (s) between PC frame and camera image.")
    p.add_argument("--color_min_depth", type=float, default=0.1)
    p.add_argument("--color_max_depth", type=float, default=None)
    p.add_argument("--gray_filter_radius", type=float, default=0.05,
        help="Gray-fill points with a real-color neighbor within this "
             "radius (m) are removed. 0 = disable.")

    # Registration
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--odom_max_latency", type=float, default=0.5)

    p.add_argument(
        "--enable_icp_refinement",
        action="store_true",
        default=False,
        help=(
            "Optional, strongly-gated local ICP refinement of the odom-derived "
            "pose. ICP can only nudge an already-valid pose or be a no-op -- "
            "it can never remove a frame from the merge."
        ),
    )
    p.add_argument("--icp_dist_thresh", type=float, default=0.2)
    p.add_argument(
        "--icp_fitness_thresh", type=float, default=0.7,
        help="Fitness bar for accepting an ICP refinement (raised from 0.6: this is now a correction gate, not the primary motion estimate).",
    )
    p.add_argument(
        "--max_icp_translation_correction", type=float, default=0.3,
        help="Max allowed ICP correction translation (meters) relative to the odom guess.",
    )
    p.add_argument(
        "--max_icp_rotation_correction_deg", type=float, default=15.0,
        help="Max allowed ICP correction rotation (degrees) relative to the odom guess.",
    )
    p.add_argument("--enable_loop_closure", action="store_true", default=False)
    p.add_argument("--loop_closure_radius", type=float, default=10.0)
    p.add_argument(
        "--loop_closure_fitness_thresh", type=float, default=0.7,
        help="Defaults to the same bar as --icp_fitness_thresh, not a separate looser value.",
    )
    p.add_argument("--loop_closure_search_interval", type=int, default=10)
    p.add_argument(
        "--loop_closure_temporal_window", type=int, default=100,
        help="Bounded number of most-recent candidate frames considered for loop closure.",
    )
    p.add_argument("--frame_stride", type=int, default=1,
        help="Process every Nth frame.")
    p.add_argument("--max_registration_frames", type=int, default=0,
        help="Cap total frames used for registration (0 = all).")
    p.add_argument("--merge_chunk_frames", type=int, default=16,
        help="Number of frames per merge chunk (reserved for future streaming use).")

    # Performance
    p.add_argument("--workers", type=int, default=4)

    p.set_defaults(func=run)
    return p
