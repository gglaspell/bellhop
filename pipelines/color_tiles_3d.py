#!/usr/bin/env python3
"""
color_tiles_3d.py - ROS 2 bag -> registered COLORED point cloud
-> georeferenced -> Cesium 3D Tiles.

This pipeline is the union of color_mesh coloring logic and tiles_3d
georeferencing logic. No mesh is produced; the output is a colored
ECEF point cloud converted to tileset.json via py3dtiles. Because there
is no mesh reconstruction step at all, this pipeline never had a
`--remesh` flag to begin with -- there is nothing here for remeshing to
destroy.

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
9. Write colored ECEF PLY per LOD level -> py3dtiles convert -> one
tileset_/ directory per level (coarse/medium/fine by default).

REFACTOR NOTE: Bag-reading, world-frame merge, and the ENU->ECEF->PLY->
py3dtiles tail-end logic live in `shared/tiles_common.py` and are shared
with `tiles_3d.py`, so bugfixes to that shared logic no longer need to be
applied twice by hand.

REFACTOR NOTE (color-projection overlap with color_mesh.py):
The camera-projection coloring math, the fallback-gray guarantee, the
"is this pixel a placeholder gray" heuristic, and the Open3D
`+`/`+=`-avoiding merge helper were previously duplicated almost verbatim
in `color_mesh.py`. That duplication is exactly how this file ended up
missing the total-color-loss fix for several revisions after it landed in
`color_mesh.py` (see the BUGFIX history this note replaces). Those four
pieces now live in `shared/color_projection.py` and are imported from
there, so this pipeline's "is this gray" detection is guaranteed to stay
in lockstep with color_mesh.py's -- a future fix to either only needs to
be made once. Only `_merge_colored_pcds()` (this pipeline's specific
"merge a list of per-frame colored clouds with gray-fill filtering, then
voxel-downsample" combination) remains local to this file, now built on
top of the shared `is_gray_fill()`/`remove_gray_fill_near_color()`
helpers instead of an inline copy of the same logic.

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

PATCH NOTE (motion-gated frame selection + odometry health check):
Ported from `mesh.py`/`color_mesh.py`/`gazebo_world.py`/`tiles_3d.py`:
`--frame_stride` has been removed in favor of motion-gated selection.
`--min_move_distance`/`--min_rotation_angle_deg` now keep a frame only if
it has actually moved/turned relative to the last KEPT frame's odometry
pose (OR'd together), instead of thinning purely by message count. An
odometry health check now runs automatically (unless
--disable_odom_health_check) and truncates registration at the first
detected tracking loss/teleport in the raw odometry stream, before any
frame is selected -- both changes are handled internally by
`run_odom_anchored_registration()` for the 'local' branch, and via
`select_registration_frames_by_motion()` directly for the 'global'
branch. See `shared/registration.py` for the full design. This also
fixes a latent breakage: `select_registration_frames` (the old
--frame_stride-based selector this file used to import) had already been
removed from `shared/registration.py` when the other four pipelines were
migrated, so this file's import of it was dead and would have raised an
ImportError the first time this module was actually loaded.

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

FEATURE NOTE (three LOD tileset layers):
The final export step now calls
`tiles_common.georeference_and_export_lod_tilesets()` instead of the
single-layer `georeference_and_export_tileset()`. The same cleaned/merged
colored cloud is voxel-downsampled at three densities (coarse/medium/fine
by default, configurable via `--lod_multipliers`) and each density is
written into its own `tileset_/` subfolder under `outputdir`, so a
Cesium viewer can offer a quality/performance toggle between layers
instead of being stuck with one fixed-resolution tileset. Color is
preserved at every LOD level since `voxel_down_sample()` averages colors
of merged points rather than dropping them.

PATCH NOTE (defaults aligned to tiles_3d):
`--pc_topic`, `--loop_closure_radius`, and `--workers` now default to the
same values `tiles_3d.py` uses (`/points`, `3.0`, and `1` respectively),
instead of this pipeline's previously divergent defaults (`points`,
`10.0`, `4`), so a bare CLI invocation of either pipeline against the
same bag behaves consistently out of the box.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from .shared.color_projection import (
    color_pcd_from_image,
    concat_point_clouds,
    fill_fallback_color,
    is_gray_fill,
    remove_gray_fill_near_color,
)
from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud
from .shared.registration import (
    _safe_normalize,
    run_odom_anchored_registration,
    select_registration_frames_by_motion,
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
    DEFAULT_LOD_LEVELS,
    georeference_and_export_lod_tilesets,
    read_bag_topics,
    transform_frame_to_world,
)


# ---------------------------------------------------------------------------
# CLI helper: parse "--lod_multipliers" into (name, multiplier) pairs
# ---------------------------------------------------------------------------
def _build_lod_levels(names: list[str], multipliers_csv: str) -> tuple[tuple[str, float], ...]:
    try:
        multipliers = [float(x) for x in multipliers_csv.split(",")]
    except ValueError:
        sys.exit(f"Error: --lod_multipliers must be a comma-separated list of numbers, got: {multipliers_csv!r}")
    if len(multipliers) != len(names):
        sys.exit(
            f"Error: --lod_multipliers must have exactly {len(names)} values "
            f"(one per LOD level: {', '.join(names)}), got {len(multipliers)}."
        )
    return tuple(zip(names, multipliers))


# ---------------------------------------------------------------------------
# Merge: per-frame colored clouds -> gray-fill filtered, voxel-downsampled
# ---------------------------------------------------------------------------
def _merge_colored_pcds(
    colored_pcds: list,
    voxel_size: float,
    gray_filter_radius: float,
) -> o3d.geometry.PointCloud:
    """Concatenate per-frame colored clouds, remove gray fill near real color,
    then voxel-downsample. Returns a single merged PointCloud."""
    all_pts, all_cols, all_nors = [], [], []

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

        all_pts.append(pts)
        all_cols.append(cols)
        all_nors.append(nors)

    if not all_pts:
        raise ValueError("No valid colored point clouds to merge.")

    pts = np.vstack(all_pts)
    cols = np.vstack(all_cols)
    nors = np.vstack(all_nors)

    if gray_filter_radius > 0:
        gray = is_gray_fill(cols)
        if not gray.any():
            pass
        elif gray.all():
            print("  Warning: no colored points found; keeping all gray fills.")
        else:
            print(f"  Gray-fill filtering (radius={gray_filter_radius} m)...")
            pts, cols, nors = remove_gray_fill_near_color(
                pts, cols, gray_filter_radius, (nors,)
            )

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

    lod_level_names = [name for name, _ in DEFAULT_LOD_LEVELS]
    lod_levels = _build_lod_levels(lod_level_names, args.lod_multipliers)

    required = [args.pc_topic, args.gps_topic]
    if args.odom_topic:
        required.append(args.odom_topic)
    if args.camera_topic:
        required += [args.camera_topic, args.camera_info_topic]
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(
            f"Error: Required topics missing from bag: {missing}\n"
            "Check topic names with: ros2 bag info "
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

    # -- Registration --------------------------------------------------
    # NOTE: frame selection (motion gating + the odometry health check) is
    # NOT applied here up front. For the 'local' branch it now happens once,
    # internally, inside `run_odom_anchored_registration()` (which needs the
    # FULL raw frame/odom lists to run its health check and motion gate in
    # the correct order). For the 'global' branch it is applied directly via
    # `select_registration_frames_by_motion()` below.
    if frame_mode == "global":
        print(
            "\n[3/7] Point cloud already in a global/fixed frame: skipping "
            "ICP/pose-graph registration and per-frame transform application."
        )
        selected_frames, _original_indices = select_registration_frames_by_motion(
            pointclouds,
            odom_data,
            min_move_distance=args.min_move_distance,
            min_rotation_angle_deg=args.min_rotation_angle_deg,
            max_registration_frames=args.max_registration_frames,
            odom_max_latency_ns=int(args.odom_max_latency * 1e9),
        )
        print(
            f"  Coverage: frames selected = {len(selected_frames):,} / {len(pointclouds):,} "
            f"(min_move_distance={args.min_move_distance}m, "
            f"min_rotation_angle_deg={args.min_rotation_angle_deg}deg, "
            f"max={args.max_registration_frames})."
        )
        pose_by_timestamp: dict[int, np.ndarray] = {
            timestamp: np.eye(4, dtype=np.float64) for timestamp, _ in selected_frames
        }
        del selected_frames
        print(f"  Coverage: frames with valid pose = {len(pose_by_timestamp):,} (identity; no registration needed).")
    else:
        print("\n[3/7] Odom-anchored registration...")
        pose_by_timestamp, _stats = run_odom_anchored_registration(pointclouds, odom_data, args)

    if not pose_by_timestamp:
        sys.exit("Error: Registration produced no usable poses.")

    odom_max_ns = int(args.odom_max_latency * 1e9)
    odom_ts_sorted = sorted(odom_data.keys())
    cam_ts_sorted = sorted(camera_images.keys())

    # -- Merge + color projection -------------------------------------------
    # BUGFIX: every frame that reaches this merge -- whether it gets a real
    # camera projection or not -- is now guaranteed an explicit colors
    # array (`fill_fallback_color` for the ones that don't project), and
    # the final combine uses `concat_point_clouds` (manual numpy
    # concatenation, shared with color_mesh.py) instead of Open3D's
    # `+`/`+=`, which would otherwise silently strip color from the entire
    # result whenever an uncolored frame was mixed with a colored one.
    #
    # NOTE: this loop iterates over the FULL raw `pointclouds` list and
    # relies on `pose_by_timestamp.get(timestamp)` to skip any frame that
    # motion gating, the odometry health check, or odometry-coverage
    # filtering excluded -- selection is no longer pre-applied above.
    print(f"\n[4/7] Merging registered frames{' with color projection' if color_mode else ''}...")
    uncolored_frames: list[o3d.geometry.PointCloud] = []
    colored_frames: list[o3d.geometry.PointCloud] = []
    merged_frame_count = 0
    camera_colored_count = 0
    fallback_gray_count = 0

    for timestamp, pcd_raw in pointclouds:
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
                pcd_world = color_pcd_from_image(
                    pcd_world, camera_images[cam_ts], cam_pose,
                    intrinsics, args.color_min_depth, args.color_max_depth,
                )
                colored_frames.append(pcd_world)
                colored_this_frame = True
                camera_colored_count += 1

        if not colored_this_frame:
            # BUGFIX (root cause 1): explicitly fallback-color this frame
            # instead of letting it enter the merge with no colors array
            # at all.
            pcd_world = fill_fallback_color(pcd_world)
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
    pcd_combined = concat_point_clouds(merged_pieces)

    print(f"  Merge: {merged_frame_count:,} frame(s) actually merged into the combined cloud.")
    del pointclouds, pose_by_timestamp

    # -- Clean -------------------------------------------------------------
    print("\n[5/7] Cleaning merged cloud...")
    pcd_clean = clean_point_cloud(pcd_combined, args.voxel_size, do_voxel_downsample=False)
    if len(pcd_clean.points) == 0:
        sys.exit("Error: No points remain after cleaning.")
    print(f"  Final cloud: {len(pcd_clean.points):,} points")

    # -- Georeference + export (3 LOD layers) -------------------------------
    print("\n[6/7] Georeferencing (local ENU -> ECEF)...")
    print("\n[7/7] Writing 3D Tiles LOD layers...")
    tiles_dirs, _ = georeference_and_export_lod_tilesets(
        pcd_clean, lat0, lon0, alt0, out_dir, bag_path.stem, args.workers,
        base_voxel_size=args.voxel_size, lod_levels=lod_levels,
    )

    print("\nColored 3D Tiles LOD layers written:")
    for name, tiles_dir in tiles_dirs.items():
        print(f"  {name}: {tiles_dir}")
    print("Done.")


def build_parser(sub):
    p = sub.add_parser(
        "color_tiles_3d",
        help="ROS 2 bag -> colored georeferenced point cloud -> Cesium 3D Tiles (3 LOD layers)"
    )
    p.add_argument("bagpath", help="Path to the ROS 2 bag directory.")
    p.add_argument("outputdir", help="Output directory.")

    # Topics
    p.add_argument("--pc_topic", default="/points",
                    help="PointCloud2 topic (default: /points).")
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
        "--min_move_distance", type=float, default=0.10,
        help="Minimum translation (m), relative to the last KEPT registration frame's "
        "odometry pose, required to keep a new frame. Replaces the old index-based "
        "--frame_stride: odom is the primary pose source, so a frame that hasn't moved "
        "(or turned) adds no new spatial coverage and is skipped instead of thinning "
        "purely by message count. A frame is kept if EITHER this OR "
        "--min_rotation_angle_deg is satisfied. Set to 0 to disable this half of the gate.",
    )
    p.add_argument(
        "--min_rotation_angle_deg", type=float, default=5.0,
        help="Minimum rotation (deg), relative to the last KEPT registration frame's "
        "odometry pose, required to keep a new frame (OR'd with --min_move_distance). "
        "Lets a robot that spins in place without translating still accumulate new "
        "frames to cover the swept field of view. Set to 0 to disable this half of "
        "the gate.",
    )

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
    p.add_argument("--loop_closure_radius", type=float, default=3.0)
    p.add_argument(
        "--loop_closure_fitness_thresh", type=float, default=0.7,
        help="Defaults to the same bar as --icp_fitness_thresh, not a separate looser value.",
    )
    p.add_argument("--loop_closure_search_interval", type=int, default=10)
    p.add_argument(
        "--loop_closure_temporal_window", type=int, default=100,
        help="Bounded number of most-recent candidate frames considered for loop closure.",
    )
    p.add_argument(
        "--max_registration_frames", type=int, default=0,
        help=(
            "Cap total frames used for registration (0 = all). Bounds BOTH raw "
            "frames read ahead of time AND the number of frames kept after the "
            "motion gate and the odometry health check, applied in temporal order."
        ))
    p.add_argument("--merge_chunk_frames", type=int, default=16,
                    help="Number of frames per merge chunk (reserved for future streaming use).")

    p.add_argument(
        "--disable_odom_health_check",
        action="store_true",
        default=False,
        help="Disable the automatic odometry health check that truncates registration "
        "at the first detected tracking loss/teleport (implied speed or rotation rate "
        "far above the bag's own 95th-percentile baseline). Enabled by default so a "
        "lost-odom segment cannot silently skew the output. The segment after a "
        "detected loss is never auto-spliced back in.",
    )
    p.add_argument(
        "--odom_loss_speed_multiplier",
        type=float,
        default=6.0,
        help="Sensitivity of the odometry health check: a consecutive odometry sample "
        "pair is flagged as a tracking loss/teleport if its implied linear speed OR "
        "angular rate exceeds this multiplier times the bag's own 95th-percentile "
        "baseline. Lower = more sensitive (may false-positive on genuinely fast "
        "motion); higher = less sensitive. Ignored if --disable_odom_health_check is set.",
    )

    # LOD / multi-layer output
    p.add_argument(
        "--lod_multipliers", type=str, default="4.0,2.0,1.0",
        help=(
            "Comma-separated voxel-size multipliers (relative to --voxel_size) "
            "for the three output LOD layers, in "
            f"{'/'.join(name for name, _ in DEFAULT_LOD_LEVELS)} order "
            "(default: 4.0,2.0,1.0). Each layer is voxel-downsampled at "
            "voxel_size * multiplier and written to its own "
            "outputdir/tileset_/ folder; a multiplier of 1.0 uses the "
            "cloud's own (already-cleaned) resolution with no extra "
            "downsampling."
        ),
    )

    # Performance
    p.add_argument("--workers", type=int, default=1,
                    help="Parallel workers for KDTree queries and py3dtiles convert.")

    p.set_defaults(func=run)
    return p
