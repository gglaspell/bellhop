#!/usr/bin/env python3
"""
tiles_3d.py - ROS 2 bag -> georeferenced registered point cloud -> Cesium 3D Tiles.

Pipeline:
1. Pre-flight: verify required topics exist.
2. Average GPS fixes to establish ENU origin (lat0, lon0, alt0).
3. Read PointCloud2 + optional Odometry messages.
4. Odom-anchored registration (shared), or identity transform if the point
   cloud is already published in a global/fixed frame.
5. Merge transformed frames into one world-frame cloud.
6. Clean (voxel -> ROR -> SOR -> DBSCAN).
7. Convert local ENU coords to ECEF (EPSG:4978).
8. Write a temp .ply then call py3dtiles convert -> tileset.json output.

REFACTOR NOTE: Bag-reading, world-frame merge, and the ENU->ECEF->PLY->
py3dtiles tail-end logic live in `shared/tiles_common.py` and are shared
with `color_tiles_3d.py`, so bugfixes to that shared logic no longer need
to be applied twice by hand.

PATCH NOTE (odom-anchored registration + frame-awareness):
See odom-anchored-registration-fix-prompt.md and
pointcloud-frame-check-prompt-2.md. Mirrors the same change already applied
to `mesh.py`, `color_mesh.py`, and `gazebo_world.py`:

- The point cloud's frame_id is detected (or overridden via
  --pc_frame_mode) and classified as 'global' (odom/map/world) or 'local'
  (a moving sensor/base frame).
- 'local' frames are registered via `run_odom_anchored_registration()`
  (odometry primary, ICP optional/gated, loop closure gated at the same
  bar and weighted so it can only nudge the graph).
- 'global' frames use an identity transform (a correct no-op -- no ICP,
  no pose graph), so frames are streamed/filtered/downsampled/merged
  directly. Odom (if provided) is still used for view-ray normal
  orientation via `tiles_common.transform_frame_to_world`.

CLEANUP NOTE: the previous backward-compatibility re-exports of
`run_py3dtiles_convert`, `write_ply_ecef`, and `write_colored_ply_ecef`
(as `_run_py3dtiles_convert`, `_write_ply_ecef`, `_write_colored_ply_ecef`)
have been removed. `color_tiles_3d.py` -- their only documented consumer --
already imports these directly from `shared/tiles_common` and does not
reference this module, so the aliases were dead code.

PERF FIX (unbounded, unchunked merge -- see gazebo_world.py's bottleneck
analysis, which surfaced the identical bug here):
The merge loop previously did `pcd_combined += pcd_world` for every single
selected frame, with NO periodic voxel-reduction in between. This
pipeline's own `--merge_chunk_frames` help text openly admitted the flag
was "reserved for future streaming use" -- i.e. accepted but never wired
up. Two compounding costs resulted on large bags: (1) `pcd_combined` grew
unboundedly across all selected frames, so each `+=` got progressively
more expensive as the cloud grew (Open3D point-cloud concatenation scales
with current size + new size, trending total merge cost toward O(n^2) in
the accumulated point count rather than O(n)); (2) that oversized,
never-downsampled cloud then fed directly into `clean_point_cloud()`
(called with `do_voxel_downsample=False`) and the ENU->ECEF->PLY->
py3dtiles export tail, both of which scale with point count. Fixed by
porting `mesh.py`/`gazebo_world.py`'s chunked-merge pattern: frames are
now batched into a bounded list and flushed (concatenated +
voxel-downsampled) every `--merge_chunk_frames` frames, so the flag now
actually does what its help text describes instead of being reserved and
unused.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud
from .shared.registration import run_odom_anchored_registration, select_registration_frames
from .shared.ros_io import convert_ros_pc2_to_o3d, get_odom_transform, parse_gps_fixes, resolve_pc_frame_mode
from .shared.tiles_common import (
    georeference_and_export_tileset,
    read_bag_topics,
    transform_frame_to_world,
)


# ---------------------------------------------------------------------------
# PERF FIX: bounded, chunked merge helper (ported from mesh.py's
# append_chunk() / gazebo_world.py's _append_chunk()). Frames accumulate in
# `chunk` until it reaches `chunk_size`, at which point they're
# concatenated together and voxel-downsampled once as a batch, keeping
# `target`'s point count close to its true (deduplicated) size throughout
# the merge instead of only reducing it once at the very end.
# ---------------------------------------------------------------------------
def _append_chunk(
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
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

    required = [args.pc_topic, args.gps_topic]
    if args.odom_topic:
        required.append(args.odom_topic)
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(
            f"Error: Required topics missing from bag: {missing}\n"
            "Check topic names with: ros2 bag info <bag_path>"
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
        print("Warning: --odom_topic set but no messages found.")
    print(f"  Coverage: frames read = {len(pointclouds):,} | odom poses = {len(odom_data):,}")

    selected_frames, _original_indices = select_registration_frames(
        pointclouds, frame_stride=args.frame_stride, max_registration_frames=args.max_registration_frames
    )
    print(
        f"  Coverage: frames selected = {len(selected_frames):,} / {len(pointclouds):,} "
        f"(stride={args.frame_stride}, max={args.max_registration_frames})."
    )
    del pointclouds
    gc.collect()

    # -- Registration --------------------------------------------------
    if frame_mode == "global":
        print(
            "\n[3/6] Point cloud already in a global/fixed frame: skipping "
            "ICP/pose-graph registration and per-frame transform application."
        )
        pose_by_timestamp: dict[int, np.ndarray] = {
            timestamp: np.eye(4, dtype=np.float64) for timestamp, _ in selected_frames
        }
        print(f"  Coverage: frames with valid pose = {len(pose_by_timestamp):,} (identity; no registration needed).")
    else:
        print("\n[3/6] Odom-anchored registration...")
        pose_by_timestamp, _stats = run_odom_anchored_registration(selected_frames, odom_data, args)

    if not pose_by_timestamp:
        sys.exit("Error: Registration produced no usable poses.")

    odom_max_ns = int(args.odom_max_latency * 1e9)
    odom_ts_sorted = sorted(odom_data.keys())

    # -- Merge world-frame cloud -------------------------------------------
    # PERF FIX: frames are now batched into a bounded `chunk` and flushed via
    # `_append_chunk()` every `args.merge_chunk_frames` frames, instead of
    # calling `pcd_combined += pcd_world` unconditionally for every single
    # selected frame. This keeps the merged cloud's point count close to its
    # true (voxel-deduplicated) size throughout the merge, so `+=` cost stays
    # roughly linear in the number of frames instead of growing toward O(n^2)
    # as an ever-larger, never-reduced cloud gets re-concatenated on every
    # iteration. `--merge_chunk_frames` now actually does what its help text
    # describes, instead of being accepted-but-unused.
    print("\n[4/6] Merging registered frames (chunked, bounded memory)...")
    pcd_combined = o3d.geometry.PointCloud()
    chunk: list[o3d.geometry.PointCloud] = []
    chunk_size = max(1, int(args.merge_chunk_frames))
    merged_frame_count = 0

    for timestamp, pcd_raw in selected_frames:
        transform_world = pose_by_timestamp.get(timestamp)
        if transform_world is None:
            continue
        pcd_world = transform_frame_to_world(
            pcd_raw, transform_world, args.voxel_size,
            timestamp=timestamp, odom_data=odom_data,
            odom_ts_sorted=odom_ts_sorted, odom_max_ns=odom_max_ns,
        )

        chunk.append(pcd_world)
        merged_frame_count += 1

        if len(chunk) >= chunk_size:
            pcd_combined = _append_chunk(pcd_combined, chunk, args.voxel_size)
            gc.collect()

    pcd_combined = _append_chunk(pcd_combined, chunk, args.voxel_size)

    print(f"  Merge: {merged_frame_count:,} frame(s) actually merged into the combined cloud.")
    del selected_frames, pose_by_timestamp
    gc.collect()

    # -- Clean -------------------------------------------------------------
    print("\n[5/6] Cleaning merged cloud...")
    pcd_clean = clean_point_cloud(
        pcd_combined, args.voxel_size, do_voxel_downsample=False
    )
    del pcd_combined
    gc.collect()

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
        help=(
            "Odometry topic (nav_msgs/Odometry). Required unless the "
            "point cloud is already published in a global/fixed frame "
            "(see --pc_frame_mode)."
        ))
    p.add_argument("--gps_topic", default="/gps/fix",
        help="NavSatFix topic for GPS origin (default: /gps/fix).")
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
    p.add_argument(
        "--merge_chunk_frames", type=int, default=16,
        help=(
            "Frames merged per batch before each voxel reduction in the "
            "streaming merge pass (default 16). PERF FIX: previously "
            "accepted but documented as \"reserved for future streaming "
            "use\" -- never actually wired into the merge loop, which did "
            "an unbounded, unchunked pcd_combined += pcd_world for every "
            "frame instead. This flag now actually controls how often the "
            "accumulated cloud is voxel-downsampled during merging."
        ),
    )

    # Performance
    p.add_argument("--workers", type=int, default=4,
        help="Parallel workers for KDTree queries and py3dtiles convert.")

    p.set_defaults(func=run)
    return p
