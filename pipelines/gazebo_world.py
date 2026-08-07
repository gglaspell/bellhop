#!/usr/bin/env python3
"""
gazebo_world.py - ROS 2 bag -> registered point cloud -> Poisson mesh
-> Gazebo simulation world (.stl + .sdf + model.config + .world).

PATCH NOTE (odom-anchored registration + frame-awareness):
See odom-anchored-registration-fix-prompt.md and
pointcloud-frame-check-prompt-2.md. Mirrors the same change already applied
to `mesh.py` and `color_mesh.py`, since all three pipelines shared the same
ICP-primary registration pattern via `shared/registration.py`:

- The point cloud's frame_id is detected (or overridden via
  --pc_frame_mode) and classified as 'global' (already in a fixed frame:
  odom/map/world) or 'local' (a moving sensor/base frame).
- 'local' frames are registered via `run_odom_anchored_registration()`,
  which makes timestamped odometry the PRIMARY pose source and demotes ICP
  to an optional, strongly-gated local refinement (--enable_icp_refinement,
  default off). Frames are only ever dropped for lacking odometry coverage,
  never for failing an ICP fitness check.
- 'global' frames skip registration entirely: an identity transform is
  used (a correct no-op), so frames are streamed/filtered/downsampled/
  merged directly with no ICP/pose-graph step. Odom (if provided) is still
  used for view-ray normal orientation.
- CAVEAT: the frame check only detects whether the cloud is already in a
  fixed/global frame -- it does NOT correct for a real sensor-to-base_link
  extrinsic offset (lever arm). See `ros_io.classify_frame_mode` docstring.

FIX (--remesh/--no-remesh flag mismatch):
Previously this pipeline registered the remesh-disable flag as
`--no_remesh` (underscore) via a manual `dest="remesh", action="store_false"`
argument, separate from `--remesh` (`action="store_true"`). gui.py's
generic command builder, however, always emits the hyphenated
`--no-remesh` for any unchecked "remesh" checkbox regardless of which
pipeline profile is active -- the same convention `argparse.BooleanOptionalAction`
produces for `mesh.py`/`color_mesh.py`. Since this pipeline's argparse never
recognized `--no-remesh` (only `--no_remesh`), unchecking "Remesh + smooth"
in the GUI's Gazebo World profile produced a `docker run` command this
pipeline rejected with "unrecognized arguments: --no-remesh". Switching to
`argparse.BooleanOptionalAction` (matching `mesh.py`/`color_mesh.py`) fixes
this by making `--remesh`/`--no-remesh` (hyphenated) the actual registered
flags.

PERF FIX (unbounded, unchunked merge -- see bottleneck analysis):
The merge loop previously did `pcd_combined += pcd_world` for every single
selected frame, with NO periodic voxel-reduction in between. `mesh.py`'s
equivalent merge loop batches frames and calls `voxel_down_sample()` every
`--merge_chunk_frames` frames to keep the accumulated cloud bounded --
this pipeline accepted `--merge_chunk_frames` as a CLI argument but never
actually used it, so the flag was silently a no-op. Two compounding costs
resulted on large bags: (1) `pcd_combined` grew unboundedly across all
selected frames, so each `+=` got progressively more expensive as the
cloud grew (Open3D point-cloud concatenation scales with current size +
new size, so total merge cost trends toward O(n^2) in the accumulated
point count rather than O(n)); (2) that oversized, never-downsampled cloud
then fed directly into `clean_point_cloud()` (called with
`do_voxel_downsample=False`) and Poisson reconstruction, both of which
scale with point count, making every downstream step slower than
necessary. Fixed by porting `mesh.py`'s chunked-merge pattern: frames are
now batched into a bounded list and flushed (concatenated +
voxel-downsampled) every `--merge_chunk_frames` frames, so
`--merge_chunk_frames` now does what its help text always claimed and the
merged cloud's size stays close to the true, no-redundant-points size
throughout the merge rather than only being reduced once at the very end.
"""

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

# ---------------------------------------------------------------------------
# Gazebo template strings
# ---------------------------------------------------------------------------
_CONFIG_TEMPLATE = """<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>bellhop</name>
    <email>bag_to_gazebo@auto@generated.com</email>
  </author>
  <description>3D environment mesh generated from a ROS 2 bag file.</description>
</model>
"""

_SDF_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh><uri>model://{model_name}/meshes/model.stl</uri></mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh><uri>model://{model_name}/meshes/model.stl</uri></mesh>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>{gazebo_material}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

_WORLD_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <include>
      <uri>model://{model_name}</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>
  </world>
</sdf>
"""


# ---------------------------------------------------------------------------
# Gazebo export helper
# ---------------------------------------------------------------------------
def _export_gazebo(mesh: o3d.geometry.TriangleMesh, out_dir: Path,
                    model_name: str, gazebo_material: str) -> None:
    """Write STL mesh + Gazebo model.config, model.sdf, and .world files."""
    models_dir = out_dir / "models" / model_name
    meshes_dir = models_dir / "meshes"
    worlds_dir = out_dir / "worlds"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir.mkdir(parents=True, exist_ok=True)

    # Centre mesh XY at origin; lowest Z sits on ground plane
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = verts.mean(axis=0)
    centroid[2] = verts[:, 2].min()
    mesh.vertices = o3d.utility.Vector3dVector(verts - centroid)
    mesh.compute_triangle_normals()

    stl_path = meshes_dir / "model.stl"
    o3d.io.write_triangle_mesh(str(stl_path), mesh)
    print(f"  STL: {stl_path}")

    (models_dir / "model.config").write_text(
        _CONFIG_TEMPLATE.format(model_name=model_name)
    )
    (models_dir / "model.sdf").write_text(
        _SDF_TEMPLATE.format(model_name=model_name, gazebo_material=gazebo_material)
    )
    (worlds_dir / f"{model_name}.world").write_text(
        _WORLD_TEMPLATE.format(model_name=model_name)
    )

    print(f"  Config: {models_dir / 'model.config'}")
    print(f"  SDF: {models_dir / 'model.sdf'}")
    print(f"  World: {worlds_dir / model_name}.world")


# ---------------------------------------------------------------------------
# PERF FIX: bounded, chunked merge helper (ported from mesh.py's
# append_chunk()). Frames accumulate in `chunk` until it reaches
# `chunk_size`, at which point they're concatenated together and
# voxel-downsampled once as a batch, keeping `target`'s point count close
# to its true (deduplicated) size throughout the merge instead of only
# reducing it once at the very end.
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

    required = [args.pc_topic] + ([args.odom_topic] if args.odom_topic else [])
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(
            f"Error: Required topics missing from bag: {missing}\n"
            "Check topic names with: ros2 bag info <bag_path>"
        )

    # -- Read bag ------------------------------------------------------------
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
        sys.exit("Error: No valid point clouds extracted from bag.")
    if args.odom_topic and not odom_data:
        print("Warning: --odom_topic set but no messages found.")

    print(f"Coverage: frames read = {len(pointclouds):,}, odom poses = {len(odom_data):,}.")

    selected_frames, _original_indices = select_registration_frames(
        pointclouds, frame_stride=args.frame_stride, max_registration_frames=args.max_registration_frames
    )
    print(
        f"Coverage: frames selected = {len(selected_frames):,} / {len(pointclouds):,} "
        f"(stride={args.frame_stride}, max={args.max_registration_frames})."
    )
    del pointclouds
    gc.collect()

    # -- Registration ---------------------------------------------------------
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
        print(f"Registering {len(selected_frames)} point-cloud frames (odom-anchored)...")
        pose_by_timestamp, _stats = run_odom_anchored_registration(selected_frames, odom_data, args)

    if not pose_by_timestamp:
        sys.exit("Error: Registration produced no usable poses.")

    odom_max_ns = int(args.odom_max_latency * 1e9)
    odom_ts_sorted = sorted(odom_data.keys())

    # -- Merge with view-ray normals -------------------------------------------
    # PERF FIX: frames are now batched into a bounded `chunk` and flushed via
    # `_append_chunk()` every `args.merge_chunk_frames` frames, instead of
    # calling `pcd_combined += pcd_world` unconditionally for every single
    # selected frame. This keeps the merged cloud's point count close to its
    # true (voxel-deduplicated) size throughout the merge, so `+=` cost stays
    # roughly linear in the number of frames instead of growing toward O(n^2)
    # as an ever-larger, never-reduced cloud gets re-concatenated on every
    # iteration. `--merge_chunk_frames` now actually does what its help text
    # always described.
    print("Merging registered frames (chunked, bounded memory)...")
    pcd_combined = o3d.geometry.PointCloud()
    chunk: list[o3d.geometry.PointCloud] = []
    chunk_size = max(1, int(args.merge_chunk_frames))
    merged_frame_count = 0

    for timestamp, pcd_raw in tqdm(selected_frames, desc="Merging"):
        transform_world = pose_by_timestamp.get(timestamp)
        if transform_world is None:
            continue

        pcd_world = pcd_raw.voxel_down_sample(args.voxel_size)
        pcd_world.transform(transform_world)

        if odom_ts_sorted:
            interpolated_pose = interpolate_odom_pose(
                timestamp, odom_ts_sorted, odom_data, odom_max_ns
            )
            if interpolated_pose is not None:
                attach_view_rays_as_normals(pcd_world, interpolated_pose[:3, 3])

        chunk.append(pcd_world)
        merged_frame_count += 1

        if len(chunk) >= chunk_size:
            pcd_combined = _append_chunk(pcd_combined, chunk, args.voxel_size)
            gc.collect()

    pcd_combined = _append_chunk(pcd_combined, chunk, args.voxel_size)

    print(f"Merge: {merged_frame_count:,} frame(s) actually merged into the combined cloud.")
    del selected_frames, pose_by_timestamp
    gc.collect()

    if not len(pcd_combined.points):
        sys.exit("Error: No registered points were produced.")

    # -- Clean ------------------------------------------------------------------
    print("Cleaning merged cloud...")
    if args.level_floor:
        pcd_combined = level_floor(pcd_combined)

    pcd_clean = clean_point_cloud(
        pcd_combined, args.voxel_size, do_voxel_downsample=False
    )
    del pcd_combined
    gc.collect()

    view_rays = (
        np.asarray(pcd_clean.normals, dtype=np.float64).copy()
        if pcd_clean.has_normals() else None
    )

    estimate_geometric_normals_oriented(pcd_clean, args.voxel_size, view_rays)

    # -- Reconstruct --------------------------------------------------------------
    print("Running Poisson reconstruction...")
    poisson_depth = args.poisson_depth if args.poisson_depth > 0 else None
    mesh = create_mesh(
        pcd_clean,
        poisson_depth=poisson_depth,
        min_density_percentile=args.min_density_percentile,
        distance_multiplier=args.distance_multiplier,
        max_vertex_distance=args.max_vertex_distance if args.max_vertex_distance > 0 else None,
        remesh=args.remesh,
        remesh_smooth_iterations=args.remesh_smooth_iterations,
        workers=args.workers,
        decimate_target=args.decimate_target,
        curvature_percentile=args.curvature_percentile,
        curvature_protect_rings=args.curvature_protect_rings,
    )

    # -- Save point cloud for reference ------------------------------------------
    ply_path = out_dir / f"{bag_path.stem}_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    print(f"Saved cloud: {ply_path}")

    # -- Gazebo export -------------------------------------------------------------
    print("\nExporting Gazebo world...")
    _export_gazebo(mesh, out_dir, args.model_name, args.gazebo_material)
    print(f"\nGazebo environment written to: {out_dir}")
    print("Done.")


def build_parser(sub):
    p = sub.add_parser(
        "gazebo_world",
        help="ROS 2 bag -> Poisson mesh -> Gazebo simulation world"
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

    # Gazebo
    p.add_argument("--model_name", default="bag_environment",
        help="Gazebo model name (default: bag_environment).")
    p.add_argument("--gazebo_material", default="Gazebo/Grey",
        help="Gazebo material (e.g. Gazebo/White, Gazebo/Wood).")

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
            "streaming merge pass (default 16). Previously accepted but "
            "unused by this pipeline's merge loop; now actually controls "
            "how often the accumulated cloud is voxel-downsampled during "
            "merging."
        ),
    )

    # Reconstruction
    p.add_argument("--poisson_depth", type=int, default=0,
        help="Poisson depth (0 = auto).")
    p.add_argument("--min_density_percentile", type=float, default=1.0,
        help="Bottom %% of Poisson vertex densities to trim (default 1.0).")
    p.add_argument("--distance_multiplier", type=float, default=3.0,
        help="Adaptive vertex distance trim multiplier (default 3.0).")
    p.add_argument("--max_vertex_distance", type=float, default=0.0,
        help="Hard cap on vertex distance (m); 0 = disabled.")
    # FIX: previously registered as two separate manual flags,
    # `--remesh` (store_true) and `--no_remesh` (dest="remesh",
    # store_false, underscore). gui.py always emits the hyphenated
    # `--no-remesh` for an unchecked "remesh" checkbox on every profile
    # (matching the argparse.BooleanOptionalAction convention used by
    # mesh.py/color_mesh.py), which this pipeline's argparse never
    # recognized -- unchecking "Remesh + smooth" in the GUI's Gazebo
    # World profile produced "unrecognized arguments: --no-remesh".
    # BooleanOptionalAction registers both --remesh and --no-remesh
    # (hyphenated) from a single declaration, matching gui.py exactly.
    p.add_argument(
        "--remesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run isotropic remesh + smooth after Poisson (default: on).",
    )
    p.add_argument("--remesh_smooth_iterations", type=int, default=5,
        help="Laplacian smooth iterations during remesh (default 5).")
    p.add_argument("--decimate_target", type=float, default=None,
        help="<=1.0 = fraction of triangles; >1 = absolute count; None = skip.")
    p.add_argument("--curvature_percentile", type=float, default=80.0,
        help="Percentile threshold for curvature-aware decimation (default 80.0).")
    p.add_argument("--curvature_protect_rings", type=int, default=1,
        help="Ring dilation for curvature protection (default 1).")
    p.add_argument("--level_floor", action="store_true", default=False)
    p.add_argument("--workers", type=int, default=4)

    p.set_defaults(func=run)
    return p
