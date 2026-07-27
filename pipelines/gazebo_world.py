#!/usr/bin/env python3
"""
gazebo_world.py – ROS 2 bag -> registered point cloud -> Poisson mesh
-> Gazebo simulation world (.stl + .sdf + model.config + .world).
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
    get_closest_timestamp,
)

# ---------------------------------------------------------------------------
# Gazebo template strings
# ---------------------------------------------------------------------------
_CONFIG_TEMPLATE = """
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>bellhop bag_to_gazebo</name>
    <email>auto@generated.com</email>
  </author>
  <description>3D environment mesh generated from a ROS 2 bag file.</description>
</model>
"""

_SDF_TEMPLATE = """
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><mesh><uri>model://{model_name}/meshes/model.stl</uri></mesh></geometry>
        <material><script><name>{gazebo_material}</name></script></material>
      </visual>
      <collision name="collision">
        <geometry><mesh><uri>model://{model_name}/meshes/model.stl</uri></mesh></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""

_WORLD_TEMPLATE = """
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
    print(f"  SDF:    {models_dir / 'model.sdf'}")
    print(f"  World:  {worlds_dir / model_name}.world")

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args) -> None:
    bag_path = Path(args.bagpath)
    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    required = [args.pc_topic] + ([args.odom_topic] if args.odom_topic else [])
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(
            f"Error: Required topics missing from bag: {missing}\n"
            "Check topic names with: ros2 bag info "
        )

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
        sys.exit("Error: No valid point clouds extracted from bag.")
    if args.odom_topic and not odom_data:
        print("Warning: --odom_topic set but no messages found; using identity guesses.")

    print(f"Extracted {len(pointclouds)} frames, {len(odom_data)} odom poses.")

    # ── ICP + pose graph ──────────────────────────────────────────────────
    posegraph, good_idx = run_icp_posegraph(pointclouds, odom_data, args)
    odom_max_ns = int(args.odom_max_latency * 1e9)
    odom_ts_sorted = sorted(odom_data.keys())

    # ── Merge with view-ray normals ───────────────────────────────────────
    print("Merging registered frames...")
    pcd_combined = o3d.geometry.PointCloud()

    for node_i, pc_i in enumerate(good_idx):
        if node_i >= len(posegraph.nodes):
            break
        T_world = np.linalg.inv(posegraph.nodes[node_i].pose)
        ts, pcd_raw = pointclouds[pc_i]
        pcd_world = pcd_raw.voxel_down_sample(args.voxel_size)
        pcd_world.transform(T_world)

        if odom_ts_sorted:
            cts = get_closest_timestamp(ts, odom_ts_sorted)
            if cts is not None and abs(cts - ts) < odom_max_ns:
                attach_view_rays_as_normals(pcd_world, odom_data[cts][:3, 3])

        pcd_combined += pcd_world

    # ── Clean ─────────────────────────────────────────────────────────────
    print("Cleaning merged cloud...")
    if args.level_floor:
        pcd_combined = level_floor(pcd_combined)

    pcd_clean = clean_point_cloud(
        pcd_combined, args.voxel_size, do_voxel_downsample=False
    )

    view_rays = (
        np.asarray(pcd_clean.normals, dtype=np.float64).copy()
        if pcd_clean.has_normals() else None
    )
    estimate_geometric_normals_oriented(pcd_clean, args.voxel_size, view_rays)

    # ── Reconstruct ───────────────────────────────────────────────────────
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

    # ── Save point cloud for reference ───────────────────────────────────
    ply_path = out_dir / f"{bag_path.stem}_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    print(f"Saved cloud: {ply_path}")

    # ── Gazebo export ─────────────────────────────────────────────────────
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
                   help="Odometry topic (nav_msgs/Odometry). Optional.")

    # Gazebo
    p.add_argument("--model_name", default="bag_environment",
                   help="Gazebo model name (default: bag_environment).")
    p.add_argument("--gazebo_material", default="Gazebo/Grey",
                   help="Gazebo material (e.g. Gazebo/White, Gazebo/Wood).")

    # Registration
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--icp_dist_thresh", type=float, default=0.2)
    p.add_argument("--icp_fitness_thresh", type=float, default=0.6)
    p.add_argument("--odom_max_latency", type=float, default=0.5)
    p.add_argument("--enable_loop_closure", action="store_true", default=False)
    p.add_argument("--loop_closure_radius", type=float, default=10.0)
    p.add_argument("--loop_closure_fitness_thresh", type=float, default=0.3)
    p.add_argument("--loop_closure_search_interval", type=int, default=10)
    p.add_argument("--frame_stride", type=int, default=0,
                   help="Process every Nth frame (0 = all frames).")
    p.add_argument("--max_registration_frames", type=int, default=0,
                   help="Cap total frames used for registration (0 = all).")
    p.add_argument("--merge_chunk_frames", type=int, default=16,
                   help="Number of frames per merge chunk.")

    # Reconstruction
    p.add_argument("--poisson_depth", type=int, default=0,
                   help="Poisson depth (0 = auto).")
    p.add_argument("--min_density_percentile", type=float, default=1.0,
                   help="Bottom %% of Poisson vertex densities to trim (default 1.0).")
    p.add_argument("--distance_multiplier", type=float, default=3.0,
                   help="Adaptive vertex distance trim multiplier (default 3.0).")
    p.add_argument("--max_vertex_distance", type=float, default=0.0,
                   help="Hard cap on vertex distance (m); 0 = disabled.")
    p.add_argument("--remesh", action="store_true", default=True,
                   help="Run isotropic remesh + smooth after Poisson (default: on).")
    p.add_argument("--no_remesh", dest="remesh", action="store_false",
                   help="Disable remesh + smooth.")
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
