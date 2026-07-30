#!/usr/bin/env python3
"""ROS 2 bag -> keyframe-baked textured mesh (atlas UV pipeline)."""

from __future__ import annotations
import argparse, csv, gc, sys
from pathlib import Path

from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R

from .shared.preflight import check_topics
from .shared.ros_io import TYPESTORE, get_odom_transform, intrinsics_from_camera_info

from .atlas_pipeline.common.packaging import create_atak_zip, setup_workspace
from .atlas_pipeline.common.trajectory import load_trajectory, build_trajectory_tree
from .atlas_pipeline.keyframeselector import KeyframeSelector
from .atlas_pipeline.meshgenerator import MeshGenerator
from .atlas_pipeline.pointcloudutils import PointCloudProcessor
from .atlas_pipeline.viewassignment import ViewAssigner
from .atlas_pipeline.atlaspacker import AtlasPacker
from .atlas_pipeline.texturebaker import TextureBaker


def _read_intrinsics_and_trajectory(bag_path, args, trajectory_csv):
    """Single cheap pass: CameraInfo -> intrinsics, odom -> trajectory.csv."""
    intrinsics_tuple, rows = None, []
    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections
                 if c.topic in (args.camera_info_topic, args.odom_topic)]
        for connection, ts_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, connection.msgtype)
            if connection.topic == args.camera_info_topic and intrinsics_tuple is None:
                intrinsics_tuple = intrinsics_from_camera_info(msg)
            elif connection.topic == args.odom_topic:
                transform = get_odom_transform(msg)
                if transform is not None:
                    q = R.from_matrix(transform[:3, :3]).as_quat()
                    rows.append({
                        "timestamp": ts_ns * 1e-9,
                        "pos.x": transform[0, 3], "pos.y": transform[1, 3], "pos.z": transform[2, 3],
                        "orient.x": q[0], "orient.y": q[1], "orient.z": q[2], "orient.w": q[3],
                    })

    if intrinsics_tuple is None:
        sys.exit("Error: No CameraInfo message found.")
    if not rows:
        sys.exit("Error: No usable odometry messages found.")

    rows.sort(key=lambda r: r["timestamp"])
    with trajectory_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    fx, fy, cx, cy, width, height = intrinsics_tuple
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}


def run(args: argparse.Namespace) -> None:
    bag_path = Path(args.bagpath)
    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    required = [args.pc_topic, args.camera_topic, args.camera_info_topic, args.odom_topic]
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(f"Error: Required topics missing from bag: {missing}")

    paths = setup_workspace(Path(args.outputdir), overwrite=args.overwrite)
    workspace = paths["workspace"]
    keyframe_images_dir = workspace / "keyframe_images"
    intermediate_dir = workspace / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    trajectory_csv = workspace / "trajectory.csv"

    print("Reading intrinsics and trajectory from bag...")
    intrinsics = _read_intrinsics_and_trajectory(bag_path, args, trajectory_csv)

    print("Selecting keyframes directly from bag...")
    selector = KeyframeSelector.for_bag(args.min_movement_m, args.min_rotation_deg)
    keyframes = selector.select_keyframes_from_bag(
        bag_path, args.camera_topic, args.odom_topic, keyframe_images_dir,
    )
    if not keyframes:
        sys.exit("Error: No keyframes could be selected from bag.")
    selector.save_keyframes(keyframes, workspace / "keyframes.csv")
    print(f"Selected {len(keyframes)} keyframes.")

    traj_df = load_trajectory(trajectory_csv)
    traj_tree = build_trajectory_tree(traj_df)

    print("Merging and frustum-filtering point clouds directly from bag...")
    pc_processor = PointCloudProcessor()
    pc_processor.merge_point_clouds_from_bag(
        bag_path, args.pc_topic, keyframes, traj_df, traj_tree, intrinsics,
        paths["merged_ply"], paths["timestamps"],
        min_frame_points=args.min_frame_points,
        ror_radius=args.ror_radius, ror_min_neighbors=args.ror_min_neighbors,
        sor_neighbors=args.sor_neighbors, sor_std_ratio=args.sor_std_ratio,
    )
    pc_processor.subsample(
        paths["merged_ply"], paths["timestamps"],
        paths["subsampled_ply"], paths["subsampled_timestamps"], res=args.voxel_size,
    )
    pc_processor.compute_normals(
        paths["subsampled_ply"], paths["subsampled_timestamps"],
        trajectory_csv, paths["normals_ply"],
    )
    gc.collect()

    mesh_gen = MeshGenerator()
    print("Running Poisson reconstruction...")
    mesh_gen.poisson_reconstruct(
        paths["normals_ply"], paths["raw_mesh"],
        depth=args.poisson_depth, max_distance=args.poisson_max_distance,
    )
    print("Smoothing mesh...")
    mesh_gen.smooth_mesh(
        paths["raw_mesh"], paths["cleaned_mesh"],
        iterations=args.smooth_iterations, method=args.smooth_method,
        lambda_filter=args.smooth_lambda,
    )
    print("Culling invisible faces...")
    culled_mesh_path = intermediate_dir / "culled_mesh.ply"
    mesh_gen.cull_invisible_faces(
        paths["cleaned_mesh"], culled_mesh_path, keyframes, trajectory_csv,
        intrinsics, min_angle=args.cull_min_angle, target_faces=args.target_faces,
    )

    print("Assigning faces to keyframe views...")
    assigner = ViewAssigner(
        culled_mesh_path, keyframes, trajectory_csv, intrinsics,
        min_angle=args.assign_min_angle, max_bake_distance=args.max_bake_distance,
        min_bake_distance=args.min_bake_distance,
        assignment_smooth_iterations=args.assignment_smooth_iterations,
    )
    assignments = assigner.assign_faces_to_views()

    print("Packing UV atlas...")
    uv_mesh_path = intermediate_dir / "uv_mesh.obj"
    packer = AtlasPacker(
        culled_mesh_path, assignments, keyframes, trajectory_csv, intrinsics, args.atlas_size,
    )
    packer.pack_and_generate_uvs(uv_mesh_path)

    print("Baking texture atlas...")
    stem = bag_path.stem
    final_mesh_obj = workspace / f"{stem}_baked_mesh.obj"
    final_texture_png = workspace / f"{stem}_baked_mesh_texture.png"
    baker = TextureBaker(
        uv_mesh_path, assignments, keyframes, keyframe_images_dir,
        trajectory_csv, intrinsics, args.atlas_size,
    )
    baker.bake_texture(final_mesh_obj, final_texture_png)

    print("Packaging ATAK zip...")
    final_zip = create_atak_zip(final_mesh_obj, paths["final_zip"], png_path=final_texture_png)

    print(f"Saved baked mesh: {final_mesh_obj}")
    print(f"Saved baked texture: {final_texture_png}")
    print(f"Saved ATAK zip: {final_zip}")
    print("Done.")


def build_parser(sub):
    parser = sub.add_parser(
        "texture_baking",
        help="ROS 2 bag -> keyframe-baked textured mesh (atlas UV pipeline)",
    )
    parser.add_argument("bagpath", help="Path to the ROS 2 bag.")
    parser.add_argument("outputdir", help="Output directory (workspace).")

    parser.add_argument("--pc_topic", default="points")
    parser.add_argument("--camera_topic", required=True)
    parser.add_argument("--camera_info_topic", required=True)
    parser.add_argument("--odom_topic", required=True)

    parser.add_argument("--min_frame_points", type=int, default=100)
    parser.add_argument("--voxel_size", type=float, default=0.05)

    parser.add_argument("--ror_radius", type=float, default=0.0)
    parser.add_argument("--ror_min_neighbors", type=int, default=10)
    parser.add_argument("--sor_neighbors", type=int, default=20)
    parser.add_argument("--sor_std_ratio", type=float, default=2.0)

    parser.add_argument("--min_movement_m", type=float, default=0.5)
    parser.add_argument("--min_rotation_deg", type=float, default=15.0)

    parser.add_argument("--poisson_depth", type=int, default=8)
    parser.add_argument("--poisson_max_distance", type=float, default=0.5)

    parser.add_argument("--smooth_method", choices=("taubin", "laplacian"), default="taubin")
    parser.add_argument("--smooth_iterations", type=int, default=5)
    parser.add_argument("--smooth_lambda", type=float, default=0.5)

    parser.add_argument("--cull_min_angle", type=float, default=75.0)
    parser.add_argument("--target_faces", type=int, default=None)

    parser.add_argument("--assign_min_angle", type=float, default=75.0)
    parser.add_argument("--max_bake_distance", type=float, default=4.0)
    parser.add_argument("--min_bake_distance", type=float, default=0.4)
    parser.add_argument("--assignment_smooth_iterations", type=int, default=3)

    parser.add_argument("--atlas_size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true", default=False)

    parser.set_defaults(func=run)
    return parser
