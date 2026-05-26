#!/usr/bin/env python3
"""
color_mesh.py – ROS 2 bag -> registered colored point cloud -> Poisson mesh.
"""

import sys
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
    convert_ros_pc2_to_o3d,
    convert_ros_image,
    get_odom_transform,
    get_closest_timestamp,
    intrinsics_from_camera_info,
)


# ---------------------------------------------------------------------------
# Color projection helpers
# ---------------------------------------------------------------------------
def _color_pcd_from_image(
    pcd, img: Image.Image, camera_pose: np.ndarray,
    intrinsics: tuple, color_min_depth=0.1, color_max_depth=None,
) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy, img_w, img_h = intrinsics
    pts    = np.asarray(pcd.points)
    img_arr = np.asarray(img)
    cam_pos = camera_pose[:3, 3]
    cam_rot = R.from_matrix(camera_pose[:3, :3])

    body  = cam_rot.inv().apply(pts - cam_pos)
    opt_x = -body[:, 1]
    opt_y = -body[:, 2]
    opt_z =  body[:, 0]
    depth = np.linalg.norm(body, axis=1)

    valid = (opt_z > 1e-6) & (depth >= color_min_depth)
    if color_max_depth is not None:
        valid &= (depth <= color_max_depth)

    z_safe = np.where(opt_z > 1e-6, opt_z, 1e-6)
    u = fx * (opt_x / z_safe) + cx
    v = fy * (opt_y / z_safe) + cy
    valid &= (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    colors = np.full((len(pts), 3), 0.5, dtype=np.float64)
    if np.any(valid):
        ui = np.clip(u[valid].astype(np.int32), 0, img_w - 1)
        vi = np.clip(v[valid].astype(np.int32), 0, img_h - 1)
        colors[valid] = img_arr[vi, ui] / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _merge_colored_pcds(
    colored_pcds: list, voxel_size: float, gray_filter_radius: float, workers: int = 4
) -> o3d.geometry.PointCloud:
    all_pts, all_cols, all_nors, all_gray = [], [], [], []
    for pcd in colored_pcds:
        if len(pcd.points) == 0 or not pcd.has_colors():
            continue
        pts  = np.asarray(pcd.points,  dtype=np.float64)
        cols = np.asarray(pcd.colors,  dtype=np.float64)
        nors = (np.asarray(pcd.normals, dtype=np.float64)
                if pcd.has_normals() else np.zeros((len(pts), 3)))
        std  = np.std(cols, axis=1)
        mean = np.mean(cols, axis=1)
        is_gray = (std < 0.08) & (np.abs(mean - 0.5) < 0.15)
        all_pts.append(pts); all_cols.append(cols)
        all_nors.append(nors); all_gray.append(is_gray)

    if not all_pts:
        raise ValueError("No valid colored point clouds to merge.")

    pts   = np.vstack(all_pts)
    cols  = np.vstack(all_cols)
    nors  = np.vstack(all_nors)
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

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(pts)
    merged.colors = o3d.utility.Vector3dVector(cols)
    if np.any(np.linalg.norm(nors, axis=1) > 0):
        from .shared.registration import _safe_normalize
        merged.normals = o3d.utility.Vector3dVector(_safe_normalize(nors))

    return merged.voxel_down_sample(voxel_size)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args) -> None:
    bag_path = Path(args.bagpath)
    out_dir  = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    required = [args.pc_topic] + ([args.odom_topic] if args.odom_topic else [])
    if args.camera_topic:
        required += [args.camera_topic, args.camera_info_topic]
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(f"Error: Required topics missing from bag: {missing}")

    if args.camera_topic and not args.camera_info_topic:
        sys.exit("Error: --camera_topic requires --camera_info_topic.")

    topics = [args.pc_topic]
    if args.odom_topic:    topics.append(args.odom_topic)
    if args.camera_topic:  topics += [args.camera_topic, args.camera_info_topic]

    pointclouds:   list = []
    odom_data:     dict = {}
    camera_images: dict = {}
    cam_info_data: dict = {}

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
                    if T is not None: odom_data[ts] = T
                elif args.camera_topic and conn.topic == args.camera_topic:
                    img = convert_ros_image(msg)
                    if img is not None: camera_images[ts] = img
                elif args.camera_info_topic and conn.topic == args.camera_info_topic:
                    intr = intrinsics_from_camera_info(msg)
                    if intr is not None: cam_info_data[ts] = intr
            except Exception:
                continue

    if not pointclouds:
        sys.exit("Error: No valid point clouds extracted.")

    color_mode = bool(args.camera_topic and camera_images and cam_info_data)
    intrinsics = None
    if color_mode:
        first_ts   = min(cam_info_data.keys())
        intrinsics = cam_info_data[first_ts]

    odom_max_ns = int(args.odom_max_latency * 1e9)
    posegraph, good_idx = run_icp_posegraph(pointclouds, odom_data, args)
    odom_ts_sorted = sorted(odom_data.keys())
    cam_ts_sorted  = sorted(camera_images.keys())

    print("Merging registered frames...")
    pcd_combined = o3d.geometry.PointCloud()
    colored_frames = []

    for node_i, pc_i in enumerate(good_idx):
        if node_i >= len(posegraph.nodes):
            break
        T_world = np.linalg.inv(posegraph.nodes[node_i].pose)
        ts, pcd_raw = pointclouds[pc_i]

        pcd_world = pcd_raw.voxel_down_sample(args.voxel_size)
        pcd_world.transform(T_world)

        if odom_ts_sorted:
            cts = get_closest_timestamp(ts, odom_ts_sorted)
            if cts and abs(cts - ts) < odom_max_ns:
                attach_view_rays_as_normals(pcd_world, odom_data[cts][:3, 3])

        if color_mode and cam_ts_sorted:
            cam_ts = get_closest_timestamp(ts, cam_ts_sorted)
            if cam_ts and abs(cam_ts - ts) < int(args.max_time_diff * 1e9):
                cam_pose = odom_data.get(
                    get_closest_timestamp(ts, odom_ts_sorted), np.eye(4)
                ) if odom_ts_sorted else np.eye(4)
                pcd_world = _color_pcd_from_image(
                    pcd_world, camera_images[cam_ts], cam_pose,
                    intrinsics, args.color_min_depth, args.color_max_depth,
                )
                colored_frames.append(pcd_world)
                continue

        pcd_combined += pcd_world

    if color_mode and colored_frames:
        print(f"Merging {len(colored_frames)} colored frames...")
        merged_color = _merge_colored_pcds(
            colored_frames, args.voxel_size, args.gray_filter_radius, args.workers
        )
        pcd_combined += merged_color

    if args.level_floor:
        pcd_combined = level_floor(pcd_combined)

    pcd_clean = clean_point_cloud(pcd_combined, args.voxel_size, do_voxel_downsample=False)
    view_rays = (
        np.asarray(pcd_clean.normals, dtype=np.float64).copy()
        if pcd_clean.has_normals() else None
    )
    estimate_geometric_normals_oriented(pcd_clean, args.voxel_size, view_rays)

    print("Running Poisson reconstruction...")
    mesh = create_mesh(
        pcd_clean,
        poisson_depth=args.poisson_depth,
        min_density_percentile=args.min_density_percentile,
        max_vertex_distance=args.max_vertex_distance,
        workers=args.workers,
        decimate_target=args.decimate_target,
    )

    stem = bag_path.stem
    ply_path = out_dir / f"{stem}_cloud.ply"
    obj_path = out_dir / f"{stem}_mesh.obj"
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    print(f"Saved cloud: {ply_path}")
    print(f"Saved mesh:  {obj_path}")
    print("Done.")


def build_parser(sub):
    p = sub.add_parser("color_mesh", help="ROS 2 bag -> colored Poisson mesh")
    p.add_argument("bagpath");  p.add_argument("outputdir")
    p.add_argument("--pc_topic",          default="points")
    p.add_argument("--odom_topic",        default=None)
    p.add_argument("--camera_topic",      default=None)
    p.add_argument("--camera_info_topic", default=None)
    p.add_argument("--camera_fx",   type=float, default=None)
    p.add_argument("--camera_fy",   type=float, default=None)
    p.add_argument("--camera_cx",   type=float, default=None)
    p.add_argument("--camera_cy",   type=float, default=None)
    p.add_argument("--camera_width",  type=int, default=None)
    p.add_argument("--camera_height", type=int, default=None)
    p.add_argument("--max_time_diff",      type=float, default=0.1)
    p.add_argument("--voxel_size",         type=float, default=0.05)
    p.add_argument("--icp_dist_thresh",    type=float, default=0.2)
    p.add_argument("--icp_fitness_thresh", type=float, default=0.6)
    p.add_argument("--odom_max_latency",   type=float, default=0.5)
    p.add_argument("--enable_loop_closure",          action="store_true")
    p.add_argument("--loop_closure_radius",          type=float, default=10.0)
    p.add_argument("--loop_closure_fitness_thresh",  type=float, default=0.3)
    p.add_argument("--loop_closure_search_interval", type=int,   default=10)
    p.add_argument("--gray_filter_radius",    type=float, default=0.05)
    p.add_argument("--color_min_depth",       type=float, default=0.1)
    p.add_argument("--color_max_depth",       type=float, default=None)
    p.add_argument("--poisson_depth",          type=int,   default=9)
    p.add_argument("--min_density_percentile", type=float, default=1.0)
    p.add_argument("--max_vertex_distance",    type=float, default=0.15)
    p.add_argument("--decimate_target",        type=float, default=None)
    p.add_argument("--level_floor",            action="store_true")
    p.add_argument("--workers",                type=int,   default=4)
    p.set_defaults(func=run)
    return p
