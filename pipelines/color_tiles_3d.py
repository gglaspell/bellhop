#!/usr/bin/env python3
"""
color_tiles_3d.py – ROS 2 bag -> registered COLORED point cloud
                    -> georeferenced -> Cesium 3D Tiles.

This pipeline is the union of color_mesh coloring logic and tiles_3d
georeferencing logic.  No mesh is produced; the output is a colored
ECEF point cloud converted to tileset.json via py3dtiles.

Pipeline:
  1. Pre-flight: verify required topics.
  2. Average GPS fixes to establish ENU origin.
  3. Read PointCloud2 + optional Odometry + Camera + CameraInfo.
  4. ICP + pose-graph registration (shared).
  5. Project camera images onto each registered frame.
  6. Merge colored frames (gray-fill filtering + voxel downsample).
  7. Clean merged cloud.
  8. ENU -> ECEF conversion.
  9. Write colored ECEF PLY -> py3dtiles convert -> tileset.json.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from rosbags.highlevel import AnyReader
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from .shared.preflight import check_topics
from .shared.reconstruction import clean_point_cloud, transform_local_enu_to_ecef
from .shared.registration import (
    _safe_normalize,
    attach_view_rays_as_normals,
    run_icp_posegraph,
)
from .shared.ros_io import (
    TYPESTORE,
    convert_ros_image,
    convert_ros_pc2_to_o3d,
    get_closest_timestamp,
    get_odom_transform,
    intrinsics_from_camera_info,
    parse_gps_fixes,
)
from .tiles_3d import _run_py3dtiles_convert, _write_colored_ply_ecef, _write_ply_ecef


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
    pts     = np.asarray(pcd.points)
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
        valid &= depth <= color_max_depth

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
    colored_pcds: list,
    voxel_size: float,
    gray_filter_radius: float,
) -> o3d.geometry.PointCloud:
    """Concatenate per-frame colored clouds, remove gray fill near real color,
    then voxel-downsample.  Returns a single merged PointCloud."""
    all_pts, all_cols, all_nors, all_gray = [], [], [], []

    for pcd in colored_pcds:
        if len(pcd.points) == 0 or not pcd.has_colors():
            continue
        pts  = np.asarray(pcd.points,  dtype=np.float64)
        cols = np.asarray(pcd.colors,  dtype=np.float64)
        nors = (
            np.asarray(pcd.normals, dtype=np.float64)
            if pcd.has_normals()
            else np.zeros((len(pts), 3), dtype=np.float64)
        )
        std  = np.std(cols,  axis=1)
        mean = np.mean(cols, axis=1)
        is_gray = (std < 0.08) & (np.abs(mean - 0.5) < 0.15)
        all_pts.append(pts); all_cols.append(cols)
        all_nors.append(nors); all_gray.append(is_gray)

    if not all_pts:
        raise ValueError("No valid colored point clouds to merge.")

    pts     = np.vstack(all_pts)
    cols    = np.vstack(all_cols)
    nors    = np.vstack(all_nors)
    is_gray = np.hstack(all_gray)

    colored_pts = pts[~is_gray]
    if len(colored_pts) > 0 and gray_filter_radius > 0:
        print(f"  Gray-fill filtering (radius={gray_filter_radius} m)...")
        tree     = cKDTree(colored_pts)
        gray_idx = np.where(is_gray)[0]
        nbrs     = tree.query_ball_point(pts[gray_idx], r=gray_filter_radius)
        has_col  = np.array([len(n) > 0 for n in nbrs], dtype=bool)
        keep     = np.ones(len(pts), dtype=bool)
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
    out_dir  = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    required = [args.pc_topic, args.gps_topic]
    if args.odom_topic:
        required.append(args.odom_topic)
    if args.camera_topic:
        required += [args.camera_topic, args.camera_info_topic]
    missing = check_topics(bag_path, required)
    if missing:
        sys.exit(
            f"Error: Required topics missing from bag: {missing}\n"
            "Check topic names with: ros2 bag info <bag>"
        )

    if args.camera_topic and not args.camera_info_topic:
        sys.exit("Error: --camera_topic requires --camera_info_topic.")

    # ── GPS origin ────────────────────────────────────────────────────────
    print(f"[1/7] Reading GPS fixes from '{args.gps_topic}'...")
    lat0, lon0, alt0 = parse_gps_fixes(bag_path, args.gps_topic)

    # ── Read bag ──────────────────────────────────────────────────────────
    topics = [args.pc_topic]
    if args.odom_topic:       topics.append(args.odom_topic)
    if args.camera_topic:     topics += [args.camera_topic, args.camera_info_topic]

    pointclouds:   list[tuple[int, o3d.geometry.PointCloud]] = []
    odom_data:     dict[int, np.ndarray]  = {}
    camera_images: dict[int, Image.Image] = {}
    cam_info_data: dict[int, tuple]       = {}

    print(f"\n[2/7] Reading messages from: {bag_path}")
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
                elif args.camera_topic and conn.topic == args.camera_topic:
                    img = convert_ros_image(msg)
                    if img is not None:
                        camera_images[ts] = img
                elif args.camera_info_topic and conn.topic == args.camera_info_topic:
                    intr = intrinsics_from_camera_info(msg)
                    if intr is not None:
                        cam_info_data[ts] = intr
            except Exception:
                continue

    if not pointclouds:
        sys.exit("Error: No valid point clouds extracted.")
    if args.odom_topic and not odom_data:
        print("Warning: --odom_topic set but no messages found; using identity guesses.")

    color_mode = bool(args.camera_topic and camera_images and cam_info_data)
    intrinsics = None
    if color_mode:
        first_ts   = min(cam_info_data.keys())
        intrinsics = cam_info_data[first_ts]
        fx, fy, cx, cy, cw, ch = intrinsics
        print(f"  Camera: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} {cw}x{ch}")
        print(f"  Images: {len(camera_images)}")
    else:
        print("  No camera data found; producing XYZ-only tiles.")

    print(f"  Frames: {len(pointclouds)} | Odom: {len(odom_data)}")

    # ── ICP + pose graph ──────────────────────────────────────────────────
    print(f"\n[3/7] ICP registration + pose-graph optimisation...")
    posegraph, good_idx = run_icp_posegraph(pointclouds, odom_data, args)
    odom_max_ns    = int(args.odom_max_latency * 1e9)
    odom_ts_sorted = sorted(odom_data.keys())
    cam_ts_sorted  = sorted(camera_images.keys())

    # ── Merge + color projection ──────────────────────────────────────────
    print(f"\n[4/7] Merging registered frames{' with color projection' if color_mode else ''}...")
    pcd_combined   = o3d.geometry.PointCloud()
    colored_frames: list[o3d.geometry.PointCloud] = []

    for node_i, pc_i in enumerate(good_idx):
        if node_i >= len(posegraph.nodes):
            break
        T_world     = np.linalg.inv(posegraph.nodes[node_i].pose)
        ts, pcd_raw = pointclouds[pc_i]
        pcd_world   = pcd_raw.voxel_down_sample(args.voxel_size)
        pcd_world.transform(T_world)

        if odom_ts_sorted:
            cts = get_closest_timestamp(ts, odom_ts_sorted)
            if cts is not None and abs(cts - ts) < odom_max_ns:
                attach_view_rays_as_normals(pcd_world, odom_data[cts][:3, 3])

        if color_mode and cam_ts_sorted:
            cam_ts = get_closest_timestamp(ts, cam_ts_sorted)
            if cam_ts is not None and abs(cam_ts - ts) < int(args.max_time_diff * 1e9):
                # Use the odometry pose at frame time as the camera pose
                cam_pose = np.eye(4, dtype=np.float64)
                if odom_ts_sorted:
                    cts2 = get_closest_timestamp(ts, odom_ts_sorted)
                    if cts2 is not None and abs(cts2 - ts) < odom_max_ns:
                        cam_pose = odom_data[cts2]
                pcd_world = _color_pcd_from_image(
                    pcd_world, camera_images[cam_ts], cam_pose,
                    intrinsics, args.color_min_depth, args.color_max_depth,
                )
                colored_frames.append(pcd_world)
                continue

        pcd_combined += pcd_world

    if color_mode and colored_frames:
        print(f"  Merging {len(colored_frames)} colored frames...")
        merged_color = _merge_colored_pcds(
            colored_frames, args.voxel_size, args.gray_filter_radius
        )
        pcd_combined += merged_color
    elif color_mode:
        print("  Warning: no colored frames produced; check --max_time_diff.")

    # ── Clean ─────────────────────────────────────────────────────────────
    print("\n[5/7] Cleaning merged cloud...")
    pcd_clean = clean_point_cloud(
        pcd_combined, args.voxel_size, do_voxel_downsample=False
    )
    if len(pcd_clean.points) == 0:
        sys.exit("Error: No points remain after cleaning.")
    print(f"  Final cloud: {len(pcd_clean.points):,} points")

    # ── ENU -> ECEF ───────────────────────────────────────────────────────
    print("\n[6/7] Georeferencing (local ENU -> ECEF)...")
    pts_enu  = np.asarray(pcd_clean.points, dtype=np.float64)
    pts_ecef = transform_local_enu_to_ecef(pts_enu, lat0, lon0, alt0)

    # ── Write ECEF PLY + py3dtiles ────────────────────────────────────────
    print("\n[7/7] Writing 3D Tiles...")
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        ply_path = Path(tmp.name)

    has_colors = pcd_clean.has_colors()
    if has_colors:
        colors = np.asarray(pcd_clean.colors, dtype=np.float64)
        _write_colored_ply_ecef(pts_ecef, colors, ply_path)
    else:
        _write_ply_ecef(pts_ecef, ply_path)

    tiles_dir = out_dir / "tileset"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_py3dtiles_convert(ply_path, tiles_dir, jobs=args.workers)
    finally:
        ply_path.unlink(missing_ok=True)

    enu_ply = out_dir / f"{bag_path.stem}_cloud_enu.ply"
    o3d.io.write_point_cloud(str(enu_ply), pcd_clean)
    print(f"  ENU cloud: {enu_ply}")
    print(f"\nColored 3D Tiles written to: {tiles_dir}")
    print("Done.")


def build_parser(sub):
    p = sub.add_parser(
        "color_tiles_3d",
        help="ROS 2 bag -> colored georeferenced point cloud -> Cesium 3D Tiles"
    )
    p.add_argument("bagpath",   help="Path to the ROS 2 bag directory.")
    p.add_argument("outputdir", help="Output directory.")

    # Topics
    p.add_argument("--pc_topic",          default="points")
    p.add_argument("--odom_topic",        default=None)
    p.add_argument("--gps_topic",         default="/gps/fix")
    p.add_argument("--camera_topic",      default=None,
                   help="sensor_msgs/Image or CompressedImage topic. Optional.")
    p.add_argument("--camera_info_topic", default=None,
                   help="sensor_msgs/CameraInfo topic. Required with --camera_topic.")

    # Color
    p.add_argument("--max_time_diff",      type=float, default=0.1,
                   help="Max timestamp diff (s) between PC frame and camera image.")
    p.add_argument("--color_min_depth",    type=float, default=0.1)
    p.add_argument("--color_max_depth",    type=float, default=None)
    p.add_argument("--gray_filter_radius", type=float, default=0.05,
                   help="Gray-fill points with a real-color neighbor within this "
                        "radius (m) are removed. 0 = disable.")

    # Registration
    p.add_argument("--voxel_size",           type=float, default=0.05)
    p.add_argument("--icp_dist_thresh",      type=float, default=0.2)
    p.add_argument("--icp_fitness_thresh",   type=float, default=0.6)
    p.add_argument("--odom_max_latency",     type=float, default=0.5)
    p.add_argument("--enable_loop_closure",  action="store_true", default=False)
    p.add_argument("--loop_closure_radius",          type=float, default=10.0)
    p.add_argument("--loop_closure_fitness_thresh",  type=float, default=0.3)
    p.add_argument("--loop_closure_search_interval", type=int,   default=10)

    # Performance
    p.add_argument("--workers", type=int, default=4)

    p.set_defaults(func=run)
    return p
