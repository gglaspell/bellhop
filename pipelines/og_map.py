#!/usr/bin/env python3
"""
og_map.py – ROS 2 bag -> 2D Nav2 occupancy grid.

Algorithm:
  1. Load odometry poses.
  2. Ray-cast each point-cloud frame into a 3D OcTree.
  3. Separate ground vs. obstacles using surface normals.
  4. Build a 2D ground-height map.
  5. Project obstacles into the occupancy grid.
  6. Denoise (morphological closing + connected-component filter).
  7. Write .pgm + .yaml.
"""

import bisect
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import open3d as o3d
import pyoctomap
import yaml
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.interpolate import griddata
from scipy.ndimage import binary_closing, label

from .shared.preflight import check_topics
from .shared.ros_io import (
    TYPESTORE,
    get_closest_timestamp,
    get_odom_transform_matrix,
    pointcloud2_to_numpy,
)

# ---------------------------------------------------------------------------
# Ground / obstacle separation
# ---------------------------------------------------------------------------
def _separate_ground_and_obstacles(pcd, slope_deg, normal_radius, voxel_size):
    print("Separating ground from obstacles...")
    if len(pcd.points) < 3:
        print("  Warning: too few points – treating all as obstacles.")
        return np.array([]), np.asarray(pcd.points)

    pcd_down = pcd.voxel_down_sample(voxel_size) if voxel_size > 0 else pcd
    try:
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=30
            )
        )
        pcd_down.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
    except Exception as e:
        print(f"  Warning: normal estimation failed ({e}) – treating all as obstacles.")
        return np.array([]), np.asarray(pcd.points)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd_down)
    indices  = np.array(
        [pcd_tree.search_knn_vector_3d(pt, 1)[1][0] for pt in pcd.points]
    )
    normals   = np.asarray(pcd_down.normals)[indices]
    dots      = np.abs(np.dot(normals, [0.0, 0.0, 1.0]))
    angles    = np.arccos(np.clip(dots, -1, 1))
    thresh    = np.deg2rad(slope_deg)
    ground_m  = angles <= thresh
    gp = np.asarray(pcd.points)[ground_m]
    op = np.asarray(pcd.points)[~ground_m]
    print(f"  Ground: {len(gp)} pts | Obstacles: {len(op)} pts")
    return gp, op


# ---------------------------------------------------------------------------
# Ground height map
# ---------------------------------------------------------------------------
def _build_ground_height_map(ground_pts, grid_res, bbx_min, x_size, y_size):
    print("Building ground height map...")
    if len(ground_pts) == 0:
        print("  Warning: no ground points – flat map at Z=0.")
        return np.zeros((y_size, x_size), dtype=np.float64)

    sum_z  = np.zeros((y_size, x_size), dtype=np.float64)
    counts = np.zeros((y_size, x_size), dtype=int)
    for pt in ground_pts:
        gx = int((pt[0] - bbx_min[0]) / grid_res)
        gy = int((pt[1] - bbx_min[1]) / grid_res)
        if 0 <= gx < x_size and 0 <= gy < y_size:
            sum_z[gy, gx]  += pt[2]
            counts[gy, gx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_z = np.nan_to_num(sum_z / counts)

    valid = np.where(counts > 0)
    if len(valid[0]) < 3:
        return np.full((y_size, x_size), ground_pts[:, 2].min(), dtype=np.float64)

    gy_g, gx_g = np.mgrid[0:y_size, 0:x_size]
    print("  Interpolating gaps in ground height map...")
    filled = griddata(
        (valid[0], valid[1]), avg_z[valid], (gy_g, gx_g), method="nearest"
    )
    return filled


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------
def _create_occupancy_grid_hybrid(
    obstacle_tree, ground_height_map, grid_res, bbx_min,
    z_min, z_max, workers=4
):
    y_size, x_size = ground_height_map.shape
    print(f"Generating 2D occupancy grid ({x_size}×{y_size})...")
    grid = np.full((y_size, x_size), 205, dtype=np.uint8)  # unknown=205

    def _process_cell(args_cell):
        gx, gy = args_cell
        wx = bbx_min[0] + (gx + 0.5) * grid_res
        wy = bbx_min[1] + (gy + 0.5) * grid_res
        gz = ground_height_map[gy, gx]
        min_z_w = gz + z_min
        max_z_w = gz + z_max
        try:
            node = obstacle_tree.search(
                pyoctomap.point3d(wx, wy, (min_z_w + max_z_w) / 2)
            )
            if node and obstacle_tree.isNodeOccupied(node):
                return gx, gy, 0   # obstacle
        except Exception:
            pass
        return gx, gy, 254  # free

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(_process_cell, (gx, gy))
            for gy in range(y_size)
            for gx in range(x_size)
        ]
        for fut in as_completed(futs):
            gx, gy, val = fut.result()
            grid[gy, gx] = val

    return grid


def _filter_small_clusters_2d(grid, min_cluster_size=20, closing_iters=1):
    obstacle_mask = grid == 0
    if closing_iters > 0:
        obstacle_mask = binary_closing(
            obstacle_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=closing_iters,
        )
    labeled, n_labels = label(obstacle_mask)
    if n_labels == 0:
        return grid
    sizes = np.bincount(labeled.ravel())
    small = np.where(sizes < min_cluster_size)[0]
    remove_mask = np.isin(labeled, small[small > 0])
    result = grid.copy()
    result[remove_mask] = 254
    return result


# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------
def run(args) -> None:
    bag_path = Path(args.input_bag)
    out_path = Path(args.output_path)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    # Pre-flight check
    required = [args.pc_topic, args.odom_topic]
    missing  = check_topics(bag_path, required)
    if missing:
        sys.exit(f"Error: Required topics not found in bag: {missing}\n"
                 "Check topic names with: ros2 bag info <bag>")

    print("--- Configuration ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("---------------------\n")

    # Step 1 – Load odometry
    print(f"[1/6] Loading odometry from '{args.odom_topic}'...")
    odom_times, odom_poses = [], []
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == args.odom_topic]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            mat = get_odom_transform_matrix(msg)
            if mat is not None:
                odom_times.append(ts * 1e-9)
                odom_poses.append(mat)

    if not odom_times:
        sys.exit("Error: No odometry messages found.")

    odom_times  = np.array(odom_times)
    odom_poses  = np.array(odom_poses)
    sort_idx    = np.argsort(odom_times)
    odom_times_sorted = odom_times[sort_idx].tolist()
    odom_poses_sorted = odom_poses[sort_idx]
    print(f"  Loaded {len(odom_times_sorted)} odometry poses.")

    # Step 2 – Build OcTree + aggregate points
    print(f"\n[2/6] Building 3D OcTree (octree_res={args.octree_res} m)...")
    obstacle_tree = pyoctomap.OcTree(args.octree_res)
    aggregated    = []

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == args.pc_topic]
        pc_msgs = []
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            pc_msgs.append((ts * 1e-9, msg))

    if not pc_msgs:
        sys.exit("Error: No point cloud messages found.")

    print(f"  {len(pc_msgs)} frames. Processing...")
    for i, (ts_sec, msg) in enumerate(pc_msgs):
        pts = pointcloud2_to_numpy(msg)
        if pts is None or len(pts) == 0:
            continue
        pts = pts[:, :3]

        best_key = get_closest_timestamp(ts_sec, odom_times_sorted)
        if best_key is None or abs(best_key - ts_sec) > args.odom_max_latency:
            print(f"\r  [{i+1}/{len(pc_msgs)}] Skipped (stale odom)", end="")
            continue

        idx = odom_times_sorted.index(best_key)
        sensor_origin = odom_poses_sorted[idx][:3, 3]
        print(f"\r  [{i+1}/{len(pc_msgs)}] {len(pts)} pts", end="")

        try:
            obstacle_tree.insertPointCloud(
                pts.astype(np.float64), sensor_origin.astype(np.float64), -1.0
            )
        except Exception as e:
            print(f" – insertion error: {e}")
            continue

        if args.voxel_size > 0:
            pcd_tmp        = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(pts)
            pcd_tmp        = pcd_tmp.voxel_down_sample(args.voxel_size)
            aggregated.append(np.asarray(pcd_tmp.points))
        else:
            aggregated.append(pts)

    print(f"\n  OcTree: {obstacle_tree.size()} nodes.")

    if not aggregated:
        sys.exit("Error: No valid point clouds processed.")

    # Step 3 – Ground / obstacle separation
    print("\n[3/6] Separating ground from obstacles...")
    full_np    = np.vstack(aggregated)
    full_pcd   = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(full_np)
    ground_pts, _ = _separate_ground_and_obstacles(
        full_pcd, args.slope_deg, args.normal_radius, args.voxel_size
    )

    # Step 4 – Ground height map
    print("\n[4/6] Building ground height map...")
    bbx_min = full_np.min(axis=0)
    bbx_max = full_np.max(axis=0)
    x_size  = int(np.ceil((bbx_max[0] - bbx_min[0]) / args.grid_res))
    y_size  = int(np.ceil((bbx_max[1] - bbx_min[1]) / args.grid_res))
    print(f"  Grid: {x_size} × {y_size} cells")
    height_map = _build_ground_height_map(
        ground_pts, args.grid_res, bbx_min, x_size, y_size
    )

    # Step 5 – 2D occupancy grid
    print("\n[5/6] Generating occupancy grid...")
    occ_grid = _create_occupancy_grid_hybrid(
        obstacle_tree, height_map,
        args.grid_res, bbx_min,
        args.z_min, args.z_max,
        args.workers,
    )

    # Step 5.5 – Denoise
    if args.min_cluster_size > 0:
        print(f"\n[5.5/6] Denoising (min_cluster_size={args.min_cluster_size})...")
        occ_grid = _filter_small_clusters_2d(
            occ_grid,
            min_cluster_size=args.min_cluster_size,
            closing_iters=args.closing_iters,
        )

    # Step 6 – Save
    print("\n[6/6] Saving output files...")
    pgm_path  = out_path.with_suffix(".pgm")
    yaml_path = out_path.with_suffix(".yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(occ_grid, "L").save(str(pgm_path))
    yaml_data = {
        "image":      pgm_path.name,
        "resolution": args.grid_res,
        "origin":     [float(bbx_min[0]), float(bbx_min[1]), 0.0],
        "negate":     0,
        "occupied_thresh": 0.65,
        "free_thresh":     0.196,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f"Saved: {pgm_path}")
    print(f"Saved: {yaml_path}")
    print("Done.")


def build_parser(sub):
    p = sub.add_parser("og_map", help="ROS 2 bag -> 2D Nav2 occupancy grid (.pgm + .yaml)")
    p.add_argument("input_bag",   help="Path to the ROS 2 bag directory.")
    p.add_argument("output_path", help="Base output path (extensions added automatically).")
    # topics
    p.add_argument("--pc_topic",   default="/dlio/odom_node/pointcloud/deskewed")
    p.add_argument("--odom_topic", default="/dlio/odom_node/odom")
    # resolution
    p.add_argument("--octree_res",    type=float, default=0.1,  help="3D OcTree resolution (m).")
    p.add_argument("--grid_res",      type=float, default=0.05, help="2D grid resolution (m).")
    p.add_argument("--slope_deg",     type=float, default=15.0, help="Max slope (deg) for ground.")
    p.add_argument("--normal_radius", type=float, default=0.2,  help="Normal estimation radius (m).")
    p.add_argument("--z_min",         type=float, default=0.1,  help="Min obstacle height above ground (m).")
    p.add_argument("--z_max",         type=float, default=2.0,  help="Max obstacle height above ground (m).")
    p.add_argument("--voxel_size",    type=float, default=0.05, help="Voxel size for downsampling (m). 0=disable.")
    # performance
    p.add_argument("--workers",           type=int,   default=4)
    p.add_argument("--min_cluster_size",  type=int,   default=20)
    p.add_argument("--closing_iters",     type=int,   default=1)
    p.add_argument("--odom_max_latency",  type=float, default=0.5)
    p.set_defaults(func=run)
    return p
