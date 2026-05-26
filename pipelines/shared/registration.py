#!/usr/bin/env python3
"""
registration.py – Shared ICP + pose-graph + loop-closure helpers.

Used by: mesh, color_mesh, gazebo_world, tiles_3d, color_tiles_3d.
NOT used by: og_map (uses OcTree ray-casting instead).
"""

import copy

import numpy as np
import open3d as o3d
from tqdm import tqdm

from .ros_io import get_odom_transform, get_closest_timestamp


# ---------------------------------------------------------------------------
# View-ray normal helpers
# ---------------------------------------------------------------------------
def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize an (N,3) array of vectors row-wise; zero rows stay zero."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norms > eps, v / norms, v)


def attach_view_rays_as_normals(
    pcd_world: o3d.geometry.PointCloud, sensor_origin_world: np.ndarray
) -> None:
    """Store unnormalized view rays in pcd.normals before voxel downsampling."""
    pts = np.asarray(pcd_world.points, dtype=np.float64)
    if len(pts) == 0:
        return
    ray_dirs = sensor_origin_world.reshape(1, 3) - pts
    pcd_world.normals = o3d.utility.Vector3dVector(ray_dirs)


def orient_geometric_normals_with_view_rays(
    pcd: o3d.geometry.PointCloud, view_rays: np.ndarray
) -> None:
    """Flip geometric normals so they point toward the camera."""
    if len(view_rays) == 0 or not pcd.has_normals():
        return
    normals = np.asarray(pcd.normals, dtype=np.float64)
    rays    = _safe_normalize(view_rays.astype(np.float64))
    dot     = np.einsum("ij,ij->i", normals, rays)
    flip    = dot < 0
    normals[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(normals)


def estimate_geometric_normals_oriented(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    view_rays: np.ndarray | None = None,
) -> None:
    """Estimate geometric normals and orient them using view rays or tangent plane."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0, max_nn=30
        )
    )
    if view_rays is not None and len(view_rays) == len(pcd.points):
        orient_geometric_normals_with_view_rays(pcd, view_rays)
    else:
        try:
            pcd.orient_normals_consistent_tangent_plane(100)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FPFH + RANSAC coarse alignment
# ---------------------------------------------------------------------------
def compute_fpfh_descriptor(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> o3d.pipelines.registration.Feature:
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.0, max_nn=30
            )
        )
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )


def ransac_coarse_alignment(
    source, target, source_fpfh, target_fpfh, voxel_size, ransac_thresh_mult=5.0
):
    dist = voxel_size * ransac_thresh_mult
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
    )


# ---------------------------------------------------------------------------
# Loop closure detection
# ---------------------------------------------------------------------------
def detect_loop_closure(
    current_idx, current_pcd, current_fpfh,
    historical_pcds, historical_fpfhs, historical_poses,
    voxel_size, search_radius=10.0, loop_fitness_thresh=0.3, temporal_window=100,
):
    results = []
    current_pos = historical_poses[current_idx][:3, 3]
    for j in range(max(0, current_idx - temporal_window)):
        if historical_fpfhs[j] is None:
            continue
        if np.linalg.norm(current_pos - historical_poses[j][:3, 3]) > search_radius:
            continue
        try:
            r = ransac_coarse_alignment(
                copy.deepcopy(current_pcd), copy.deepcopy(historical_pcds[j]),
                current_fpfh, historical_fpfhs[j], voxel_size,
            )
            if r.fitness < loop_fitness_thresh * 0.5:
                continue
            icp = o3d.pipelines.registration.registration_icp(
                copy.deepcopy(current_pcd), copy.deepcopy(historical_pcds[j]),
                voxel_size * 1.5, r.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
            if icp.fitness >= loop_fitness_thresh:
                results.append((j, icp.transformation, icp.fitness))
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Full ICP + pose-graph pipeline
# ---------------------------------------------------------------------------
def run_icp_posegraph(
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]],
    odom_data: dict,
    args,
) -> tuple[o3d.pipelines.registration.PoseGraph, list[int]]:
    """
    Pairwise ICP + global pose-graph optimisation.

    Returns (PoseGraph, list_of_successful_pc_indices).
    """
    odom_max_latency_ns = int(args.odom_max_latency * 1e9)
    odom_ts_sorted = sorted(odom_data.keys())

    posegraph = o3d.pipelines.registration.PoseGraph()
    current_transform = np.eye(4, dtype=np.float64)
    posegraph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(current_transform.copy())
    )

    _, source_raw = pointclouds[0]
    source = source_raw.voxel_down_sample(args.voxel_size)
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.voxel_size * 2.0, max_nn=30
        )
    )

    lc_pcds, lc_fpfhs, lc_poses = [], [], []
    if args.enable_loop_closure:
        lc_pcds.append(source)
        lc_fpfhs.append(compute_fpfh_descriptor(copy.deepcopy(source), args.voxel_size))
        lc_poses.append(current_transform.copy())

    previous_odom_T = None
    if odom_ts_sorted:
        cts = get_closest_timestamp(pointclouds[0][0], odom_ts_sorted)
        if cts is not None and abs(cts - pointclouds[0][0]) < odom_max_latency_ns:
            previous_odom_T = odom_data[cts]

    successful_pc_indices = [0]
    loop_closures_found   = 0

    print("Registering point clouds...")
    for i in tqdm(range(1, len(pointclouds)), desc="Registering"):
        ts, target_raw = pointclouds[i]
        target = target_raw.voxel_down_sample(args.voxel_size)
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.voxel_size * 2.0, max_nn=30
            )
        )

        initial_guess = np.eye(4, dtype=np.float64)
        if odom_ts_sorted:
            cts = get_closest_timestamp(ts, odom_ts_sorted)
            if cts is not None and abs(cts - ts) < odom_max_latency_ns:
                current_odom_T = odom_data[cts]
                if previous_odom_T is not None:
                    initial_guess = np.linalg.inv(previous_odom_T) @ current_odom_T
                previous_odom_T = current_odom_T
            else:
                previous_odom_T = None

        try:
            reg = o3d.pipelines.registration.registration_icp(
                source, target,
                args.icp_dist_thresh,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
        except Exception:
            continue

        if reg.fitness < args.icp_fitness_thresh:
            continue

        current_transform = reg.transformation @ current_transform
        posegraph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(current_transform))
        )
        info = np.eye(6, dtype=np.float64) * max(float(reg.fitness), 1e-6)
        posegraph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                len(posegraph.nodes) - 2,
                len(posegraph.nodes) - 1,
                reg.transformation, info, uncertain=False,
            )
        )

        if args.enable_loop_closure:
            do_lc    = (i % args.loop_closure_search_interval == 0)
            tgt_fpfh = (
                compute_fpfh_descriptor(copy.deepcopy(target), args.voxel_size)
                if do_lc else None
            )
            lc_pcds.append(target); lc_fpfhs.append(tgt_fpfh); lc_poses.append(current_transform.copy())
            if do_lc:
                closures = detect_loop_closure(
                    len(lc_pcds) - 1, target, tgt_fpfh,
                    lc_pcds, lc_fpfhs, lc_poses,
                    args.voxel_size,
                    search_radius=args.loop_closure_radius,
                    loop_fitness_thresh=args.loop_closure_fitness_thresh,
                )
                for cand_idx, lc_T, lc_fit in closures:
                    lc_info = np.eye(6, dtype=np.float64) * max(lc_fit * 100.0, 1e-6)
                    posegraph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            cand_idx, len(posegraph.nodes) - 1,
                            lc_T, lc_info, uncertain=True,
                        )
                    )
                    loop_closures_found += 1

        successful_pc_indices.append(i)
        source = target

    if len(posegraph.nodes) < 2:
        raise RuntimeError(
            "Registration failed — too few successful frames. "
            "Try lowering --icp_fitness_thresh or increasing --icp_dist_thresh."
        )

    if args.enable_loop_closure:
        print(f"Loop closures detected: {loop_closures_found}")

    print("Optimising pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=args.icp_dist_thresh,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    try:
        o3d.pipelines.registration.global_optimization(
            posegraph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
        print("Pose graph optimisation complete.")
    except Exception as e:
        print(f"Warning: Pose graph optimisation failed ({e}); using unoptimised poses.")

    return posegraph, successful_pc_indices
