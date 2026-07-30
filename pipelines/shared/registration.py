#!/usr/bin/env python3
"""
registration.py - Shared ICP + pose-graph + loop-closure helpers.

Used by: mesh, color_mesh, gazebo_world, tiles_3d, color_tiles_3d.
NOT used by: og_map (uses OcTree ray-casting instead).

Registration controls
---------------------
--frame_stride:
    Register every Nth input PointCloud2 frame. Values <= 1 mean all frames.

--max_registration_frames:
    Limit the number of frames after stride selection. 0 means no limit.

--merge_chunk_frames:
    Used by iter_registered_frame_chunks() to process successful registered
    frame indices in bounded batches during pipeline-side merging. This avoids
    keeping a large temporary merge batch alive at once.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import open3d as o3d
from tqdm import tqdm

from .ros_io import get_closest_timestamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------
def _arg_int(args: Any, name: str, default: int) -> int:
    """Read an integer argument safely for pipelines with older parsers."""
    try:
        return int(getattr(args, name, default))
    except (TypeError, ValueError):
        return default


def _arg_float(args: Any, name: str, default: float) -> float:
    """Read a float argument safely for pipelines with older parsers."""
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError):
        return default


def _arg_bool(args: Any, name: str, default: bool) -> bool:
    """Read a bool argument safely for pipelines with older parsers."""
    return bool(getattr(args, name, default))


# ---------------------------------------------------------------------------
# Registration / merge selection helpers
# ---------------------------------------------------------------------------
def select_registration_frames(
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]],
    frame_stride: int = 0,
    max_registration_frames: int = 0,
) -> tuple[list[tuple[int, o3d.geometry.PointCloud]], list[int]]:
    """
    Select input clouds for pose-graph registration.

    Returns:
        selected_pointclouds:
            Point-cloud records used internally by ICP.

        original_indices:
            Index of each selected record within the original `pointclouds`
            list. The caller needs these after registration because pipeline
            merge stages index the original, full input list.

    Semantics:
        frame_stride <= 1 (including 0) selects every input frame.
        max_registration_frames <= 0 means no post-stride cap.
    """
    if not pointclouds:
        raise RuntimeError("No point clouds available for registration.")

    stride = max(1, int(frame_stride))
    original_indices = list(range(0, len(pointclouds), stride))

    if max_registration_frames > 0:
        original_indices = original_indices[:int(max_registration_frames)]

    if len(original_indices) < 2:
        raise RuntimeError(
            "Too few frames after registration selection. "
            "Use --frame_stride 0/1, increase --max_registration_frames, "
            "or provide a bag containing at least two point-cloud frames."
        )

    return [pointclouds[i] for i in original_indices], original_indices


def iter_registered_frame_chunks(
    successful_original_indices: list[int],
    merge_chunk_frames: int = 16,
) -> Iterator[list[int]]:
    """
    Yield successful original frame indices in bounded merge batches.

    Pipeline merge code can use this to keep a batch-local temporary cloud,
    append it to the final output, then discard it before processing the next
    batch. A non-positive value means one chunk containing all frames.
    """
    if not successful_original_indices:
        return

    chunk_size = int(merge_chunk_frames)
    if chunk_size <= 0:
        chunk_size = len(successful_original_indices)

    for start in range(0, len(successful_original_indices), chunk_size):
        yield successful_original_indices[start:start + chunk_size]


# ---------------------------------------------------------------------------
# View-ray normal helpers
# ---------------------------------------------------------------------------
def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize an (N, 3) array row-wise; zero rows remain zero."""
    values = np.asarray(v, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > eps,
    )


def attach_view_rays_as_normals(
    pcd_world: o3d.geometry.PointCloud,
    sensor_origin_world: np.ndarray,
) -> None:
    """
    Store unnormalised sensor-to-point view rays in `pcd.normals`.

    These are intentionally temporary: downstream code retains them before
    geometric normal estimation, then uses them to orient final normals.

    IMPORTANT: Point clouds carrying these view rays must NEVER be passed
    directly into point-to-plane ICP or FPFH feature computation. Use
    `_ensure_unit_geometric_normals()` (or `estimate_geometric_normals_oriented`)
    to replace them with true unit-length geometric normals first.
    """
    points = np.asarray(pcd_world.points, dtype=np.float64)
    if len(points) == 0:
        return

    sensor_origin = np.asarray(sensor_origin_world, dtype=np.float64).reshape(1, 3)
    view_rays = sensor_origin - points
    pcd_world.normals = o3d.utility.Vector3dVector(view_rays)


def orient_geometric_normals_with_view_rays(
    pcd: o3d.geometry.PointCloud,
    view_rays: np.ndarray,
) -> None:
    """Flip geometric normals to face toward the corresponding view ray."""
    if not pcd.has_normals():
        return

    normals = np.asarray(pcd.normals, dtype=np.float64)
    rays = np.asarray(view_rays, dtype=np.float64)

    if len(normals) == 0 or len(rays) != len(normals):
        return

    rays = _safe_normalize(rays)
    valid = np.linalg.norm(rays, axis=1) > 1e-12
    dot = np.einsum("ij,ij->i", normals, rays)
    flip = valid & (dot < 0.0)
    normals[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(normals)


def estimate_geometric_normals_oriented(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    view_rays: np.ndarray | None = None,
) -> None:
    """Estimate geometric normals and orient them with rays or topology."""
    if len(pcd.points) == 0:
        return

    radius = max(float(voxel_size) * 2.0, 1e-4)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30,
        )
    )

    if view_rays is not None and len(view_rays) == len(pcd.points):
        orient_geometric_normals_with_view_rays(pcd, view_rays)
        return

    try:
        pcd.orient_normals_consistent_tangent_plane(100)
    except RuntimeError as exc:
        # Open3D raises RuntimeError (often wrapping a C++ exception) when
        # the tangent-plane graph is degenerate -- e.g. too few neighbors,
        # a fully planar/collinear point set, or a disconnected KNN graph.
        # Normals already computed by estimate_normals() above are kept;
        # only their *global* orientation consistency is lost.
        logger.warning(
            "orient_normals_consistent_tangent_plane failed for a cloud "
            "with %d points (voxel_size=%.4g): %s. Falling back to "
            "per-point normals without global orientation consistency.",
            len(pcd.points), voxel_size, exc,
        )


def _ensure_unit_geometric_normals(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> None:
    """
    Force `pcd` to carry true unit-length geometric normals before it is
    used for point-to-plane ICP or FPFH feature computation.

    This helper always (re)computes normals via `estimate_normals()`
    regardless of whether normals are already present, guaranteeing that
    every normal consumed by ICP/FPFH in this module is a genuine unit-
    length geometric normal (not a leftover, unnormalised view ray from
    `attach_view_rays_as_normals()`).
    """
    if len(pcd.points) == 0:
        return

    radius = max(float(voxel_size) * 2.0, 1e-4)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30,
        )
    )


# ---------------------------------------------------------------------------
# FPFH + RANSAC coarse alignment
# ---------------------------------------------------------------------------
def compute_fpfh_descriptor(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.pipelines.registration.Feature:
    """Compute an FPFH descriptor for loop-closure candidate matching."""
    if len(pcd.points) == 0:
        raise ValueError("Cannot compute FPFH for an empty point cloud.")

    voxel_size = max(float(voxel_size), 1e-4)

    # Always (re)compute geometric normals rather than trusting
    # has_normals(). A cloud carrying leftover view-ray "normals" from
    # attach_view_rays_as_normals() would otherwise be fed straight into
    # compute_fpfh_feature(), which requires unit-length surface normals.
    _ensure_unit_geometric_normals(pcd, voxel_size)

    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5.0,
            max_nn=100,
        ),
    )


def ransac_coarse_alignment(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
    ransac_thresh_mult: float = 5.0,
) -> o3d.pipelines.registration.RegistrationResult:
    """Compute a feature-based coarse transform for loop-closure validation."""
    distance = max(float(voxel_size) * float(ransac_thresh_mult), 1e-4)

    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance,
        estimation_method=(
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            100_000,
            0.999,
        ),
    )


# ---------------------------------------------------------------------------
# Loop closure detection
# ---------------------------------------------------------------------------
def detect_loop_closure(
    current_idx: int,
    current_pcd: o3d.geometry.PointCloud,
    current_fpfh: o3d.pipelines.registration.Feature,
    historical_pcds: list[o3d.geometry.PointCloud],
    historical_fpfhs: list[o3d.pipelines.registration.Feature | None],
    historical_poses: list[np.ndarray],
    voxel_size: float,
    search_radius: float = 10.0,
    loop_fitness_thresh: float = 0.3,
    temporal_window: int = 100,
) -> list[tuple[int, np.ndarray, float]]:
    """
    Detect valid historical loop closures for the current successful node.

    Candidate cloud history contains successful registrations only, so every
    returned index is a valid pose-graph node index.
    """
    if current_idx <= 0 or current_fpfh is None:
        return []

    current_pos = historical_poses[current_idx][:3, 3]
    candidates: list[tuple[int, np.ndarray, float]] = []

    stop = max(0, current_idx - max(1, int(temporal_window)))
    for historical_idx in range(stop):
        historical_fpfh = historical_fpfhs[historical_idx]
        if historical_fpfh is None:
            continue

        historical_pos = historical_poses[historical_idx][:3, 3]
        if np.linalg.norm(current_pos - historical_pos) > float(search_radius):
            continue

        try:
            source_copy = copy.deepcopy(current_pcd)
            target_copy = copy.deepcopy(historical_pcds[historical_idx])

            coarse = ransac_coarse_alignment(
                source_copy,
                target_copy,
                current_fpfh,
                historical_fpfh,
                voxel_size,
            )

            if coarse.fitness < float(loop_fitness_thresh) * 0.5:
                continue

            # Point-to-plane ICP requires unit-length geometric normals on
            # both source and target. Force correct normals on the
            # deep-copied clouds used for refinement, since whatever
            # normals were attached earlier (view rays or geometric) are
            # not guaranteed to match what this stage needs.
            _ensure_unit_geometric_normals(source_copy, voxel_size)
            _ensure_unit_geometric_normals(target_copy, voxel_size)

            refined = o3d.pipelines.registration.registration_icp(
                source_copy,
                target_copy,
                max(float(voxel_size) * 1.5, 1e-4),
                coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50
                ),
            )

            if refined.fitness >= float(loop_fitness_thresh):
                candidates.append((
                    historical_idx,
                    refined.transformation,
                    float(refined.fitness),
                ))

        except (RuntimeError, ValueError) as exc:
            # RuntimeError: Open3D RANSAC/ICP failure on degenerate or
            # mismatched feature/point data.
            # ValueError: malformed inputs (e.g. empty point cloud) that
            # slipped through upstream checks.
            logger.debug(
                "Loop-closure candidate against historical node %d rejected: %s",
                historical_idx, exc,
            )
            continue

    return candidates


# ---------------------------------------------------------------------------
# Full ICP + pose-graph pipeline
# ---------------------------------------------------------------------------
def run_icp_posegraph(
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]],
    odom_data: dict[int, np.ndarray],
    args: Any,
) -> tuple[o3d.pipelines.registration.PoseGraph, list[int]]:
    """
    Run pairwise ICP and global pose-graph optimisation.

    Returns:
        posegraph:
            One node per successfully registered selected frame.

        successful_original_indices:
            Original indices into the caller's complete `pointclouds` list,
            ordered exactly like `posegraph.nodes`. This preserves compatibility
            with mesh, color_mesh, gazebo_world, tiles_3d, and color_tiles_3d.
    """
    voxel_size = _arg_float(args, "voxel_size", 0.05)
    if voxel_size <= 0.0:
        raise ValueError("--voxel_size must be greater than zero.")

    icp_dist_thresh = _arg_float(args, "icp_dist_thresh", 0.2)
    if icp_dist_thresh <= 0.0:
        raise ValueError("--icp_dist_thresh must be greater than zero.")

    icp_fitness_thresh = _arg_float(args, "icp_fitness_thresh", 0.6)
    odom_max_latency_ns = int(_arg_float(args, "odom_max_latency", 0.5) * 1e9)

    frame_stride = _arg_int(args, "frame_stride", 0)
    max_registration_frames = _arg_int(args, "max_registration_frames", 0)
    merge_chunk_frames = _arg_int(args, "merge_chunk_frames", 16)

    selected_clouds, selected_original_indices = select_registration_frames(
        pointclouds,
        frame_stride=frame_stride,
        max_registration_frames=max_registration_frames,
    )

    print(
        f"Registration selection: {len(selected_clouds):,} / {len(pointclouds):,} "
        f"frames (stride={frame_stride}, max={max_registration_frames})."
    )
    print(
        f"Merge chunk setting: {merge_chunk_frames} frame(s). "
        "Pipeline merge stages may use iter_registered_frame_chunks()."
    )

    odom_ts_sorted = sorted(odom_data.keys())
    enable_loop_closure = _arg_bool(args, "enable_loop_closure", False)
    loop_search_interval = max(
        1,
        _arg_int(args, "loop_closure_search_interval", 10),
    )
    loop_closure_radius = _arg_float(args, "loop_closure_radius", 10.0)
    loop_fitness_thresh = _arg_float(
        args,
        "loop_closure_fitness_thresh",
        0.3,
    )

    posegraph = o3d.pipelines.registration.PoseGraph()
    current_transform = np.eye(4, dtype=np.float64)
    posegraph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(current_transform.copy())
    )

    first_ts, first_raw = selected_clouds[0]
    source = first_raw.voxel_down_sample(voxel_size)
    if len(source.points) == 0:
        raise RuntimeError("The first selected registration frame is empty.")

    # Use the shared helper so behaviour matches compute_fpfh_descriptor
    # and detect_loop_closure -- always a fresh geometric-normal estimate,
    # never a stale/leftover view-ray normal.
    _ensure_unit_geometric_normals(source, voxel_size)

    previous_odom_transform: np.ndarray | None = None
    if odom_ts_sorted:
        closest_ts = get_closest_timestamp(first_ts, odom_ts_sorted)
        if (
            closest_ts is not None
            and abs(closest_ts - first_ts) < odom_max_latency_ns
        ):
            previous_odom_transform = odom_data[closest_ts]

    # These lists are indexed by successful pose-graph node index.
    loop_clouds: list[o3d.geometry.PointCloud] = []
    loop_fpfhs: list[o3d.pipelines.registration.Feature | None] = []
    loop_poses: list[np.ndarray] = []

    if enable_loop_closure:
        loop_clouds.append(copy.deepcopy(source))
        loop_fpfhs.append(compute_fpfh_descriptor(copy.deepcopy(source), voxel_size))
        loop_poses.append(current_transform.copy())

    # Local selected-frame indices. Convert these to original indices at return.
    successful_selected_indices = [0]
    loop_closures_found = 0

    print("Registering point clouds...")
    for selected_idx in tqdm(
        range(1, len(selected_clouds)),
        desc="Registering",
    ):
        timestamp, target_raw = selected_clouds[selected_idx]
        target = target_raw.voxel_down_sample(voxel_size)
        if len(target.points) == 0:
            continue

        # Same normal-safety guarantee as `source` above.
        _ensure_unit_geometric_normals(target, voxel_size)

        initial_guess = np.eye(4, dtype=np.float64)
        current_odom_transform: np.ndarray | None = None

        if odom_ts_sorted:
            closest_ts = get_closest_timestamp(timestamp, odom_ts_sorted)
            if (
                closest_ts is not None
                and abs(closest_ts - timestamp) < odom_max_latency_ns
            ):
                current_odom_transform = odom_data[closest_ts]
                if previous_odom_transform is not None:
                    initial_guess = (
                        np.linalg.inv(previous_odom_transform)
                        @ current_odom_transform
                    )

        try:
            registration = o3d.pipelines.registration.registration_icp(
                source,
                target,
                icp_dist_thresh,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50
                ),
            )
        except RuntimeError as exc:
            # Open3D ICP raises RuntimeError on degenerate correspondence
            # sets (e.g. no points within icp_dist_thresh of each other).
            logger.debug(
                "ICP failed for frame at selected index %d (timestamp=%d): %s",
                selected_idx, timestamp, exc,
            )
            continue

        if registration.fitness < icp_fitness_thresh:
            continue

        # Promote odometry only after a frame was actually accepted. If ICP
        # fails, the next successful frame still receives an odometry delta
        # relative to the last accepted source frame.
        if current_odom_transform is not None:
            previous_odom_transform = current_odom_transform

        current_transform = registration.transformation @ current_transform
        posegraph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(current_transform)
            )
        )

        information = (
            np.eye(6, dtype=np.float64)
            * max(float(registration.fitness), 1e-6)
        )
        posegraph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                len(posegraph.nodes) - 2,
                len(posegraph.nodes) - 1,
                registration.transformation,
                information,
                uncertain=False,
            )
        )

        successful_selected_indices.append(selected_idx)

        if enable_loop_closure:
            successful_node_idx = len(loop_clouds)
            do_loop_search = (
                successful_node_idx > 0
                and successful_node_idx % loop_search_interval == 0
            )

            target_fpfh = (
                compute_fpfh_descriptor(copy.deepcopy(target), voxel_size)
                if do_loop_search
                else None
            )

            loop_clouds.append(copy.deepcopy(target))
            loop_fpfhs.append(target_fpfh)
            loop_poses.append(current_transform.copy())

            if do_loop_search and target_fpfh is not None:
                closures = detect_loop_closure(
                    current_idx=successful_node_idx,
                    current_pcd=target,
                    current_fpfh=target_fpfh,
                    historical_pcds=loop_clouds,
                    historical_fpfhs=loop_fpfhs,
                    historical_poses=loop_poses,
                    voxel_size=voxel_size,
                    search_radius=loop_closure_radius,
                    loop_fitness_thresh=loop_fitness_thresh,
                )

                for candidate_idx, transform, fitness in closures:
                    loop_information = (
                        np.eye(6, dtype=np.float64)
                        * max(fitness * 100.0, 1e-6)
                    )
                    posegraph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            candidate_idx,
                            len(posegraph.nodes) - 1,
                            transform,
                            loop_information,
                            uncertain=True,
                        )
                    )
                    loop_closures_found += 1

        source = target

    if len(posegraph.nodes) < 2:
        raise RuntimeError(
            "Registration failed - fewer than two frames met the ICP fitness "
            "threshold. Try lowering --icp_fitness_thresh, increasing "
            "--icp_dist_thresh, or reducing --frame_stride."
        )

    if enable_loop_closure:
        print(f"Loop closures detected: {loop_closures_found}")

    print(
        f"Optimising pose graph ({len(posegraph.nodes):,} nodes, "
        f"{len(posegraph.edges):,} edges)..."
    )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=icp_dist_thresh,
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
    except RuntimeError as exc:
        # Open3D's global optimizer can raise RuntimeError on a
        # numerically ill-conditioned pose graph (e.g. disconnected
        # components, degenerate loop-closure edges). Falling back to
        # the unoptimised, purely sequential poses keeps the pipeline
        # usable instead of aborting the whole run.
        logger.warning(
            "Pose graph optimisation failed (%s); using unoptimised poses.",
            exc,
        )

    successful_original_indices = [
        selected_original_indices[selected_idx]
        for selected_idx in successful_selected_indices
    ]

    return posegraph, successful_original_indices
