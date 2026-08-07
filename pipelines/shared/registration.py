#!/usr/bin/env python3
"""
registration.py - Shared odom-anchored registration + loop-closure helpers.

Used by: mesh, color_mesh, gazebo_world, tiles_3d, color_tiles_3d.
NOT used by: og_map (uses OcTree ray-casting instead).

Registration controls
---------------------
--frame_stride:
    Register every Nth input PointCloud2 frame. Values <= 1 mean all frames.

--max_registration_frames:
    Limit the number of frames after stride selection. 0 means no limit.

PATCH NOTE (odom-anchored registration; loop-closure information weighting)
-----------------------------------------------------------------------------
All five pipelines that use this module now call
`run_odom_anchored_registration()` (see odom-anchored-registration-fix-prompt.md
for the full design), which makes timestamped odometry the PRIMARY pose
source and demotes ICP to an optional, strongly-gated local refinement.
This fixed two failure modes present in the previous ICP-primary design:

1. Silent coverage collapse: frames that failed the ICP fitness bar against
   the last *successful* frame used to be dropped with no accounting, and
   the ICP chain could never recover once broken. Odom-anchored pose lookup
   means registration/ICP quality is never the reason a frame is excluded --
   only missing odometry coverage is, and that is now logged loudly with an
   explicit count.
2. Loop-closure overcorrection: loop edges used to be accepted at a looser
   fitness bar than sequential edges and weighted 100x higher, letting
   perceptual aliasing warp the whole trajectory. Loop closure now defaults
   to the SAME fitness bar as local refinement and is weighted on the same
   scale as sequential edges (which are themselves weighted far higher than
   any loop edge), so a bad loop match can only nudge the graph.

CLEANUP NOTE (unused functions removed)
----------------------------------------
Now that every caller has migrated to `run_odom_anchored_registration()`,
the following are no longer referenced anywhere in this codebase and have
been removed:
  - `run_icp_posegraph()` -- the old ICP-primary pose-graph pipeline.
  - `detect_loop_closure()` -- only ever called from `run_icp_posegraph()`;
    `run_odom_anchored_registration()` performs loop-closure detection
    inline so each candidate is captured and reused exactly once (see its
    docstring).
  - `iter_registered_frame_chunks()` -- documented as a merge-batching
    helper but never actually called by any pipeline; each pipeline's
    streaming merge loop manages its own bounded chunk instead.
If you are restoring ICP-as-primary behaviour for a one-off comparison,
pull these from version control history rather than re-adding them here.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Any

import numpy as np
import open3d as o3d
from tqdm import tqdm

from .ros_io import interpolate_odom_pose

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
    Select input clouds for registration.

    Returns:
        selected_pointclouds:
            Point-cloud records used internally by registration.

        original_indices:
            Index of each selected record within the original `pointclouds`
            list. Most callers now look poses up by timestamp instead, but
            this is kept for callers that still index the original list.

    Semantics:
        frame_stride <= 1 (including 0) selects every input frame.
        max_registration_frames <= 0 means no post-stride cap.
    """
    if not pointclouds:
        raise RuntimeError("No point clouds available for registration.")

    stride = max(1, int(frame_stride))
    original_indices = list(range(0, len(pointclouds), stride))

    if max_registration_frames > 0:
        original_indices = original_indices[: int(max_registration_frames)]

    if len(original_indices) < 2:
        raise RuntimeError(
            "Too few frames after registration selection. "
            "Use --frame_stride 0/1, increase --max_registration_frames, "
            "or provide a bag containing at least two point-cloud frames."
        )

    return [pointclouds[i] for i in original_indices], original_indices


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
# Odom-anchored registration (odom-primary)
# ---------------------------------------------------------------------------
def refine_pose_with_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    odom_relative_transform: np.ndarray,
    icp_dist_thresh: float,
    icp_fitness_thresh: float,
    max_translation_correction: float,
    max_rotation_correction_deg: float,
) -> tuple[np.ndarray, bool]:
    """
    Attempt point-to-plane ICP refinement of an odom-derived relative pose.

    `source`/`target` must already carry true unit-length geometric normals
    (see `_ensure_unit_geometric_normals`). `odom_relative_transform` is used
    as BOTH the initial guess and the fallback result.

    Accepted only if BOTH hold: ICP fitness clears `icp_fitness_thresh`, and
    the correction's translation/rotation magnitude relative to the odom
    guess is within the configured bounds. On any rejection or exception,
    the original odom transform is returned unchanged -- ICP can only nudge
    an already-valid pose here, never remove a frame from the merge.

    Returns:
        (relative_transform, accepted)
    """
    try:
        registration = o3d.pipelines.registration.registration_icp(
            source,
            target,
            icp_dist_thresh,
            odom_relative_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
    except RuntimeError as exc:
        logger.debug("ICP refinement raised an exception; keeping odom pose: %s", exc)
        return odom_relative_transform, False

    if registration.fitness < icp_fitness_thresh:
        return odom_relative_transform, False

    correction = np.linalg.inv(odom_relative_transform) @ registration.transformation
    translation_delta = float(np.linalg.norm(correction[:3, 3]))
    trace = np.clip((np.trace(correction[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rotation_delta_deg = float(np.degrees(np.arccos(trace)))

    if translation_delta > max_translation_correction:
        return odom_relative_transform, False
    if rotation_delta_deg > max_rotation_correction_deg:
        return odom_relative_transform, False

    return registration.transformation, True


def run_odom_anchored_registration(
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]],
    odom_data: dict[int, np.ndarray],
    args: Any,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """
    Odom-anchored registration: timestamped odometry is the PRIMARY pose
    source for every frame; ICP is only ever an optional, strongly-gated
    local refinement; loop closure is gated at the same bar as local
    refinement and weighted so it can only nudge, never dominate, the pose
    graph. See odom-anchored-registration-fix-prompt.md for the full design.

    Frames are included whenever they fall within `--odom_max_latency` of
    odometry coverage -- ICP/registration quality is NEVER the reason a
    frame is excluded. Frames without odometry coverage are dropped and
    counted loudly (logged and printed), never silently.

    Memory/perf safeguards (see design-doc section 6):
      - Only the current and previous frame's normal-computed cloud are
        held for sequential refinement (no whole-trajectory cloud list).
      - Normals are computed exactly once per frame and carried forward.
      - Loop-closure candidates are held in a `deque(maxlen=...)` bounded
        window, and each closure is detected exactly once (the accepted
        transform/fitness is reused directly as a pose-graph edge, never
        re-computed for a summary count).

    Args:
        pointclouds: (timestamp, cloud) records already selected/bounded by
            the caller (e.g. via `select_registration_frames`).
        odom_data: dict mapping odometry timestamp -> 4x4 transform.
        args: namespace exposing (at least) `voxel_size`, `odom_max_latency`,
            `enable_icp_refinement`, `icp_dist_thresh`, `icp_fitness_thresh`,
            `max_icp_translation_correction`, `max_icp_rotation_correction_deg`,
            `enable_loop_closure`, `loop_closure_radius`,
            `loop_closure_fitness_thresh`, `loop_closure_search_interval`,
            `loop_closure_temporal_window`.

    Returns:
        (final_poses, stats)
        final_poses: timestamp -> 4x4 transform that maps that frame's own
            points into the merge/world frame (apply via `cloud.transform(pose)`).
        stats: coverage counters for end-to-end logging -- total_selected,
            dropped_no_odom, with_pose, icp_attempted, icp_accepted,
            loop_closures_found, merged.
    """
    if not pointclouds:
        raise RuntimeError("No point clouds available for odom-anchored registration.")

    if not odom_data:
        raise RuntimeError(
            "Odom-anchored registration requires odometry data, but no usable "
            "odometry messages were found. Fix --odom_topic, or pass "
            "--pc_frame_mode global if the point cloud is already published "
            "in a fixed/global frame (no registration needed in that case)."
        )

    voxel_size = _arg_float(args, "voxel_size", 0.05)
    if voxel_size <= 0.0:
        raise ValueError("--voxel_size must be greater than zero.")

    odom_max_latency_ns = int(_arg_float(args, "odom_max_latency", 0.5) * 1e9)

    enable_icp_refinement = _arg_bool(args, "enable_icp_refinement", False)
    icp_dist_thresh = _arg_float(args, "icp_dist_thresh", 0.2)
    icp_fitness_thresh = _arg_float(args, "icp_fitness_thresh", 0.7)
    max_translation_correction = _arg_float(args, "max_icp_translation_correction", 0.3)
    max_rotation_correction_deg = _arg_float(args, "max_icp_rotation_correction_deg", 15.0)

    enable_loop_closure = _arg_bool(args, "enable_loop_closure", False)
    loop_closure_radius = _arg_float(args, "loop_closure_radius", 10.0)
    # Defaults to the SAME bar as local ICP refinement (not a separate,
    # looser value) -- a loop match that only clears a low bar is exactly
    # the failure mode that lets visually similar but different locations
    # get matched together.
    loop_fitness_thresh = _arg_float(args, "loop_closure_fitness_thresh", icp_fitness_thresh)
    loop_search_interval = max(1, _arg_int(args, "loop_closure_search_interval", 10))
    loop_temporal_window = max(1, _arg_int(args, "loop_closure_temporal_window", 100))

    odom_ts_sorted = sorted(odom_data.keys())
    total_selected = len(pointclouds)

    raw_odom_poses: dict[int, np.ndarray] = {}
    dropped_no_odom = 0
    kept_records: list[tuple[int, o3d.geometry.PointCloud]] = []

    for timestamp, cloud in pointclouds:
        pose = interpolate_odom_pose(timestamp, odom_ts_sorted, odom_data, odom_max_latency_ns)
        if pose is None:
            dropped_no_odom += 1
            continue
        raw_odom_poses[timestamp] = pose
        kept_records.append((timestamp, cloud))

    if dropped_no_odom:
        message = (
            f"{dropped_no_odom} / {total_selected} frames had no odometry "
            "coverage and were dropped"
        )
        logger.warning(message)
        print(f"Warning: {message}")

    if not kept_records:
        raise RuntimeError("No frames had usable odometry coverage; nothing to register.")

    print(
        f"Coverage: frames selected={total_selected:,} -> frames with valid "
        f"pose={len(kept_records):,} (odom_max_latency={odom_max_latency_ns / 1e9:.3f}s)."
    )

    # FINAL poses start as a COPY of the raw odom poses. Refinement below is
    # written into this copy only -- the raw dict is NEVER mutated -- so
    # every relative-motion delta between consecutive frames is always
    # computed from the untouched odometry chain, never from a refinement
    # applied to the previous frame (see design-doc section 1).
    final_poses: dict[int, np.ndarray] = dict(raw_odom_poses)
    node_poses: list[np.ndarray] = [raw_odom_poses[ts].copy() for ts, _ in kept_records]
    node_index_by_timestamp: dict[int, int] = {
        ts: idx for idx, (ts, _) in enumerate(kept_records)
    }

    icp_attempted = 0
    icp_accepted = 0
    loop_closures_found = 0

    # Only the current and previous frame's normal-computed cloud are ever
    # held at once -- never a list growing with trajectory length.
    prev_timestamp: int | None = None
    prev_cloud_normals: o3d.geometry.PointCloud | None = None

    # Bounded structure: holds only the most recent N candidate frames for
    # loop closure, never a list that grows with bag length.
    loop_history: deque[tuple[int, o3d.geometry.PointCloud, o3d.pipelines.registration.Feature | None]] = (
        deque(maxlen=loop_temporal_window)
    )

    # Sequential edges are stored alongside loop-closure edges so a single
    # pose-graph optimisation pass (only run if a loop closure was actually
    # found) can weight them far higher -- a bad loop match can then only
    # nudge the graph, never disconnect or collapse it.
    edges: list[tuple[int, int, np.ndarray, float, bool]] = []
    SEQUENTIAL_EDGE_INFORMATION_SCALE = 50.0

    needs_normals = enable_icp_refinement or enable_loop_closure

    for position, (timestamp, cloud) in enumerate(
        tqdm(kept_records, desc="Odom-anchored registration")
    ):
        current_cloud_normals: o3d.geometry.PointCloud | None = None
        if needs_normals:
            # Normals computed exactly once per frame; carried forward as
            # `prev_cloud_normals` on the next iteration instead of being
            # recomputed when this frame is later used as a "target".
            current_cloud_normals = cloud.voxel_down_sample(voxel_size)
            _ensure_unit_geometric_normals(current_cloud_normals, voxel_size)

        if position > 0 and prev_timestamp is not None:
            odom_relative = np.linalg.inv(raw_odom_poses[prev_timestamp]) @ raw_odom_poses[timestamp]
            relative_transform = odom_relative

            if (
                enable_icp_refinement
                and prev_cloud_normals is not None
                and current_cloud_normals is not None
            ):
                icp_attempted += 1
                refined_relative, accepted = refine_pose_with_icp(
                    prev_cloud_normals,
                    current_cloud_normals,
                    odom_relative,
                    icp_dist_thresh,
                    icp_fitness_thresh,
                    max_translation_correction,
                    max_rotation_correction_deg,
                )
                if accepted:
                    icp_accepted += 1
                    relative_transform = refined_relative
                    world_pose = raw_odom_poses[prev_timestamp] @ relative_transform
                    final_poses[timestamp] = world_pose
                    node_poses[position] = world_pose.copy()

            edges.append((position - 1, position, relative_transform, SEQUENTIAL_EDGE_INFORMATION_SCALE, False))

        if enable_loop_closure and current_cloud_normals is not None:
            do_loop_search = position > 0 and position % loop_search_interval == 0
            current_fpfh = None

            if do_loop_search:
                current_fpfh = compute_fpfh_descriptor(copy.deepcopy(current_cloud_normals), voxel_size)
                current_pos_xyz = node_poses[position][:3, 3]

                for hist_timestamp, hist_cloud, hist_fpfh in loop_history:
                    if hist_fpfh is None:
                        continue
                    hist_index = node_index_by_timestamp[hist_timestamp]
                    hist_pos_xyz = node_poses[hist_index][:3, 3]
                    if np.linalg.norm(current_pos_xyz - hist_pos_xyz) > loop_closure_radius:
                        continue

                    try:
                        source_copy = copy.deepcopy(current_cloud_normals)
                        target_copy = copy.deepcopy(hist_cloud)
                        coarse = ransac_coarse_alignment(
                            source_copy, target_copy, current_fpfh, hist_fpfh, voxel_size
                        )
                        if coarse.fitness < loop_fitness_thresh * 0.5:
                            continue

                        refined = o3d.pipelines.registration.registration_icp(
                            source_copy,
                            target_copy,
                            max(voxel_size * 1.5, 1e-4),
                            coarse.transformation,
                            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
                        )
                        # Detected exactly once: the accepted transform/fitness
                        # is captured here and reused directly as the
                        # pose-graph edge below -- never re-detected later
                        # just to build a summary count.
                        if refined.fitness >= loop_fitness_thresh:
                            edges.append(
                                (hist_index, position, refined.transformation, float(refined.fitness), True)
                            )
                            loop_closures_found += 1
                    except (RuntimeError, ValueError) as exc:
                        logger.debug("Loop-closure candidate rejected: %s", exc)
                        continue

            loop_history.append((timestamp, current_cloud_normals, current_fpfh))

        prev_timestamp = timestamp
        prev_cloud_normals = current_cloud_normals

    if enable_icp_refinement:
        print(f"ICP refinement: attempted {icp_attempted:,} pair(s), accepted {icp_accepted:,}.")
    if enable_loop_closure:
        print(f"Loop closures detected: {loop_closures_found:,}")

    if enable_loop_closure and loop_closures_found > 0:
        posegraph = o3d.pipelines.registration.PoseGraph()
        for pose in node_poses:
            posegraph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(pose)))

        for source_idx, target_idx, transform, weight, uncertain in edges:
            # Sequential edges are weighted significantly higher than any
            # loop-closure edge (fixed high scale vs. the loop edge's own
            # fitness, un-inflated) so a single bad loop-closure match can
            # only nudge the graph, never disconnect or collapse it.
            info_scale = weight if not uncertain else max(float(weight), 1e-6)
            information = np.eye(6, dtype=np.float64) * info_scale
            posegraph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_idx, target_idx, transform, information, uncertain=uncertain,
                )
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
            for timestamp, node_idx in node_index_by_timestamp.items():
                final_poses[timestamp] = np.linalg.inv(posegraph.nodes[node_idx].pose)
            print("Pose graph optimisation complete (loop closure applied).")
        except RuntimeError as exc:
            logger.warning(
                "Pose graph optimisation failed (%s); using odom-anchored/ICP-refined poses.",
                exc,
            )

    print(f"Merge: {len(kept_records):,} frame(s) will be merged into the combined cloud.")

    stats = {
        "total_selected": total_selected,
        "dropped_no_odom": dropped_no_odom,
        "with_pose": len(kept_records),
        "icp_attempted": icp_attempted,
        "icp_accepted": icp_accepted,
        "loop_closures_found": loop_closures_found,
        "merged": len(kept_records),
    }
    return final_poses, stats
