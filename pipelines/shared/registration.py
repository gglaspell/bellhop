#!/usr/bin/env python3
"""
registration.py - Shared odom-anchored registration + loop-closure helpers.

Used by: mesh, color_mesh, gazebo_world, tiles_3d, color_tiles_3d.
NOT used by: og_map (uses OcTree ray-casting instead).

Registration controls
----------------------
--min_move_distance / --min_rotation_angle_deg:
    Motion-gated frame selection. A frame is kept only if it has translated
    >= --min_move_distance meters OR rotated >= --min_rotation_angle_deg
    degrees, relative to the last KEPT frame's odometry pose. Replaces the
    old index-based --frame_stride: odom is the primary pose source, so a
    frame that hasn't moved (or turned) adds no new spatial coverage and is
    skipped instead of being thinned purely by message count. Set either to
    0 to disable that half of the gate; set both to 0 to disable motion
    gating entirely (keep every frame, subject only to
    --max_registration_frames).

--max_registration_frames:
    Hard cap on the number of KEPT frames, applied after the motion gate
    (and after the odometry health check, below). 0 means no cap. This cap
    also bounds how many raw frames a pipeline's bag-reading pass needs to
    read ahead of time.

--disable_odom_health_check / --odom_loss_speed_multiplier:
    Odometry health check (see `detect_odom_loss()` below). Enabled by
    default; truncates registration at the first detected tracking
    loss/teleport in the raw odometry stream, before frame selection runs,
    so a lost-odom segment can never reach the motion gate, ICP refinement,
    or loop closure.

PATCH NOTE (motion-gated frame selection; odometry health check)
-----------------------------------------------------------------------------
Two changes replace the old index-stride frame selection:

1. Frame selection: index stride -> motion gating.
   Problem solved: `--frame_stride` selected "every Nth point-cloud
   message," which wastes registration/merge work (and memory) when the
   robot isn't moving, and under-samples when it's moving fast, because
   message rate has nothing to do with spatial coverage once odometry (not
   sequential ICP overlap) is the primary pose source.
   New behavior: `select_registration_frames_by_motion()` walks
   odometry-covered frames in temporal order and keeps a frame only if it
   has translated >= --min_move_distance meters OR rotated >=
   --min_rotation_angle_deg degrees since the last KEPT frame's odometry
   pose. `--max_registration_frames` (0 = no cap) is retained as a hard cap
   on the kept count, applied after the motion gate.

2. Odometry health check: auto-detect and truncate on tracking loss.
   Problem solved: once odometry is lost (SLAM/VIO relocalization failure,
   wheel-odom teleport, etc.), every frame anchored to it afterward is
   placed at a wrong pose, and the merged output comes out skewed --
   silently, unless something catches it.
   New behavior: `detect_odom_loss(odom_data, outlier_multiplier=6.0)` scans
   every consecutive pair in the FULL raw odometry stream, computing implied
   linear speed and angular rate. It is self-calibrating: the "normal"
   baseline is the bag's own 95th-percentile speed/rate, not a hardcoded
   absolute limit, so it works across platforms/speeds without per-bag
   tuning. Any consecutive pair exceeding `outlier_multiplier x baseline` on
   either metric is flagged, and that pair's earlier timestamp becomes the
   cutoff. `run_odom_anchored_registration()` runs this check (unless
   --disable_odom_health_check) immediately on the full raw odometry stream,
   BEFORE frame selection -- so the motion gate, ICP refinement, and loop
   closure never see a single frame past the detected loss point. The
   segment after a detected loss is never auto-spliced back in.

PATCH NOTE (odom-anchored registration; loop-closure information weighting)
-----------------------------------------------------------------------------
All five pipelines that use this module call `run_odom_anchored_registration()`
which makes timestamped odometry the PRIMARY pose source and demotes ICP to
an optional, strongly-gated local refinement. This fixed two failure modes
present in the previous ICP-primary design:

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
- `run_icp_posegraph()` -- the old ICP-primary pose-graph pipeline.
- `detect_loop_closure()` -- only ever called from `run_icp_posegraph()`.
- `iter_registered_frame_chunks()` -- never actually called by any pipeline.
- `select_registration_frames()` -- the old --frame_stride-based selector,
  replaced by `select_registration_frames_by_motion()` above. Do not
  reintroduce --frame_stride; if you are restoring stride-based behaviour
  for a one-off comparison, pull it from version control history rather
  than re-adding it here.

PERF NOTE (loop-closure hot path: RANSAC iterations + redundant deepcopy)
-----------------------------------------------------------------------------
Loop closure defaults to OFF (--enable_loop_closure defaults False in every
pipeline), so this only matters for runs that opt in. When enabled, two
changes reduce cost with no behavior change to the accepted/rejected outcome
of any candidate:

1. `ransac_coarse_alignment()`'s `RANSACConvergenceCriteria` dropped from
   (100_000, 0.999) to (10_000, 0.99). This is a pre-screening step only --
   every candidate that clears it still goes through a fitness-thresholded
   point-to-plane ICP refinement below, gated at `loop_closure_fitness_thresh`
   (defaulting to the same bar as local ICP refinement). 10,000 iterations at
   a 0.99 confidence level is already generous for a 3-point RANSAC estimate
   and cuts the single most expensive call in this path by roughly an order
   of magnitude.
2. Removed two `copy.deepcopy()` calls inside the per-candidate loop
   (`current_fpfh = compute_fpfh_descriptor(copy.deepcopy(current_cloud_normals), ...)`
   and the `source_copy`/`target_copy` pair built before RANSAC+ICP).
   Neither Open3D's `registration_ransac_based_on_feature_matching` nor
   `registration_icp` mutate their `source`/`target` point clouds --
   they only read points/normals/features and return a RegistrationResult.
   `compute_fpfh_descriptor()`'s internal `_ensure_unit_geometric_normals()`
   call does mutate its input's `.normals` in place, but `current_cloud_normals`
   already has unit geometric normals from the same call earlier in the main
   registration loop, so re-running it is deterministic and redundant, never
   destructive. Since `current_cloud_normals` is also the exact object later
   pushed into `loop_history` and reused as `prev_cloud_normals`/a future
   `hist_cloud`, dropping the copies means those call sites now operate on
   the same object rather than a throwaway clone -- with no change in the
   values any of them read.

   NOTE: the `loop_history` candidate pool is already a
   `deque(maxlen=loop_closure_temporal_window)` (default 100) and is
   filtered by `loop_closure_radius` before RANSAC ever runs, so the
   candidate-selection scan itself is already bounded and was not further
   optimized here -- a linear scan over <=100 items is not a measurable
   cost next to RANSAC/ICP.
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
# Odometry health check (tracking loss / teleport detection)
# ---------------------------------------------------------------------------
def detect_odom_loss(
    odom_data: dict[int, np.ndarray],
    outlier_multiplier: float = 6.0,
) -> int | None:
    """
    Scan the FULL raw odometry stream for a tracking loss / teleport.

    Computes implied linear speed (m/s) and angular rate (deg/s) between
    every consecutive pair of odometry samples (sorted by timestamp). The
    "normal" baseline is THIS bag's own 95th-percentile speed/rate --
    self-calibrating, not a hardcoded absolute limit, so it works across
    platforms/speeds without per-bag tuning. The first consecutive pair
    whose speed OR rate exceeds `outlier_multiplier * baseline` is flagged
    as a tracking loss/teleport.

    Args:
        odom_data: dict mapping odometry timestamp (ns) -> 4x4 transform.
        outlier_multiplier: sensitivity; lower = more sensitive.

    Returns:
        The cutoff timestamp (the last known-good odometry sample, i.e.
        the EARLIER timestamp of the flagged pair), or None if no loss was
        detected (including when there are too few samples to establish a
        baseline).
    """
    ts_sorted = sorted(odom_data.keys())
    if len(ts_sorted) < 3:
        return None

    speeds: list[float] = []
    rates: list[float] = []

    for i in range(1, len(ts_sorted)):
        t0, t1 = ts_sorted[i - 1], ts_sorted[i]
        dt = (t1 - t0) / 1e9
        if dt <= 0:
            speeds.append(0.0)
            rates.append(0.0)
            continue

        pose0, pose1 = odom_data[t0], odom_data[t1]
        translation = float(np.linalg.norm(pose1[:3, 3] - pose0[:3, 3]))

        relative_rotation = pose0[:3, :3].T @ pose1[:3, :3]
        trace = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = float(np.degrees(np.arccos(trace)))

        speeds.append(translation / dt)
        rates.append(rotation_deg / dt)

    speeds_arr = np.asarray(speeds, dtype=np.float64)
    rates_arr = np.asarray(rates, dtype=np.float64)

    speed_baseline = float(np.percentile(speeds_arr, 95)) if len(speeds_arr) else 0.0
    rate_baseline = float(np.percentile(rates_arr, 95)) if len(rates_arr) else 0.0

    speed_thresh = speed_baseline * outlier_multiplier
    rate_thresh = rate_baseline * outlier_multiplier

    for i, (speed, rate) in enumerate(zip(speeds, rates)):
        speed_flagged = speed_thresh > 0.0 and speed > speed_thresh
        rate_flagged = rate_thresh > 0.0 and rate > rate_thresh
        if not (speed_flagged or rate_flagged):
            continue

        cutoff_ts = ts_sorted[i]  # last known-good sample, before the jump
        excluded_count = len(ts_sorted) - (i + 1)
        message = (
            f"Odometry health check: tracking loss/teleport detected at "
            f"t={ts_sorted[i + 1] / 1e9:.3f}s (speed={speed:.3f} m/s vs "
            f"baseline {speed_baseline:.3f} m/s, rate={rate:.3f} deg/s vs "
            f"baseline {rate_baseline:.3f} deg/s; multiplier="
            f"{outlier_multiplier:.1f}). Truncating at last known-good "
            f"sample t={cutoff_ts / 1e9:.3f}s -- {excluded_count} "
            "odometry sample(s) after this point will be excluded and "
            "never spliced back in."
        )
        logger.warning(message)
        print(f"Warning: {message}")
        return cutoff_ts

    return None


# ---------------------------------------------------------------------------
# Registration / merge selection helpers
# ---------------------------------------------------------------------------
def select_registration_frames_by_motion(
    pointclouds: list[tuple[int, o3d.geometry.PointCloud]],
    odom_data: dict[int, np.ndarray],
    min_move_distance: float = 0.10,
    min_rotation_angle_deg: float = 5.0,
    max_registration_frames: int = 0,
    odom_max_latency_ns: int = int(0.5 * 1e9),
) -> tuple[list[tuple[int, o3d.geometry.PointCloud]], list[int]]:
    """
    Select input clouds for registration by odometry-relative motion
    instead of a fixed index stride.

    Walks `pointclouds` in temporal order (the order they were provided
    in) and keeps a frame only if it has translated >= `min_move_distance`
    meters OR rotated >= `min_rotation_angle_deg` degrees relative to the
    last KEPT frame's odometry pose. A frame whose odometry pose cannot be
    interpolated (no coverage within `odom_max_latency_ns`) is kept as-is
    and NOT motion-gated -- it is passed through so a later odometry-
    coverage check (e.g. in `run_odom_anchored_registration`) can decide
    whether to drop it; motion gating and odometry-coverage gating are
    deliberately kept as separate concerns.

    If `odom_data` is empty, or both thresholds are <= 0 (gate disabled),
    every input frame is kept in order, subject only to the hard cap.

    Args:
        pointclouds: (timestamp, cloud) records in temporal order.
        odom_data: dict mapping odometry timestamp (ns) -> 4x4 transform.
        min_move_distance: meters; <= 0 disables this half of the gate.
        min_rotation_angle_deg: degrees; <= 0 disables this half of the gate.
        max_registration_frames: hard cap on KEPT frames; <= 0 = no cap.
        odom_max_latency_ns: max allowed gap to odometry coverage for the
            per-frame pose lookup used by the gate itself.

    Returns:
        selected_pointclouds: the kept (timestamp, cloud) records.
        original_indices: index of each kept record within the original
            `pointclouds` list.
    """
    if not pointclouds:
        raise RuntimeError("No point clouds available for registration.")

    min_move_distance = max(0.0, float(min_move_distance))
    min_rotation_angle_deg = max(0.0, float(min_rotation_angle_deg))
    max_registration_frames = max(0, int(max_registration_frames))
    gate_disabled = min_move_distance <= 0.0 and min_rotation_angle_deg <= 0.0

    if not odom_data or gate_disabled:
        selected = list(pointclouds)
        if max_registration_frames > 0:
            selected = selected[:max_registration_frames]
        original_indices = list(range(len(selected)))
    else:
        odom_ts_sorted = sorted(odom_data.keys())
        selected = []
        original_indices = []
        last_kept_pose: np.ndarray | None = None

        for idx, (timestamp, cloud) in enumerate(pointclouds):
            pose = interpolate_odom_pose(timestamp, odom_ts_sorted, odom_data, odom_max_latency_ns)

            if pose is None or last_kept_pose is None:
                keep = True
            else:
                relative = np.linalg.inv(last_kept_pose) @ pose
                translation = float(np.linalg.norm(relative[:3, 3]))
                trace = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
                rotation_deg = float(np.degrees(np.arccos(trace)))
                keep = (
                    (min_move_distance > 0.0 and translation >= min_move_distance)
                    or (min_rotation_angle_deg > 0.0 and rotation_deg >= min_rotation_angle_deg)
                )

            if not keep:
                continue

            selected.append((timestamp, cloud))
            original_indices.append(idx)
            if pose is not None:
                last_kept_pose = pose

            if max_registration_frames > 0 and len(selected) >= max_registration_frames:
                break

    if len(selected) < 2:
        raise RuntimeError(
            "Too few frames after motion-gated registration selection. "
            "Lower --min_move_distance/--min_rotation_angle_deg, increase "
            "--max_registration_frames, or provide a bag containing more "
            "motion / at least two point-cloud frames."
        )

    return selected, original_indices


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
    #
    # NOTE: callers no longer need to pass a deepcopy of `pcd` here. This
    # function's normal re-computation is deterministic given the same
    # points/voxel_size, so calling it on an already-unit-normal cloud
    # (as produced earlier in run_odom_anchored_registration's main loop)
    # is redundant but not destructive -- it does not need to be isolated
    # behind a defensive copy.
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
    """Compute a feature-based coarse transform for loop-closure validation.

    PERF NOTE: `RANSACConvergenceCriteria` was reduced from (100_000, 0.999)
    to (10_000, 0.99). This is only a pre-screening step -- every candidate
    that clears it is still validated by a fitness-thresholded point-to-plane
    ICP refinement in `run_odom_anchored_registration()` before being
    accepted as a pose-graph edge, so the lower iteration count trades a
    small amount of RANSAC pre-screening recall (not final-match precision)
    for roughly a 10x reduction in the single most expensive call in the
    loop-closure candidate loop.
    """
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
            10_000,
            0.99,
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
    graph.

    `pointclouds` is expected to be the FULL raw (timestamp, cloud) list in
    temporal order, NOT pre-thinned by the caller. This function internally:

    1. Runs the odometry health check (unless --disable_odom_health_check)
       over the FULL raw odometry stream and truncates both `pointclouds`
       and `odom_data` at the first detected tracking loss/teleport, BEFORE
       any frame selection -- so the motion gate, ICP refinement, and loop
       closure never see a single frame past the detected loss point.
    2. Runs motion-gated frame selection (`select_registration_frames_by_motion`)
       over the (possibly truncated) frame list.
    3. Looks up each selected frame's odometry pose; frames without odometry
       coverage within `--odom_max_latency` are dropped and counted loudly
       (logged and printed), never silently.
    4. Optionally refines sequential poses with gated ICP and/or applies
       loop closure via a weighted pose graph.

    Memory/perf safeguards:
        - Only the current and previous frame's normal-computed cloud are
          held for sequential refinement (no whole-trajectory cloud list).
        - Normals are computed exactly once per frame and carried forward.
        - Loop-closure candidates are held in a `deque(maxlen=...)` bounded
          window, and each closure is detected exactly once (the accepted
          transform/fitness is reused directly as a pose-graph edge, never
          re-computed for a summary count).

    Args:
        pointclouds: FULL raw (timestamp, cloud) records in temporal order.
        odom_data: dict mapping odometry timestamp -> 4x4 transform.
        args: namespace exposing (at least) `voxel_size`, `odom_max_latency`,
            `min_move_distance`, `min_rotation_angle_deg`,
            `max_registration_frames`, `disable_odom_health_check`,
            `odom_loss_speed_multiplier`, `enable_icp_refinement`,
            `icp_dist_thresh`, `icp_fitness_thresh`,
            `max_icp_translation_correction`, `max_icp_rotation_correction_deg`,
            `enable_loop_closure`, `loop_closure_radius`,
            `loop_closure_fitness_thresh`, `loop_closure_search_interval`,
            `loop_closure_temporal_window`.

    Returns:
        (final_poses, stats)
        final_poses: timestamp -> 4x4 transform that maps that frame's own
            points into the merge/world frame (apply via `cloud.transform(pose)`).
            Only timestamps for KEPT, odometry-covered frames are present.
        stats: coverage counters for end-to-end logging -- total_raw_frames,
            odom_loss_detected, frames_excluded_by_health_check,
            total_selected, dropped_no_odom, with_pose, icp_attempted,
            icp_accepted, loop_closures_found, merged.
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

    total_raw_frames = len(pointclouds)

    voxel_size = _arg_float(args, "voxel_size", 0.05)
    if voxel_size <= 0.0:
        raise ValueError("--voxel_size must be greater than zero.")

    odom_max_latency_ns = int(_arg_float(args, "odom_max_latency", 0.5) * 1e9)

    min_move_distance = _arg_float(args, "min_move_distance", 0.10)
    min_rotation_angle_deg = _arg_float(args, "min_rotation_angle_deg", 5.0)
    max_registration_frames = _arg_int(args, "max_registration_frames", 0)

    disable_odom_health_check = _arg_bool(args, "disable_odom_health_check", False)
    odom_loss_speed_multiplier = _arg_float(args, "odom_loss_speed_multiplier", 6.0)

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

    # -----------------------------------------------------------------
    # Step 1: odometry health check on the FULL raw odometry stream,
    # BEFORE any frame selection.
    # -----------------------------------------------------------------
    odom_loss_detected = False
    frames_excluded_by_health_check = 0

    if not disable_odom_health_check:
        cutoff_ts = detect_odom_loss(odom_data, outlier_multiplier=odom_loss_speed_multiplier)
        if cutoff_ts is not None:
            odom_loss_detected = True
            pre_cutoff_frames = [(ts, cloud) for ts, cloud in pointclouds if ts <= cutoff_ts]
            frames_excluded_by_health_check = len(pointclouds) - len(pre_cutoff_frames)
            odom_data = {ts: pose for ts, pose in odom_data.items() if ts <= cutoff_ts}

            print(
                f"Odometry health check: truncating to frames at/before "
                f"t={cutoff_ts / 1e9:.3f}s ({len(pre_cutoff_frames):,}/"
                f"{len(pointclouds):,} frame(s) retained). Pass "
                f"--max_registration_frames {len(pre_cutoff_frames)} to "
                "reproduce this exact truncation without re-running the "
                "health check, or --disable_odom_health_check to bypass it."
            )
            pointclouds = pre_cutoff_frames

        if not pointclouds or not odom_data:
            raise RuntimeError(
                "Odometry health check truncated all frames (tracking "
                "loss detected at or before the first frame). Pass "
                "--disable_odom_health_check to bypass, or inspect the "
                "bag's odometry stream."
            )

    # -----------------------------------------------------------------
    # Step 2: motion-gated frame selection on the (possibly truncated)
    # frame list.
    # -----------------------------------------------------------------
    selected_frames, _original_indices = select_registration_frames_by_motion(
        pointclouds,
        odom_data,
        min_move_distance=min_move_distance,
        min_rotation_angle_deg=min_rotation_angle_deg,
        max_registration_frames=max_registration_frames,
        odom_max_latency_ns=odom_max_latency_ns,
    )
    total_selected = len(selected_frames)
    print(
        f"Coverage: raw frames={total_raw_frames:,} -> after odom health "
        f"check={len(pointclouds):,} -> motion-gated selection="
        f"{total_selected:,} (min_move_distance={min_move_distance}m, "
        f"min_rotation_angle_deg={min_rotation_angle_deg}deg, "
        f"max_registration_frames={max_registration_frames})."
    )

    # -----------------------------------------------------------------
    # Step 3: per-frame odometry pose lookup; frames without coverage are
    # dropped and counted loudly.
    # -----------------------------------------------------------------
    odom_ts_sorted = sorted(odom_data.keys())

    raw_odom_poses: dict[int, np.ndarray] = {}
    dropped_no_odom = 0
    kept_records: list[tuple[int, o3d.geometry.PointCloud]] = []

    for timestamp, cloud in selected_frames:
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
    # applied to the previous frame.
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
                # PERF FIX: no longer wraps `current_cloud_normals` in
                # copy.deepcopy(). compute_fpfh_descriptor() mutates its
                # input's normals in place via _ensure_unit_geometric_normals,
                # but current_cloud_normals already carries the identical
                # unit geometric normals computed a few lines above in this
                # same iteration, so the in-place recompute is a no-op in
                # effect, not a destructive mutation. current_cloud_normals
                # is safe to reuse directly here and later in loop_history.
                current_fpfh = compute_fpfh_descriptor(current_cloud_normals, voxel_size)
                current_pos_xyz = node_poses[position][:3, 3]

                for hist_timestamp, hist_cloud, hist_fpfh in loop_history:
                    if hist_fpfh is None:
                        continue
                    hist_index = node_index_by_timestamp[hist_timestamp]
                    hist_pos_xyz = node_poses[hist_index][:3, 3]
                    if np.linalg.norm(current_pos_xyz - hist_pos_xyz) > loop_closure_radius:
                        continue

                    try:
                        # PERF FIX: no longer deepcopies current_cloud_normals/
                        # hist_cloud before RANSAC+ICP. Neither
                        # registration_ransac_based_on_feature_matching() nor
                        # registration_icp() mutate their source/target point
                        # clouds -- both only read points/normals/features and
                        # return a RegistrationResult -- so passing the
                        # originals directly is safe and avoids a full
                        # point-cloud copy per candidate.
                        coarse = ransac_coarse_alignment(
                            current_cloud_normals, hist_cloud, current_fpfh, hist_fpfh, voxel_size
                        )

                        if coarse.fitness < loop_fitness_thresh * 0.5:
                            continue

                        refined = o3d.pipelines.registration.registration_icp(
                            current_cloud_normals,
                            hist_cloud,
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
        "total_raw_frames": total_raw_frames,
        "odom_loss_detected": int(odom_loss_detected),
        "frames_excluded_by_health_check": frames_excluded_by_health_check,
        "total_selected": total_selected,
        "dropped_no_odom": dropped_no_odom,
        "with_pose": len(kept_records),
        "icp_attempted": icp_attempted,
        "icp_accepted": icp_accepted,
        "loop_closures_found": loop_closures_found,
        "merged": len(kept_records),
    }
    return final_poses, stats
