"""
pipelines.atlas_pipeline.pointcloudutils
=============================================
Point-cloud merge / subsample / normal-estimation utilities for the
atlas-bake pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from rosbags.highlevel import AnyReader

from .common.trajectory import (
    load_trajectory, build_trajectory_tree, parse_stem_timestamp, get_pose_at,
)
from .common.projection import world_to_optical, project_to_pixels
from ..shared.ros_io import TYPESTORE, convert_ros_pc2_to_o3d


# FIX: Range-dependent ROR/SOR outlier filtering. Previously a single
# global (ror_radius, ror_min_neighbors, sor_neighbors, sor_std_ratio)
# tuple was applied uniformly to the entire merged cloud regardless of a
# point's distance from the sensor/trajectory. Near-sensor points are
# typically dense, so tight thresholds correctly reject noise there without
# discarding valid structure; far-range points are naturally sparse, so the
# same tight thresholds over-aggressively strip legitimate far geometry
# (thin structures, distant walls, etc.), while looser thresholds would let
# too much noise through near the sensor. Bucketing by distance from the
# nearest trajectory pose and scaling the thresholds per bucket lets each
# range regime use appropriately tuned filtering.
_DEFAULT_RANGE_BINS = (
    # (near_m, far_m, ror_radius_mult, sor_std_ratio_mult)
    (0.0, 5.0, 0.6, 0.75),
    (5.0, 15.0, 1.0, 1.0),
    (15.0, float("inf"), 1.8, 1.4),
)


def _range_adaptive_outlier_removal(
    merged,
    all_ts,
    cam_positions,
    ror_radius=0.0,
    ror_min_neighbors=10,
    sor_neighbors=20,
    sor_std_ratio=2.0,
    bins=_DEFAULT_RANGE_BINS,
):
    """Apply ROR/SOR outlier removal with per-range-bucket thresholds.

    Args:
        merged: o3d.geometry.PointCloud to filter.
        all_ts: (N,) timestamp array aligned with merged.points.
        cam_positions: (M, 3) array of trajectory/camera positions used to
            compute each point's distance-from-sensor via nearest neighbour.
        ror_radius, ror_min_neighbors, sor_neighbors, sor_std_ratio: base
            thresholds, scaled per bucket by the multipliers in `bins`.
        bins: sequence of (near_m, far_m, ror_radius_mult, sor_std_ratio_mult).

    Returns:
        (filtered_merged, filtered_all_ts)
    """
    pts = np.asarray(merged.points)
    if len(pts) == 0 or cam_positions is None or len(cam_positions) == 0:
        return merged, all_ts

    tree = cKDTree(cam_positions)
    dist, _ = tree.query(pts)

    keep_mask = np.zeros(len(pts), dtype=bool)

    for near, far, r_mult, s_mult in bins:
        bucket_idx = np.where((dist >= near) & (dist < far))[0]
        if len(bucket_idx) == 0:
            continue

        sub = merged.select_by_index(bucket_idx)
        local_keep = np.arange(len(bucket_idx))

        if ror_radius is not None and float(ror_radius) > 0.0 and int(ror_min_neighbors) > 0:
            _, ind = sub.remove_radius_outlier(
                nb_points=int(ror_min_neighbors),
                radius=float(ror_radius) * r_mult,
            )
            local_keep = local_keep[np.asarray(ind, dtype=np.int64)]

        if (
            len(local_keep) > 0
            and sor_neighbors is not None and int(sor_neighbors) > 0
            and sor_std_ratio is not None and float(sor_std_ratio) > 0.0
        ):
            sub2 = sub.select_by_index(local_keep.tolist())
            _, ind2 = sub2.remove_statistical_outlier(
                nb_neighbors=int(sor_neighbors),
                std_ratio=float(sor_std_ratio) * s_mult,
            )
            local_keep = local_keep[np.asarray(ind2, dtype=np.int64)]

        if len(local_keep) > 0:
            keep_mask[bucket_idx[local_keep]] = True

    keep_global = np.where(keep_mask)[0]
    filtered = merged.select_by_index(keep_global.tolist())
    filtered_ts = all_ts[keep_global]
    return filtered, filtered_ts


class PointCloudProcessor:

    def merge_point_clouds(
        self,
        folder,
        out_ply,
        out_ts,
        timestamp_offset: float = 0.0,
        ror_radius: float = 0.0,
        ror_min_neighbors: int = 10,
        sor_neighbors: int = 20,
        sor_std_ratio: float = 2.0,
        traj_path=None,
    ) -> None:
        # FIX: added optional `traj_path` so range-dependent outlier removal
        # can bucket points by distance from the nearest trajectory pose.
        # When not provided, filtering falls back to the previous
        # global-threshold behaviour (no bucketing).
        folder = Path(folder)
        files = sorted(list(folder.glob("*.pcd")) + list(folder.glob("*.ply")))

        if not files:
            raise FileNotFoundError(f"No point cloud files found in {folder} to merge.")

        logging.info(f"Merging {len(files)} files...")
        pts_list: list[np.ndarray] = []
        ts_list: list[np.ndarray] = []

        for f in files:
            pcd = o3d.io.read_point_cloud(str(f))
            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                continue

            pts_list.append(pts)
            ts = parse_stem_timestamp(f.stem, offset=timestamp_offset)
            ts_list.append(np.full(len(pts), ts, dtype=np.float64))

        if not pts_list:
            raise ValueError("All point clouds were empty after load; nothing to merge.")

        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(np.vstack(pts_list))
        all_ts = np.concatenate(ts_list, axis=0)

        if len(merged.points) == 0:
            raise ValueError("Merged point cloud unexpectedly has 0 points.")

        # FIX: range-dependent ROR/SOR. If a trajectory is available, bucket
        # points by distance-from-nearest-pose and apply tier-specific
        # thresholds (tighter near, looser far) instead of one global
        # threshold for the whole cloud. Falls back to the previous
        # single-pass global filtering when no traj_path is given, to
        # preserve existing call sites/behaviour.
        n_before = len(merged.points)
        if traj_path is not None:
            traj_df = load_trajectory(traj_path)
            cam_positions = traj_df[["pos.x", "pos.y", "pos.z"]].values
            logging.info(
                "Applying range-adaptive ROR/SOR "
                f"(base ror_radius={ror_radius}, ror_min_neighbors={ror_min_neighbors}, "
                f"sor_neighbors={sor_neighbors}, sor_std_ratio={sor_std_ratio})..."
            )
            merged, all_ts = _range_adaptive_outlier_removal(
                merged, all_ts, cam_positions,
                ror_radius=ror_radius, ror_min_neighbors=ror_min_neighbors,
                sor_neighbors=sor_neighbors, sor_std_ratio=sor_std_ratio,
            )
            if len(merged.points) == 0:
                raise ValueError(
                    "Range-adaptive ROR/SOR removed all points. "
                    f"Try loosening thresholds (ror_radius={ror_radius}, "
                    f"ror_min_neighbors={ror_min_neighbors}, sor_neighbors={sor_neighbors}, "
                    f"sor_std_ratio={sor_std_ratio})."
                )
        else:
            if ror_radius is not None and float(ror_radius) > 0.0 and int(ror_min_neighbors) > 0:
                logging.info(f"Applying ROR (radius={ror_radius}, min_neighbors={ror_min_neighbors})...")
                _, ind_ror = merged.remove_radius_outlier(
                    nb_points=int(ror_min_neighbors), radius=float(ror_radius)
                )
                merged = merged.select_by_index(ind_ror)
                all_ts = all_ts[np.asarray(ind_ror, dtype=np.int64)]

                if len(merged.points) == 0:
                    raise ValueError(
                        "ROR removed all points. "
                        f"Try increasing ror_radius (current={ror_radius}) or "
                        f"decreasing ror_min_neighbors (current={ror_min_neighbors})."
                    )

            if (
                sor_neighbors is not None and int(sor_neighbors) > 0
                and sor_std_ratio is not None and float(sor_std_ratio) > 0.0
            ):
                logging.info(f"Applying SOR (neighbors={sor_neighbors}, std_ratio={sor_std_ratio})...")
                _, ind_sor = merged.remove_statistical_outlier(
                    nb_neighbors=int(sor_neighbors), std_ratio=float(sor_std_ratio)
                )
                merged = merged.select_by_index(ind_sor)
                all_ts = all_ts[np.asarray(ind_sor, dtype=np.int64)]

                if len(merged.points) == 0:
                    raise ValueError(
                        "SOR removed all points. "
                        f"Try increasing sor_std_ratio (current={sor_std_ratio}) or "
                        f"decreasing sor_neighbors (current={sor_neighbors})."
                    )

        logging.info(f"Outlier removal: {n_before} -> {len(merged.points)} points.")
        o3d.io.write_point_cloud(str(out_ply), merged)
        np.save(str(out_ts), all_ts)
        logging.info(f"Saved merged cloud with {len(merged.points)} points to {out_ply}")

    def subsample(self, in_ply, in_ts, out_ply, out_ts, res: float) -> None:
        in_ply = Path(in_ply)
        in_ts = Path(in_ts)

        if not in_ply.exists():
            raise FileNotFoundError(f"Input file missing: {in_ply}")

        pcd = o3d.io.read_point_cloud(str(in_ply))

        if len(pcd.points) == 0:
            raise ValueError(f"Input point cloud is empty: {in_ply}")

        if in_ts.exists():
            ts = np.load(str(in_ts))
            if len(ts) != len(pcd.points):
                logging.warning(
                    f"Timestamp length ({len(ts)}) != point count ({len(pcd.points)}). "
                    "Falling back to dummy timestamps."
                )
                ts = np.zeros(len(pcd.points), dtype=np.float64)
        else:
            logging.warning("Timestamp file missing, creating dummy timestamps.")
            ts = np.zeros(len(pcd.points), dtype=np.float64)

        has_normals = pcd.has_normals()

        down, _, voxel_map = pcd.voxel_down_sample_and_trace(
            float(res),
            pcd.get_min_bound(),
            pcd.get_max_bound(),
        )

        if len(down.points) == 0:
            raise ValueError(
                "Voxel downsampling removed all points. "
                f"voxel_size={res} may be larger than the scan extent."
            )

        representative_indices = [int(v[0]) for v in voxel_map]

        if has_normals:
            src_normals = np.asarray(pcd.normals)
            down.normals = o3d.utility.Vector3dVector(src_normals[representative_indices])

        o3d.io.write_point_cloud(str(out_ply), down)

        rep = np.asarray(representative_indices, dtype=np.int64)
        rep = rep[(rep >= 0) & (rep < len(ts))]
        np.save(
            str(out_ts),
            ts[rep] if len(rep) > 0 else np.zeros(len(down.points), dtype=np.float64),
        )

    def compute_normals(self, in_ply, in_ts, traj_path, out_ply) -> None:
        in_ply = Path(in_ply)
        in_ts = Path(in_ts)

        if not in_ply.exists():
            raise FileNotFoundError(f"Input file missing: {in_ply}")
        if not in_ts.exists():
            raise FileNotFoundError(f"Timestamp file missing: {in_ts}")

        pcd = o3d.io.read_point_cloud(str(in_ply))
        if len(pcd.points) == 0:
            raise ValueError(f"Input point cloud is empty: {in_ply}")

        ts = np.load(str(in_ts))

        df = load_trajectory(traj_path)

        required = {"pos.x", "pos.y", "pos.z", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Trajectory CSV missing required columns: {sorted(missing)}")

        pcd.estimate_normals()
        pts = np.asarray(pcd.points)
        nrms = np.asarray(pcd.normals)

        tree = build_trajectory_tree(df)
        _, idx = tree.query(ts.reshape(-1, 1))
        cam_pos = df.iloc[idx.flatten()][["pos.x", "pos.y", "pos.z"]].values

        vec = cam_pos - pts
        flip = np.sum(vec * nrms, axis=1) < 0
        nrms[flip] *= -1

        pcd.normals = o3d.utility.Vector3dVector(nrms)
        o3d.io.write_point_cloud(str(out_ply), pcd)

    def merge_point_clouds_from_bag(
        self, bag_path, pc_topic, keyframes, traj_df, traj_tree, intrinsics,
        out_ply, out_ts, min_frame_points=100,
        ror_radius=0.0, ror_min_neighbors=10, sor_neighbors=20, sor_std_ratio=2.0,
    ):
        """Stream pc_topic once; keep only clouds visible in >=1 keyframe."""
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        w, h = intrinsics["width"], intrinsics["height"]
        kf_ts = np.array([ts for _, ts, _ in keyframes])

        pts_list, ts_list = [], []
        with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
            connections = [c for c in reader.connections if c.topic == pc_topic]
            for connection, ts_ns, raw in reader.messages(connections=connections):
                ts_sec = ts_ns * 1e-9
                nearby = np.abs(kf_ts - ts_sec) <= 0.5
                if not np.any(nearby):
                    continue

                msg = reader.deserialize(raw, connection.msgtype)
                cloud = convert_ros_pc2_to_o3d(msg)
                if cloud is None or len(cloud.points) < min_frame_points:
                    continue
                points = np.asarray(cloud.points)

                visible = False
                for kf_ts_val in kf_ts[nearby]:
                    cp, cr = get_pose_at(traj_df, traj_tree, kf_ts_val)
                    opt = world_to_optical(points, cp, cr)
                    u, v, front = project_to_pixels(opt, fx, fy, cx, cy, min_z=0.1)
                    if np.any(front & (u >= 0) & (u < w) & (v >= 0) & (v < h)):
                        visible = True
                        break
                if not visible:
                    continue

                pts_list.append(points)
                ts_list.append(np.full(len(points), ts_sec, dtype=np.float64))

        if not pts_list:
            raise ValueError("No visible point clouds found in bag for these keyframes.")

        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(np.vstack(pts_list))
        all_ts = np.concatenate(ts_list, axis=0)

        if len(merged.points) == 0:
            raise ValueError("Merged point cloud unexpectedly has 0 points.")

        # FIX: Previously this block called remove_radius_outlier() /
        # remove_statistical_outlier() and immediately re-indexed `merged`
        # and `all_ts` with no check on whether any points survived. Under
        # aggressive ror_radius/ror_min_neighbors or sor_std_ratio settings,
        # `idx` can come back empty, silently producing a 0-point merged
        # cloud and a 0-length timestamp array that are then written to disk
        # with no error and no diagnostic -- downstream stages would fail
        # far later with a much more confusing error (or silently produce
        # an empty mesh/atlas). This mirrors the existing, correctly-guarded
        # behavior already used in merge_point_clouds() above.
        # FIX: switched from a single global ROR/SOR pass to range-adaptive
        # filtering bucketed by distance from the nearest trajectory pose
        # (traj_df/traj_tree are already available here). Tighter thresholds
        # apply to near/dense buckets, looser thresholds to far/sparse
        # buckets, instead of one threshold for the whole merged cloud. The
        # zero-point guard from the previous fix is preserved.
        n_before = len(merged.points)
        cam_positions = traj_df[["pos.x", "pos.y", "pos.z"]].values
        logging.info(
            "Applying range-adaptive ROR/SOR "
            f"(base ror_radius={ror_radius}, ror_min_neighbors={ror_min_neighbors}, "
            f"sor_neighbors={sor_neighbors}, sor_std_ratio={sor_std_ratio})..."
        )
        merged, all_ts = _range_adaptive_outlier_removal(
            merged, all_ts, cam_positions,
            ror_radius=ror_radius, ror_min_neighbors=ror_min_neighbors,
            sor_neighbors=sor_neighbors, sor_std_ratio=sor_std_ratio,
        )

        if len(merged.points) == 0:
            raise ValueError(
                "Range-adaptive ROR/SOR removed all points. "
                f"Try loosening thresholds (ror_radius={ror_radius}, "
                f"ror_min_neighbors={ror_min_neighbors}, sor_neighbors={sor_neighbors}, "
                f"sor_std_ratio={sor_std_ratio})."
            )

        logging.info(f"Outlier removal: {n_before} -> {len(merged.points)} points.")
        o3d.io.write_point_cloud(str(out_ply), merged)
        np.save(str(out_ts), all_ts)
        logging.info(f"Saved merged cloud with {len(merged.points)} points to {out_ply}")
