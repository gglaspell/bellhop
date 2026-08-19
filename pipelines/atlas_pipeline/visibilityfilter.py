"""
pipelines.atlas_pipeline.visibilityfilter
===========================================
Point-cloud visibility filtering (camera-frustum + keyframe-timestamp
proximity) for the atlas-bake pipeline.

REFACTOR NOTE (moved from atlas_pipeline/common to shared):
Previously imported trajectory/projection helpers from `.common.trajectory`
and `.common.projection` (i.e. `atlas_pipeline/common/`). That package has
been merged into `shared/` (see `shared/trajectory.py`'s module
docstring), so this now imports from `..shared.trajectory` and
`..shared.projection` instead. No behavior change. This module is not
currently imported by `texture_baking.py`, but it's fixed here for
consistency with the rest of the `atlas_pipeline` package and so it
doesn't break the moment something starts importing it.
"""

import logging

import numpy as np
import open3d as o3d

from ..shared.trajectory import (
    load_trajectory, build_trajectory_tree, get_pose_at, parse_stem_timestamp,
)
from ..shared.projection import world_to_optical, project_to_pixels


class VisibilityFilter:
    def __init__(self, intrinsics, traj_path, timestamp_offset=0.0):
        self.intrinsics = intrinsics
        self.traj_path = traj_path
        self.timestamp_offset = timestamp_offset

        self.traj = load_trajectory(traj_path)
        self.tree = build_trajectory_tree(self.traj)

    def _is_visible_in_any_keyframe(self, points, keyframes, ts_tolerance=0.5):
        """Return True if at least one point projects inside the image bounds
        (and in front of the camera) for at least one nearby keyframe."""
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]
        w = self.intrinsics["width"]
        h = self.intrinsics["height"]

        for _, kf_ts, _ in keyframes:
            cp, cr = get_pose_at(self.traj, self.tree, kf_ts)
            opt = world_to_optical(points, cp, cr)
            u, v, front_mask = project_to_pixels(opt, fx, fy, cx, cy, min_z=0.1)
            in_bounds = front_mask & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if np.any(in_bounds):
                return True
        return False

    def filter_point_clouds(self, cloud_dir, keyframes, out_dir):
        ts_keys = np.array([ts for _, ts, _ in keyframes])
        files = sorted(list(cloud_dir.glob('*.pcd')) + list(cloud_dir.glob('*.ply')))

        if not files:
            return

        logging.info(
            f"Frustum-filtering {len(files)} point clouds against "
            f"{len(keyframes)} keyframes..."
        )

        kept = 0
        for f in files:
            try:
                ts = parse_stem_timestamp(f.stem, offset=self.timestamp_offset)
            except Exception:
                continue

            nearby_mask = np.abs(ts_keys - ts) <= 0.5
            if not np.any(nearby_mask):
                continue
            nearby_keyframes = [kf for kf, keep in zip(keyframes, nearby_mask) if keep]

            pcd = o3d.io.read_point_cloud(str(f))
            if len(pcd.points) == 0:
                continue

            points = np.asarray(pcd.points)
            if not self._is_visible_in_any_keyframe(points, nearby_keyframes):
                continue

            o3d.io.write_point_cloud(str(out_dir / f.name), pcd)
            kept += 1

        logging.info(f"Kept {kept}/{len(files)} point clouds after frustum filtering.")
