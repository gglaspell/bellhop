"""
shared/projection.py - Shared world -> camera -> optical-frame projection
utilities.

REFACTOR NOTE (moved from atlas_pipeline/common):
This module used to live at `pipelines/atlas_pipeline/common/projection.py`.
See `shared/trajectory.py`'s module docstring for the full rationale --
in short, having a texture-baking-only `atlas_pipeline/common/` package
next to the pipeline-wide `shared/` package was a confusing split for code
that isn't atlas-bake-specific. Its contents now live directly in
`shared/`. Every caller in `atlas_pipeline/` (`meshgenerator.py`,
`atlaspacker.py`, `viewassignment.py`, `visibilityfilter.py`,
`pointcloudutils.py`) has been updated to import from `shared.projection`
instead of `atlas_pipeline.common.projection`. `atlas_pipeline/common/`
no longer exists.

Originally consolidated the ROS-body/optical-frame projection math that
was copy-pasted (identically) across four files in the two legacy repos
this pipeline was merged in from: point_cloud_coloring.py, atlaspacker.py,
viewassignment.py, and meshgenerator.py.

Frame conventions
------------------
World frame:
    Arbitrary fixed frame in which the trajectory poses (pos, orient) and
    the point cloud / mesh vertices are expressed.

ROS body frame (REP-103):
    x = forward, y = left, z = up.
    Obtained from world coordinates via: cam_rot.inv().apply(world - cam_pos)

Optical (camera) frame:
    x = right, y = down, z = forward (standard pinhole/OpenCV convention).
    Obtained from ROS body frame via the fixed axis permutation:
        x_opt = -y_body
        y_opt = -z_body
        z_opt = x_body
"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def world_to_body(points: np.ndarray, cam_pos: np.ndarray, cam_rot: R) -> np.ndarray:
    """
    Transform world-frame points/vectors into the ROS body frame
    (x=forward, y=left, z=up) relative to a camera pose.

    Args:
        points: (N, 3) array of world-frame points (or direction vectors,
            e.g. face normals — in that case cam_pos should be np.zeros(3)).
        cam_pos: (3,) camera position in world frame.
        cam_rot: camera orientation (world -> body rotation) as a
            scipy Rotation.

    Returns:
        (N, 3) array in the ROS body frame.
    """
    return cam_rot.inv().apply(points - cam_pos)


def body_to_optical(points_body: np.ndarray) -> np.ndarray:
    """
    Convert ROS body-frame (x=fwd, y=left, z=up) coordinates to the
    optical/camera frame (x=right, y=down, z=forward).

    Args:
        points_body: (N, 3) array in the ROS body frame.

    Returns:
        (N, 3) array in the optical frame.
    """
    return np.column_stack([
        -points_body[:, 1],  # x_opt = -y_body
        -points_body[:, 2],  # y_opt = -z_body
        points_body[:, 0],   # z_opt = x_body
    ])


def world_to_optical(points: np.ndarray, cam_pos: np.ndarray, cam_rot: R) -> np.ndarray:
    """
    Convenience wrapper: world frame -> ROS body frame -> optical frame,
    relative to a single camera pose.

    Args:
        points: (N, 3) world-frame points or direction vectors.
        cam_pos: (3,) camera position in world frame. Pass np.zeros(3)
            when projecting direction vectors (e.g. face normals) rather
            than absolute positions.
        cam_rot: camera orientation as a scipy Rotation.

    Returns:
        (N, 3) array in the optical frame.
    """
    body = world_to_body(points, cam_pos, cam_rot)
    return body_to_optical(body)


def project_to_pixels(
    points_optical: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    min_z: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pinhole-project optical-frame points to pixel coordinates.

    Args:
        points_optical: (N, 3) array in the optical frame (x=right, y=down,
            z=forward).
        fx, fy, cx, cy: camera intrinsics.
        min_z: minimum forward depth to be considered in front of the
            camera (avoids division blow-up / behind-camera artifacts).

    Returns:
        (u, v, valid_mask): pixel x/y coordinates (N,) each, and a boolean
        mask (N,) that is True where z > min_z (i.e. the point is in front
        of the camera). u/v are NOT yet clamped to image bounds — callers
        should additionally check 0 <= u < width and 0 <= v < height.
    """
    z = points_optical[:, 2].copy()
    valid = z > min_z
    z_safe = np.where(z == 0, 1.0, z)
    u = (fx * points_optical[:, 0] / z_safe) + cx
    v = (fy * points_optical[:, 1] / z_safe) + cy
    return u, v, valid


def euclidean_distance(points: np.ndarray, cam_pos: np.ndarray) -> np.ndarray:
    """Euclidean distance (N,) from each world-frame point to the camera
    position. Distinct from optical-frame z (pure depth along view axis)."""
    return np.linalg.norm(points - cam_pos, axis=1)
