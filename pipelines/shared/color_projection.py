"""
shared/color_projection.py - Camera-image point-cloud coloring helpers
shared by `color_mesh.py` and `color_tiles_3d.py`.

REFACTOR NOTE (overlap consolidation):
`color_mesh.py` and `color_tiles_3d.py` both independently implemented:
  - the same FALLBACK_GRAY constant,
  - the same pinhole camera-projection coloring math
    (`_color_pcd_from_image`),
  - the same "guarantee a colors array before merging" fallback
    (`_fill_fallback_color`),
  - the same "is this pixel a placeholder gray, not real color"
    std/mean heuristic, and
  - the same Open3D `+`/`+=`-avoiding manual-concatenation merge
    (`_concat_point_clouds`), because Open3D's `PointCloud.__add__`/
    `__iadd__` clears colors on the ENTIRE result if either operand
    lacks a colors array (or the arrays mismatch length).

Any bugfix to any of these had to be applied twice by hand, which is
exactly how `color_tiles_3d.py` ended up missing the total-color-loss fix
that had already landed in `color_mesh.py` (see color_tiles_3d.py's own
BUGFIX note history). Centralizing them here means a future fix only
needs to be made once, and both pipelines' "is this a placeholder gray"
detection is guaranteed to stay in lockstep since they call the same
function instead of two independently-maintained copies of the same
threshold.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

# Neutral fallback color used for any point/frame that could not be
# camera-colored (no image in time tolerance, no intrinsics yet, or a
# point that didn't project into the image). A single shared constant
# keeps every "is this a placeholder gray" heuristic in this module (and
# in both calling pipelines) consistent.
FALLBACK_GRAY = (0.5, 0.5, 0.5)


def color_pcd_from_image(
    pcd: o3d.geometry.PointCloud,
    image: Image.Image,
    camera_pose: np.ndarray,
    intrinsics: tuple,
    min_depth: float = 0.1,
    max_depth: float | None = None,
) -> o3d.geometry.PointCloud:
    """Project `image` onto `pcd` in-place using a pinhole camera model.

    `camera_pose` is the 4x4 world pose of the camera optical center.
    Points that are behind the camera, outside [min_depth, max_depth],
    or don't land inside the image bounds get `FALLBACK_GRAY` instead of
    a sampled pixel. Returns `pcd` (mutated) for convenient chaining.
    """
    fx, fy, cx, cy, width, height = intrinsics
    points = np.asarray(pcd.points)
    image_array = np.asarray(image)

    rotation = R.from_matrix(camera_pose[:3, :3])
    body = rotation.inv().apply(points - camera_pose[:3, 3])

    optical_x = -body[:, 1]
    optical_y = -body[:, 2]
    optical_z = body[:, 0]
    distance = np.linalg.norm(body, axis=1)

    valid = (optical_z > 1e-6) & (distance >= min_depth)
    if max_depth is not None:
        valid &= distance <= max_depth

    safe_z = np.where(optical_z > 1e-6, optical_z, 1e-6)
    u = fx * optical_x / safe_z + cx
    v = fy * optical_y / safe_z + cy
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

    colors = np.full((len(points), 3), FALLBACK_GRAY, dtype=np.float64)

    if np.any(valid):
        colors[valid] = (
            image_array[
                np.clip(v[valid].astype(np.int32), 0, height - 1),
                np.clip(u[valid].astype(np.int32), 0, width - 1),
                :3,
            ]
            / 255.0
        )

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def fill_fallback_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Guarantee `pcd` has an explicit `FALLBACK_GRAY` colors array.

    Any frame that skips camera projection entirely (no image within
    time tolerance, no intrinsics yet, or color mode is off) must still
    carry an explicit colors array of matching length before it reaches
    `concat_point_clouds` -- otherwise a mix of colored and uncolored
    frames can silently strip color from the whole merged result (see
    `concat_point_clouds`'s docstring).
    """
    colors = np.full((len(pcd.points), 3), FALLBACK_GRAY, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def is_gray_fill(colors: np.ndarray) -> np.ndarray:
    """Boolean mask: rows of `colors` that look like `FALLBACK_GRAY`.

    Uses a std-below-threshold + mean-near-0.5 test (instead of an exact
    equality check) so it still matches gray fill after it has passed
    through a voxel-downsample average with other gray-filled points.
    """
    colors = np.asarray(colors, dtype=np.float64)
    std = np.std(colors, axis=1)
    mean = np.mean(colors, axis=1)
    return (std < 0.08) & (np.abs(mean - 0.5) < 0.15)


def remove_gray_fill_near_color(
    points: np.ndarray,
    colors: np.ndarray,
    gray_filter_radius: float,
    extra_arrays: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, ...]:
    """Drop gray-fallback points that have a genuinely colored neighbor.

    Gray-fallback points with NO nearby real color are kept (there is
    nothing better to show there); gray-fallback points near real color
    are removed so real camera color always wins in overlap regions.

    `extra_arrays` are any additional per-point arrays (e.g. normals)
    that must be filtered in lockstep with `points`/`colors`. Returns
    `(points, colors, *extra_arrays)` filtered to the surviving rows; if
    `gray_filter_radius <= 0`, there are no points, or there is no gray
    fill to remove, the inputs are returned unchanged.
    """
    if gray_filter_radius <= 0 or not len(points):
        return (points, colors, *extra_arrays)

    gray = is_gray_fill(colors)
    colored_points = points[~gray]

    if not len(colored_points) or not gray.any():
        return (points, colors, *extra_arrays)

    neighbors = cKDTree(colored_points).query_ball_point(
        points[gray], gray_filter_radius
    )
    near_color = np.array(
        [len(neighbors_for_point) > 0 for neighbors_for_point in neighbors],
        dtype=bool,
    )

    keep = np.ones(len(points), dtype=bool)
    keep[np.flatnonzero(gray)[near_color]] = False

    kept_idx = np.flatnonzero(keep)
    return (
        points[kept_idx],
        colors[kept_idx],
        *(arr[kept_idx] for arr in extra_arrays),
    )


def concat_point_clouds(
    clouds: list[o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    """Merge point clouds via manual array concatenation.

    Open3D's `PointCloud.__add__`/`__iadd__` clears the ENTIRE result's
    colors if either operand lacks a colors array (or the arrays
    mismatch length). This function never touches `+`/`+=`, so it is
    immune to that behavior. Every incoming cloud is expected to already
    carry a `.colors` array of matching length -- either real
    camera-projected color or `fill_fallback_color`'s fallback -- so the
    `np.vstack` calls below are always safe.
    """
    non_empty = [c for c in clouds if len(c.points)]

    if not non_empty:
        return o3d.geometry.PointCloud()

    points = np.vstack([np.asarray(c.points) for c in non_empty])
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(points)

    if all(c.has_colors() for c in non_empty):
        colors = np.vstack([np.asarray(c.colors) for c in non_empty])
        merged.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Should not happen once every frame is fallback-colored before
        # reaching this function, but guard against it defensively
        # rather than silently emitting an uncolored merged cloud.
        merged = fill_fallback_color(merged)

    if non_empty and all(c.has_normals() for c in non_empty):
        normals = np.vstack([np.asarray(c.normals) for c in non_empty])
        if len(normals) == len(points):
            merged.normals = o3d.utility.Vector3dVector(normals)

    return merged
