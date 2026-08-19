"""
shared/merge_utils.py - Bounded, chunked point-cloud merge helper shared by
`mesh.py`, `gazebo_world.py`, and `tiles_3d.py`.

REFACTOR NOTE (overlap consolidation):
All three pipelines independently defined the exact same function --
`append_chunk()` in mesh.py, `_append_chunk()` in gazebo_world.py and
tiles_3d.py -- byte-for-byte identical bodies:

    def _append_chunk(target, chunk, voxel_size):
        if not chunk:
            return target
        local = o3d.geometry.PointCloud()
        for cloud in chunk:
            local += cloud
        chunk.clear()
        if len(target.points):
            target += local
        else:
            target = local
        return target.voxel_down_sample(voxel_size)

This is the "batch frames into a bounded list, concatenate + voxel-reduce
every --merge_chunk_frames frames" pattern that keeps a streaming merge's
accumulated point count close to its true (deduplicated) size throughout
the merge, instead of growing unboundedly and only being reduced once at
the very end (see gazebo_world.py's/tiles_3d.py's own PERF FIX notes,
which independently diagnosed and fixed the identical O(n^2)-trending
merge cost by porting mesh.py's pattern -- by hand, in two separate
files). Centralizing it here means any future fix to the chunking logic
itself only needs to be made once, and all three pipelines are
guaranteed to keep behaving identically.

NOTE ON SCOPE: `color_mesh.py`/`color_tiles_3d.py` do NOT use this helper.
Their merge chunks may contain per-point colors that must never be
combined with Open3D's `+`/`+=` (see `shared/color_projection.py`'s
docstring for why), so they use `color_projection.concat_point_clouds()`
instead, which explicitly avoids `+`/`+=` for that reason. This helper is
for the plain (uncolored) point-cloud merges in `mesh.py`, `gazebo_world.py`,
and `tiles_3d.py`, where every chunk's clouds are guaranteed to either all
carry colors/normals or none do, so `+`/`+=` is safe as originally written.

NOTE ON SCOPE: `texture_baking.py` does NOT use this helper either. It
merges point clouds via a fundamentally different, disk-file-based
architecture (`atlas_pipeline.pointcloudutils.PointCloudProcessor`,
which reads/writes intermediate `.ply`/`.npy` files rather than holding a
bounded in-memory chunk list), so there is no equivalent in-memory
chunked-merge step here to extract.
"""

from __future__ import annotations

import open3d as o3d


def merge_chunk(
    target: o3d.geometry.PointCloud,
    chunk: list[o3d.geometry.PointCloud],
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Merge a bounded chunk into `target` and immediately downsample it.

    `chunk` is cleared in-place once merged (callers reuse the same list
    object across iterations). If `chunk` is empty, `target` is returned
    unchanged. This function uses Open3D's `+=`/`+` directly, which is
    safe here because callers only ever put clouds of uniformly matching
    attributes (all with normals or all without, none with per-point
    color) into a chunk -- see this module's docstring for why colored
    pipelines must NOT use this helper.
    """
    if not chunk:
        return target

    local = o3d.geometry.PointCloud()
    for cloud in chunk:
        local += cloud
    chunk.clear()

    if len(target.points):
        target += local
    else:
        target = local

    return target.voxel_down_sample(voxel_size)
