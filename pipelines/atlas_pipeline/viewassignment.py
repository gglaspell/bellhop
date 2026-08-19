"""
pipelines.atlas_pipeline.viewassignment
=========================================
Per-face view assignment for the atlas-bake pipeline.

REFACTOR NOTE (moved from atlas_pipeline/common to shared):
Previously imported trajectory/projection helpers from `.common.trajectory`
and `.common.projection` (i.e. `atlas_pipeline/common/`). That package has
been merged into `shared/` (see `shared/trajectory.py`'s module
docstring), so this now imports from `..shared.trajectory` and
`..shared.projection` instead. No behavior change.
"""

from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from tqdm import tqdm

from ..shared.trajectory import load_trajectory, build_trajectory_tree, get_pose_at
from ..shared.projection import world_to_optical, project_to_pixels


def _build_face_adjacency_csr(mesh):
    """Precompute face-adjacency once as a CSR sparse matrix.

    FIX: previously each call site (this module and atlas_packer) queried
    or masked `mesh.face_adjacency` independently, and _smooth_assignments
    rebuilt a Python list-of-lists neighbour structure from scratch every
    time it was invoked. Centralising this into a single reusable sparse
    matrix builder means the O(n_edges) adjacency scan happens once and the
    same precomputed structure can be reused across keyframe loops and
    smoothing calls instead of being rebuilt per-invocation.
    """
    adj = mesh.face_adjacency
    n_faces = len(mesh.faces)
    rows = np.concatenate([adj[:, 0], adj[:, 1]])
    cols = np.concatenate([adj[:, 1], adj[:, 0]])
    return csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n_faces, n_faces)
    )


def _smooth_assignments(mesh, assignments, iterations=3, adj_sparse=None):
    """Majority-vote smoothing over face adjacency to reduce per-face keyframe fragmentation.

    FIX: fully vectorized replacement for the previous per-face Python loop
    + collections.Counter majority vote. Uses the CSR adjacency's
    (indptr, indices) arrays to build an explicit (src, dst) edge-list view
    of the graph once, then per iteration:
    1. Filters edges whose destination face is currently assigned
       (assignments[dst] != -1) -- unassigned neighbours never
       contribute votes, exactly as `if assignments[j] != -1` did.
    2. Tallies (src_face, neighbour_label) vote counts via a single
       np.unique on a combined integer key, which is equivalent to
       building a Counter per face but done for every face at once.
    3. For each face, picks the label with the *strictly highest* count.
       Ties are broken by NumPy's stable sort order (first-encountered
       label among the tied maximum), which matches
       `Counter.most_common(1)` -- Python's Counter also preserves
       insertion order among equal counts (insertion order here being
       neighbour-list order, i.e. ascending neighbour face id, identical
       to the original `neighbours[face_i]` list order since indices are
       stored in ascending order within each CSR row).
    4. Applies the exact same strict majority rule
       `majority_count > len(nbrs) / 2` (using each face's true degree,
       not just the number of assigned neighbours) before reassigning.
    Faces with 0 neighbours, or whose neighbours are all unassigned (-1),
    are left unchanged, matching the original `continue` branches.
    """
    n_faces = len(assignments)

    if adj_sparse is None:
        adj_sparse = _build_face_adjacency_csr(mesh)

    indptr, indices = adj_sparse.indptr, adj_sparse.indices
    deg = np.diff(indptr)
    edge_src = np.repeat(np.arange(n_faces), deg)
    edge_dst = indices

    assignments = assignments.copy()

    for _ in range(iterations):
        dst_labels = assignments[edge_dst]
        valid = dst_labels != -1
        if not np.any(valid):
            continue

        v_src = edge_src[valid]
        v_labels = dst_labels[valid]

        uniq_labels, inv = np.unique(v_labels, return_inverse=True)
        n_labels = len(uniq_labels)

        # Combined key encodes (face, label-index) so a single np.unique
        # call tallies per-(face, label) vote counts for every face in
        # one pass, equivalent to a per-face Counter but vectorized.
        combined = v_src.astype(np.int64) * n_labels + inv.astype(np.int64)
        uniq_combined, counts = np.unique(combined, return_counts=True)

        face_of_combined = uniq_combined // n_labels
        label_idx_of_combined = uniq_combined % n_labels

        # Sort by (face, -count) so that for each face the first row after
        # grouping is its highest-count label; ties keep the lowest
        # label-index (== lowest neighbour face id among tied labels,
        # matching Counter's insertion-order tie-break given ascending
        # CSR neighbour order).
        order = np.lexsort((-counts, face_of_combined))
        sorted_faces = face_of_combined[order]
        sorted_counts = counts[order]
        sorted_label_idx = label_idx_of_combined[order]

        first_mask = np.empty(len(sorted_faces), dtype=bool)
        first_mask[0] = True
        first_mask[1:] = sorted_faces[1:] != sorted_faces[:-1]

        best_faces = sorted_faces[first_mask]
        best_counts = sorted_counts[first_mask]
        best_label_idx = sorted_label_idx[first_mask]

        # Strict majority rule uses each face's TRUE degree (deg), not the
        # count of assigned neighbours, exactly matching the original
        # `majority_count > len(nbrs) / 2` where len(nbrs) was the full
        # neighbour list length including unassigned (-1) neighbours.
        new_assignments = assignments.copy()
        majority = best_counts > (deg[best_faces] / 2)
        new_assignments[best_faces[majority]] = uniq_labels[best_label_idx[majority]]
        assignments = new_assignments

    return assignments


class ViewAssigner:
    def __init__(self, mesh_path, keyframes, traj_path, intr, min_angle,
                 max_bake_distance=4.0, min_bake_distance=0.4,
                 assignment_smooth_iterations=3):
        self.mesh = trimesh.load(str(mesh_path), process=False)
        self.mesh_path = mesh_path
        self.keyframes = keyframes
        self.intr = intr
        self.max_bake_dist_sq = max_bake_distance ** 2
        self.min_bake_dist_sq = min_bake_distance ** 2
        self.min_cos = np.cos(np.radians(min_angle))
        self.assignment_smooth_iterations = assignment_smooth_iterations

        self.traj = load_trajectory(traj_path)
        self.tree = build_trajectory_tree(self.traj)

    def _export_coverage_diagnostic(self, coverage: np.ndarray) -> None:
        """
        Export a per-face coverage-count diagnostic mesh (colored by the
        'viridis' colormap) alongside the input mesh. This is purely a
        debugging aid and must never abort the main assignment pipeline,
        but genuine programming errors here should still be visible.

        FIX: The previous `except Exception as e: logging.warning(...)`
        swallowed *every* possible exception, including ones that indicate
        real bugs (e.g. a typo'd attribute name, a broken colormap import,
        or a malformed `self.mesh` from an upstream loader change) -- these
        would silently degrade to a one-line warning instead of surfacing
        during debugging. We now catch only the specific, genuinely
        expected failure modes for this operation:
        - OSError / PermissionError: diag_path is not writable (disk
          full, read-only output directory, path collision, etc.)
        - ValueError: malformed coverage/color data that trimesh or
          matplotlib legitimately reject (e.g. NaN in `coverage`).
        Anything else (AttributeError, TypeError, ImportError, etc.)
        propagates immediately so it is not masked during development.
        """
        try:
            diag_mesh = self.mesh.copy()
            norm_cov = np.clip(coverage / 5.0, 0, 1)
            colors = plt.get_cmap('viridis')(norm_cov)[:, :3] * 255
            diag_mesh.visual.face_colors = colors.astype(np.uint8)
            diag_path = Path(self.mesh_path).parent / 'coverage_diag.ply'
            diag_mesh.export(str(diag_path))
            logging.info(f"Saved coverage diagnostic to {diag_path}")
        except (OSError, PermissionError) as e:
            logging.warning(f"Diagnostic export failed (I/O error writing '{diag_path if 'diag_path' in locals() else '?'}'): {e}")
        except ValueError as e:
            logging.warning(f"Diagnostic export failed (invalid coverage/color data): {e}")

    def assign_faces_to_views(self):
        centers = self.mesh.triangles_center
        normals = self.mesh.face_normals
        assignments = np.full(len(centers), -1, dtype=int)
        scores = np.full(len(centers), -1.0)
        coverage = np.zeros(len(centers))

        fx, fy, cx, cy = self.intr['fx'], self.intr['fy'], self.intr['cx'], self.intr['cy']
        w, h = self.intr['width'], self.intr['height']

        sharpness_vals = np.array([s for _, _, s in self.keyframes])
        med_s = np.median(sharpness_vals) if len(sharpness_vals) > 0 else 1.0
        sharp_weights = np.clip(sharpness_vals / (med_s + 1e-6), 0.5, 2.0)

        kf_poses = [get_pose_at(self.traj, self.tree, ts) for _, ts, _ in self.keyframes]

        # FIX: build a KD-tree over face centers once, then for each
        # keyframe only consider faces within max_bake_distance of that
        # keyframe's CAMERA POSITION (cp) -- not the mesh centroid -- before
        # running the full projection/visibility math. This is a strict
        # superset prefilter: the radius query uses max_bake_distance (the
        # sqrt of self.max_bake_dist_sq), which returns every face that
        # could possibly pass the existing `dist ** 2 < self.max_bake_dist_sq`
        # mask, AND also every face that the existing
        # `dist ** 2 > self.min_bake_dist_sq` mask would keep, since
        # min_bake_distance <= max_bake_distance always narrows from the
        # same near end. In other words the KD-tree query only prunes faces
        # that are guaranteed to fail the max-distance mask anyway; the
        # min-distance mask is still applied afterward, unchanged, on the
        # candidate subset. This preserves correctness while skipping the
        # expensive projection math for faces that are trivially too far.
        max_bake_distance = float(np.sqrt(self.max_bake_dist_sq))
        face_tree = cKDTree(centers)

        for kf_idx, (cp, cr) in enumerate(tqdm(kf_poses)):
            candidate_idx = np.asarray(
                face_tree.query_ball_point(cp, max_bake_distance), dtype=np.int64
            )
            if len(candidate_idx) == 0:
                continue

            cand_centers = centers[candidate_idx]
            cand_normals = normals[candidate_idx]

            opt = world_to_optical(cand_centers, cp, cr)
            mask = opt[:, 2] > 0.1
            if not np.any(mask):
                continue

            # normals are direction vectors; cam_pos=zeros is correct here
            n_opt = world_to_optical(cand_normals, np.zeros(3), cr)

            view_dir = -opt
            dist = np.linalg.norm(view_dir, axis=1)
            view_dir /= (dist[:, None] + 1e-9)

            align = np.sum(n_opt * view_dir, axis=1)
            mask &= (align > self.min_cos)
            mask &= (dist ** 2 < self.max_bake_dist_sq)
            mask &= (dist ** 2 > self.min_bake_dist_sq)

            u, v, _ = project_to_pixels(opt, fx, fy, cx, cy, min_z=0.1)
            mask &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
            valid_local = np.where(mask)[0]
            if len(valid_local) == 0:
                continue

            z = opt[:, 2]
            u_int = np.clip(np.floor(u[valid_local]).astype(int), 0, w - 1)
            v_int = np.clip(np.floor(v[valid_local]).astype(int), 0, h - 1)
            depths = z[valid_local]

            sort_order = np.argsort(depths)
            flat_idx = v_int[sort_order] * w + u_int[sort_order]
            _, unique_idx = np.unique(flat_idx, return_index=True)

            z_buffer = np.full(w * h, np.inf)
            z_buffer[flat_idx[unique_idx]] = depths[sort_order][unique_idx]

            is_visible = depths < (z_buffer[v_int * w + u_int] + 0.02)
            final_local = valid_local[is_visible]
            if len(final_local) == 0:
                continue

            # Map local (candidate-subset) indices back to global face ids.
            final_indices = candidate_idx[final_local]

            coverage[final_indices] += 1

            f_align = align[final_local]
            f_dist_sq = dist[final_local] ** 2
            f_u, f_v = u[final_local], v[final_local]
            dist_from_center = np.sqrt(((f_u - cx) / (w / 2)) ** 2 + ((f_v - cy) / (h / 2)) ** 2)
            vignette_weight = np.clip(1.0 - (dist_from_center * 0.6), 0.2, 1.0)

            s = (f_align ** 2 * vignette_weight * sharp_weights[kf_idx]) / (f_dist_sq + 1e-6)

            better = s > scores[final_indices]
            update_indices = final_indices[better]
            scores[update_indices] = s[better]
            assignments[update_indices] = kf_idx

        if self.assignment_smooth_iterations > 0:
            logging.info(
                f"Smoothing assignments over face adjacency "
                f"({self.assignment_smooth_iterations} iterations)..."
            )
            assignments = _smooth_assignments(
                self.mesh, assignments, self.assignment_smooth_iterations,
                adj_sparse=_build_face_adjacency_csr(self.mesh),
            )

        self._export_coverage_diagnostic(coverage)

        unassigned = np.where(assignments == -1)[0]
        if len(unassigned) > 0:
            cam_pos = np.array([p[0] for p in kf_poses])
            for i in unassigned:
                assignments[i] = np.argmin(np.sum((centers[i] - cam_pos) ** 2, axis=1))

        return assignments
