"""
colormesh3d.atlas_pipeline.viewassignment
============================================
Per-face view assignment for the atlas-bake pipeline.
"""

from collections import Counter
from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from tqdm import tqdm

from colormesh3d.common.trajectory import load_trajectory, build_trajectory_tree, get_pose_at
from colormesh3d.common.projection import world_to_optical, project_to_pixels


def _smooth_assignments(mesh, assignments, iterations=3):
    """Majority-vote smoothing over face adjacency to reduce per-face keyframe fragmentation."""
    adj = mesh.face_adjacency
    n_faces = len(assignments)

    neighbours = [[] for _ in range(n_faces)]
    for a, b in adj:
        neighbours[a].append(b)
        neighbours[b].append(a)

    for _ in range(iterations):
        new_assignments = assignments.copy()
        for face_i in range(n_faces):
            nbrs = neighbours[face_i]
            if not nbrs:
                continue
            assigned_nbrs = [assignments[j] for j in nbrs if assignments[j] != -1]
            if not assigned_nbrs:
                continue
            vote = Counter(assigned_nbrs)
            majority_kf, majority_count = vote.most_common(1)[0]
            if majority_count > len(nbrs) / 2:
                new_assignments[face_i] = majority_kf
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

        for kf_idx, (cp, cr) in enumerate(tqdm(kf_poses)):
            opt = world_to_optical(centers, cp, cr)
            mask = opt[:, 2] > 0.1
            if not np.any(mask):
                continue

            # normals are direction vectors; cam_pos=zeros is correct here
            n_opt = world_to_optical(normals, np.zeros(3), cr)

            view_dir = -opt
            dist = np.linalg.norm(view_dir, axis=1)
            view_dir /= (dist[:, None] + 1e-9)

            align = np.sum(n_opt * view_dir, axis=1)
            mask &= (align > self.min_cos)
            mask &= (dist ** 2 < self.max_bake_dist_sq)
            mask &= (dist ** 2 > self.min_bake_dist_sq)

            u, v, _ = project_to_pixels(opt, fx, fy, cx, cy, min_z=0.1)
            mask &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                continue

            z = opt[:, 2]
            u_int = np.clip(np.floor(u[valid_indices]).astype(int), 0, w - 1)
            v_int = np.clip(np.floor(v[valid_indices]).astype(int), 0, h - 1)
            depths = z[valid_indices]

            sort_order = np.argsort(depths)
            flat_idx = v_int[sort_order] * w + u_int[sort_order]
            _, unique_idx = np.unique(flat_idx, return_index=True)

            z_buffer = np.full(w * h, np.inf)
            z_buffer[flat_idx[unique_idx]] = depths[sort_order][unique_idx]

            is_visible = depths < (z_buffer[v_int * w + u_int] + 0.02)
            final_indices = valid_indices[is_visible]
            if len(final_indices) == 0:
                continue

            coverage[final_indices] += 1

            f_align = align[final_indices]
            f_dist_sq = dist[final_indices] ** 2
            f_u, f_v = u[final_indices], v[final_indices]
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
                self.mesh, assignments, self.assignment_smooth_iterations
            )

        self._export_coverage_diagnostic(coverage)

        unassigned = np.where(assignments == -1)[0]
        if len(unassigned) > 0:
            cam_pos = np.array([p[0] for p in kf_poses])
            for i in unassigned:
                assignments[i] = np.argmin(np.sum((centers[i] - cam_pos) ** 2, axis=1))

        return assignments
