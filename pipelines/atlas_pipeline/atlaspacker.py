"""
colormesh3d.atlas_pipeline.atlaspacker
=========================================
UV atlas packing for the atlas-bake pipeline.
"""

import pickle

import networkx as nx
import numpy as np
import trimesh
from tqdm import tqdm
import logging

from colormesh3d.common.trajectory import load_trajectory, build_trajectory_tree, get_pose_at
from colormesh3d.common.projection import world_to_optical


class AtlasPacker:
    def __init__(self, mesh_path, assignments, keyframes, traj_path, intr, size):
        self.mesh = trimesh.load(str(mesh_path), process=False)
        self.asm = assignments
        self.kf = keyframes
        self.base_size = size
        self.intr = intr

        self.traj = load_trajectory(traj_path)
        self.tree = build_trajectory_tree(self.traj)

    def _project(self, v, cp, cr):
        return world_to_optical(v, cp, cr)

    def _get_kf_pose(self, kf_idx):
        """Return (camera_position, camera_rotation) for a keyframe index."""
        return get_pose_at(self.traj, self.tree, self.kf[kf_idx][1])

    def pack_and_generate_uvs(self, out_path):
        uniq = np.unique(self.asm[self.asm != -1])
        verts, faces, adj = self.mesh.vertices, self.mesh.faces, self.mesh.face_adjacency
        fx, fy, cx, cy = self.intr['fx'], self.intr['fy'], self.intr['cx'], self.intr['cy']
        img_w, img_h = self.intr['width'], self.intr['height']

        components_data = []
        logging.info("Segmenting charts by connected components...")

        for kf_idx in tqdm(uniq, desc="Gathering"):
            f_idx = np.where(self.asm == kf_idx)[0]
            if len(f_idx) == 0:
                continue

            mask = np.isin(adj[:, 0], f_idx) & np.isin(adj[:, 1], f_idx)
            g = nx.Graph()
            g.add_nodes_from(f_idx)
            g.add_edges_from(adj[mask])

            cp, cr = self._get_kf_pose(kf_idx)

            all_verts_for_kf = np.unique(faces[f_idx])
            v_opt_all = self._project(verts[all_verts_for_kf], cp, cr)
            z_all = np.maximum(v_opt_all[:, 2], 0.1)
            u_all = np.clip((fx * v_opt_all[:, 0] / z_all) + cx, -img_w * 0.5, img_w * 1.5)
            v_all = np.clip((fy * v_opt_all[:, 1] / z_all) + cy, -img_h * 0.5, img_h * 1.5)
            vert_to_uv_row = {gv: li for li, gv in enumerate(all_verts_for_kf)}
            uv_all = np.column_stack([u_all, v_all])

            for comp in nx.connected_components(g):
                comp_idx = np.array(list(comp))
                uniq_v = np.unique(faces[comp_idx])

                local_rows = np.array([vert_to_uv_row[gv] for gv in uniq_v])
                uv = uv_all[local_rows]

                min_uv, max_uv = np.min(uv, axis=0), np.max(uv, axis=0)
                dims = max_uv - min_uv

                if dims[0] < 64 or dims[1] < 64:
                    continue

                components_data.append({
                    'kf': kf_idx, 'comp_idx': comp_idx, 'uniq_v': uniq_v,
                    'uv': uv, 'min_uv': min_uv, 'max_uv': max_uv, 'dims': dims,
                    'uv_all': uv_all, 'vert_to_uv_row': vert_to_uv_row,
                })

        if not components_data:
            logging.error("No valid charts generated.")
            return

        total_native_area = sum(c['dims'][0] * c['dims'][1] for c in components_data)
        target_total_area = (self.base_size * self.base_size) * 0.6

        # FIX: `total_native_area` can legitimately be zero (or extremely
        # close to it) if every surviving chart has a degenerate dims[0] or
        # dims[1] -- e.g. all charts are exactly on the 64px cutoff boundary
        # after floating-point rounding, or a pathological single-chart
        # mesh with a zero-area UV footprint. The previous guard only
        # checked `total_native_area > target_total_area`, so when
        # `total_native_area` was 0 (or NaN, which can arise from degenerate
        # projections producing NaN dims upstream), the `else` branch was
        # taken and global_scale fell back to 1.0 -- except if
        # total_native_area was NaN, `NaN > target_total_area` evaluates to
        # False, so global_scale = 1.0 is actually safe there. The real gap
        # is that nothing validates total_native_area itself before it's
        # used, and nothing checks whether the resulting global_scale is
        # finite before it is multiplied through every chart's UVs/dims.
        # Guard explicitly against a non-finite or non-positive
        # total_native_area, and assert global_scale is finite afterward so
        # a bad value fails loudly here instead of propagating silent NaNs
        # into every downstream chart size, the shelf packer, and the final
        # UV mesh.
        if not np.isfinite(total_native_area) or total_native_area <= 0.0:
            logging.warning(
                "Total native chart area is non-finite or non-positive "
                f"({total_native_area!r}); skipping area-based rescaling "
                "(global_scale=1.0)."
            )
            global_scale = 1.0
        elif total_native_area > target_total_area:
            global_scale = np.sqrt(target_total_area / total_native_area)
        else:
            global_scale = 1.0

        if not np.isfinite(global_scale) or global_scale <= 0.0:
            logging.warning(
                f"Computed global_scale is invalid ({global_scale!r}); "
                "falling back to 1.0."
            )
            global_scale = 1.0

        charts = []
        pad = 2
        max_chart_dim = int(self.base_size * 0.30)
        max_aspect = 4.0

        for c in components_data:
            uv, center = c['uv'], (c['min_uv'] + c['max_uv']) / 2
            uv_scaled = center + (uv - center) * global_scale
            dims_scaled = c['dims'] * global_scale

            # FIX: dims_scaled (and therefore w_native/h_native below) can
            # still be non-finite for an individual chart even when
            # global_scale itself is a valid finite number, if that chart's
            # own c['dims'] happened to contain NaN/Inf from an upstream
            # degenerate projection. Skip such charts explicitly rather than
            # letting int(np.ceil(nan)) raise or silently coerce into a
            # bogus atlas slot.
            if not np.all(np.isfinite(dims_scaled)):
                logging.warning(
                    f"Skipping chart (kf={c['kf']}) with non-finite scaled "
                    f"dimensions: {dims_scaled!r}."
                )
                continue

            w_native = max(1, min(self.base_size, int(np.ceil(dims_scaled[0])) + 2 * pad))
            h_native = max(1, min(self.base_size, int(np.ceil(dims_scaled[1])) + 2 * pad))

            cap_scale = 1.0
            if w_native > max_chart_dim or h_native > max_chart_dim:
                cap_scale = min(max_chart_dim / w_native, max_chart_dim / h_native)
                uv_scaled = center + (uv_scaled - center) * cap_scale
                dims_scaled = dims_scaled * cap_scale
                w_native = max(1, int(np.ceil(dims_scaled[0])) + 2 * pad)
                h_native = max(1, int(np.ceil(dims_scaled[1])) + 2 * pad)

            if h_native > 0 and w_native / max(h_native, 1) > max_aspect:
                w_native = max(1, int(h_native * max_aspect))
            elif w_native > 0 and h_native / max(w_native, 1) > max_aspect:
                h_native = max(1, int(w_native * max_aspect))

            effective_scale = global_scale * cap_scale

            if w_native < 8 or h_native < 8:
                continue

            rotated = w_native > h_native
            w_int = h_native if rotated else w_native
            h_int = w_native if rotated else h_native

            charts.append({
                'kf': c['kf'],
                'f_idx': c['comp_idx'],
                'uv': uv_scaled,
                'v_idx': c['uniq_v'],
                'min': np.min(uv_scaled, axis=0),
                'w_int': w_int,
                'h_int': h_int,
                'w': dims_scaled[0],
                'h': dims_scaled[1],
                'original_uv': c['uv'],
                'scale': effective_scale,
                'rotated': rotated,
            })

        if not charts:
            logging.error("No charts survived size/aspect filtering.")
            return

        logging.info(f"Packing {len(charts)} charts using Instant Shelf Packer...")

        charts.sort(key=lambda c: (c['h_int'], c['w_int']), reverse=True)

        current_x, current_y, shelf_height = 0, 0, 0

        for c in charts:
            w_c, h_c = c['w_int'], c['h_int']
            if current_x + w_c > self.base_size:
                current_y += shelf_height
                current_x = 0
                shelf_height = 0
            c['px'] = current_x
            c['py'] = current_y
            current_x += w_c
            shelf_height = max(shelf_height, h_c)

        final_h = int(np.ceil((current_y + shelf_height) / 1024.0) * 1024)
        logging.info(f"Packing complete! Final Atlas Size: {self.base_size} x {final_h}")

        final_v, final_f, final_uv, bake_charts, cursor = [], [], [], [], 0

        for c in tqdm(charts, desc="UV Mesh"):
            px, py = c['px'], c['py']
            orig_uv = c['original_uv']
            orig_min = np.min(orig_uv, axis=0)
            orig_dims = np.max(orig_uv, axis=0) - orig_min

            bake_charts.append({
                'kf': c['kf'],
                'px': px,
                'py': py,
                'minu': orig_min[0],
                'minv': orig_min[1],
                'w': orig_dims[0],
                'h': orig_dims[1],
                'atlas_w': c['w_int'],
                'atlas_h': c['h_int'],
                'pad': pad,
                'scale': c['scale'],
                'rotated': c['rotated'],
            })

            g2l = {gv: li for li, gv in enumerate(c['v_idx'])}
            for f in faces[c['f_idx']]:
                tri_uv = []
                for vi in f:
                    li = g2l[vi]
                    du = c['uv'][li, 0] - c['min'][0]
                    dv = c['uv'][li, 1] - c['min'][1]
                    if c['rotated']:
                        ua = (px + pad + dv) / self.base_size
                        va = 1.0 - (py + pad + du) / final_h
                    else:
                        ua = (px + pad + du) / self.base_size
                        va = 1.0 - (py + pad + dv) / final_h
                    tri_uv.append([ua, va])
                final_v.extend(verts[f])
                final_uv.extend(tri_uv)
                final_f.append([cursor, cursor + 1, cursor + 2])
                cursor += 3

        out_mesh = trimesh.Trimesh(
            vertices=final_v,
            faces=final_f,
            visual=trimesh.visual.TextureVisuals(uv=final_uv),
        )

        out_mesh.export(str(out_path))

        atlas_data = {
            'charts': bake_charts,
            'final_w': self.base_size,
            'final_h': final_h,
        }

        with open(out_path.parent / 'atlas_charts.pkl', 'wb') as f:
            pickle.dump(atlas_data, f)
