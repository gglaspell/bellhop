"""
colormesh3d.atlas_pipeline.meshgenerator
===========================================
Mesh generation, smoothing, and culling utilities for the atlas-bake pipeline.
"""

import logging

import numpy as np
import open3d as o3d
import trimesh
import trimesh.repair
from scipy.spatial import cKDTree
from tqdm import tqdm

from colormesh3d.common.trajectory import load_trajectory, build_trajectory_tree, get_pose_at
from colormesh3d.common.projection import world_to_optical, project_to_pixels


class MeshGenerator:

    def poisson_reconstruct(self, in_pcd, out_mesh, depth=8, max_distance=0.5):
        logging.info(f"Running Poisson reconstruction (depth={depth}, max_distance={max_distance})...")
        pcd = o3d.io.read_point_cloud(str(in_pcd))

        # FIX: added missing closing ) on create_from_point_cloud_poisson call
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )

        dens = np.asarray(dens)
        dens_threshold = np.quantile(dens, 0.01)
        mesh.remove_vertices_by_mask(dens < dens_threshold)

        vertices = np.asarray(mesh.vertices)
        tree = cKDTree(np.asarray(pcd.points))
        distances, _ = tree.query(vertices, k=1)

        close_to_points = distances < max_distance
        mesh.remove_vertices_by_mask(~close_to_points)

        o3d.io.write_triangle_mesh(str(out_mesh), mesh, write_vertex_normals=True)

    def smooth_mesh(
        self,
        in_mesh,
        out_mesh,
        iterations: int = 5,
        method: str = "taubin",
        lambda_filter: float = 0.5,
    ) -> None:
        logging.info(
            f"Smoothing mesh ({method}, iterations={iterations}, lambda={lambda_filter})..."
        )

        mesh = o3d.io.read_triangle_mesh(str(in_mesh))
        if len(mesh.vertices) == 0:
            logging.warning("smooth_mesh: input mesh is empty -- skipping.")
            import shutil
            shutil.copy(str(in_mesh), str(out_mesh))
            return

        mesh.compute_vertex_normals()

        if method == "taubin":
            smoothed = mesh.filter_smooth_taubin(
                number_of_iterations=iterations,
                lambda_filter=lambda_filter,
            )
        else:
            smoothed = mesh.filter_smooth_laplacian(
                number_of_iterations=iterations,
                lambda_filter=lambda_filter,
            )

        smoothed.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_mesh), smoothed, write_vertex_normals=True)
        logging.info(f"Smoothed mesh saved: {out_mesh}")

    def cull_invisible_faces(
        self,
        in_mesh,
        out_mesh,
        keyframes,
        traj_path,
        intr,
        min_angle,
        target_faces=None,
    ):
        mesh = trimesh.load(str(in_mesh), process=False)
        if mesh.face_normals is None:
            mesh.compute_face_normals()

        centers = mesh.triangles_center
        normals = mesh.face_normals

        df = load_trajectory(traj_path)
        tree = build_trajectory_tree(df)

        min_cos = np.cos(np.radians(min_angle))
        visible = np.zeros(len(centers), dtype=bool)

        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        img_w, img_h = intr["width"], intr["height"]

        for _, ts, _ in tqdm(keyframes, desc="Culling"):
            cp, cr = get_pose_at(df, tree, ts)

            opt = world_to_optical(centers, cp, cr)
            n_opt = world_to_optical(normals, np.zeros(3), cr)

            mask = opt[:, 2] > 0.1

            view = -opt
            dist = np.linalg.norm(view, axis=1)
            view /= dist[:, None] + 1e-9

            align = np.sum(n_opt * view, axis=1)
            mask &= align > min_cos

            u, v, _ = project_to_pixels(opt, fx, fy, cx, cy, min_z=0.0)

            mask &= (u >= -img_w * 0.5) & (u < img_w * 1.5)
            mask &= (v >= -img_h * 0.5) & (v < img_h * 1.5)

            visible |= mask

        mesh.update_faces(visible)

        logging.info("Filling small holes...")
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception as e:
            logging.warning(f"Hole filling failed: {e}")

        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_watertight:
            try:
                trimesh.repair.broken_faces(mesh)
            except Exception:
                pass
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()

        if target_faces and len(mesh.faces) > target_faces:
            logging.info(f"Simplifying mesh down to {target_faces} faces...")
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
            mesh = trimesh.Trimesh(
                vertices=np.asarray(o3d_mesh.vertices),
                faces=np.asarray(o3d_mesh.triangles),
                process=False,
            )
            trimesh.repair.fix_normals(mesh)
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()

        trimesh.repair.fix_normals(mesh)
        mesh.export(str(out_mesh))
        logging.info(f"Culled mesh saved: {len(mesh.faces)} faces -> {out_mesh}")
