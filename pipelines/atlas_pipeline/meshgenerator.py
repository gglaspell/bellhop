"""
pipelines.atlas_pipeline.meshgenerator
=======================================
Mesh generation, smoothing, and culling utilities for the atlas-bake pipeline.

REFACTOR NOTE (moved from atlas_pipeline/common to shared):
Previously imported trajectory/projection helpers from `.common.trajectory`
and `.common.projection` (i.e. `atlas_pipeline/common/`). That package has
been merged into `shared/` (see `shared/trajectory.py`'s module
docstring), so this now imports from `..shared.trajectory` and
`..shared.projection` instead. No behavior change.

PATCH NOTE (isotropic remeshing before UV/atlas work):
Added `remesh_isotropic()`, mirroring `shared/reconstruction.py`'s
`remesh_isotropic()` (used by `mesh`/`color_mesh`/`gazebo_world`) but kept
in this module's file-in/file-out style to match `poisson_reconstruct()`
and `smooth_mesh()`.

Raw Poisson reconstruction (and Taubin/Laplacian smoothing, which only
relaxes vertex positions) tends to leave irregular, sometimes sliver-thin
triangles, especially near low-density regions and mesh boundaries.
Irregular triangle shape/size is a direct cause of UV-unwrap distortion,
seams, and blurry/stretched baked textures -- exactly the failure mode
this pipeline's final output (a UV-textured, baked mesh) is most
sensitive to. Regularizing triangle topology here, after smoothing but
before `cull_invisible_faces()` / view assignment / UV atlas packing,
improves downstream bake quality without disturbing the existing
Poisson -> smooth -> cull -> assign -> pack -> bake ordering.

BUGFIX (unguarded trimesh.repair.fix_normals crash):
`cull_invisible_faces()` already wraps `trimesh.repair.fill_holes()` in a
try/except that logs a warning on failure -- but the very next call,
`trimesh.repair.fix_normals()` (invoked three times in this function),
had no equivalent guard. Both functions depend on `networkx` internally
(`fix_normals()` -> `fix_winding()` -> `networkx.from_edgelist()`), so an
environment missing `networkx` hit BOTH failures back to back: a
harmless logged warning from `fill_holes()`, immediately followed by an
unhandled `ModuleNotFoundError` from `fix_normals()` that crashed the
entire pipeline with exit code 1. `networkx` is now declared directly in
requirements.txt so this shouldn't happen in a correctly built image --
but every `fix_normals()` call here is now wrapped the same way
`fill_holes()` already was, so a missing/broken `networkx` (or any other
trimesh-internal repair failure) degrades to an unoriented-normals
warning instead of aborting the run. Normals already computed upstream
(from Poisson reconstruction / Open3D) are kept as-is if this step can't
run; culling and export continue regardless.
"""

import logging

import numpy as np
import open3d as o3d
import trimesh
import trimesh.repair
from scipy.spatial import cKDTree
from tqdm import tqdm

from ..shared.trajectory import load_trajectory, build_trajectory_tree, get_pose_at
from ..shared.projection import world_to_optical, project_to_pixels


def _safe_fix_normals(mesh: trimesh.Trimesh, context: str) -> None:
    """Attempt `trimesh.repair.fix_normals()`; never let it abort the run.

    See the BUGFIX note in this module's docstring: `fix_normals()` ->
    `fix_winding()` requires `networkx` internally, and unlike the
    `fill_holes()` call right before it in `cull_invisible_faces()`, it
    previously had no try/except at all. A missing/broken `networkx` (or
    any other trimesh-internal repair failure) now logs a warning and
    leaves the mesh's existing normals untouched, instead of raising an
    unhandled exception that kills the whole pipeline.
    """
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception as exc:
        logging.warning(f"fix_normals failed ({context}): {exc}")


class MeshGenerator:

    def poisson_reconstruct(self, in_pcd, out_mesh, depth=8, max_distance=0.5):
        logging.info(f"Running Poisson reconstruction (depth={depth}, max_distance={max_distance})...")
        pcd = o3d.io.read_point_cloud(str(in_pcd))

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

    def remesh_isotropic(self, in_mesh, out_mesh, smooth_iterations: int = 5) -> None:
        """
        Regularize the mesh into evenly-sized triangles via PyMeshLab's
        isotropic explicit remeshing, then apply a conservative Laplacian
        smoothing pass -- the same technique `shared/reconstruction.py`
        uses for `mesh`/`color_mesh`/`gazebo_world` behind `--remesh`.

        Falls back to copying the input mesh through unchanged if it's
        empty or if remeshing happens to strip it to nothing, so a bad
        remesh can never abort the run or silently produce an empty file.
        """
        try:
            import pymeshlab as ml
        except ImportError as exc:
            raise RuntimeError(
                "--remesh requires pymeshlab. Install Bellhop's updated requirements."
            ) from exc

        mesh = o3d.io.read_triangle_mesh(str(in_mesh))
        if len(mesh.vertices) == 0:
            logging.warning("remesh_isotropic: input mesh is empty -- skipping.")
            import shutil
            shutil.copy(str(in_mesh), str(out_mesh))
            return

        face_count_before = len(mesh.triangles)
        logging.info(
            f"Isotropic remeshing ({face_count_before} faces, "
            f"smooth_iterations={smooth_iterations})..."
        )

        mesh_set = ml.MeshSet()
        mesh_set.add_mesh(
            ml.Mesh(
                vertex_matrix=np.asarray(mesh.vertices),
                face_matrix=np.asarray(mesh.triangles),
            )
        )
        mesh_set.meshing_isotropic_explicit_remeshing(featuredeg=15.0, adaptive=True)
        if smooth_iterations > 0:
            mesh_set.apply_coord_laplacian_smoothing(
                stepsmoothnum=int(smooth_iterations), selected=False
            )

        current = mesh_set.current_mesh()
        result = o3d.geometry.TriangleMesh()
        result.vertices = o3d.utility.Vector3dVector(
            np.asarray(current.vertex_matrix(), dtype=np.float64)
        )
        result.triangles = o3d.utility.Vector3iVector(
            np.asarray(current.face_matrix(), dtype=np.int32)
        )
        result.remove_duplicated_vertices()
        result.remove_duplicated_triangles()
        result.remove_degenerate_triangles()
        result.remove_non_manifold_edges()
        result.remove_unreferenced_vertices()

        if not len(result.triangles):
            logging.warning(
                "remesh_isotropic produced an empty mesh; falling back to the "
                "pre-remesh input mesh."
            )
            import shutil
            shutil.copy(str(in_mesh), str(out_mesh))
            return

        result.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_mesh), result, write_vertex_normals=True)
        logging.info(
            f"Isotropic remesh: {face_count_before} -> {len(result.triangles)} faces."
        )

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
            view /= (dist[:, None] + 1e-9)

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
        # BUGFIX: was an unguarded `trimesh.repair.fix_normals(mesh)` call --
        # see this module's docstring. Now wrapped like `fill_holes()` above.
        _safe_fix_normals(mesh, context="post-fill_holes")
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
            _safe_fix_normals(mesh, context="post-decimation")
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()

        _safe_fix_normals(mesh, context="pre-export")
        mesh.export(str(out_mesh))
        logging.info(f"Culled mesh saved: {len(mesh.faces)} faces -> {out_mesh}")
