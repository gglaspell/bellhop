#!/usr/bin/env python3
"""
reconstruction.py – Shared point-cloud cleaning, Poisson mesh reconstruction,
floor levelling, and ENU->ECEF georeferencing.
"""

import logging
import math
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Point-cloud cleaning
# ---------------------------------------------------------------------------
def clean_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    do_voxel_downsample: bool = True,
) -> o3d.geometry.PointCloud:
    """Voxel -> ROR -> SOR -> DBSCAN noise removal."""
    pcd_clean = pcd

    if do_voxel_downsample:
        logging.info("Voxel downsampling...")
        pcd_clean = pcd_clean.voxel_down_sample(voxel_size)

    logging.info("Radius outlier removal...")
    try:
        tmp, _ = pcd_clean.remove_radius_outlier(nb_points=12, radius=voxel_size * 3.0)
        if len(tmp.points) > 0:
            pcd_clean = tmp
        else:
            logging.warning("ROR produced empty cloud; skipping.")
    except Exception as e:
        logging.warning(f"ROR failed ({e}); skipping.")

    logging.info("Statistical outlier removal...")
    try:
        tmp, _ = pcd_clean.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(tmp.points) > 0:
            pcd_clean = tmp
        else:
            logging.warning("SOR produced empty cloud; skipping.")
    except Exception as e:
        logging.warning(f"SOR failed ({e}); skipping.")

    logging.info("DBSCAN noise removal...")
    try:
        labels = np.array(
            pcd_clean.cluster_dbscan(eps=voxel_size * 4.0, min_points=30, print_progress=False)
        )
        if len(labels) > 0 and labels.max() >= 0:
            tmp = pcd_clean.select_by_index(np.where(labels >= 0)[0])
            if len(tmp.points) > 0:
                pcd_clean = tmp
    except Exception as e:
        logging.warning(f"DBSCAN failed ({e}); skipping.")

    logging.info(f"Cleaned cloud: {len(pcd_clean.points):,} points.")
    return pcd_clean


# ---------------------------------------------------------------------------
# Poisson mesh reconstruction
# ---------------------------------------------------------------------------
def create_mesh(
    pcd: o3d.geometry.PointCloud,
    poisson_depth: int = 9,
    min_density_percentile: float = 1.0,
    max_vertex_distance: float = 0.15,
    workers: int = 4,
    decimate_target: float | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Poisson reconstruction + density trim + distance trim + cleanup.

    Args:
        min_density_percentile: Remove bottom N% of vertices by Poisson density.
                                Default 1.0 = remove bottom 1%.
        max_vertex_distance:    Remove vertices > N metres from any input point.
        decimate_target:        <=1.0 = fraction to keep; >1 = absolute count; None = skip.
    """
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(100)
        except Exception:
            pass

    logging.info(f"Poisson reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(poisson_depth), linear_fit=True
    )
    logging.info(f"Initial mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces")

    # Density trim
    densities = np.asarray(densities, dtype=np.float64)
    if densities.size > 0:
        thr = np.percentile(densities, float(min_density_percentile))
        logging.info(f"Density trim: bottom {min_density_percentile}% (threshold={thr:.4f})")
        mesh.remove_vertices_by_mask(densities < thr)
    if len(mesh.triangles) == 0:
        raise ValueError("Mesh empty after density trim. Try a lower --min_density_percentile.")

    # Distance trim
    pts   = np.asarray(pcd.points)
    verts = np.asarray(mesh.vertices)
    if len(pts) > 0 and len(verts) > 0 and max_vertex_distance and float(max_vertex_distance) > 0:
        logging.info(f"Distance trim: removing vertices > {max_vertex_distance} m...")
        tree = cKDTree(pts)
        d, _ = tree.query(verts, k=1, workers=workers)
        mesh.remove_vertices_by_mask(d > float(max_vertex_distance))
    if len(mesh.triangles) == 0:
        raise ValueError("Mesh empty after distance trim. Try a larger --max_vertex_distance.")

    # Cleanup
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    # Optional decimation
    if decimate_target is not None:
        n_before = len(mesh.triangles)
        target_tris = (
            max(1, int(n_before * decimate_target))
            if decimate_target <= 1.0
            else int(decimate_target)
        )
        logging.info(f"Decimating: {n_before:,} -> ~{target_tris:,} triangles...")
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_tris, boundary_weight=10.0
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        logging.info(f"Decimated: {len(mesh.triangles):,} triangles")

    logging.info(f"Final mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces")
    return mesh


# ---------------------------------------------------------------------------
# Floor levelling
# ---------------------------------------------------------------------------
def level_floor(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """RANSAC plane detection; rotate cloud so detected floor aligns with Z=0."""
    try:
        _, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        if len(inliers) < 10:
            logging.warning("Floor levelling: too few inliers; skipping.")
            return pcd

        plane_pts = np.asarray(pcd.select_by_index(inliers).points, dtype=np.float64)
        centroid  = plane_pts.mean(axis=0)
        _, _, Vt  = np.linalg.svd(plane_pts - centroid)
        normal    = Vt[-1]
        if normal[2] < 0:
            normal = -normal

        target_n = np.array([0.0, 0.0, 1.0])
        v    = np.cross(normal, target_n)
        s    = np.linalg.norm(v)
        cang = float(np.dot(normal, target_n))

        if s > 1e-12:
            vx = np.array(
                [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
                dtype=np.float64,
            )
            R3  = np.eye(3) + vx + vx @ vx * ((1.0 - cang) / (s * s))
            pts = np.asarray(pcd.points, dtype=np.float64) @ R3.T
            pcd.points = o3d.utility.Vector3dVector(pts)
            if pcd.has_normals():
                nr = np.asarray(pcd.normals, dtype=np.float64) @ R3.T
                pcd.normals = o3d.utility.Vector3dVector(nr)
            logging.info("Floor levelling applied.")
        else:
            logging.info("Map already level; skipping rotation.")
    except Exception as e:
        logging.warning(f"Floor levelling failed: {e}")
    return pcd


# ---------------------------------------------------------------------------
# Georeferencing: local ENU -> ECEF (EPSG:4978)
# ---------------------------------------------------------------------------
def _enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [[-so,       co,       0.0],
         [-sl * co, -sl * so,  cl ],
         [ cl * co,  cl * so,  sl ]],
        dtype=np.float64,
    )


def transform_local_enu_to_ecef(
    pts_enu: np.ndarray, lat0_deg: float, lon0_deg: float, alt0_m: float
) -> np.ndarray:
    """Convert (N, 3) local-ENU metres to ECEF XYZ float64."""
    from pyproj import CRS, Transformer
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(4978), always_xy=True
    )
    ox, oy, oz = transformer.transform(lon0_deg, lat0_deg, alt0_m)
    ecef_origin = np.array([ox, oy, oz], dtype=np.float64)
    R_mat = _enu_to_ecef_rotation(lat0_deg, lon0_deg)
    return pts_enu @ R_mat.T + ecef_origin
