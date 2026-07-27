#!/usr/bin/env python3
"""Shared point-cloud cleanup, adaptive Poisson reconstruction, and georeferencing."""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def clean_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float, do_voxel_downsample: bool = True) -> o3d.geometry.PointCloud:
    """Voxel -> radius outlier -> statistical outlier -> DBSCAN noise removal."""
    clean = pcd
    if do_voxel_downsample:
        clean = clean.voxel_down_sample(voxel_size)
    for name, operation in (
        ("ROR", lambda cloud: cloud.remove_radius_outlier(nb_points=12, radius=voxel_size * 3.0)[0]),
        ("SOR", lambda cloud: cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]),
    ):
        try:
            candidate = operation(clean)
            if len(candidate.points):
                clean = candidate
            else:
                logging.warning("%s produced an empty cloud; retaining the prior cloud.", name)
        except Exception as exc:
            logging.warning("%s failed: %s", name, exc)
    try:
        labels = np.asarray(clean.cluster_dbscan(eps=voxel_size * 4.0, min_points=30, print_progress=False))
        if labels.size and labels.max() >= 0:
            candidate = clean.select_by_index(np.flatnonzero(labels >= 0))
            if len(candidate.points):
                clean = candidate
    except Exception as exc:
        logging.warning("DBSCAN failed: %s", exc)
    logging.info("Cleaned cloud: %s points.", f"{len(clean.points):,}")
    return clean


def _average_point_spacing(pcd: o3d.geometry.PointCloud, sample_size: int = 20_000) -> float:
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) < 10:
        return 0.0
    sample = points if len(points) <= sample_size else points[np.linspace(0, len(points) - 1, sample_size, dtype=np.int64)]
    distances, _ = cKDTree(points).query(sample, k=2, workers=-1)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    return float(np.mean(nearest)) if nearest.size else 0.0


def suggest_poisson_depth(pcd: o3d.geometry.PointCloud, min_depth: int = 6, max_auto_depth: int = 11) -> int:
    """Select an automatic depth, never exceeding ``max_auto_depth`` (11 by default)."""
    spacing = _average_point_spacing(pcd)
    if spacing <= 0.0:
        logging.warning("Could not estimate point spacing; using automatic Poisson depth 9.")
        return min(9, max_auto_depth)
    bounds = pcd.get_axis_aligned_bounding_box()
    diagonal = float(np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound()))
    if diagonal <= 0.0:
        return min_depth
    depth = int(np.clip(np.rint(np.log2(diagonal / spacing)), min_depth, max_auto_depth))
    logging.info("Automatic Poisson depth=%d (extent=%.3fm, mean spacing=%.4fm; cap=%d).", depth, diagonal, spacing, max_auto_depth)
    return depth


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def _mesh_from_arrays(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    result.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    return result


def remesh_isotropic(mesh: o3d.geometry.TriangleMesh, smooth_iterations: int = 5) -> o3d.geometry.TriangleMesh:
    """Use PyMeshLab isotropic remeshing, then conservative pre-decimation smoothing."""
    try:
        import pymeshlab as ml
    except ImportError as exc:
        raise RuntimeError("--remesh requires pymeshlab. Install Bellhop's updated requirements.") from exc
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(mesh.vertices), face_matrix=np.asarray(mesh.triangles)))
    ms.meshing_isotropic_explicit_remeshing(featuredeg=15.0, adaptive=True)
    if smooth_iterations > 0:
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=int(smooth_iterations), selected=False)
    current = ms.current_mesh()
    result = _mesh_from_arrays(current.vertex_matrix(), current.face_matrix())
    result.compute_vertex_normals()
    return _cleanup_mesh(result)


def _vertex_curvature_proxy(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Mean incident normal deviation; high values identify edges and corners."""
    values = np.zeros(len(vertices), dtype=np.float64)
    counts = np.zeros(len(vertices), dtype=np.int32)
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    for face, normal in zip(faces[valid], normals[valid]):
        for vertex in face:
            values[vertex] += np.linalg.norm(normal)
            counts[vertex] += 1
    # Angular defect is expensive without topology; accumulated incident-face
    # normal variation below is a stable curvature proxy for protection.
    adjacency = [[] for _ in range(len(vertices))]
    for face_index, face in enumerate(faces[valid]):
        for vertex in face:
            adjacency[vertex].append(normals[face_index])
    for index, incident in enumerate(adjacency):
        if len(incident) > 1:
            unit = np.asarray(incident)
            mean = unit.mean(axis=0)
            norm = np.linalg.norm(mean)
            values[index] = 1.0 - (norm / len(unit) if norm else 0.0)
    return values


def _dilate_face_mask(faces: np.ndarray, mask: np.ndarray, rings: int) -> np.ndarray:
    if rings <= 0 or not mask.any():
        return mask
    by_vertex = [[] for _ in range(int(faces.max()) + 1)]
    for face_index, face in enumerate(faces):
        for vertex in face:
            by_vertex[vertex].append(face_index)
    expanded = mask.copy()
    for _ in range(rings):
        selected_vertices = np.unique(faces[expanded].ravel())
        for vertex in selected_vertices:
            expanded[by_vertex[vertex]] = True
    return expanded


def _submesh(vertices: np.ndarray, faces: np.ndarray, face_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = faces[face_mask]
    used, inverse = np.unique(selected.ravel(), return_inverse=True)
    return vertices[used], inverse.reshape((-1, 3)).astype(np.int32)


def simplify_curvature_aware(mesh: o3d.geometry.TriangleMesh, target_triangles: int, curvature_percentile: float = 80.0, protect_rings: int = 1) -> o3d.geometry.TriangleMesh:
    """Decimate low-curvature faces with pyfqmr while preserving edges and corners."""
    try:
        import pyfqmr
    except ImportError as exc:
        raise RuntimeError("Curvature-aware decimation requires pyfqmr.") from exc
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(faces) <= max(4, target_triangles):
        return mesh
    curvature = _vertex_curvature_proxy(vertices, faces)
    threshold = np.percentile(curvature, float(curvature_percentile))
    protected = (curvature[faces] >= threshold).any(axis=1)
    protected = _dilate_face_mask(faces, protected, int(protect_rings))
    protected_count = int(protected.sum())
    if protected_count >= target_triangles or (~protected).sum() < 4:
        logging.warning("Protected geometry already meets/exceeds target; skipping decimation.")
        return mesh
    low_vertices, low_faces = _submesh(vertices, faces, ~protected)
    low_target = max(4, min(len(low_faces), target_triangles - protected_count))
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(low_vertices, low_faces)
    simplifier.simplify_mesh(target_count=low_target, aggressiveness=7, verbose=0)
    simple_vertices, simple_faces, _ = simplifier.getMesh()
    keep_vertices, keep_faces = _submesh(vertices, faces, protected)
    offset = len(simple_vertices)
    result = _mesh_from_arrays(np.vstack((simple_vertices, keep_vertices)), np.vstack((simple_faces, keep_faces + offset)))
    result.compute_vertex_normals()
    logging.info("Curvature-aware decimation: %d -> %d faces (%d protected).", len(faces), len(result.triangles), protected_count)
    return _cleanup_mesh(result)


def create_mesh(pcd: o3d.geometry.PointCloud, poisson_depth: Optional[int] = None, min_density_percentile: float = 1.0, distance_multiplier: float = 3.0, max_vertex_distance: Optional[float] = None, remesh: bool = True, remesh_smooth_iterations: int = 5, workers: int = 4, decimate_target: Optional[float] = None, curvature_percentile: float = 80.0, curvature_protect_rings: int = 1) -> o3d.geometry.TriangleMesh:
    """Reconstruct a cleaned mesh; automatic depth is capped at 11, manual depth is not."""
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        try:
            pcd.orient_normals_consistent_tangent_plane(100)
        except Exception:
            pass
    depth = suggest_poisson_depth(pcd) if poisson_depth is None else int(poisson_depth)
    if depth < 2:
        raise ValueError("--poisson_depth must be at least 2.")
    if poisson_depth is not None and depth > 11:
        logging.warning("Using explicit Poisson depth %d; automatic mode is capped at 11.", depth)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=True)
    densities = np.asarray(densities, dtype=np.float64)
    low_density = densities < np.percentile(densities, float(min_density_percentile)) if densities.size else np.zeros(len(mesh.vertices), bool)
    points, vertices = np.asarray(pcd.points), np.asarray(mesh.vertices)
    spacing = _average_point_spacing(pcd)
    adaptive_limit = float(distance_multiplier) * spacing if spacing > 0 and distance_multiplier > 0 else np.inf
    cap = float(max_vertex_distance) if max_vertex_distance is not None and float(max_vertex_distance) > 0 else np.inf
    limit = min(adaptive_limit, cap)
    if np.isfinite(limit) and len(vertices):
        distances, _ = cKDTree(points).query(vertices, k=1, workers=workers)
        far = distances > limit
    else:
        far = np.zeros(len(vertices), dtype=bool)
    logging.info("Hybrid trim: %d low-density, %d farther than %.4fm.", int(low_density.sum()), int(far.sum()), limit if np.isfinite(limit) else 0.0)
    mesh.remove_vertices_by_mask(low_density | far)
    mesh = _cleanup_mesh(mesh)
    if not len(mesh.triangles):
        raise ValueError("Mesh empty after hybrid trim; reduce density trim or increase distance multiplier.")
    if remesh:
        mesh = remesh_isotropic(mesh, remesh_smooth_iterations)
    if decimate_target is not None:
        target = max(4, int(len(mesh.triangles) * decimate_target)) if decimate_target <= 1.0 else int(decimate_target)
        mesh = simplify_curvature_aware(mesh, target, curvature_percentile, curvature_protect_rings)
    mesh.compute_vertex_normals()
    logging.info("Final mesh: %s verts, %s faces.", f"{len(mesh.vertices):,}", f"{len(mesh.triangles):,}")
    return mesh


def level_floor(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    try:
        _, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        if len(inliers) < 10:
            return pcd
        points = np.asarray(pcd.select_by_index(inliers).points)
        center = points.mean(axis=0)
        normal = np.linalg.svd(points - center)[2][-1]
        if normal[2] < 0:
            normal = -normal
        target = np.array([0.0, 0.0, 1.0])
        cross = np.cross(normal, target); sine = np.linalg.norm(cross); cosine = float(normal @ target)
        if sine > 1e-12:
            skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
            rotation = np.eye(3) + skew + skew @ skew * ((1 - cosine) / sine**2)
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) @ rotation.T)
            if pcd.has_normals():
                pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) @ rotation.T)
    except Exception as exc:
        logging.warning("Floor levelling failed: %s", exc)
    return pcd


def _enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]], dtype=np.float64)


def transform_local_enu_to_ecef(pts_enu: np.ndarray, lat0_deg: float, lon0_deg: float, alt0_m: float) -> np.ndarray:
    from pyproj import CRS, Transformer
    x, y, z = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(4978), always_xy=True).transform(lon0_deg, lat0_deg, alt0_m)
    return np.asarray(pts_enu, dtype=np.float64) @ _enu_to_ecef_rotation(lat0_deg, lon0_deg).T + np.array([x, y, z])
