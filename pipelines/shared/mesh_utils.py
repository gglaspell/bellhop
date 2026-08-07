"""Height-based false-color PLY export."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

_VALID_COLORMAPS = {"jet", "hot", "cool", "gray"}


def _colormap(values: np.ndarray, name: str = "jet") -> np.ndarray:
    """Map normalized values in [0, 1] to float RGB colors in [0, 1]."""
    values = np.clip(values, 0.0, 1.0).astype(np.float64)

    if name == "jet":
        red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    elif name == "hot":
        red = np.clip(values * 3.0, 0.0, 1.0)
        green = np.clip(values * 3.0 - 1.0, 0.0, 1.0)
        blue = np.clip(values * 3.0 - 2.0, 0.0, 1.0)
    elif name == "cool":
        red = values
        green = 1.0 - values
        blue = np.ones_like(values)
    elif name == "gray":
        red = green = blue = values
    else:
        choices = ", ".join(sorted(_VALID_COLORMAPS))
        raise ValueError(
            f"Unsupported height colormap '{name}'. "
            f"Choose one of: {choices}."
        )

    return np.stack((red, green, blue), axis=-1)


def apply_height_colormap(
    input_mesh_path: Path,
    output_ply_path: Path,
    colormap: str = "jet",
) -> Path:
    """
    Export a PLY colored by mesh Z height, using per-vertex colors.

    The lowest mesh vertex maps to the beginning of the color ramp and the
    highest mesh vertex maps to its end. Colors are baked directly as
    per-vertex RGB -- no UV texture, no MTL, no PNG. The original mesh
    geometry is not modified.
    """
    colormap = colormap.lower()

    if colormap not in _VALID_COLORMAPS:
        choices = ", ".join(sorted(_VALID_COLORMAPS))
        raise ValueError(
            f"Unsupported height colormap '{colormap}'. "
            f"Choose one of: {choices}."
        )

    input_mesh_path = Path(input_mesh_path)
    output_ply_path = Path(output_ply_path)

    if not input_mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {input_mesh_path}")

    logging.info(
        "Applying '%s' height colormap to '%s' as per-vertex PLY colors.",
        colormap,
        input_mesh_path.name,
    )

    mesh = o3d.io.read_triangle_mesh(str(input_mesh_path))

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(
            f"'{input_mesh_path}' must contain vertices and triangular faces."
        )

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    heights = vertices[:, 2]
    height_range = float(heights.max() - heights.min())

    if height_range <= 1e-12:
        normalized_heights = np.full(len(heights), 0.5, dtype=np.float64)
        logging.warning(
            "Mesh has no measurable height range; using the midpoint color."
        )
    else:
        normalized_heights = (heights - heights.min()) / height_range

    colors = _colormap(normalized_heights, colormap)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    output_ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(
        str(output_ply_path),
        mesh,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )

    logging.info("Height-colored PLY written to '%s'.", output_ply_path)

    return output_ply_path
