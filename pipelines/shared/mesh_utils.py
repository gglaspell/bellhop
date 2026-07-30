"""Height-based false-color OBJ export."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

_VALID_COLORMAPS = {"jet", "hot", "cool", "gray"}


def _colormap(values: np.ndarray, name: str = "jet") -> np.ndarray:
    """Map normalized values in [0, 1] to uint8 RGB colors."""
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

    return (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)


def _write_obj(
    output_obj_path: Path,
    mtl_name: str,
    colormap: str,
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    flat_uvs: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Write an OBJ with explicit vertex-normal and texture indices."""
    buffer = io.BytesIO()

    header = (
        f"# Height-coloured mesh; colormap: {colormap}\n"
        f"mtllib {mtl_name}\n\n"
    ).encode("utf-8")
    buffer.write(header)

    np.savetxt(
        buffer,
        np.asarray(vertices, dtype=np.float32),
        fmt="v %.6f %.6f %.6f",
    )
    buffer.write(b"\n")

    np.savetxt(
        buffer,
        np.asarray(vertex_normals, dtype=np.float32),
        fmt="vn %.6f %.6f %.6f",
    )
    buffer.write(b"\n")

    np.savetxt(
        buffer,
        np.asarray(flat_uvs, dtype=np.float32),
        fmt="vt %.6f %.6f",
    )
    buffer.write(b"\nusemtl material0\n")

    face_count = len(faces)
    vertex_indices = np.asarray(faces, dtype=np.int32) + 1
    uv_indices = np.arange(face_count, dtype=np.int32) * 3 + 1

    # FIX: np.column_stack() accepts a single sequence argument (a tuple or
    # list of the columns to stack), not multiple bare positional arguments.
    # The previous call passed nine separate positional args -- e.g.
    # np.column_stack(vertex_indices[:, 0], uv_indices, ...) -- which raises
    # `TypeError: column_stack() takes 1 positional argument but 9 were
    # given`. Wrapping all nine arrays in a single tuple fixes this.
    face_rows = np.column_stack((
        vertex_indices[:, 0],
        uv_indices,
        vertex_indices[:, 0],
        vertex_indices[:, 1],
        uv_indices + 1,
        vertex_indices[:, 1],
        vertex_indices[:, 2],
        uv_indices + 2,
        vertex_indices[:, 2],
    ))

    np.savetxt(
        buffer,
        face_rows,
        fmt="f %d/%d/%d %d/%d/%d %d/%d/%d",
        comments="",
    )
    buffer.write(b"\n")

    output_obj_path.parent.mkdir(parents=True, exist_ok=True)

    with output_obj_path.open("wb") as file:
        file.write(buffer.getvalue())


def apply_height_colormap(
    input_mesh_path: Path,
    output_obj_path: Path,
    colormap: str = "jet",
    texture_size: int = 1024,
) -> tuple[Path, Path]:
    """
    Export an OBJ, MTL, and PNG texture bundle colored by mesh Z height.

    The lowest mesh vertex maps to the beginning of the color ramp and the
    highest mesh vertex maps to its end. The original mesh geometry is not
    modified.
    """
    colormap = colormap.lower()

    if colormap not in _VALID_COLORMAPS:
        choices = ", ".join(sorted(_VALID_COLORMAPS))
        raise ValueError(
            f"Unsupported height colormap '{colormap}'. "
            f"Choose one of: {choices}."
        )

    if texture_size < 2:
        raise ValueError("--height_texture_size must be at least 2.")

    input_mesh_path = Path(input_mesh_path)
    output_obj_path = Path(output_obj_path)

    if not input_mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {input_mesh_path}")

    logging.info(
        "Applying '%s' height colormap to '%s' using a %d-pixel LUT.",
        colormap,
        input_mesh_path.name,
        texture_size,
    )

    mesh = o3d.io.read_triangle_mesh(str(input_mesh_path))

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(
            f"'{input_mesh_path}' must contain vertices and triangular faces."
        )

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    if len(vertex_normals) != len(vertices):
        mesh.compute_vertex_normals()
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    heights = vertices[:, 2]
    height_range = float(heights.max() - heights.min())

    if height_range <= 1e-12:
        normalized_heights = np.full(len(heights), 0.5, dtype=np.float64)
        logging.warning(
            "Mesh has no measurable height range; using the midpoint color."
        )
    else:
        normalized_heights = (
            heights - heights.min()
        ) / height_range

    gradient_values = np.linspace(1.0, 0.0, texture_size)
    gradient_colors = _colormap(gradient_values, colormap)

    texture_array = np.tile(
        gradient_colors[:, np.newaxis, :],
        (1, 4, 1),
    )

    texture_path = output_obj_path.with_name(
        f"{output_obj_path.stem}_texture.png"
    )
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(texture_array, mode="RGB").save(texture_path)

    # FIX: same np.column_stack() positional-args bug as in _write_obj().
    # Wrap the two UV columns in a tuple so column_stack receives a single
    # sequence argument instead of two bare positional args.
    vertex_uvs = np.column_stack((
        np.full(len(normalized_heights), 0.5, dtype=np.float64),
        normalized_heights,
    ))

    flat_uvs = vertex_uvs[faces].reshape(-1, 2)

    mtl_path = output_obj_path.with_suffix(".mtl")

    _write_obj(
        output_obj_path=output_obj_path,
        mtl_name=mtl_path.name,
        colormap=colormap,
        vertices=vertices,
        vertex_normals=vertex_normals,
        flat_uvs=flat_uvs,
        faces=faces,
    )

    with mtl_path.open("w", encoding="utf-8") as file:
        file.write("newmtl material0\n")
        file.write("Ka 1.0 1.0 1.0\n")
        file.write("Kd 1.0 1.0 1.0\n")
        file.write("Ks 0.1 0.1 0.1\n")
        file.write("Ns 16.0\n")
        file.write(f"map_Kd {texture_path.name}\n")

    logging.info("Height-colored OBJ written to '%s'.", output_obj_path)
    logging.info("Material file written to '%s'.", mtl_path)
    logging.info("Texture written to '%s'.", texture_path)

    return output_obj_path, texture_path
