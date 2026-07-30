"""
colormesh3d.common.packaging
==============================
Shared output-packaging utilities for the ColorMesh3D pipeline.

Consolidates two previously-duplicated implementations:
  - io_utils.create_atak_zip()      (legacy mesh3d-photoreal / vertex mode)
  - inline zipfile logic in pipeline.py (legacy mesh3d-photoreal2 / atlas mode)

Both legacy versions built a zip containing an OBJ + MTL + PNG (plus
optional georef .xyz/.prj sidecars). This module provides a single
`create_atak_zip()` that covers both use-cases via an `extra_files`
parameter, plus `setup_workspace()` for consistent output-directory
initialization.
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def setup_workspace(workspace_path: Path, overwrite: bool = False) -> Dict[str, Path]:
    """
    Create the workspace directory structure and return a dict of key
    intermediate/output paths. Only creates directories as needed.

    Raises:
        FileExistsError: if workspace_path already exists and overwrite=False.
    """
    workspace_path = Path(workspace_path)

    if workspace_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Workspace {workspace_path} already exists. Use --overwrite to replace it."
            )
        logger.warning(f"Workspace {workspace_path} exists. Overwriting...")

    workspace_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Workspace {workspace_path} created.")

    paths = {
        "workspace": workspace_path,
        # Vertex-mode intermediates
        "merged_ply": workspace_path / "merged.ply",
        "timestamps": workspace_path / "timestamps.npy",
        "subsampled_ply": workspace_path / "subsampled.ply",
        "subsampled_timestamps": workspace_path / "subsampled_timestamps.npy",
        "normals_ply": workspace_path / "normals.ply",
        # Meshing stages (shared naming convention)
        "raw_mesh": workspace_path / "01_raw_mesh.ply",
        "cleaned_mesh": workspace_path / "02_cleaned_mesh.ply",
        "simplified_mesh": workspace_path / "03_simplified_mesh.ply",
        "final_mesh": workspace_path / "04_final_mesh.ply",
        # Final output
        "final_mesh_obj": workspace_path / "05_final_mesh.obj",
        "final_mesh_mtl": workspace_path / "05_final_mesh.mtl",
        "final_zip": workspace_path / "06_final_mesh_atak.zip",
    }
    return paths


def create_atak_zip(
    obj_path: Path,
    zip_path: Path,
    mtl_path: Optional[Path] = None,
    png_path: Optional[Path] = None,
    extra_files: Optional[List[Path]] = None,
) -> Path:
    """
    Create an ATAK-compatible zip file containing the OBJ file, its
    corresponding MTL, the baked PNG texture, and any extra sidecar
    files (e.g. georef.xyz / georef.prj).

    If mtl_path is not supplied, it is inferred as obj_path.with_suffix('.mtl').
    If png_path is not supplied, it is inferred as obj_path.with_suffix('.png').

    If no MTL file exists on disk, a default one is generated and a
    `mtllib`/`usemtl` reference is injected into the OBJ file so it
    remains valid even without a texture-baking step having run.

    Args:
        obj_path: Path to the final OBJ mesh (must exist).
        zip_path: Output path for the zip file.
        mtl_path: Optional explicit MTL path (inferred from obj_path if omitted).
        png_path: Optional explicit PNG texture path (inferred if omitted).
        extra_files: Optional list of additional files to include
            (e.g. georef.xyz, georef.prj).

    Returns:
        The path to the created zip file.

    Raises:
        FileNotFoundError: if obj_path does not exist.
    """
    obj_path = Path(obj_path)
    zip_path = Path(zip_path)

    logger.info(f"Creating zip file: {zip_path.name}")

    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    mtl_path = Path(mtl_path) if mtl_path is not None else obj_path.with_suffix(".mtl")
    png_path = Path(png_path) if png_path is not None else obj_path.with_suffix(".png")
    extra_files = extra_files or []

    # Force-create a default MTL file if one doesn't exist, and inject
    # the material reference into the OBJ so downstream viewers don't
    # silently render an untextured mesh.
    if not mtl_path.exists():
        logger.warning(f"No MTL file found for {obj_path.name}. Generating a default MTL file.")
        with open(mtl_path, "w") as f:
            f.write("# Default MTL file generated for ATAK compatibility\n")
            f.write("newmtl default_material\n")
            f.write("Ka 1.0 1.0 1.0 # Ambient color\n")
            f.write("Kd 0.8 0.8 0.8 # Diffuse color (light gray)\n")

        try:
            with open(obj_path, "r") as f:
                obj_content = f.read()
            if "mtllib" not in obj_content.splitlines()[0:3].__str__():
                with open(obj_path, "w") as f:
                    f.write(f"mtllib {mtl_path.name}\n")
                    f.write("usemtl default_material\n")
                    f.write(obj_content)
                logger.info(f"Injected material reference into {obj_path.name}")
        except Exception as e:
            logger.error(f"Failed to link default MTL inside OBJ file: {e}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(obj_path, arcname=obj_path.name)
        logger.info(f"Added {obj_path.name} to zip.")

        if mtl_path.exists():
            zipf.write(mtl_path, arcname=mtl_path.name)
            logger.info(f"Added {mtl_path.name} to zip.")
        else:
            logger.warning(f"No MTL file found for {obj_path.name}.")

        if png_path.exists():
            zipf.write(png_path, arcname=png_path.name)
            logger.info(f"Added {png_path.name} to zip.")
        else:
            logger.warning(f"No PNG file found for {obj_path.name}.")

        for extra in extra_files:
            extra = Path(extra)
            if extra.exists():
                zipf.write(extra, arcname=extra.name)
                logger.info(f"Added {extra.name} to zip.")
            else:
                logger.warning(f"Expected extra file not found, skipping: {extra}")

    logger.info(f"Successfully created zip file: {zip_path}")
    return zip_path
