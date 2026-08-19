"""
shared/packaging.py - Shared output-workspace setup for the texture-baking
(atlas-bake) pipeline.

REFACTOR NOTE (moved from atlas_pipeline/common):
This module used to live at `pipelines/atlas_pipeline/common/packaging.py`.
See `shared/trajectory.py`'s module docstring for the full rationale for
moving `atlas_pipeline/common/`'s contents into `shared/`. `texture_baking.py`
now imports `setup_workspace` from `shared.packaging` instead of
`atlas_pipeline.common.packaging`. `atlas_pipeline/common/` no longer
exists.

PATCH NOTE (ATAK export removed):
This module previously also exported `create_atak_zip()`, which bundled
the baked OBJ + MTL + PNG (plus optional georef sidecars) into an
ATAK-compatible zip. The texture_baking pipeline no longer produces an
ATAK zip at all -- it now stops after writing the baked OBJ + PNG texture
(see texture_baking.py's PATCH NOTE). `create_atak_zip()` and its
`zipfile` dependency have been removed entirely rather than left as dead
code. `setup_workspace()`'s returned path dict no longer reserves
ATAK-specific output filenames (`final_mesh_obj`, `final_mesh_mtl`,
`final_zip`) -- none of these were actually read by any caller even
before this change (texture_baking.py computes its own
`{stem}_baked_mesh.obj` / `_baked_mesh_texture.png` paths directly in the
workspace root), so removing them is a pure cleanup with no behavior
change.
"""

import logging
from pathlib import Path
from typing import Dict

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
    }

    return paths
