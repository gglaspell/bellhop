"""Texture baking with photometric normalisation and vertex-normal OBJ export."""
import trimesh
import trimesh.repair
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import logging
import cv2
from pathlib import Path


def _inject_normals_into_obj(obj_path: Path, mesh: trimesh.Trimesh) -> None:
    """Post-process the exported OBJ to add ``vn`` lines and update face format.

    Converts ``f v/vt v/vt v/vt`` entries to ``f v/vt/vn v/vt/vn v/vt/vn`` so
    that renderers such as ATAK can rely on explicit vertex normals rather than
    inferring winding order (Fix 2 from normal-fix spec).
    """
    trimesh.repair.fix_normals(mesh)
    normals = mesh.vertex_normals  # (N, 3) -- per-exploded-vertex for UV meshes

    with open(obj_path, "r") as fh:
        lines = fh.readlines()

    # If vn lines are already present trimesh wrote them -- nothing to do.
    if any(l.startswith("vn ") for l in lines):
        return

    new_lines: list[str] = []
    vn_injected = False

    for line in lines:
        stripped = line.strip()

        # Inject all vn entries immediately before the first face line.
        if stripped.startswith("f ") and not vn_injected:
            for n in normals:
                new_lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            vn_injected = True

        if stripped.startswith("f "):
            # Rewrite: f v/vt ... -> f v/vt/vn ... (vertex and normal share same index)
            parts = stripped.split()
            new_parts = ["f"]
            for p in parts[1:]:
                segs = p.split("/")
                if len(segs) == 2:  # v/vt -- add normal index
                    new_parts.append(f"{segs[0]}/{segs[1]}/{segs[0]}")
                else:  # already v/vt/vn or plain v
                    new_parts.append(p)
            new_lines.append(" ".join(new_parts) + "\n")
        else:
            new_lines.append(line)

    with open(obj_path, "w") as fh:
        fh.writelines(new_lines)


def _build_chart_seam_mask(texture_np: np.ndarray, mask_np: np.ndarray,
                            dilation_px: int = 8) -> np.ndarray:
    """Return a blended seam region for cross-chart colour continuity.

    At every atlas chart boundary the hard edge between two independently
    sampled regions can produce a visible step in colour/brightness even after
    photometric normalisation. This function:

    1. Dilates the filled mask by ``dilation_px`` pixels to find the seam band.
    2. Inpaints the seam band using OpenCV's Telea algorithm so that colour
       blends inward from both filled sides of the seam.

    This is applied *after* the main bake loop and *before* Pillow export.

    Args:
        texture_np: (H, W, 3) uint8 baked texture.
        mask_np: (H, W) uint8 mask, 255 = filled, 0 = empty.
        dilation_px: width of the seam blend band in pixels.

    Returns:
        Corrected texture_np with softened chart seams (in-place modification
        is avoided -- a new array is returned).
    """
    kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
    seam_band = (dilated_mask == 255) & (mask_np == 0)

    if not np.any(seam_band):
        return texture_np

    # FIX: The inpaint mask must mark the pixels to be filled (the seam band
    # itself), not its complement. The previous version passed the inverse
    # of the seam band, which told OpenCV to "inpaint everything except the
    # seam", leaving the actual seam untouched and corrupting unrelated
    # already-filled pixels.
    inpaint_mask = seam_band.astype(np.uint8) * 255

    inpainted = cv2.inpaint(
        texture_np,
        inpaint_mask,
        inpaintRadius=dilation_px,
        flags=cv2.INPAINT_TELEA,
    )

    out = texture_np.copy()
    out[seam_band] = inpainted[seam_band]
    return out


class TextureBaker:
    def __init__(
        self,
        uv_mesh_path,
        assignments,
        keyframes,
        image_folder,
        traj_path,
        intr,
        size,
    ):
        # FIX: previously only `self.mesh` and `self.kf` were stored; the
        # remaining constructor arguments were silently dropped even though
        # some (uv_mesh_path in particular) are needed later in bake_texture().
        self.uv_mesh_path = Path(uv_mesh_path)
        self.mesh = trimesh.load(str(self.uv_mesh_path), process=False)
        self.assignments = assignments
        self.kf = keyframes
        self.image_folder = Path(image_folder) if image_folder is not None else None
        self.traj_path = traj_path
        self.intr = intr
        # Atlas dimensions are driven by atlas_packer; size kept for API compat.
        self.size = size

    def bake_texture(self, out_mesh_path, out_texture_path):
        out_mesh_path = Path(out_mesh_path)
        charts_path = out_mesh_path.parent / "intermediate" / "atlas_charts.pkl"
        with open(charts_path, "rb") as f:
            atlas_data = pickle.load(f)

        if isinstance(atlas_data, list):
            charts = atlas_data
            final_w = final_h = 8192
        else:
            charts = atlas_data["charts"]
            final_w = atlas_data["final_w"]
            final_h = atlas_data["final_h"]

        logging.info(f"Baking into a {final_w}x{final_h} atlas...")

        texture_np = np.full((final_h, final_w, 3), 128, dtype=np.uint8)
        mask_np = np.zeros((final_h, final_w), dtype=np.uint8)

        logging.info("Preparing photometric reference...")
        ref_img = cv2.imread(str(self.kf[0][0]))
        if ref_img is None:
            raise FileNotFoundError(
                f"Could not read reference keyframe image: {self.kf[0][0]}"
            )
        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
        ref_l_mean, ref_l_std = cv2.meanStdDev(ref_lab[:, :, 0])

        # Cache loaded + normalised images so that consecutive charts from
        # the same keyframe do not reload and re-normalise the same file.
        # Previously every chart -- including the many small charts that all
        # map to the same keyframe on a long wall -- read and converted the
        # source image independently, introducing floating-point rounding
        # differences that could shift mortar-line pixel values by +/-1 LSB
        # and create visible discontinuities at chart boundaries.
        img_cache: dict[str, np.ndarray] = {}

        for chart in tqdm(charts, desc="Baking"):
            img_path = str(self.kf[chart["kf"]][0])

            if img_path not in img_cache:
                raw = cv2.imread(img_path)
                if raw is None:
                    img_cache[img_path] = None
                    continue

                # Photometric normalisation -- done once per source image so all
                # charts from the same keyframe see identical pixel values.
                lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)
                l_mean, l_std = cv2.meanStdDev(lab[:, :, 0])
                lab[:, :, 0] = np.clip(
                    (lab[:, :, 0] - l_mean) * (ref_l_std / (l_std + 1e-5)) + ref_l_mean,
                    0,
                    255,
                ).astype(np.uint8)
                normalised = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                normalised = cv2.cvtColor(normalised, cv2.COLOR_BGR2RGB)
                img_cache[img_path] = normalised

            img = img_cache[img_path]
            if img is None:
                continue

            px, py = chart["px"], chart["py"]
            minu, minv = chart["minu"], chart["minv"]
            pad, scale = chart["pad"], chart.get("scale", 1.0)
            rotated = chart.get("rotated", False)

            # Use the actual reserved atlas block dimensions stored by atlas_packer.
            atlas_w = chart.get("atlas_w", max(1, int(chart["w"] * scale) + 2 * pad))
            atlas_h = chart.get("atlas_h", max(1, int(chart["h"] * scale) + 2 * pad))

            # Build index grids over the atlas block (rows=j, cols=i)
            j_idx, i_idx = np.mgrid[0:atlas_h, 0:atlas_w]

            if rotated:
                img_u = (minu + (j_idx - pad) / scale).astype(int)
                img_v = (minv + (i_idx - pad) / scale).astype(int)
            else:
                img_u = (minu + (i_idx - pad) / scale).astype(int)
                img_v = (minv + (j_idx - pad) / scale).astype(int)

            atlas_x = px + i_idx
            atlas_y = py + j_idx

            valid = (
                (img_u >= 0) & (img_u < img.shape[1])
                & (img_v >= 0) & (img_v < img.shape[0])
                & (atlas_x >= 0) & (atlas_x < final_w)
                & (atlas_y >= 0) & (atlas_y < final_h)
            )

            texture_np[atlas_y[valid], atlas_x[valid]] = img[img_v[valid], img_u[valid]]
            mask_np[atlas_y[valid], atlas_x[valid]] = 255

        logging.info("Applying post-bake seam dilation...")
        # Standard 5-px dilation fills empty border pixels of each chart.
        dilated = cv2.dilate(texture_np, np.ones((5, 5), np.uint8), iterations=3)
        texture_np = np.where(mask_np[:, :, None] == 255, texture_np, dilated)

        # Apply Telea inpainting in the seam band between adjacent charts
        # from *different* keyframes. Pure dilation propagates each chart's
        # border colour outward independently, leaving a hard step at the
        # meeting point. Telea inpainting blends inward from both sides so
        # the transition is smooth enough that mortar lines appear continuous
        # even where chart boundaries cross them.
        logging.info("Blending cross-chart seams...")
        texture_np = _build_chart_seam_mask(texture_np, mask_np, dilation_px=6)

        out_tex = Path(out_texture_path)
        Image.MAX_IMAGE_PIXELS = None
        Image.fromarray(texture_np).save(out_tex)

        logging.info("Generating mipmaps...")
        hh, ww = texture_np.shape[:2]
        Image.fromarray(
            cv2.resize(texture_np, (ww // 2, hh // 2), interpolation=cv2.INTER_AREA)
        ).save(out_tex.with_name(f"{out_tex.stem}_mip1.png"))
        Image.fromarray(
            cv2.resize(texture_np, (ww // 4, hh // 4), interpolation=cv2.INTER_AREA)
        ).save(out_tex.with_name(f"{out_tex.stem}_mip2.png"))

        # FIX: AtlasPacker writes the intermediate UV mesh as "uv_mesh.obj"
        # (see atlas_pipeline/atlaspacker.py's `pack_and_generate_uvs` output
        # path, and texture_baking.py's `uv_mesh_path = intermediate_dir /
        # "uv_mesh.obj"`). The previous version hardcoded "mesh_uv.obj" here,
        # which does not exist on disk and raised FileNotFoundError on every
        # run. Reuse self.uv_mesh_path (stored in __init__) instead of
        # re-deriving a wrong path.
        final_mesh = trimesh.load(str(self.uv_mesh_path))

        final_mesh.visual = trimesh.visual.TextureVisuals(
            uv=final_mesh.visual.uv, image=Image.fromarray(texture_np)
        )

        # FIX 3 (normal-fix): fix_normals as the last step before export.
        trimesh.repair.fix_normals(final_mesh)
        _ = final_mesh.vertex_normals  # force normal computation before export

        final_mesh.export(str(out_mesh_path))

        # FIX 2 (normal-fix): inject vn lines + update face format to v/vt/vn.
        _inject_normals_into_obj(Path(out_mesh_path), final_mesh)
        logging.info("Vertex normals injected into OBJ.")
