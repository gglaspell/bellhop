# 🛎️ bellhop

**bellhop** converts ROS 2 bag files into spatial outputs: Nav2 occupancy maps, Poisson meshes, Gazebo worlds, georeferenced Cesium 3D Tiles, and keyframe-baked textured meshes. Use the Docker-backed CLI or the host-native Tk GUI.

## Features

- Seven pipelines behind one CLI: `og_map`, `mesh`, `color_mesh`, `gazebo_world`, `tiles_3d`, `color_tiles_3d`, and `texture_baking`.
- Automatic point-cloud frame handling. `--pc_frame_mode auto` detects fixed/global (`odom`, `map`, or `world`) versus moving/local cloud frames; use `global` or `local` to override detection.
- Odometry-anchored registration for mesh, color-mesh, Gazebo, and Tiles pipelines. ICP refinement and loop closure are optional, correction-gated refinements—not the primary pose source.
- Motion-gated frame selection (`--min_move_distance`, `--min_rotation_angle_deg`) replaces the old registration `--frame_stride` behavior. An odometry health check stops processing at an apparent tracking-loss/teleport segment by default.
- Memory-bounded chunked merging controlled by `--merge_chunk_frames` for the registration-based pipelines.
- Three Cesium Tiles detail layers—`coarse`, `medium`, and `fine`—with configurable voxel multipliers.
- Camera-color workflows preserve per-vertex colors and explicitly use neutral-gray fallback when a frame cannot be projected from a camera image.
- A host-native GUI with pipeline-specific forms, topic pre-flight checks, a live Docker log, and light/dark themes.

## Requirements

### Docker CLI

Docker is the only host prerequisite for CLI usage.

```bash
git clone https://github.com/gglaspell/bellhop.git
cd bellhop
docker build -t bellhop:latest .
```

### GUI

The GUI runs on the host and requires Python 3 with `tkinter`, Docker, and a local clone of this repository. It has no third-party Python dependencies on the host.

```bash
python3 gui.py
```

On Ubuntu/Debian, install Tk if needed:

```bash
sudo apt install python3-tk
```

## Quick start

All CLI pipelines use this container form:

```bash
docker run --rm \
  -v /path/to/bag:/data/bag:ro \
  -v /path/to/output:/data/output \
  bellhop:latest <pipeline> /data/bag /data/output [options]
```

For example, create a mesh from a local-frame cloud using odometry:

```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  -v ~/output:/data/output \
  bellhop:latest mesh /data/bag /data/output \
    --pc_topic /points \
    --odom_topic /odom \
    --pc_frame_mode local
```

To inspect available commands and parameters:

```bash
docker run --rm bellhop:latest --help
docker run --rm bellhop:latest mesh --help
```

## Pipelines

| Pipeline | Required topics | Output |
|---|---|---|
| `og_map` | PointCloud2, Odometry | Nav2 `.pgm` + `.yaml`, plus PNG preview |
| `mesh` | PointCloud2; Odometry for local clouds | `<bag>_cloud.ply`, `<bag>_mesh.ply`, optional height-colored PLY |
| `color_mesh` | PointCloud2, Image/CompressedImage, CameraInfo; Odometry for local clouds | `<bag>_colored_cloud.ply`, `<bag>_colored_mesh.ply` |
| `gazebo_world` | PointCloud2; Odometry for local clouds | STL mesh, `model.config`, `model.sdf`, and Gazebo `.world` |
| `tiles_3d` | PointCloud2, NavSatFix; Odometry for local clouds | Three georeferenced Cesium Tiles layers |
| `color_tiles_3d` | PointCloud2, NavSatFix; camera topics are optional; Odometry for local clouds | Three colored Cesium Tiles layers when camera data is available |
| `texture_baking` | PointCloud2, Image/CompressedImage, CameraInfo, Odometry | Keyframe-baked OBJ mesh and PNG texture in an output workspace |

### Frame handling and registration

All pipelines support:

```text
--pc_frame_mode {auto,global,local}
```

- `auto` inspects the point-cloud frame ID. Known fixed frames (`odom`, `map`, `world`) are treated as global; other or missing frame IDs are treated as local.
- `global` streams points without per-frame pose transformation or registration.
- `local` transforms clouds into the world frame using interpolated odometry. It requires `--odom_topic`.

`og_map` and `texture_baking` use odometry directly when a local transform is needed; they do not have ICP or loop closure. The other five pipelines use odometry as the primary motion estimate and may optionally enable correction-gated ICP:

```text
--enable_icp_refinement
--icp_dist_thresh 0.2
--icp_fitness_thresh 0.7
--max_icp_translation_correction 0.3
--max_icp_rotation_correction_deg 15.0
--enable_loop_closure
```

Registration-based pipelines select frames based on movement or rotation relative to the last kept pose:

```text
--min_move_distance 0.10
--min_rotation_angle_deg 5.0
--merge_chunk_frames 16
--disable_odom_health_check
--odom_loss_speed_multiplier 6.0
```

Set either motion threshold to `0` to disable that half of the motion gate. The automatic odometry health check is enabled by default; it truncates processing at the first likely tracking-loss/teleport segment. `--disable_odom_health_check` disables that safeguard.

> Frame classification only determines whether a cloud needs a world-frame transform. It does not apply a sensor-to-base extrinsic transform from `tf_static`.

### `og_map` — Nav2 occupancy grid

Builds an OctoMap-backed occupancy grid, estimates ground from surface normals, and writes a Nav2 map. It supports raw local-frame clouds by transforming each frame with interpolated odometry before insertion.

Useful options:

```text
--pc_topic /points
--odom_topic /odom
--octree_res 0.10
--grid_res 0.10
--octree_max_range 40
--octree_lazy_eval
--octree_discretize
--frame_stride 2
--max_frames 0
```

`--octree_lazy_eval` defers inner-node updates until insertion is complete (default: **on**). Use `--octree_max_range -1` for unlimited ray range.

> **`--octree_discretize` defaults to OFF and should generally be left off.** It is meant to reduce duplicate rays on dense scans by snapping each frame onto the octree's own voxel grid before ray casting, but the `pyoctomap` build currently pinned in the Docker image crashes on every insertion when it's enabled:
>
> ```text
> AttributeError: 'pyoctomap.octree_base.OcTreeKey' object has no attribute 'thisptr'
> ```
>
> This is a bug inside `pyoctomap`'s own `_discretizePointCloud`/`coordToKeyChecked` code (an `OcTreeKey` object gets constructed without its underlying C++ pointer allocated), not something `og_map.py` can fix on its own. The flag is still exposed — passing `--octree_discretize` explicitly re-enables the old behavior, and `og_map.py` logs a loud warning when you do — in case a future `pyoctomap` release fixes the bug and you want the discretization speedup back. Until then, leave it off; per-point ray casting (the current default) is unaffected by the bug.

### `mesh` and `color_mesh`

`mesh` reconstructs a Poisson mesh from the merged cloud. Its optional height false-color export uses vertex RGB stored directly in PLY:

```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  -v ~/output:/data/output \
  bellhop:latest mesh /data/bag /data/output \
    --pc_topic /points --odom_topic /odom \
    --height_colormap gray
```

`color_mesh` projects synchronized images onto points, writes PLY-only colored cloud and mesh outputs, and retains neutral-gray fallback points when imagery is missing. It intentionally does not offer mesh remeshing because that operation would discard the per-vertex colors.

### `gazebo_world`

Creates a static Gazebo model and world from the reconstructed mesh:

```text
models/<model_name>/meshes/model.stl
models/<model_name>/model.config
models/<model_name>/model.sdf
worlds/<model_name>.world
```

Configure `--model_name`, `--gazebo_material`, and `--level_floor` as needed.

### `tiles_3d` and `color_tiles_3d`

Both Tiles pipelines use NavSatFix data as an ENU origin, convert the cleaned cloud to ECEF, and write three Tiles layers. Customize their relative voxel resolutions with:

```text
--lod_multipliers 4.0,2.0,1.0
```

The values correspond to the `coarse`, `medium`, and `fine` layers, respectively. `color_tiles_3d` can run without camera data and will produce XYZ-only tiles; provide both `--camera_topic` and `--camera_info_topic` to create colored tiles.

### `texture_baking`

This pipeline selects well-spaced camera keyframes, filters and reconstructs a mesh, assigns faces to camera views, packs a UV atlas, and bakes a PNG texture. Odometry is always required.

```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  -v ~/output:/data/output \
  bellhop:latest texture_baking /data/bag /data/output \
    --pc_topic /points \
    --camera_topic /camera/image_raw \
    --camera_info_topic /camera/camera_info \
    --odom_topic /odom
```

The final artifacts are `<bag>_baked_mesh.obj` and `<bag>_baked_mesh_texture.png` inside the workspace. Use `--overwrite` to reuse an existing output workspace.

## GUI

Run `python3 gui.py` from the repository root. The GUI:

- Provides profiles in this order: OG Map, Mesh, 3D Tiles, Gazebo World, Color Mesh, Color Tiles, and Texture Baking
- Maps host bag and output paths to `/data/bag` and `/data/output` automatically
- Lets you select a Docker image tag
- Checks the selected profile's required ROS topics in a short-lived container before running
- Builds and echoes the Docker command, then streams its combined output into the log panel
- Shows only parameters accepted by the selected pipeline

For `og_map`, the GUI passes an output base path under `/data/output/occupancy_map`; all other pipelines receive `/data/output` as their output directory. The OG Map profile's "OcTree discretize scan before insertion" checkbox is unchecked by default, matching `og_map.py`'s own default — see the `--octree_discretize` note under [`og_map` — Nav2 occupancy grid](#og_map--nav2-occupancy-grid) before enabling it.

## Changes from earlier versions

- Registration-based pipelines no longer accept registration `--frame_stride`; use movement/rotation thresholds instead. `og_map` retains its independent index-based `--frame_stride` and `--max_frames` controls.
- `mesh` and `color_mesh` no longer write OBJ meshes. Their mesh outputs are PLY.
- Height false-color is a per-vertex PLY export; there is no UV texture, MTL file, PNG texture, or `--height_texture_size` option.
- `color_mesh` no longer supports `--remesh` or `--remesh_smooth_iterations`, preserving camera-derived vertex color.
- `texture_baking` no longer creates an ATAK zip or ATAK-specific package. It stops after writing the baked OBJ and PNG texture.
- `texture_baking` does not accept camera-projection color options (`--max_time_diff`, `--color_min_depth`, `--color_max_depth`, or `--gray_filter_radius`) or remeshing options.
- Cesium output is no longer a single fixed-resolution tileset: it is produced as three configurable LOD layers.
- `og_map`'s `--octree_discretize` now defaults to **off** (previously on). The `pyoctomap` build pinned in the Docker image crashes with `AttributeError: 'OcTreeKey' object has no attribute 'thisptr'` whenever `insertPointCloud()` is called with `discretize=True`. The flag still exists — pass `--octree_discretize` explicitly to re-enable it, and expect it to crash until that upstream `pyoctomap` bug is fixed. The GUI's OG Map profile checkbox for this option is unchecked by default to match.

## Project layout

```text
bellhop/
├── cli.py
├── gui.py
├── Dockerfile
├── requirements.txt
└── pipelines/
    ├── og_map.py
    ├── mesh.py
    ├── color_mesh.py
    ├── gazebo_world.py
    ├── tiles_3d.py
    ├── color_tiles_3d.py
    ├── texture_baking.py
    ├── atlas_pipeline/
    └── shared/
```

## Local installation

Docker is the supported easy path. For local development, use a virtual environment and install `requirements.txt`; ROS bag support and native/system dependencies still need to be available on the host.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cli.py --help
```
