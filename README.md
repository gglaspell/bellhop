# 🛎️ bellhop

**bellhop** converts ROS 2 bag files into spatial outputs (occupancy grids, surface meshes, Gazebo worlds, and Cesium 3D Tiles) through a unified CLI and graphical launcher.

---

## 📋 Table of Contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Quick start with Docker (CLI)](#quick-start-with-docker-cli)
- [Graphical launcher (GUI)](#graphical-launcher-gui)
- [Project layout](#project-layout)
- [Pipelines](#pipelines)
  - [og_map — 2D Occupancy Grid](#og_map--2d-occupancy-grid)
  - [mesh — Poisson Surface Mesh](#mesh--poisson-surface-mesh)
  - [color_mesh — Camera-Colored Mesh](#color_mesh--camera-colored-mesh)
  - [gazebo_world — Gazebo Simulation World](#gazebo_world--gazebo-simulation-world)
  - [tiles_3d — Cesium 3D Tiles](#tiles_3d--cesium-3d-tiles)
  - [color_tiles_3d — Colored Cesium 3D Tiles](#color_tiles_3d--colored-cesium-3d-tiles)
- [Shared parameters](#shared-parameters)
  - [Registration](#registration)
  - [Reconstruction](#reconstruction)
- [Pre-flight topic check](#pre-flight-topic-check)
- [Running without Docker](#running-without-docker)
- [Per-pipeline documentation](#per-pipeline-documentation)

---

## 🦾 What it does

Each pipeline reads one or more topics from a ROS 2 bag, registers the point-cloud frames with ICP, and writes a ready-to-use spatial artifact:

| Pipeline | Input topics | Output |
|---|---|---|
| `og_map` | PointCloud2, Odometry | `.pgm` + `.yaml` (Nav2 map) |
| `mesh` | PointCloud2, Odometry (opt.) | `.ply` cloud + `.obj` mesh |
| `color_mesh` | PointCloud2, Camera, CameraInfo, Odometry (opt.) | `.ply` cloud + `.obj` colored mesh |
| `gazebo_world` | PointCloud2, Odometry (opt.) | `.stl` + `.sdf` + `.world` |
| `tiles_3d` | PointCloud2, NavSatFix, Odometry (opt.) | `tileset.json` (Cesium) |
| `color_tiles_3d` | PointCloud2, NavSatFix, Camera, CameraInfo, Odometry (opt.) | Colored `tileset.json` (Cesium) |

---

## ⚙️ Requirements

### To use the CLI or run pipelines via Docker

- [Docker](https://docs.docker.com/get-docker/) — the only requirement

### To use the graphical launcher (GUI)

- Python 3.8+ with `tkinter` (included in standard CPython — no extra install needed)
- Docker (same as above)
- The bellhop repo cloned locally

> **The GUI has zero third-party Python dependencies.** All pipeline work runs inside the Docker container. `tkinter` is part of the Python standard library and is available on Linux, macOS, and Windows without any `pip install`.

To check that tkinter is available on your system:
```bash
python3 -c "import tkinter; print('tkinter OK')"
```

On Ubuntu/Debian, if tkinter is missing:
```bash
sudo apt install python3-tk
```

To list all topics in a bag before running:
```bash
ros2 bag info /path/to/bag
```

---

## 💨 Quick start with Docker (CLI)

### 1. Clone and build

```bash
git clone https://github.com/gglaspell/bellhop.git
cd bellhop
docker build -t bellhop:latest .
```

### 2. Run a pipeline

```bash
docker run --rm \
  -v /path/to/your/bag:/data/bag:ro \
  -v /path/to/output:/data/output \
  bellhop:latest <pipeline> /data/bag /data/output [options]
```

Your host paths are mounted into the container at fixed locations:

| Host path | Container path |
|---|---|
| Your bag directory | `/data/bag` (read-only) |
| Your output directory | `/data/output` |

**Example — Nav2 occupancy grid:**

```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  -v ~/output:/data/output \
  bellhop:latest og_map /data/bag /data/output \
    --pc_topic /dlio/odom_node/pointcloud/deskewed \
    --odom_topic /dlio/odom_node/odom
```

**Example — Poisson mesh (with stride and memory controls):**

```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  -v ~/output:/data/output \
  bellhop:latest mesh /data/bag /data/output \
    --pc_topic /points \
    --voxel_size 0.05 \
    --frame_stride 4 \
    --max_registration_frames 500
```

**Example — georeferenced 3D Tiles:**

```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  -v ~/output:/data/output \
  bellhop:latest tiles_3d /data/bag /data/output \
    --pc_topic /points \
    --gps_topic /gps/fix
```

---

## 🖥️ Graphical launcher (GUI)

The GUI runs **on your host machine** — not inside Docker. It requires only Python's built-in `tkinter` and a working `docker` installation. No ROS, no Open3D, no heavy dependencies on the host.

### Start the GUI

```bash
# From the cloned repo root:
python3 gui.py
```

No virtual environment or `pip install` required.

### How it works

```
Your machine
├── python3 gui.py          ← runs natively (tkinter only)
│     │
│     ├─ Browse bag dir     → e.g. /home/user/bags/my_scan
│     ├─ Browse output dir  → e.g. /home/user/output
│     ├─ Fill parameters
│     │
│     ├─ [Check Topics] ──► docker run --rm
│     │                       -v /home/user/bags/my_scan:/data/bag:ro
│     │                       bellhop:latest preflight /data/bag /points /gps/fix
│     │                     (parses MISSING:/OK: lines, shows result inline)
│     │
│     └─ [Run Pipeline] ──► docker run --rm
│                             -v /home/user/bags/my_scan:/data/bag:ro
│                             -v /home/user/output:/data/output
│                             bellhop:latest tiles_3d /data/bag /data/output ...
│                           (streams stdout live into the log panel)
│
└── Docker container        ← all heavy computation happens here
```

The full `docker run` command is echoed in the log panel before execution so you can copy it for scripting.

### GUI layout

The left sidebar selects the pipeline profile. Each profile shows only the parameters relevant to that output. The interface provides:

- **Docker image tag** — defaults to `bellhop:latest`; change this if you have tagged builds (e.g. `bellhop:v1.2`)
- **Bag directory (host path)** — the folder produced by `ros2 bag record`; Browse button for convenience
- **Output directory (host path)** — where results are written; created automatically if it does not exist
- **Profile-specific parameters** — pre-filled with sensible defaults
- **Check Topics** — runs the pre-flight check inside a short-lived container; reports missing topics before committing to a full run
- **Run Pipeline** — assembles and fires the `docker run` command; streams stdout/stderr live into the log panel
- **Light / dark theme toggle** — top-right corner (◑)

### Path note

Enter **host-side absolute paths** in the Bag and Output fields (e.g. `/home/user/bags/run1`). The GUI automatically maps them to the fixed container paths `/data/bag` and `/data/output` when building the `docker run` command — you never need to think about container paths.

---

## 📊 Project layout

```
bellhop/
├── cli.py                        # Unified CLI entry point (runs inside container)
├── gui.py                        # Host-native graphical launcher (tkinter only)
├── Dockerfile
├── requirements.txt
├── pipelines/
│   ├── og_map.py
│   ├── mesh.py
│   ├── color_mesh.py
│   ├── gazebo_world.py
│   ├── tiles_3d.py
│   ├── color_tiles_3d.py
│   └── shared/
│       ├── preflight.py          # Topic existence check
│       ├── ros_io.py             # Bag reading helpers
│       ├── registration.py       # ICP pose-graph registration
│       └── reconstruction.py     # Poisson mesh + point-cloud cleaning
└── docs/                         # Per-pipeline reference docs (pre-merge)
```

---

## 🧠 Pipelines

All pipelines follow the same calling convention inside the container:

```
cli.py <pipeline> /data/bag /data/output [options]
```

### og_map — 2D Occupancy Grid

Produces a Nav2-compatible `.pgm` image and `.yaml` map file from a LiDAR scan.

**Algorithm:** loads odometry poses → ray-casts each point-cloud frame into a 3D OcTree → separates ground from obstacles using surface normals → builds a 2D ground-height map → projects obstacles into a 2D grid → denoises with morphological closing.

**Required topics:** PointCloud2, Odometry

| Option | Default | Description |
|---|---|---|
| `--pc_topic` | `/dlio/odom_node/pointcloud/deskewed` | PointCloud2 topic |
| `--odom_topic` | `/dlio/odom_node/odom` | Odometry topic |
| `--octree_res` | `0.1` | 3D OcTree resolution (m) |
| `--grid_res` | `0.10` | 2D grid resolution (m) |
| `--slope_deg` | `15.0` | Max slope angle (°) for ground classification |
| `--normal_radius` | `0.2` | Normal estimation radius (m) |
| `--z_min` | `0.1` | Min obstacle height above ground (m) |
| `--z_max` | `2.0` | Max obstacle height above ground (m) |
| `--voxel_size` | `0.05` | Voxel downsampling size (m); `0` disables |
| `--odom_max_latency` | `0.5` | Max tolerated timestamp gap between odom and cloud (s) |
| `--frame_stride` | `1` | Use every Nth cloud frame; `1` uses all frames |
| `--max_frames` | `0` | Maximum frames to process; `0` means unlimited |
| `--min_cluster_size` | `20` | Minimum obstacle cluster size (cells); `0` disables |
| `--closing_iters` | `1` | Morphological closing iterations |
| `--workers` | `4` | Parallel worker threads |

**Outputs:** `<output_base>.pgm`, `<output_base>.yaml`

---

### mesh — Poisson Surface Mesh

Registers LiDAR frames with ICP, merges them into a world-frame cloud, and runs Poisson surface reconstruction. Uses a two-pass streaming merge to keep memory usage bounded regardless of bag size.

**Required topics:** PointCloud2. Odometry is optional but improves registration.

| Option | Default | Description |
|---|---|---|
| `--pc_topic` | `points` | PointCloud2 topic |
| `--odom_topic` | _(none)_ | Odometry topic (optional) |
| `--voxel_size` | `0.10` | Voxel downsampling size (m) |
| `--min_frame_points` | `100` | Skip frames with fewer points than this |
| `--frame_stride` | `1` | Use every Nth cloud for registration and merging |
| `--max_registration_frames` | `500` | Maximum frames retained for ICP registration; `0` means unlimited |
| `--merge_chunk_frames` | `16` | Frames merged per batch before each voxel reduction |
| `--height_colormap` | _(none)_ | Also export a textured height false-color OBJ bundle; choices: `jet`, `hot`, `cool`, `gray` |
| `--height_texture_size` | `1024` | Height lookup texture size in pixels |

All [shared registration](#registration) and [shared reconstruction](#reconstruction) options also apply.

**Outputs:** `<stem>_cloud.ply`, `<stem>_mesh.ply`, `<stem>_mesh.obj`.

When `--height_colormap` is selected, Bellhop also writes:

- `<stem>_height_<colormap>.obj`
- `<stem>_height_<colormap>.mtl`
- `<stem>_height_<colormap>_texture.png`

---

### color_mesh — Camera-Colored Mesh

Extends `mesh` by projecting camera images onto each registered frame before reconstruction. Produces a textured mesh with per-vertex color.

**Required topics:** PointCloud2, Camera image, CameraInfo. Odometry is optional.

| Option | Default | Description |
|---|---|---|
| `--camera_topic` | _(none)_ | `sensor_msgs/Image` or `CompressedImage` topic |
| `--camera_info_topic` | _(none)_ | `sensor_msgs/CameraInfo` topic (required with `--camera_topic`) |
| `--max_time_diff` | `0.1` | Max timestamp gap between cloud frame and image (s) |
| `--color_min_depth` | `0.1` | Min projection depth (m) |
| `--color_max_depth` | _(none)_ | Max projection depth (m); blank = unlimited |
| `--gray_filter_radius` | `0.05` | Remove gray-fill points within this radius of a real-color point (m); `0` disables |

All `mesh` options also apply. **Outputs:** `<stem>_cloud.ply`, `<stem>_mesh.obj`

---

### gazebo_world — Gazebo Simulation World

Produces a Gazebo-ready static mesh (STL) plus `.sdf`, `model.config`, and `.world` files.

**Required topics:** PointCloud2. Odometry is optional.

| Option | Default | Description |
|---|---|---|
| `--model_name` | `bag_environment` | Gazebo model name |
| `--gazebo_material` | `Gazebo/Grey` | Gazebo material string (e.g. `Gazebo/White`, `Gazebo/Wood`, `Gazebo/Bricks`, `Gazebo/Grass`) |
| `--level_floor` | _(off)_ | Level the mesh to the ground plane before export |

All `mesh` registration and reconstruction options also apply.

**Outputs (inside `<output_dir>`):**
```
models/<model_name>/meshes/model.stl
models/<model_name>/model.config
models/<model_name>/model.sdf
worlds/<model_name>.world
```

---

### tiles_3d — Cesium 3D Tiles

Registers frames, converts the cloud from local ENU to ECEF using a GPS origin, and writes a `tileset.json` for Cesium streaming.

**Required topics:** PointCloud2, `sensor_msgs/NavSatFix`. Odometry is optional.

| Option | Default | Description |
|---|---|---|
| `--gps_topic` | `/gps/fix` | `sensor_msgs/NavSatFix` topic for GPS origin |
| `--workers` | `4` | Parallel workers for KDTree and `py3dtiles convert` |

All standard registration options also apply.

**Outputs:** `<output_dir>/tileset/tileset.json` + tile content files, `<stem>_cloud_enu.ply`

---

### color_tiles_3d — Colored Cesium 3D Tiles

Combines the camera-coloring logic of `color_mesh` with the georeferencing of `tiles_3d`. Output is a colored ECEF point cloud converted to a Cesium tileset — no mesh is produced.

**Required topics:** PointCloud2, NavSatFix, Camera image, CameraInfo. Odometry is optional.

All `tiles_3d` and `color_mesh` options apply.

**Outputs:** `<output_dir>/tileset/tileset.json` + tile content files, `<stem>_cloud_enu.ply`

---

## 🗘 Shared parameters

### Registration

These options are accepted by all pipelines that run ICP registration (`mesh`, `color_mesh`, `gazebo_world`, `tiles_3d`, `color_tiles_3d`):

| Option | Default | Description |
|---|---|---|
| `--voxel_size` | `0.05` | Voxel size for per-frame downsampling (m) |
| `--icp_dist_thresh` | `0.2` | ICP max correspondence distance (m) |
| `--icp_fitness_thresh` | `0.6` | ICP fitness score threshold; frames below this are skipped |
| `--odom_max_latency` | `0.5` | Max timestamp gap between odometry and cloud (s) |
| `--frame_stride` | `4` | Register every Nth input frame; `1` (or `0`) uses all frames |
| `--max_registration_frames` | `0` | Cap on frames passed to ICP after stride selection; `0` = unlimited |
| `--merge_chunk_frames` | `16` | Frames merged per batch before each voxel reduction in the streaming merge pass |
| `--enable_loop_closure` | _(off)_ | Enable pose-graph loop closure |
| `--loop_closure_radius` | `10.0` | Spatial search radius for loop closure candidates (m) |
| `--loop_closure_fitness_thresh` | `0.3` | Fitness threshold for accepting a loop closure edge |
| `--loop_closure_search_interval` | `10` | Check every N frames for loop closure candidates |
| `--workers` | `4` | Parallel worker threads |

> **Frame selection order:** frames are first thinned by `--frame_stride`, then capped by `--max_registration_frames`. The full (un-strided) frame list is used during the subsequent streaming merge pass, so merging is always complete regardless of stride.

### Reconstruction

These options control Poisson reconstruction and mesh post-processing for `mesh`, `color_mesh`, and `gazebo_world`:

| Option | Default | Description |
|---|---|---|
| `--poisson_depth` | _(Auto)_ | Poisson octree depth; omit for automatic selection capped at 11. Explicit depths ≥ 12 are permitted. |
| `--min_density_percentile` | `1.0` | Percentile of low-density vertices to trim after Poisson |
| `--distance_multiplier` | `3.0` | Adaptive trim threshold = this multiplier × mean point spacing (m); `0` disables adaptive trim |
| `--max_vertex_distance` | _(none)_ | Hard cap on vertex distance from input cloud (m); blank = unlimited |
| `--remesh` | `True` | Run isotropic remeshing + Laplacian smoothing (requires `pymeshlab`); pass `--no-remesh` to skip |
| `--remesh_smooth_iterations` | `5` | Laplacian smoothing iterations applied during remeshing |
| `--decimate_target` | _(none)_ | Triangle count target: `< 1.0` = fraction of current count; `≥ 1` = absolute count; blank = skip |
| `--curvature_percentile` | `80.0` | Protect the top N% of high-curvature faces from decimation |
| `--curvature_protect_rings` | `1` | Dilation rings added around protected high-curvature faces |
| `--level_floor` | _(off)_ | Rotate cloud so dominant ground plane is horizontal before reconstruction |

---

## ✈️ Pre-flight topic check

Before running any computationally heavy pipeline, bellhop verifies that all required topics are present in the bag. If a topic is missing, the run stops immediately with a clear error rather than failing partway through.

**From the CLI:**
```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  bellhop:latest preflight /data/bag /points /gps/fix
```

Output:
```
OK: /points
MISSING: /gps/fix
```

**From the GUI:** click **Check Topics** on any profile. The GUI runs the same `docker run ... preflight` command and shows the result inline — no separate terminal needed.

To list all topics available in your bag:
```bash
ros2 bag info /path/to/bag
```

---

## 🦿 Running without Docker

If you want to run pipelines directly on the host (e.g. for development):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `pyoctomap` (required by `og_map`) needs the system library `liboctomap-dev`. On Ubuntu:
> ```bash
> sudo apt install liboctomap-dev
> ```
>
> `--remesh` requires `pymeshlab` and `--decimate_target` requires `pyfqmr`. Both are included in `requirements.txt`.

Run any pipeline:
```bash
python cli.py mesh /path/to/bag ./output --pc_topic /points
```

The GUI can still be used on the host in this mode — it will call `docker run` as usual. The GUI itself never needs the pipeline dependencies installed.

---

## 📝 Per-pipeline documentation

Original per-pipeline README files from before the merge are preserved in `docs/`:

| File | Pipeline |
|---|---|
| `docs/bag_to_og_README.md` | `og_map` |
| `docs/bag_to_mesh_README.md` | `mesh` |
| `docs/bag_to_color_mesh.md` | `color_mesh` |
| `docs/bag_to_gazebo_world_README.md` | `gazebo_world` |
| `docs/bag_to_3Dtileset_README.md` | `tiles_3d` |
| `docs/bag_to_color_3Dtileset_README.md` | `color_tiles_3d` |
