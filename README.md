# 🛎️ bellhop

**bellhop** converts ROS 2 bag files into spatial outputs (occupancy grids, surface meshes, Gazebo worlds, textured atlas meshes, and Cesium 3D Tiles) through a unified CLI and graphical launcher.

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
  - [texture_baking — Keyframe-Baked Textured Mesh](#texture_baking--keyframe-baked-textured-mesh)
- [Shared parameters](#shared-parameters)
  - [Point-cloud frame handling](#point-cloud-frame-handling)
  - [Registration](#registration)
  - [Reconstruction](#reconstruction)
- [Pre-flight topic check](#pre-flight-topic-check)
- [Running without Docker](#running-without-docker)
- [Per-pipeline documentation](#per-pipeline-documentation)

---

## 🦾 What it does

Each pipeline reads one or more topics from a ROS 2 bag and writes a ready-to-use spatial artifact. For point clouds that arrive in a moving sensor/base frame, five of the pipelines register frames using **odometry as the primary pose source**, with ICP available as an optional, tightly-gated local refinement (see [Registration](#registration)). All seven pipelines detect — or let you override — whether the point cloud is already in a global/fixed frame or needs that odometry-anchored transform (see [Point-cloud frame handling](#point-cloud-frame-handling)).

| Pipeline | Input topics | Output | Registers frames? |
|---|---|---|---|
| `og_map` | PointCloud2, Odometry | `.pgm` + `.yaml` (Nav2 map) | Odometry-only (no ICP) |
| `mesh` | PointCloud2, Odometry (opt.) | `.ply` point cloud + `.ply` mesh | Odom-anchored + optional ICP |
| `color_mesh` | PointCloud2, Camera, CameraInfo, Odometry (opt.) | `.ply` point cloud + `.ply` colored mesh | Odom-anchored + optional ICP |
| `gazebo_world` | PointCloud2, Odometry (opt.) | `.stl` + `.sdf` + `.world` | Odom-anchored + optional ICP |
| `tiles_3d` | PointCloud2, NavSatFix, Odometry (opt.) | `tileset.json` (Cesium) | Odom-anchored + optional ICP |
| `color_tiles_3d` | PointCloud2, NavSatFix, Camera, CameraInfo, Odometry (opt.) | Colored `tileset.json` (Cesium) | Odom-anchored + optional ICP |
| `texture_baking` | PointCloud2, Camera, CameraInfo, Odometry (required) | Keyframe-baked textured mesh (ATAK zip) | Odometry-only (no ICP) |

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

**Example — Poisson mesh (odom-anchored, with optional ICP refinement):**

```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  -v ~/output:/data/output \
  bellhop:latest mesh /data/bag /data/output \
    --pc_topic /points \
    --odom_topic /odom \
    --voxel_size 0.05 \
    --enable_icp_refinement
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

**Example — keyframe-baked textured mesh (ATAK zip):**

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

**Example — force local-frame handling on a raw (non-deskewed) point cloud:**

```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  -v ~/output:/data/output \
  bellhop:latest mesh /data/bag /data/output \
    --pc_topic /velodyne_points \
    --odom_topic /odom \
    --pc_frame_mode local
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
│     ├─ [Check Topics] ──► docker run --rm --entrypoint python
│     │                       -v /home/user/bags/my_scan:/data/bag:ro
│     │                       bellhop:latest -c "<inline check_topics(...) call>"
│     │                     (parses MISSING:/OK output, shows result inline)
│     │
│     └─ [Run Pipeline] ──► docker run --rm
│                             -v /home/user/bags/my_scan:/data/bag:ro
│                             -v /home/user/output:/data/output
│                             bellhop:latest tiles_3d /data/bag /data/output ...
│                           (streams stdout live into the log panel)
│
└── Docker container        ← all heavy computation happens here
```

> **Note:** `preflight` is **not** a `bellhop` pipeline subcommand — there is no `bellhop:latest preflight ...` invocation. The GUI's **Check Topics** button instead runs a short inline Python snippet inside the container (`--entrypoint python ... -c "..."`) that imports `pipelines.shared.preflight.check_topics()` directly and prints `OK` or `MISSING: <topics>`. See [Pre-flight topic check](#pre-flight-topic-check) for the equivalent manual command.

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

`Point cloud frame mode` (`pc_frame_mode`) is available on every profile. `Enable ICP refinement`, its correction-bound fields, and `LC temporal window` only appear on the five profiles that actually register frames (`Mesh`, `Color Mesh`, `Gazebo World`, `3D Tiles`, `Color Tiles`) — `OG Map` and `Texture Baking` have no ICP/loop-closure step, so those fields are intentionally absent from their forms.

The `Mesh` profile's `Height false-color` field applies a per-vertex colored PLY export (see [mesh — Poisson Surface Mesh](#mesh--poisson-surface-mesh)); there is no separate texture-size field, since colors are baked directly onto mesh vertices rather than into a UV texture.

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
│   ├── texture_baking.py
│   ├── atlas_pipeline/               # texture_baking's UV-atlas bake pipeline
│   │   ├── keyframeselector.py
│   │   ├── pointcloudutils.py
│   │   ├── meshgenerator.py
│   │   ├── viewassignment.py
│   │   ├── visibilityfilter.py
│   │   ├── atlaspacker.py
│   │   ├── texturebaker.py
│   │   └── common/
│   │       ├── trajectory.py         # Trajectory CSV load/interp + pose lookups
│   │       ├── projection.py         # World<->optical projection helpers
│   │       └── packaging.py          # ATAK zip / workspace setup
│   └── shared/
│       ├── preflight.py              # Topic existence check
│       ├── ros_io.py                 # Bag reading, odom pose interpolation, frame detection
│       ├── registration.py           # Odom-anchored registration + optional ICP refinement
│       ├── reconstruction.py         # Poisson mesh + point-cloud cleaning
│       ├── mesh_utils.py             # Height false-color, baked as per-vertex PLY colors
│       └── tiles_common.py           # Shared bag-reading/merge/ECEF export for tiles pipelines
└── docs/                              # Per-pipeline reference docs (pre-merge)
```

---

## 🧠 Pipelines

All pipelines follow the same calling convention inside the container:

```
cli.py <pipeline> /data/bag /data/output [options]
```

### og_map — 2D Occupancy Grid

Produces a Nav2-compatible `.pgm` image and `.yaml` map file from a LiDAR scan.

**Algorithm:** loads odometry poses (interpolated between bracketing samples, not nearest-neighbor) → ray-casts each point-cloud frame into a 3D OcTree → separates ground from obstacles using surface normals → builds a 2D ground-height map → projects obstacles into a 2D grid → denoises with morphological closing. Odometry is this pipeline's sole pose source; there is no ICP step.

**Required topics:** PointCloud2, Odometry

| Option | Default | Description |
|---|---|---|
| `--pc_topic` | `/dlio/odom_node/pointcloud/deskewed` | PointCloud2 topic |
| `--odom_topic` | `/dlio/odom_node/odom` | Odometry topic |
| `--pc_frame_mode` | `auto` | `auto`/`global`/`local` — see [Point-cloud frame handling](#point-cloud-frame-handling). `local` transforms each frame into the world frame via an odom pose before OcTree insertion; `global` (the original, still-default-detected behavior) passes points through unchanged |
| `--octree_res` | `0.05` | 3D OcTree resolution (m) |
| `--grid_res` | `0.05` | 2D grid resolution (m) |
| `--slope_deg` | `15.0` | Max slope angle (°) for ground classification |
| `--normal_radius` | `0.2` | Normal estimation radius (m) |
| `--z_min` | `0.1` | Min obstacle height above ground (m) |
| `--z_max` | `1.0` | Max obstacle height above ground (m) |
| `--voxel_size` | `0.05` | Voxel downsampling size (m); `0` disables |
| `--odom_max_latency` | `0.5` | Max tolerated timestamp gap between odom and cloud (s); frames outside this are dropped and counted in the log |
| `--frame_stride` | `1` | Use every Nth cloud frame; `1` uses all frames |
| `--max_frames` | `0` | Maximum frames to process; `0` means unlimited |
| `--min_cluster_size` | `20` | Minimum obstacle cluster size (cells); `0` disables |
| `--closing_iters` | `1` | Morphological closing iterations |
| `--workers` | `4` | Parallel worker threads |

**Outputs:** `<output_base>.pgm`, `<output_base>.yaml`

---

### mesh — Poisson Surface Mesh

Registers LiDAR frames using odometry as the primary pose source (with optional, tightly-gated ICP refinement), merges them into a world-frame cloud, and runs Poisson surface reconstruction. Uses a two-pass streaming merge to keep memory usage bounded regardless of bag size.

**Required topics:** PointCloud2. Odometry is required unless `--pc_frame_mode` resolves to `global` (point cloud already in a fixed frame).

| Option | Default | Description |
|---|---|---|
| `--pc_topic` | `points` | PointCloud2 topic |
| `--odom_topic` | _(none)_ | Odometry topic — required unless the point cloud is already in a global/fixed frame |
| `--pc_frame_mode` | `auto` | See [Point-cloud frame handling](#point-cloud-frame-handling) |
| `--voxel_size` | `0.10` | Voxel downsampling size (m) |
| `--min_frame_points` | `100` | Skip frames with fewer points than this |
| `--frame_stride` | `1` | Use every Nth cloud for registration and merging |
| `--max_registration_frames` | `0` | Maximum frames retained for registration; `0` means unlimited (odom-anchored pose lookup is cheap, so no artificial cap is needed) |
| `--merge_chunk_frames` | `16` | Frames merged per batch before each voxel reduction |
| `--height_colormap` | _(none)_ | Also export a height-colored PLY with colors baked directly onto per-vertex colors (no UV texture); choices: `jet`, `hot`, `cool`, `gray` |

All [shared registration](#registration) and [shared reconstruction](#reconstruction) options also apply.

**Outputs:** `<stem>_cloud.ply` (point cloud), `<stem>_mesh.ply` (mesh). No `.obj` mesh file is written.

When `--height_colormap` is selected, Bellhop also writes:

- `<stem>_height_<colormap>.ply` — the same mesh with per-vertex RGB baked in from the Z-height colormap (lowest vertex → start of the ramp, highest vertex → end). No `.mtl` or texture PNG is produced; the color lives directly on the mesh vertices.

---

### color_mesh — Camera-Colored Mesh

Extends `mesh` by projecting camera images onto each registered frame before reconstruction. Produces a textured mesh with per-vertex color.

**Required topics:** PointCloud2, Camera image, CameraInfo. Odometry is required unless `--pc_frame_mode` resolves to `global`.

| Option | Default | Description |
|---|---|---|
| `--camera_topic` | _(none)_ | `sensor_msgs/Image` or `CompressedImage` topic |
| `--camera_info_topic` | _(none)_ | `sensor_msgs/CameraInfo` topic (required with `--camera_topic`) |
| `--pc_frame_mode` | `auto` | See [Point-cloud frame handling](#point-cloud-frame-handling) |
| `--max_time_diff` | `0.1` | Max timestamp gap between cloud frame and image (s) |
| `--color_min_depth` | `0.1` | Min projection depth (m) |
| `--color_max_depth` | _(none)_ | Max projection depth (m); blank = unlimited |
| `--gray_filter_radius` | `0.05` | Remove gray-fill points within this radius of a real-color point (m); `0` disables |

All `mesh` options also apply. **Outputs:** `<stem>_colored_cloud.ply` (point cloud), `<stem>_colored_mesh.ply` (mesh with native per-vertex PLY colors). No `.obj` mesh file is written — PLY has a native per-vertex-color field, so colors round-trip cleanly without the non-standard OBJ vertex-color extension that many viewers ignored.

---

### gazebo_world — Gazebo Simulation World

Produces a Gazebo-ready static mesh (STL) plus `.sdf`, `model.config`, and `.world` files.

**Required topics:** PointCloud2. Odometry is required unless `--pc_frame_mode` resolves to `global`.

| Option | Default | Description |
|---|---|---|
| `--pc_frame_mode` | `auto` | See [Point-cloud frame handling](#point-cloud-frame-handling) |
| `--model_name` | `bag_environment` | Gazebo model name |
| `--gazebo_material` | `Gazebo/Grey` | Gazebo material string (e.g. `Gazebo/White`, `Gazebo/Wood`) |
| `--level_floor` | _(off)_ | Level the mesh to the ground plane before export |

All [shared registration](#registration) and [shared reconstruction](#reconstruction) options also apply.

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

**Required topics:** PointCloud2, `sensor_msgs/NavSatFix`. Odometry is required unless `--pc_frame_mode` resolves to `global`.

| Option | Default | Description |
|---|---|---|
| `--gps_topic` | `/gps/fix` | `sensor_msgs/NavSatFix` topic for GPS origin |
| `--pc_frame_mode` | `auto` | See [Point-cloud frame handling](#point-cloud-frame-handling) |
| `--workers` | `4` | Parallel workers for KDTree and `py3dtiles convert` |

All [shared registration](#registration) options also apply.

**Outputs:** `<output_dir>/tileset/tileset.json` + tile content files, `<stem>_cloud_enu.ply`

---

### color_tiles_3d — Colored Cesium 3D Tiles

Combines the camera-coloring logic of `color_mesh` with the georeferencing of `tiles_3d`. Output is a colored ECEF point cloud converted to a Cesium tileset — no mesh is produced.

**Required topics:** PointCloud2, NavSatFix, Camera image, CameraInfo. Odometry is required unless `--pc_frame_mode` resolves to `global`.

All `tiles_3d` and `color_mesh` options apply, including `--pc_frame_mode`.

**Outputs:** `<output_dir>/tileset/tileset.json` + tile content files, `<stem>_cloud_enu.ply`

---

### texture_baking — Keyframe-Baked Textured Mesh

Selects sharp, well-spaced camera keyframes from the bag, merges and frustum-filters the point cloud against those keyframes, runs Poisson reconstruction, culls invisible faces, assigns mesh faces to the best keyframe view, packs a UV atlas, and bakes a texture — producing a self-contained ATAK-compatible zip. This pipeline has no ICP/registration step; odometry (via a trajectory CSV built up front) is its sole pose source.

**Required topics:** PointCloud2, Camera image, CameraInfo, Odometry. Unlike the other pipelines, `--odom_topic` is **required unconditionally** here (not optional), since keyframe selection and view assignment both depend on it directly.

| Option | Default | Description |
|---|---|---|
| `--pc_topic` | `points` | PointCloud2 topic |
| `--camera_topic` | _(required)_ | `sensor_msgs/Image` or `CompressedImage` topic |
| `--camera_info_topic` | _(required)_ | `sensor_msgs/CameraInfo` topic |
| `--odom_topic` | _(required)_ | Odometry topic |
| `--pc_frame_mode` | `auto` | See [Point-cloud frame handling](#point-cloud-frame-handling). `local` transforms each point-cloud frame into the world frame via an odom pose (interpolated from the trajectory CSV) before frustum-visibility filtering, meshing, and normal orientation |
| `--odom_max_latency` | `0.5` | Only used when `--pc_frame_mode` resolves to `local`: max seconds a frame's timestamp may fall outside trajectory coverage before it's dropped |
| `--min_frame_points` | `100` | Skip frames with fewer points than this |
| `--voxel_size` | `0.05` | Voxel size used when subsampling the merged cloud |
| `--ror_radius` | `0.0` | Range-adaptive radius outlier removal base radius (m); `0` disables |
| `--ror_min_neighbors` | `10` | ROR minimum neighbor count |
| `--sor_neighbors` | `20` | Statistical outlier removal neighbor count |
| `--sor_std_ratio` | `2.0` | SOR standard deviation ratio |
| `--min_movement_m` | `0.5` | Minimum camera movement (m) to accept a new keyframe |
| `--min_rotation_deg` | `15.0` | Minimum camera rotation (°) to accept a new keyframe |
| `--poisson_depth` | `8` | Poisson octree depth |
| `--poisson_max_distance` | `0.5` | Poisson reconstruction max point distance (m) |
| `--smooth_method` | `taubin` | `taubin` or `laplacian` mesh smoothing |
| `--smooth_iterations` | `5` | Smoothing iterations |
| `--smooth_lambda` | `0.5` | Smoothing lambda factor |
| `--cull_min_angle` | `75.0` | Minimum view angle (°) to keep a face as visible |
| `--target_faces` | _(none)_ | Optional target triangle count after culling |
| `--assign_min_angle` | `75.0` | Minimum view angle (°) for face-to-keyframe assignment |
| `--max_bake_distance` | `4.0` | Maximum camera distance (m) eligible for baking a face |
| `--min_bake_distance` | `0.4` | Minimum camera distance (m) eligible for baking a face |
| `--assignment_smooth_iterations` | `3` | Smoothing passes over the face-to-view assignment map |
| `--atlas_size` | `8192` | UV atlas texture size (px) |
| `--overwrite` | _(off)_ | Overwrite an existing output workspace |

**Outputs:** `<stem>_baked_mesh.obj`, `<stem>_baked_mesh_texture.png`, and an ATAK-compatible zip bundling both, plus intermediate files (`trajectory.csv`, `keyframes.csv`, keyframe images) under the output workspace.

---

## 🗘 Shared parameters

### Point-cloud frame handling

**All seven pipelines** now detect — or let you override — whether `--pc_topic` is already published in a global/fixed frame or needs an odometry-anchored transform:

```
--pc_frame_mode {auto,global,local}   (default: auto)
```

- **`auto`** (default): peeks the first message on `--pc_topic`, classifies its `frame_id` as **global** if it's `odom`, `map`, or `world` (case-insensitive, leading `/` tolerated) or **local** otherwise (e.g. `base_link`, a lidar/camera frame, or empty), and prints what it found: `Point cloud frame_id: '<id>' -> <global|local>`.
- **`global`**: no per-frame transform is applied — points are streamed/filtered/downsampled/merged directly. Odometry (if provided) is still used for view-ray normal orientation or camera-origin lookups where relevant.
- **`local`**: each frame is transformed into the world/fixed frame using an odometry-derived pose before it's used for anything else. In `mesh`, `color_mesh`, `gazebo_world`, `tiles_3d`, and `color_tiles_3d` this pose additionally feeds the full odom-anchored registration pass described below; in `og_map` and `texture_baking` (which have no registration step) it's a direct odometry transform with no ICP involved.

Use the override when a bag's `frame_id` is missing, wrong, or empty — for example, force `local` to run directly against a raw, non-deskewed sensor-frame point cloud instead of requiring a pre-deskewed one.

> **Caveat:** this only detects whether the cloud is already in a fixed/global frame. It does **not** correct for a real sensor-to-base_link extrinsic offset (lever arm) — odometry only carries the transform from a fixed frame to the robot's base frame, never static sensor extrinsics. Fixing a lever-arm offset requires a separate static transform (e.g. from `tf_static`), which none of these pipelines currently apply.

### Registration

These options are accepted by the five pipelines that register frames (`mesh`, `color_mesh`, `gazebo_world`, `tiles_3d`, `color_tiles_3d`). **Odometry is the primary pose source for every frame** — a frame is included whenever it falls within `--odom_max_latency` of odometry coverage, looked up by interpolating between the two bracketing odometry samples (linear translation + SLERP rotation), never by nearest-neighbor snapping. Registration/ICP quality is never the reason a frame is dropped; only missing odometry coverage is, and that's always logged loudly with a frame count, never silently.

ICP is available as an **optional, strongly-gated local refinement** of the odometry pose (off by default). When enabled, it's accepted only if its fitness clears `--icp_fitness_thresh` *and* its correction stays within the configured translation/rotation bounds — otherwise the raw odometry pose is kept unchanged. ICP can nudge an already-valid pose or be a no-op; it can never remove a frame from the merge.

| Option | Default | Description |
|---|---|---|
| `--voxel_size` | `0.05`–`0.10` (pipeline-dependent) | Voxel size for per-frame downsampling (m) |
| `--odom_max_latency` | `0.5` | Max timestamp gap between odometry and a frame before that frame is dropped (s) |
| `--enable_icp_refinement` | _(off)_ | Turn on optional local ICP refinement of the odometry pose |
| `--icp_dist_thresh` | `0.2` | ICP max correspondence distance (m) |
| `--icp_fitness_thresh` | `0.7` | Fitness bar for *accepting* an ICP refinement (raised from a legacy `0.6`: this now gates a correction, not the primary motion estimate) |
| `--max_icp_translation_correction` | `0.3` | Max allowed ICP correction translation (m) relative to the odometry guess |
| `--max_icp_rotation_correction_deg` | `15.0` | Max allowed ICP correction rotation (°) relative to the odometry guess |
| `--enable_loop_closure` | _(off)_ | Enable pose-graph loop closure (only takes effect if `--enable_icp_refinement` finds a candidate pair) |
| `--loop_closure_radius` | `10.0` | Spatial search radius for loop closure candidates (m) |
| `--loop_closure_fitness_thresh` | `0.7` | Fitness threshold for accepting a loop-closure edge — defaults to the **same** bar as `--icp_fitness_thresh`, not a separate looser value, so a low-confidence loop match can't sneak in |
| `--loop_closure_search_interval` | `10` | Check every N frames for loop-closure candidates |
| `--loop_closure_temporal_window` | `100` | Bounded number of most-recent candidate frames considered for loop closure (memory-safe; never grows with bag length) |
| `--frame_stride` | `1` | Register every Nth input frame; odometry-anchored pose lookup is cheap, so `1` (all frames) is now the default |
| `--max_registration_frames` | `0` | Cap on frames after stride selection; `0` = unlimited |
| `--merge_chunk_frames` | `16` | Frames merged per batch before each voxel reduction in the streaming merge pass |
| `--workers` | `1`–`4` (pipeline-dependent) | Parallel worker threads |

> **Frame selection order:** frames are first thinned by `--frame_stride`, then capped by `--max_registration_frames`, then filtered by odometry coverage (`--odom_max_latency`). Every stage's frame count is printed (read → selected → with valid pose → merged) so a coverage problem is visible in the console output, not hidden behind a generic "done" message.

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

Before running any computationally heavy pipeline, bellhop can verify that all required topics are present in the bag, so a run fails immediately with a clear error rather than partway through.

There is **no dedicated `preflight` pipeline subcommand** in `cli.py`. The check is the `check_topics()` function in `pipelines/shared/preflight.py`, invoked directly:

**From the CLI (equivalent to what the GUI runs):**
```bash
docker run --rm \
  -v ~/bags/my_scan:/data/bag:ro \
  --entrypoint python \
  bellhop:latest -c "
from pathlib import Path
from pipelines.shared.preflight import check_topics
missing = check_topics(Path('/data/bag'), ['/points', '/gps/fix'])
print('MISSING: ' + ', '.join(missing) if missing else 'OK')
"
```

**From the GUI:** click **Check Topics** on any profile. The GUI runs the same inline check internally and shows the result next to the button — no separate terminal needed.

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
> `--remesh` requires `pymeshlab` and `--decimate_target` requires `pyfqmr`. Both are included in `requirements.txt`. Odometry pose interpolation (SLERP) uses `scipy.spatial.transform`, already required by the existing rotation handling — no new dependency was added.

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
| _(none yet)_ | `texture_baking` — this pipeline exists in `pipelines/` and is documented above, but predates/postdates the original per-pipeline doc set and has no standalone legacy file in `docs/`. Consider adding `docs/bag_to_texture_baking_README.md` if you want parity with the others. |

---

## 🔄 Recent changes

- **`mesh`/`color_mesh` no longer produce `.obj` mesh files.** Both now write only a point-cloud `.ply` and a mesh `.ply`. PLY natively supports per-vertex color, so `color_mesh`'s colored mesh no longer relies on Open3D's non-standard OBJ vertex-color extension that many viewers (e.g. Blender's default OBJ importer) silently ignored.
- **`mesh`'s optional height false-color export (`--height_colormap`) is now a per-vertex-colored PLY**, not a UV-textured OBJ+MTL+PNG bundle. `mesh_utils.apply_height_colormap()` was rewritten accordingly: it computes a colormap value per vertex directly from Z-height and writes it straight into the mesh's PLY vertex-color field — no texture, UV coordinates, or material file are produced. As a result, `--height_texture_size` was removed (there is no texture/LUT left to size) from both the CLI and the GUI's Mesh profile.
- **Odom-anchored registration** (`mesh`, `color_mesh`, `gazebo_world`, `tiles_3d`, `color_tiles_3d`): odometry replaced ICP as the primary pose source, fixing two failure modes in the old ICP-primary design — silent coverage collapse (the ICP chain breaking and never recovering, silently shrinking the merged output) and loop-closure overcorrection (a looser loop-closure fitness bar letting perceptually-similar-but-different locations warp the whole pose graph). See [Registration](#registration) for the new/changed flags and defaults.
- **Point-cloud frame detection** (`--pc_frame_mode`, all seven pipelines): every pipeline now detects or lets you override whether the point cloud is already in a global/fixed frame, closing a gap where `og_map` and `texture_baking` previously assumed this with no runtime check. See [Point-cloud frame handling](#point-cloud-frame-handling).
- **`texture_baking` documented for the first time** in this README — it existed in the codebase but was previously missing from these docs entirely.
- **GUI**: all new flags wired into the parameter forms (scoped correctly — ICP/loop-closure fields only appear on the five pipelines that actually register frames); stale GUI defaults that had drifted from the pipelines' actual argparse defaults were corrected (`icp_fitness_thresh`/`loop_closure_fitness_thresh` 0.6→0.7, `loop_closure_radius` 3.0→10.0, `frame_stride` 2→1); the Mesh profile's height false-color field was updated to match the PLY-only export and its now-removed texture-size field.
- **Removed as dead code**: the old ICP-primary `run_icp_posegraph()`, its `detect_loop_closure()` helper, and the never-called `iter_registered_frame_chunks()` (all in `shared/registration.py`); backward-compatibility re-exports in `tiles_3d.py` (`_run_py3dtiles_convert`, `_write_ply_ecef`, `_write_colored_ply_ecef`) whose only consumer already imported from `shared/tiles_common` directly; `og_map`'s nearest-neighbor-only `_nearest_index()` helper, superseded by the shared interpolating odometry lookup; `mesh_utils.py`'s `_write_obj()` OBJ writer and its `PIL.Image`/texture-baking code, superseded by the per-vertex PLY color path.
- **Corrected in this pass**: the "Pre-flight topic check" section previously documented a `bellhop:latest preflight ...` subcommand that does not exist in `cli.py` — it's now documented as the inline Python invocation the GUI actually runs. The project layout tree was also updated to include previously-undocumented modules (`mesh_utils.py`, `tiles_common.py`, the whole `atlas_pipeline/` package).
