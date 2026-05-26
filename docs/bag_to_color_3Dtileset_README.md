# ROS 2 Bag to RGB 3D Tileset Converter

Converts a ROS 2 bag file into a georeferenced, optionally RGB-coloured
3D Tiles tileset (`tileset.json` + `*.pnts`).

## Files

| File | Purpose |
|---|---|
| `bag_to_tileset.py` | Core conversion script (runs inside Docker) |
| `bag_to_tileset_gui.py` | Tkinter GUI — builds and launches the Docker command |
| `Dockerfile` | Container definition for the pipeline |

## Quick Start

```bash
# 1. Build the image
docker build -t bag-to-tileset .

# 2. Run (XYZ-only, no camera)
docker run --rm \
  -v "$(pwd)/data/input:/bag" \
  -v "$(pwd)/data/output:/output" \
  bag-to-tileset \
  /bag/your_bag.mcap /output \
  --pc_topic /points \
  --gps_topic /gps/fix

# 3. Run with RGB colorization
docker run --rm \
  -v "$(pwd)/data/input:/bag" \
  -v "$(pwd)/data/output:/output" \
  bag-to-tileset \
  /bag/your_bag.mcap /output \
  --pc_topic /points \
  --gps_topic /gps/fix \
  --odom_topic /odom \
  --camera_topic /camera/image_raw \
  --camera_info_topic /camera/camera_info \
  --max_time_diff 0.2 \
  --color_min_depth 0.2 \
  --gray_filter_radius 0.05
```

## All Parameters

### Required (positional)

| Argument | Description |
|---|---|
| `bagpath` | Path to the ROS 2 `.bag` / `.db3` / `.mcap` file |
| `outputdir` | Output directory for `tileset.json` and `*.pnts` |

### ROS Topics

| Flag | Default | Description |
|---|---|---|
| `--pc_topic` | `/points` | `sensor_msgs/PointCloud2` topic |
| `--gps_topic` | `/gps/fix` | `sensor_msgs/NavSatFix` topic for georeferencing |
| `--odom_topic` | *(none)* | `nav_msgs/Odometry` topic — omit if unavailable |
| `--camera_topic` | *(none)* | `sensor_msgs/Image` or `CompressedImage` — omit to disable colorization |
| `--camera_info_topic` | *(none)* | `sensor_msgs/CameraInfo` — required when `--camera_topic` is set |

### Camera Colorization

| Flag | Default | Description |
|---|---|---|
| `--max_time_diff` | `0.1` | Max camera–lidar timestamp gap in seconds |
| `--color_min_depth` | `0.1` | Min projection depth (m); closer points get gray fill |
| `--color_max_depth` | *(none)* | Max projection depth (m); no limit if omitted |
| `--gray_filter_radius` | `0.05` | Remove gray fill points within this radius of coloured points (m) |

### Registration (ICP)

| Flag | Default | Description |
|---|---|---|
| `--voxel_size` | `0.05` | Downsampling voxel size (m) |
| `--icp_dist_thresh` | `0.2` | ICP max correspondence distance (m) |
| `--icp_fitness_thresh` | `0.6` | Min ICP fitness to accept a frame `[0–1]` |
| `--odom_max_latency` | `0.5` | Max odometry–pointcloud timestamp gap (s) |

### Loop Closure

| Flag | Default | Description |
|---|---|---|
| `--enable_loop_closure` | *(off)* | Enable RANSAC+ICP loop closure detection |
| `--loop_closure_radius` | `5.0` | Spatial search radius for loop closure candidates (m) |
| `--loop_closure_fitness_thresh` | `0.3` | Min ICP fitness to accept a loop closure edge |
| `--loop_closure_search_interval` | `10` | Check for loop closures every N frames |

### Outlier Removal & Cleaning

| Flag | Default | Description |
|---|---|---|
| `--ror_nb_points` | `6` | ROR: min neighbours within radius |
| `--ror_radius` | `0.5` | ROR: search radius (m) |
| `--sor_nb_neighbors` | `20` | SOR: neighbour count for mean-distance estimate |
| `--sor_std_ratio` | `2.0` | SOR: std-deviation multiplier for outlier threshold |
| `--dbscan_eps` | `0.5` | DBSCAN epsilon (m) — set `0` to disable |
| `--dbscan_min_points` | `10` | DBSCAN minimum cluster size |

### Output Options

| Flag | Default | Description |
|---|---|---|
| `--level_floor` | *(off)* | Attempt RANSAC-based floor leveling before export |
| `--workers` | `4` | py3dtiles worker threads |

## GUI

Launch the GUI on any machine with Python 3 and Tkinter installed
(Docker is still used for the conversion itself):

```bash
python3 bag_to_tileset_gui.py
```

The GUI mirrors all parameters above and provides a live command
preview plus a colour-coded output log.

## Important Notes

- GPS averaging assumes all fixes are near the same meridian.
  Bags spanning the antimeridian will produce incorrect georeferencing.
- The camera pose is assumed to be coincident with the LiDAR sensor pose.
  If your camera is offset from the LiDAR, add a rigid extrinsic transform
  argument or TF lookup before using this in production.
