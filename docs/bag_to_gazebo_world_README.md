# bag_to_gazebo_world

Convert a ROS 2 bag file containing 3D LiDAR scans into a ready-to-use Gazebo simulation world — automatically.

This tool reads point cloud data from a `.bag` file, registers all the frames into a single aligned map, generates a surface mesh using Poisson reconstruction, and exports the result as a complete Gazebo model with an `.stl` mesh, `model.sdf`, `model.config`, and a `.world` file.

---

## How It Works

1. **Reads** all `PointCloud2` messages (and optionally odometry) from a ROS 2 bag file
2. **Registers** each frame together using ICP + pose graph optimization to build a globally consistent map
3. **Cleans** the combined point cloud (voxel downsampling, outlier removal, DBSCAN clustering)
4. **Reconstructs** a surface mesh using Poisson reconstruction
5. **Centers** the mesh at the origin so it spawns correctly in Gazebo
6. **Exports** a complete Gazebo model directory and `.world` file

---

## Requirements

You only need **Docker** installed. All dependencies run inside the container.

- [Docker](https://docs.docker.com/get-docker/)
- A ROS 2 bag file containing a `PointCloud2` topic (e.g. from a LiDAR sensor)
- Gazebo installed on your host machine for simulation

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/gglaspell/bag_to_gazebo_world.git
cd bag_to_gazebo_world
```


### 2. Build the Docker image

```bash
docker build -t bag-to-gazebo-world .
```

This only needs to be done once (or after any code changes).

### 3. Prepare your data

Create a `data/` folder in the repo and place your ROS 2 bag folder inside it:

```
bag_to_gazebo_world/
└── data/
    └── my_scan/          ← your ROS 2 bag folder goes here
        ├── my_scan_0.db3
        └── metadata.yaml
```


### 4. Run the tool

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-gazebo-world \
  /app/data/my_scan \
  /app/data/output \
  --pc_topic /ouster/points
```

Replace `/ouster/points` with the name of your LiDAR topic. If you are unsure what topics are in your bag, see the [Checking Your Bag File](#checking-your-bag-file) section below.

### 5. Install the model into Gazebo

Copy the generated model into Gazebo's local model directory so it can be found automatically:

```bash
cp -r data/output/models/bag_environment ~/.gazebo/models/
```


### 6. Launch in Gazebo

```bash
gazebo data/output/worlds/bag_environment.world
```


---

## Output Structure

After running, the following files will be generated in your output directory:

```
data/output/
├── models/
│   └── bag_environment/
│       ├── model.config       ← Gazebo model metadata
│       ├── model.sdf          ← Gazebo model definition (material, collision, visual)
│       └── meshes/
│           └── model.stl      ← 3D surface mesh of the scanned environment
└── worlds/
    └── bag_environment.world  ← Gazebo world file (loads the model + sun + ground plane)
```


---

## All Options

| Argument | Default | Description |
| :-- | :-- | :-- |
| `bagpath` | *(required)* | Path to the ROS 2 bag folder |
| `outputdir` | *(required)* | Directory to write output files |
| `--model_name` | `bag_environment` | Name of the generated Gazebo model |
| `--gazebo_material` | `Gazebo/Grey` | Surface material applied in Gazebo (see [Materials](#gazebo-materials)) |
| `--pc_topic` | `points` | ROS 2 topic name for the `PointCloud2` messages |
| `--odom_topic` | *(none)* | Optional odometry topic for better frame registration |
| `--voxel_size` | `0.05` | Voxel downsampling size in metres. Smaller = more detail, slower |
| `--icp_dist_thresh` | `0.2` | ICP max correspondence distance in metres |
| `--icp_fitness_thresh` | `0.6` | Minimum ICP fitness score to accept a frame (0.0–1.0) |
| `--odom_max_latency` | `0.5` | Max odometry timestamp age in seconds before falling back to identity |
| `--poisson_depth` | `9` | Poisson reconstruction depth. Higher = more detail, slower (8–11 recommended) |
| `--min_density_percentile` | `1.0` | Removes the lowest density percentage of the mesh after reconstruction |
| `--max_vertex_distance` | `0.15` | Removes mesh vertices farther than this distance from any input point (metres) |
| `--decimate_target` | *(none)* | Reduce triangle count. Values ≤ 1.0 are a ratio (e.g. `0.25` = 25%). Values > 1 are an absolute count |
| `--level_floor` | `false` | Automatically detect and align the floor plane to Z=0 |
| `--enable_loop_closure` | `false` | Enable loop closure detection for longer trajectories |
| `--loop_closure_radius` | `10.0` | Search radius for loop closures in metres |
| `--loop_closure_fitness_thresh` | `0.3` | Minimum ICP fitness for a loop closure to be accepted |
| `--loop_closure_search_interval` | `10` | Check for loop closures every N frames |
| `--workers` | `4` | Number of parallel CPU workers for KDTree queries |


---

## Example Commands

### Basic scan (no odometry)

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-gazebo-world \
  /app/data/my_scan \
  /app/data/output \
  --pc_topic /velodyne_points
```


### With odometry and floor leveling

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-gazebo-world \
  /app/data/my_scan \
  /app/data/output \
  --pc_topic /ouster/points \
  --odom_topic /odom \
  --level_floor
```


### High detail with custom name and material

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-gazebo-world \
  /app/data/my_scan \
  /app/data/output \
  --pc_topic /ouster/points \
  --model_name bunker_interior \
  --gazebo_material Gazebo/Bricks \
  --poisson_depth 10 \
  --decimate_target 0.25 \
  --level_floor
```


---

## Gazebo Materials

The `--gazebo_material` argument sets the visual surface appearance of the mesh inside Gazebo. These are built-in Gazebo materials that require no additional files:


| Value | Appearance |
| :-- | :-- |
| `Gazebo/Grey` | Flat grey (default) |
| `Gazebo/White` | Bright white |
| `Gazebo/DarkGrey` | Dark grey |
| `Gazebo/Bricks` | Brick texture |
| `Gazebo/Wood` | Wood texture |
| `Gazebo/WoodFloor` | Wood floor texture |
| `Gazebo/CeilingTiled` | Tiled ceiling |
| `Gazebo/Grass` | Grass texture |


---

## Checking Your Bag File

If you are unsure what topics are available in your bag, you can inspect it using the `rosbags` tool:

```bash
pip install rosbags
rosbags-info /path/to/your/bag/
```

Look for a topic with the message type `sensor_msgs/msg/PointCloud2` — that is your `--pc_topic` value.

---

## Troubleshooting

**Nothing appears in Gazebo**
The model may be far from the origin. Make sure you are running the latest version of the script which automatically centers the mesh. Also verify the model was copied to `~/.gazebo/models/`.

**`Write STL failed: compute normals first`**
You are running an older version of the script. Rebuild the Docker image:

```bash
docker build --no-cache -t bag-to-gazebo-world .
```

**`OSError: libgomp.so.1: cannot open shared object file`**
Rebuild the Docker image with the latest Dockerfile which includes the `libgomp1` fix:

```bash
docker build --no-cache -t bag-to-gazebo-world .
```

**`Error: No messages found for topics`**
Your `--pc_topic` argument does not match the topic name in the bag. Use `rosbags-info` to check the correct topic name.

**`Registration failed — no frames were successfully registered`**
Try lowering `--icp_fitness_thresh` (e.g. `0.5`) or increasing `--icp_dist_thresh` (e.g. `0.3`).

**Mesh has holes or is incomplete**
Try increasing `--poisson_depth` (e.g. `10` or `11`) or increasing `--max_vertex_distance` (e.g. `0.25`).

---

## License

MIT License. See [LICENSE](LICENSE) for details.


