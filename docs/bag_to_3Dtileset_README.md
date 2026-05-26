# ROS 2 Bag to 3D Tileset Converter

Convert ROS 2 bag files containing LiDAR point clouds and GPS data into georeferenced **3D Tiles** (PNTS format) ready for streaming in CesiumJS or any 3D Tiles-compatible viewer.

---

## 📋 What This Does

This tool transforms ROS 2 bag files with `sensor_msgs/PointCloud2` and `sensor_msgs/NavSatFix` data into a georeferenced **3D Tiles point cloud** (`tileset.json` + `*.pnts` files).

**Design Philosophy:** Accuracy over speed. Built for offline post-processing where quality matters more than real-time performance.

### Key Capabilities
- **Point-to-Plane ICP Registration** — Sub-centimeter alignment accuracy
- **Global Pose Graph Optimization** — Corrects accumulated drift across entire trajectories
- **Odometry Integration** — Uses wheel/visual odometry for better initial alignment
- **Loop Closure Detection** (Optional) — Finds and constrains revisited areas with FPFH feature matching
- **GPS Georeferencing** — Anchors the local ENU map to WGS84 via `sensor_msgs/NavSatFix`
- **Z-Axis Leveling** — Prevents vertical drift in planar environments
- **Adaptive Frame Filtering** — Automatically skips low-quality registrations

---

## 🛠 What You'll Need

### Required
- **Docker** ([Install Docker](https://docs.docker.com/get-docker/))
- **ROS 2 Bag File** with:
  - `sensor_msgs/PointCloud2` messages (LiDAR)
  - `sensor_msgs/NavSatFix` messages (GPS — required for georeferencing)
- **8GB+ RAM** (16GB+ recommended for large datasets)

### Optional (Improves Results)
- Odometry topic (`nav_msgs/Odometry`) — For better initial ICP alignment
- ~5–10 GB free disk space per hour of bag data

### Python Dependencies (installed automatically in Docker)
`rosbags` · `open3d` · `numpy` · `scipy` · `tqdm` · `pyproj` · `py3dtiles`

---

## 🚀 Getting Started

### 1. Set Up Your Project

```bash
mkdir bag-to-tileset && cd bag-to-tileset
# Place Dockerfile and bag_to_tileset.py here
```

### 2. Build the Docker Image

```bash
docker build -t bag-to-tileset .
```

This downloads dependencies (~2.5 GB) and takes 5–10 minutes on first build.

### 3. Organize Your Data

```bash
mkdir -p data/input data/output
# Copy your ROS 2 bag file to data/input/
```

### 4. Run Your First Conversion

The Docker container mounts two directories:
- The **bag file's parent directory** → `/bag` inside the container
- Your chosen **output directory** → `/output` inside the container

**Basic command** (assumes default topic names):

```bash
docker run --rm \
  -v "$(pwd)/data/input:/bag" \
  -v "$(pwd)/data/output:/output" \
  bag-to-tileset \
  /bag/your_bag_file \
  /output
```

**Recommended command** (with odometry and custom topics):

```bash
docker run --rm \
  -v "$(pwd)/data/input:/bag" \
  -v "$(pwd)/data/output:/output" \
  bag-to-tileset \
  /bag/your_bag_file \
  /output \
  --pc_topic /your/pointcloud/topic \
  --odom_topic /your/odom/topic \
  --gps_topic /gps/fix \
  --icp_fitness_thresh 0.4 \
  --level_floor
```

> **GUI Users:** The `bag_to_tileset_gui.py` launcher automatically constructs the correct volume mounts from the paths you select — no manual path editing required.

### 5. View Your Results

Output files will be in `data/output/`:
- `your_bag_file_cloud_ecef.ply` — Georeferenced point cloud in ECEF coordinates
- `tileset.json` — 3D Tiles entry point (load this in CesiumJS)
- `*.pnts` — Tiled point cloud data files

**Load in CesiumJS:**

```javascript
const tileset = await Cesium.Cesium3DTileset.fromUrl(
  "http://your-server/output/tileset.json"
);
viewer.scene.primitives.add(tileset);
```

---

## 💡 Common Use Cases

### Outdoor Survey with GPS (Typical)

```bash
docker run --rm \
  -v "$(pwd)/data/input:/bag" \
  -v "$(pwd)/data/output:/output" \
  bag-to-tileset \
  /bag/outdoor_survey \
  /output \
  --pc_topic /velodyne_points \
  --odom_topic /odom \
  --gps_topic /gps/fix \
  --icp_fitness_thresh 0.3 \
  --icp_dist_thresh 0.5 \
  --voxel_size 0.1 \
  --level_floor
```

---

### Outdoor Mapping with DLIO/LIO-SAM

```bash
docker run --rm \
  -v "$(pwd)/data/input:/bag" \
  -v "$(pwd)/data/output:/output" \
  bag-to-tileset \
  /bag/outdoor_survey \
  /output \
  --pc_topic /dlio/odom_node/pointcloud/deskewed \
  --odom_topic /dlio/odom_node/odom \
  --gps_topic /fix \
  --icp_fitness_thresh 0.2 \
  --icp_dist_thresh 0.5 \
  --level_floor \
  --voxel_size 0.1
```

---

### Large Campus/Warehouse (with Loop Closure)

```bash
docker run --rm \
  -v "$(pwd)/data/input:/bag" \
  -v "$(pwd)/data/output:/output" \
  bag-to-tileset \
  /bag/warehouse_full \
  /output \
  --pc_topic /velodyne_points \
  --odom_topic /integrated_odom \
  --gps_topic /gps/fix \
  --voxel_size 0.1 \
  --icp_fitness_thresh 0.3 \
  --level_floor \
  --enable_loop_closure
```

---

## 📖 Parameter Quick Reference

| Parameter | Default | Range | Purpose |
|---|---|---|---|
| `bagpath` | required | — | Path to ROS 2 bag file (inside container: `/bag/filename`) |
| `outputdir` | required | — | Output directory (inside container: `/output`) |
| `--pc_topic` | `points` | — | PointCloud2 topic name |
| `--odom_topic` | None | — | Odometry topic (nav_msgs/Odometry) |
| `--gps_topic` | `/gps/fix` | — | NavSatFix topic for GPS georeferencing |
| `--voxel_size` | 0.05 | 0.001–1.0 | Downsampling resolution (m) |
| `--icp_dist_thresh` | 0.2 | 0.01–10.0 | Max point correspondence distance (m) |
| `--icp_fitness_thresh` | 0.6 | 0.0–1.0 | Min % of points aligned to accept frame |
| `--odom_max_latency` | 0.5 | — | Max odom↔pointcloud timestamp gap (s) |
| `--enable_loop_closure` | False | — | Enable FPFH + RANSAC loop detection |
| `--loop_closure_radius` | 10.0 | 1.0–50.0 | Search radius for loop closure (m) |
| `--loop_closure_fitness_thresh` | 0.3 | 0.0–1.0 | Min fitness for loop closure acceptance |
| `--loop_closure_search_interval` | 10 | — | Loop closure search frequency (every N frames) |
| `--level_floor` | False | — | Apply Z-leveling via floor plane detection |
| `--workers` | 4 | 1–32 | Parallel threads for py3dtiles conversion |

---

## 🌍 GPS Georeferencing

The pipeline reads all valid `sensor_msgs/NavSatFix` messages from `--gps_topic`. Messages with `STATUS_NO_FIX` are discarded. The remaining fixes are **averaged** to produce a stable GPS origin — or used as-is if only one fix is present.

This origin anchors the local ENU (East-North-Up) robot frame to the Earth. The cleaned point cloud is then transformed to **ECEF (EPSG:4978)** — the geocentric coordinate system that CesiumJS and 3D Tiles natively use.

**The script hard-exits in two GPS error conditions:**
1. **Topic not found** — the topic name in `--gps_topic` does not exist in the bag at all. Use `ros2 bag info` to find the correct name.
2. **No valid fixes** — the topic exists but all messages have `STATUS_NO_FIX` or could not be parsed. Ensure the robot had GPS lock during recording.

### Finding Your GPS Topic

```bash
ros2 bag info /path/to/your/bag_file
# Look for: sensor_msgs/msg/NavSatFix
```

Common GPS topic names: `/gps/fix`, `/fix`, `/navsat/fix`, `/gps/gps`, `/ublox/fix`

---

## 🎯 Pro Tips

### Finding Your Topics

```bash
ros2 bag info /path/to/your/bag_file
```

Look for:
- `sensor_msgs/msg/PointCloud2` — Your point cloud topic
- `sensor_msgs/msg/NavSatFix` — Your GPS topic
- `nav_msgs/msg/Odometry` — Your odometry topic

### Performance Tuning

| Goal | Voxel Size | Fitness Thresh | ICP Distance |
|------|------------|----------------|--------------|
| 🚀 Fast Preview | 0.1 | 0.3 | 0.5 |
| ⚖️ Balanced | 0.05 | 0.5 | 0.2 |
| 🎨 Maximum Quality | 0.01 | 0.7 | 0.1 |

### Loop Closure

Loop closure is **disabled by default** for speed. Enable only for large-scale mapping where you revisit areas:

```bash
--enable_loop_closure
```

**When to use:**
- ✅ Large outdoor surveys (>30 min of data)
- ✅ Areas with significant loops/revisits
- ✅ When accuracy is critical

**When to skip:**
- ❌ Linear paths (roads, corridors)
- ❌ Small areas
- ❌ When speed matters

**Performance impact:**
- Disabled (default): ~2–3 min per 1000 frames
- Enabled: ~5–15 min per 1000 frames (3–8× slower)

---

## 🔧 Troubleshooting

### "Error: GPS topic not found in bag"

**Problem:** The GPS topic name doesn't match what's in your bag.

**Fix:**
```bash
ros2 bag info your_bag_file
# Find the NavSatFix topic name, then:
docker run ... --gps_topic /your/actual/gps/topic
```

---

### "Error: No valid GPS fixes found"

**Problem:** All GPS messages have `STATUS_NO_FIX`, or the topic has no messages.

**Fix:**
- Check GPS was active when the bag was recorded
- Verify the fix status in your bag: `ros2 topic echo /gps/fix --once`
- Ensure the robot had an open-sky view for GPS lock

---

### "Error: No messages found for topics"

**Problem:** Point cloud topic name doesn't match your bag.

**Fix:**
```bash
ros2 bag info your_bag_file
docker run ... --pc_topic /exact/topic/name
```

---

### "Error: Registration failed"

**Problem:** Too few successful ICP registrations.

**Fix (in order):**
1. Lower fitness threshold: `--icp_fitness_thresh 0.3`
2. Increase distance threshold: `--icp_dist_thresh 0.5`
3. Add odometry: `--odom_topic /your/odom`

---

### Point Clouds Look "Stacked" or Doubled

**Problem:** Severe registration failure causing ghost geometry.

**Fix:**
```bash
--icp_fitness_thresh 0.7 --icp_dist_thresh 0.15
```

---

### Processing is Too Slow

**Speed-up strategies (in priority order):**

1. Increase voxel size (biggest impact): `--voxel_size 0.1`
2. Don't use `--enable_loop_closure` (default: off)
3. Lower `--icp_fitness_thresh` to 0.4 (skips fewer frames)
4. Increase `--workers` to match your CPU core count

---

### Vertical Drift (Z-axis Issues)

1. Try Z-leveling (flat environments): `--level_floor`
2. Use loop closure: `--enable_loop_closure`
3. Add odometry: `--odom_topic`

---

## ⚠️ Known Limitations

### GPS Requirement
- **GPS is mandatory.** The pipeline will not produce a georeferenced tileset without a valid NavSatFix fix. Ensure the bag was recorded with GPS active.

### Loop Closure
- Searches within spatial radius and temporal window
- May produce false positives in highly repetitive environments
- **Disabled by default due to performance impact**

### Scale
- Tested up to **5000 frames** (~10 min at 10 Hz)
- Larger datasets may require **32 GB+ RAM**

### Point Cloud Types
- XYZ fields only (FLOAT32 or FLOAT64, all three must be the same type)
- Intensity/color fields are not included in the tileset

### Real-Time
- Not designed for real-time operation
- Typical speed: **~2–3 min per minute of recorded data** (without loop closure)

---

## 📊 Performance Guide

### Processing Times (Without Loop Closure)

| Dataset Size | Point Clouds | Voxel Size | Time | Peak RAM |
|---|---|---|---|---|
| Small area | ~500 frames | 0.05 m | 1–2 min | 4 GB |
| Medium survey | ~1500 frames | 0.05 m | 3–5 min | 8 GB |
| Large outdoor | ~3000 frames | 0.1 m | 8–12 min | 12 GB |
| Campus survey | ~5000 frames | 0.1 m | 15–25 min | 16 GB |

### Processing Times (With Loop Closure)

| Dataset Size | Point Clouds | Voxel Size | Time | Peak RAM |
|---|---|---|---|---|
| Medium survey | ~1500 frames | 0.05 m | 10–20 min | 8 GB |
| Large outdoor | ~3000 frames | 0.1 m | 30–60 min | 12 GB |
| Campus survey | ~5000 frames | 0.1 m | 60–120 min | 16 GB |

### Hardware Recommendations

| Scenario | CPU | RAM | Storage |
|---|---|---|---|
| Fast preview | 4 cores | 8 GB | 5 GB |
| Standard mapping | 8 cores | 16 GB | 20 GB |
| Large datasets | 16+ cores | 32 GB | 50+ GB SSD |

---

## 📝 Output Files Explained

### `*_cloud_ecef.ply`
- **Type:** Point cloud in ECEF (EPSG:4978) coordinates, float64
- **Use:** Intermediate file; input to py3dtiles
- **Tools:** CloudCompare (with CRS set to EPSG:4978), Open3D

### `tileset.json`
- **Type:** 3D Tiles root manifest
- **Use:** Entry point for CesiumJS / any 3D Tiles viewer
- **Load with:** `Cesium.Cesium3DTileset.fromUrl("…/tileset.json")`

### `*.pnts`
- **Type:** 3D Tiles Point Cloud tile files (binary)
- **Use:** Automatically streamed by the viewer — do not load directly

---

## 🔗 Viewing Your Results

### 3D Tiles Viewers
- **CesiumJS** (browser, free): https://cesium.com/cesiumjs/
- **Cesium ion** (hosted, free tier): https://cesium.com/platform/cesium-ion/
- **3D Tiles Tools** (CLI inspection): https://github.com/CesiumGS/3d-tiles-tools

### Intermediate PLY (ECEF)
- **CloudCompare** (free, cross-platform): https://www.cloudcompare.org/
  - Set coordinate system to EPSG:4978 for correct display

---

## 📦 Project Files

- `bag_to_tileset.py` — Main conversion script
- `bag_to_tileset_gui.py` — Tkinter GUI launcher
- `Dockerfile` — Container build instructions
- `README.md` — This file

---

**Built with:** Open3D · NumPy · SciPy · rosbags · pyproj · py3dtiles · tqdm · Python 3.10+
