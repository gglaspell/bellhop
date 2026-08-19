#!/usr/bin/env python3
"""
og_map.py - ROS 2 bag -> Nav2 occupancy grid.

This preserves the validated hybrid method:
- build an OcTree using ray casting
- estimate local ground height
- query a vertical OcTree slice above ground
- produce occupied / free / unknown Nav2 pixels

PATCH NOTE (frame-awareness + odom-anchored local-frame support):
See pointcloud-frame-check-prompt-2.md and
odom-anchored-registration-fix-prompt.md.

Previously this pipeline *assumed* --pc_topic was already published in a
global/fixed frame and passed points into the OcTree unchanged, with only
a code comment ("Do not transform points again unless you verify its
frame_id") flagging the risk if that assumption were ever wrong. That
assumption is now verified (or overridden) at runtime:

- The point cloud's frame_id is detected (or overridden via
  --pc_frame_mode) and classified as 'global' (odom/map/world) or 'local'
  (a moving sensor/base frame), via `ros_io.resolve_pc_frame_mode`.
- 'global' frames: unchanged behaviour -- points are passed into the
  OcTree as-is; odometry is used only to supply the ray-casting sensor
  origin.
- 'local' frames: each frame's points are now transformed into the
  world/fixed frame using an odometry pose *before* OcTree insertion and
  before ground/height-map building, so og_map can run directly against a
  raw sensor-frame point cloud instead of requiring a pre-deskewed one.
  The odom pose is looked up via `ros_io.interpolate_odom_pose()`
  (linear translation + SLERP rotation between bracketing samples), not
  nearest-neighbor snapping -- matching the pose-lookup approach used by
  the odom-anchored mesh/tiles pipelines. og_map has no ICP/registration
  step of its own and doesn't need one here: odometry is already the sole
  and sufficient pose source for this ray-casting/height-map method, so
  no loop closure or ICP refinement is added.
- Frames with no odometry coverage within --odom_max_latency are dropped
  and counted loudly (logger.warning + print), never silently, mirroring
  the coverage-visibility requirement in odom-anchored-registration-fix-prompt.md.
- CAVEAT: the frame check only detects whether the cloud is already in a
  fixed/global frame -- it does NOT correct for a real sensor-to-base_link
  extrinsic offset (lever arm). See `ros_io.classify_frame_mode` docstring.

CLEANUP NOTE: `_nearest_index()` has been removed. It was og_map's local,
nearest-neighbor-only odometry lookup helper; it's now replaced everywhere
by the shared, interpolating `ros_io.interpolate_odom_pose()`.

FIX (coverage-log accuracy): the end-of-run coverage summary previously
computed "frames with valid pose" as `selected_frames - dropped_no_odom`.
That overcounts whenever frames are also skipped earlier for malformed
messages or too few finite points -- those frames never reach the
odometry check at all, so subtracting only `dropped_no_odom` credited
them as if they *had* gotten a valid pose. An explicit `frames_with_pose`
counter is now incremented only when `interpolate_odom_pose()` actually
returns a pose, so the logged count is exact.

PERF NOTE (OcTree insertion speedup): step [2/6] previously called
`obstacle_tree.insertPointCloud(points, sensor_origin, -1.0)` once per
frame with no other options set. Profiling on dense LiDAR bags (e.g.
Ouster, ~65k-131k pts/frame) showed this step dominating total runtime,
for three compounding reasons, all fixed below without changing the
occupancy semantics used by the validated hybrid method:

1. Every single `insertPointCloud()` call recomputed inner-node occupancy
   from scratch (the default `lazy_eval=False`). With thousands of
   frames, that per-call recomputation cost dwarfs the actual ray
   casting. `--octree_lazy_eval` (default: on) defers that recomputation
   via `lazy_eval=True`, and a single `obstacle_tree.updateInnerOccupancy()`
   call is made once after the frame loop, before the tree is queried in
   step [5/6] (querying a tree with stale inner nodes under lazy_eval
   would silently return wrong occupancy -- see OctoMap's own docs on
   this flag).
2. Full-resolution points (not yet voxel-downsampled) were ray-cast into
   the tree every frame -- the existing `--voxel_size` downsample only
   ran *after* insertion, on the copy kept for ground/obstacle
   separation. `--octree_discretize` (default: on) sets `discretize=True`,
   which snaps the scan onto the octree's own voxel grid before casting
   rays, collapsing many same-voxel points into one ray. This changes
   which exact points contribute to each ray (occupied nodes still take
   precedence over free ones, per OctoMap's `computeDiscreteUpdate()`),
   which is a deliberate accuracy/speed tradeoff now enabled by default;
   pass `--no-octree_discretize`-equivalent behavior is not available as
   a flag but the operator may set `--octree_discretize` explicitly if
   they need to reason about it.
3. Every ray was previously cast the complete beam length (`maxrange=-1.0`),
   so distant returns cost proportionally more voxel traversals.
   `--octree_max_range` (default: 40.0 m) now caps beam length by default
   for the ground/obstacle band this pipeline maps; pass a larger value
   (or -1.0 for unlimited) if far returns are needed.

These defaults (`--octree_lazy_eval` on, `--octree_discretize` on,
`--octree_max_range` 40.0) match the Bellhop GUI's OG Map profile
defaults, so a run launched from the GUI with no changes and one
launched from the bare CLI with no flags now produce identical
`docker run` commands.
"""

from __future__ import annotations

import logging
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import open3d as o3d
import pyoctomap
import yaml
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.interpolate import griddata
from scipy.ndimage import binary_closing, label

from .shared.preflight import check_topics
from .shared.ros_io import (
    get_odom_transform_matrix,
    interpolate_odom_pose,
    pointcloud2_to_numpy,
    resolve_pc_frame_mode,
)

logger = logging.getLogger(__name__)


def _separate_ground_and_obstacles(
    cloud: o3d.geometry.PointCloud,
    slope_deg: float,
    normal_radius: float,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate ground and non-ground points using surface-normal slope."""
    logger.info("Separating ground from obstacles...")

    if len(cloud.points) < 3:
        return np.empty((0, 3)), np.asarray(cloud.points)

    downsampled = (
        cloud.voxel_down_sample(voxel_size)
        if voxel_size > 0
        else cloud
    )

    try:
        downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=30,
            )
        )
        downsampled.orient_normals_to_align_with_direction(
            [0.0, 0.0, 1.0]
        )
    except RuntimeError as exc:
        # Open3D raises RuntimeError on a degenerate KDTree (e.g. too few
        # neighbors within normal_radius for a sparse/downsampled cloud).
        logger.warning(
            "Normal estimation failed (%s); treating all points as obstacles.",
            exc,
        )
        return np.empty((0, 3)), np.asarray(cloud.points)

    points = np.asarray(cloud.points)
    tree = o3d.geometry.KDTreeFlann(downsampled)

    nearest_indices = np.fromiter(
        (
            tree.search_knn_vector_3d(point, 1)[1][0]
            for point in points
        ),
        dtype=np.int64,
        count=len(points),
    )

    normals = np.asarray(downsampled.normals)[nearest_indices]

    angles = np.arccos(
        np.clip(
            np.abs(normals[:, 2]),
            -1.0,
            1.0,
        )
    )

    ground_mask = angles <= np.deg2rad(slope_deg)

    ground_points = points[ground_mask]
    obstacle_points = points[~ground_mask]

    logger.info(
        "Ground: %d pts | Obstacles: %d pts",
        len(ground_points), len(obstacle_points),
    )

    return ground_points, obstacle_points


def _build_ground_height_map(
    ground_points: np.ndarray,
    grid_resolution: float,
    map_origin: np.ndarray,
    x_size: int,
    y_size: int,
) -> np.ndarray:
    """Build a local ground-height image and fill missing cells."""
    logger.info("Building ground height map...")

    if len(ground_points) == 0:
        logger.warning("No ground points; using a Z=0 ground plane.")
        return np.zeros((y_size, x_size), dtype=np.float64)

    sum_z = np.zeros((y_size, x_size), dtype=np.float64)
    counts = np.zeros((y_size, x_size), dtype=np.int32)

    grid_x = (
        (ground_points[:, 0] - map_origin[0])
        / grid_resolution
    ).astype(np.int64)

    grid_y = (
        (ground_points[:, 1] - map_origin[1])
        / grid_resolution
    ).astype(np.int64)

    valid = (
        (grid_x >= 0)
        & (grid_x < x_size)
        & (grid_y >= 0)
        & (grid_y < y_size)
    )

    np.add.at(
        sum_z,
        (grid_y[valid], grid_x[valid]),
        ground_points[valid, 2],
    )

    np.add.at(
        counts,
        (grid_y[valid], grid_x[valid]),
        1,
    )

    average_z = np.divide(
        sum_z,
        counts,
        out=np.zeros_like(sum_z),
        where=counts > 0,
    )

    known = np.where(counts > 0)

    if len(known[0]) < 3:
        return np.full(
            (y_size, x_size),
            float(ground_points[:, 2].min()),
        )

    logger.info("Interpolating gaps in ground height map...")

    yy, xx = np.mgrid[0:y_size, 0:x_size]

    return griddata(
        (known[0], known[1]),
        average_z[known],
        (yy, xx),
        method="nearest",
    )


def _process_row_block(
    row_start: int,
    row_end: int,
    x_size: int,
    ground_height_map: np.ndarray,
    obstacle_tree,
    grid_resolution: float,
    map_origin: np.ndarray,
    relative_z_min: float,
    relative_z_max: float,
    sample_count: int,
) -> tuple[int, np.ndarray]:
    """
    Compute occupancy values for a contiguous block of grid rows
    [row_start, row_end) in a single thread.

    Batching by row-block (instead of submitting one ThreadPoolExecutor
    task per individual grid cell) keeps the number of futures bounded
    by the worker count rather than by x_size * y_size, which avoids
    exhausting memory/threads on large grids while still parallelising
    across the OcTree queries.
    """
    block_height = row_end - row_start
    block = np.full((block_height, x_size), 127, dtype=np.uint8)

    for local_y, grid_y in enumerate(range(row_start, row_end)):
        for grid_x in range(x_size):
            ground_z = ground_height_map[grid_y, grid_x]

            if not np.isfinite(ground_z):
                continue

            world_x = map_origin[0] + (grid_x + 0.5) * grid_resolution
            world_y = map_origin[1] + (grid_y + 0.5) * grid_resolution

            z_slice = np.linspace(
                ground_z + relative_z_min,
                ground_z + relative_z_max,
                sample_count,
            )

            free_evidence = False

            for world_z in z_slice:
                node = obstacle_tree.search(
                    np.array(
                        [world_x, world_y, world_z],
                        dtype=np.float64,
                    )
                )

                if node is None:
                    continue

                if (
                    node.getOccupancy()
                    >= obstacle_tree.getOccupancyThres()
                ):
                    block[local_y, grid_x] = 0
                    free_evidence = False
                    break

                free_evidence = True
            else:
                if free_evidence:
                    block[local_y, grid_x] = 254

    return row_start, block


def _create_hybrid_occupancy_grid(
    obstacle_tree,
    ground_height_map: np.ndarray,
    grid_resolution: float,
    map_origin: np.ndarray,
    relative_z_min: float,
    relative_z_max: float,
    workers: int,
) -> np.ndarray:
    """
    Query OcTree occupancy in a vertical band above each ground cell.

    Values:
      0   occupied
      127 unknown
      254 free
    """
    y_size, x_size = ground_height_map.shape

    logger.info("Generating 2D occupancy grid (%d x %d)...", x_size, y_size)

    sample_count = max(
        7,
        int(
            np.ceil(
                (relative_z_max - relative_z_min)
                / grid_resolution
            )
        )
        + 1,
    )

    worker_count = max(1, workers)

    grid = np.full((y_size, x_size), 127, dtype=np.uint8)

    blocks_per_worker = 4
    num_blocks = max(1, min(y_size, worker_count * blocks_per_worker))
    row_boundaries = np.linspace(0, y_size, num_blocks + 1, dtype=np.int64)
    row_boundaries = np.unique(row_boundaries)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _process_row_block,
                int(row_boundaries[i]),
                int(row_boundaries[i + 1]),
                x_size,
                ground_height_map,
                obstacle_tree,
                grid_resolution,
                map_origin,
                relative_z_min,
                relative_z_max,
                sample_count,
            )
            for i in range(len(row_boundaries) - 1)
        ]

        for future in as_completed(futures):
            row_start, block = future.result()
            grid[row_start:row_start + block.shape[0], :] = block

    return grid


def _filter_small_clusters(
    grid: np.ndarray,
    min_cluster_size: int,
    closing_iters: int,
) -> np.ndarray:
    """
    Remove tiny occupied clusters.

    Removed clusters become unknown (127), not free (254), which is the
    conservative behavior from the validated implementation.
    """
    if min_cluster_size <= 0:
        return grid

    occupied_mask = grid == 0

    if closing_iters > 0:
        occupied_mask = binary_closing(
            occupied_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=closing_iters,
        )

    labels, number_of_labels = label(
        occupied_mask,
        structure=np.ones((3, 3), dtype=np.int8),
    )

    if number_of_labels == 0:
        return grid

    sizes = np.bincount(labels.ravel())
    too_small = sizes < min_cluster_size
    too_small[0] = False

    cleaned = grid.copy()
    cleaned[occupied_mask] = 0
    cleaned[too_small[labels]] = 127

    return cleaned


def run(args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    bag_path = Path(args.input_bag)
    output_path = Path(args.output_path)

    if not bag_path.exists():
        sys.exit(f"Error: Bag not found: {bag_path}")

    missing_topics = check_topics(
        bag_path,
        [args.pc_topic, args.odom_topic],
    )
    if missing_topics:
        sys.exit(f"Error: Required topics not found: {missing_topics}")

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    frame_mode = resolve_pc_frame_mode(bag_path, args.pc_topic, args.pc_frame_mode)
    if frame_mode == "local":
        logger.info(
            "Point cloud is in a local/moving frame: each frame will be "
            "transformed into the world frame via an odom-derived pose "
            "before OcTree insertion and ground/height-map building."
        )
    else:
        logger.info(
            "Point cloud is already in a global/fixed frame: points will be "
            "passed into the OcTree unchanged; odometry supplies only the "
            "ray-casting sensor origin."
        )

    logger.info("[1/6] Loading odometry from '%s'...", args.odom_topic)

    odom_data: dict[int, np.ndarray] = {}

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        odom_connections = [
            connection
            for connection in reader.connections
            if connection.topic == args.odom_topic
        ]

        for connection, timestamp, raw in reader.messages(
            connections=odom_connections
        ):
            try:
                message = reader.deserialize(raw, connection.msgtype)
                transform = get_odom_transform_matrix(message)
            except (AttributeError, KeyError, struct.error, ValueError) as exc:
                # AttributeError/KeyError: malformed or unexpected message
                # schema. struct.error: corrupted binary payload.
                # ValueError: bad numeric field (e.g. non-finite quaternion).
                logger.warning(
                    "Skipping malformed odometry message at t=%.3f: %s",
                    timestamp * 1e-9, exc,
                )
                continue

            if transform is not None:
                odom_data[timestamp] = transform

    if not odom_data:
        sys.exit("Error: No valid odometry messages found.")

    odom_ts_sorted = sorted(odom_data)
    odom_max_latency_ns = int(args.odom_max_latency * 1e9)

    logger.info("Loaded %d odometry poses.", len(odom_data))

    logger.info(
        "[2/6] Building 3D OcTree (octree_res=%s m, lazy_eval=%s, "
        "discretize=%s, max_range=%s)...",
        args.octree_res, args.octree_lazy_eval, args.octree_discretize,
        args.octree_max_range,
    )

    obstacle_tree = pyoctomap.OcTree(args.octree_res)

    point_chunks: list[np.ndarray] = []

    seen_frames = 0
    selected_frames = 0
    valid_frames = 0
    frames_with_pose = 0
    dropped_no_odom = 0

    frame_stride = max(1, int(args.frame_stride))
    max_frames = max(0, int(args.max_frames))

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        cloud_connections = [
            connection
            for connection in reader.connections
            if connection.topic == args.pc_topic
        ]

        for connection, timestamp, raw in reader.messages(
            connections=cloud_connections
        ):
            seen_frames += 1

            if (seen_frames - 1) % frame_stride != 0:
                continue

            if max_frames and selected_frames >= max_frames:
                break

            selected_frames += 1

            try:
                message = reader.deserialize(raw, connection.msgtype)
                points = pointcloud2_to_numpy(message)
            except (AttributeError, KeyError, struct.error, ValueError) as exc:
                logger.warning(
                    "Skipping malformed PointCloud2 message at t=%.3f: %s",
                    timestamp * 1e-9, exc,
                )
                continue

            if points is None or len(points) == 0:
                continue

            points = np.asarray(points[:, :3], dtype=np.float64)
            points = points[np.isfinite(points).all(axis=1)]

            if len(points) == 0:
                continue

            # Odometry is the sole pose source for this pipeline (no ICP/
            # registration step exists or is needed here). Pose is looked
            # up by interpolating between the two bracketing odometry
            # samples closest in time (linear translation + SLERP
            # rotation), not nearest-neighbor snapping.
            odom_pose = interpolate_odom_pose(
                timestamp, odom_ts_sorted, odom_data, odom_max_latency_ns
            )

            if odom_pose is None:
                dropped_no_odom += 1
                continue

            frames_with_pose += 1
            sensor_origin = odom_pose[:3, 3].astype(np.float64)

            if frame_mode == "local":
                # The point cloud is in a local/moving sensor or base
                # frame: transform it into the world/fixed frame using the
                # odom-derived pose before it is used for anything else.
                # CAVEAT: this only corrects for the odom-reported
                # fixed-frame-to-base transform. It does NOT correct for a
                # real sensor-to-base_link extrinsic offset (lever arm);
                # that requires a separate static transform (e.g. from
                # tf_static), which is out of scope here.
                homogeneous_points = np.hstack(
                    [points, np.ones((len(points), 1), dtype=np.float64)]
                )
                points = (odom_pose @ homogeneous_points.T).T[:, :3]
            # else: frame_mode == "global" -- the cloud is already in a
            # global/fixed frame (odom/map/world). Keep the validated
            # coordinate convention: pass points through unchanged and use
            # odometry only to supply the ray-casting sensor origin.

            try:
                # PERF: lazy_eval defers inner-node occupancy recomputation
                # until updateInnerOccupancy() is called once below (see
                # PERF NOTE at module top) instead of on every frame;
                # discretize snaps the scan onto the octree's own
                # voxel grid before ray casting to avoid redundant rays
                # from a still-full-resolution frame; max_range caps beam
                # length (40 m by default; pass -1.0 for unlimited).
                obstacle_tree.insertPointCloud(
                    points,
                    sensor_origin,
                    max_range=args.octree_max_range,
                    lazy_eval=args.octree_lazy_eval,
                    discretize=args.octree_discretize,
                )
            except RuntimeError as exc:
                # pyoctomap raises RuntimeError on malformed/degenerate
                # ray-casting input (e.g. NaN sensor origin).
                logger.warning(
                    "[%d] OcTree insertion error: %s", selected_frames, exc
                )
                continue

            if args.voxel_size > 0:
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                downsampled = cloud.voxel_down_sample(args.voxel_size)
                chunk = np.asarray(downsampled.points)
            else:
                chunk = points

            if len(chunk):
                point_chunks.append(chunk)
                valid_frames += 1

    if dropped_no_odom:
        message = (
            f"{dropped_no_odom} / {selected_frames} selected frames had no "
            "odometry coverage and were dropped"
        )
        logger.warning(message)
        print(f"Warning: {message}")

    if not point_chunks:
        sys.exit("Error: No valid point clouds processed.")

    logger.info(
        "Coverage: frames read=%d -> frames selected=%d -> frames with valid "
        "pose=%d -> frames merged=%d.",
        seen_frames, selected_frames, frames_with_pose, valid_frames,
    )

    if args.octree_lazy_eval:
        # REQUIRED when lazy_eval=True was used above: inner-node occupancy
        # was deferred on every insertPointCloud() call, so the tree's
        # inner nodes are stale until this single reconciliation pass runs.
        # Step [5/6] queries obstacle_tree.search()/getOccupancy() below,
        # so this must happen before that step, not after.
        logger.info("Reconciling deferred OcTree inner-node occupancy...")
        obstacle_tree.updateInnerOccupancy()

    logger.info("OcTree: %d nodes.", obstacle_tree.size())

    logger.info("[3/6] Separating ground from obstacles...")

    all_points = np.vstack(point_chunks)
    del point_chunks

    full_cloud = o3d.geometry.PointCloud()
    full_cloud.points = o3d.utility.Vector3dVector(all_points)

    ground_points, _ = _separate_ground_and_obstacles(
        full_cloud,
        args.slope_deg,
        args.normal_radius,
        args.voxel_size,
    )

    logger.info("[4/6] Building ground height map...")

    map_origin = all_points.min(axis=0)
    map_max = all_points.max(axis=0)

    x_size = max(
        1,
        int(np.ceil((map_max[0] - map_origin[0]) / args.grid_res)),
    )
    y_size = max(
        1,
        int(np.ceil((map_max[1] - map_origin[1]) / args.grid_res)),
    )

    logger.info("Grid: %d x %d cells", x_size, y_size)

    ground_height_map = _build_ground_height_map(
        ground_points,
        args.grid_res,
        map_origin,
        x_size,
        y_size,
    )

    logger.info("[5/6] Generating occupancy grid...")

    occupancy_grid = _create_hybrid_occupancy_grid(
        obstacle_tree,
        ground_height_map,
        args.grid_res,
        map_origin,
        args.z_min,
        args.z_max,
        args.workers,
    )

    if args.min_cluster_size > 0:
        logger.info(
            "[5.5/6] Denoising (min_cluster_size=%d)...", args.min_cluster_size
        )
        occupancy_grid = _filter_small_clusters(
            occupancy_grid,
            args.min_cluster_size,
            args.closing_iters,
        )

    logger.info("[6/6] Saving output files...")

    pgm_path = output_path.with_suffix(".pgm")
    yaml_path = output_path.with_suffix(".yaml")
    png_path = output_path.with_suffix(".png")

    pgm_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.ascontiguousarray(np.flipud(occupancy_grid))

    Image.fromarray(image, "L").save(pgm_path)
    Image.fromarray(image, "L").save(png_path)

    yaml_data = {
        "image": pgm_path.name,
        "resolution": float(args.grid_res),
        "origin": [
            float(map_origin[0]),
            float(map_origin[1]),
            0.0,
        ],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
    }

    with yaml_path.open("w", encoding="utf-8") as output_file:
        yaml.dump(yaml_data, output_file, sort_keys=False)

    logger.info("Occupied cells: %d", np.count_nonzero(occupancy_grid == 0))
    logger.info("Free cells: %d", np.count_nonzero(occupancy_grid == 254))
    logger.info("Unknown cells: %d", np.count_nonzero(occupancy_grid == 127))
    logger.info("Saved: %s", pgm_path)
    logger.info("Saved preview: %s", png_path)
    logger.info("Saved: %s", yaml_path)
    logger.info("Done.")


def build_parser(sub):
    parser = sub.add_parser(
        "og_map",
        help="ROS 2 bag -> Nav2 occupancy grid (.pgm + .yaml)",
    )
    parser.add_argument("input_bag")
    parser.add_argument("output_path")

    parser.add_argument(
        "--pc_topic",
        default="/points",
    )
    parser.add_argument(
        "--odom_topic",
        default="/odom",
    )
    parser.add_argument(
        "--pc_frame_mode",
        choices=["auto", "global", "local"],
        default="auto",
        help=(
            "Point-cloud frame handling. 'auto' detects the frame_id of the "
            "first message on --pc_topic and classifies odom/map/world as "
            "global (points passed through unchanged) and anything else as "
            "local (each frame is transformed into the world frame via an "
            "odom-derived pose before OcTree insertion). Use 'global'/'local' "
            "to override when a bag's frame_id is missing, wrong, or empty -- "
            "e.g. force 'local' to run directly against a raw, non-deskewed "
            "sensor-frame point cloud."
        ),
    )

    parser.add_argument("--octree_res", type=float, default=0.1)
    parser.add_argument("--grid_res", type=float, default=0.10)
    parser.add_argument("--slope_deg", type=float, default=15.0)
    parser.add_argument("--normal_radius", type=float, default=0.2)
    parser.add_argument("--z_min", type=float, default=0.1)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--voxel_size", type=float, default=0.10)

    parser.add_argument(
        "--frame_stride",
        type=int,
        default=2,
        help="Process every Nth PointCloud2 frame.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum selected PointCloud2 frames; 0 means all.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min_cluster_size", type=int, default=20)
    parser.add_argument("--closing_iters", type=int, default=1)

    parser.add_argument(
        "--odom_max_latency",
        type=float,
        default=0.5,
    )

    # PERF: flags controlling OcTree insertion (step [2/6]). See the
    # "PERF NOTE" in this file's module docstring for why each exists.
    # Defaults below match the Bellhop GUI's OG Map profile exactly.
    parser.add_argument(
        "--octree_max_range",
        type=float,
        default=40.0,
        help=(
            "Max ray-casting beam length (m) per point during OcTree "
            "insertion; default 40.0 caps beam length for the mapped "
            "z_min/z_max band. Pass -1.0 to cast the complete beam "
            "(unlimited) if far returns are needed."
        ),
    )
    parser.add_argument(
        "--octree_lazy_eval",
        action="store_true",
        default=True,
        help=(
            "Defer OcTree inner-node occupancy recomputation during "
            "insertion (default: on) and reconcile it once via "
            "updateInnerOccupancy() after all frames are inserted, instead "
            "of recomputing it on every single frame. This does not change "
            "output; it only reorders when the recomputation happens."
        ),
    )
    parser.add_argument(
        "--no-octree_lazy_eval",
        dest="octree_lazy_eval",
        action="store_false",
        help="Disable --octree_lazy_eval (recompute inner nodes every frame).",
    )
    parser.add_argument(
        "--octree_discretize",
        action="store_true",
        default=True,
        help=(
            "Snap each frame onto the OcTree's own voxel grid before ray "
            "casting (default: on). Collapses multiple same-voxel "
            "points into one ray per frame, which speeds up insertion on "
            "dense point clouds at the cost of using a discretized "
            "approximation of each frame rather than every raw point."
        ),
    )
    parser.add_argument(
        "--no-octree_discretize",
        dest="octree_discretize",
        action="store_false",
        help="Disable --octree_discretize (ray-cast every raw point).",
    )

    parser.set_defaults(func=run)
    return parser
