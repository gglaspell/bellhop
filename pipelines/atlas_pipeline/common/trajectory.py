"""
colormesh3d.common.trajectory
==============================
Shared trajectory-loading and column-normalization utilities.

PATCH NOTE (frame-awareness for texture_baking, additive):
See pointcloud-frame-check-prompt-2.md. Added `get_pose_at_interpolated()`
and `pose_to_matrix()` so callers that need to transform a *point cloud*
frame into the world frame (not just look up a single keyframe/camera
pose) can do so using an odometry-derived pose interpolated between the
two bracketing trajectory rows closest in time (linear translation +
SLERP rotation), instead of `get_pose_at()`'s nearest-neighbor snap.
`get_pose_at()` itself is unchanged and still used for keyframe/camera
pose lookups elsewhere (view assignment, atlas packing, texture baking).
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"pos.x", "pos.y", "pos.z",
                    "orient.x", "orient.y", "orient.z", "orient.w",
                    "timestamp"}


def load_trajectory(traj_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a trajectory CSV and normalize its columns to a canonical schema:
    timestamp, pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w

    Raises:
        ValueError: if no recognizable timestamp or position schema is found.
    """
    df = pd.read_csv(traj_path)
    df.columns = df.columns.str.strip()

    # -- Timestamp normalization --------------------------------------------
    if "pose_timestamp_sec" in df.columns:
        df["timestamp"] = df["pose_timestamp_sec"] + df["pose_timestamp_nanosec"] * 1e-9
    elif "msg_timestamp_sec" in df.columns:
        df["timestamp"] = df["msg_timestamp_sec"] + df["msg_timestamp_nanosec"] * 1e-9
    elif "timestamp" not in df.columns:
        if "msg.timestamp.sec" in df.columns:
            df["timestamp"] = df["msg.timestamp.sec"] + df["msg.timestamp.nanosec"] * 1e-9
        elif "sec" in df.columns and "nanosec" in df.columns:
            df["timestamp"] = df["sec"] + df["nanosec"] * 1e-9
        else:
            raise ValueError(
                "Trajectory CSV is missing a recognizable timestamp schema. "
                "Expected one of: pose_timestamp_sec/nanosec, "
                "msg_timestamp_sec/nanosec, msg.timestamp.sec/nanosec, "
                "sec/nanosec, or a pre-combined 'timestamp' column."
            )

    # -- Position normalization ----------------------------------------------
    if "pos.x" not in df.columns and "pos_x" in df.columns:
        df.rename(columns={"pos_x": "pos.x", "pos_y": "pos.y", "pos_z": "pos.z"},
                  inplace=True)

    # -- Orientation normalization ---------------------------------------------
    if "orient.x" not in df.columns and "orient_x" in df.columns:
        df.rename(
            columns={
                "orient_x": "orient.x", "orient_y": "orient.y",
                "orient_z": "orient.z", "orient_w": "orient.w",
            },
            inplace=True,
        )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory CSV missing required columns after "
                          f"normalization: {sorted(missing)}")

    return df.sort_values("timestamp").reset_index(drop=True)


def build_trajectory_tree(df: pd.DataFrame) -> cKDTree:
    """Build a 1-D cKDTree over the trajectory's timestamp column for
    fast nearest-pose lookups."""
    return cKDTree(df["timestamp"].values.reshape(-1, 1))


def get_pose_at(df: pd.DataFrame, tree: cKDTree, ts: float):
    """
    Look up the trajectory row nearest to timestamp `ts`.

    Returns:
        (cam_pos, cam_rot): cam_pos is a (3,) np.ndarray; cam_rot is a
        scipy.spatial.transform.Rotation.
    """
    _, idx = tree.query([[ts]], k=1)
    row = df.iloc[int(np.asarray(idx).flat[0])]
    cam_pos = np.array([row["pos.x"], row["pos.y"], row["pos.z"]])
    cam_rot = R.from_quat(
        [row["orient.x"], row["orient.y"], row["orient.z"], row["orient.w"]]
    )
    return cam_pos, cam_rot


def _row_pose(df: pd.DataFrame, row_index: int):
    row = df.iloc[row_index]
    pos = np.array([row["pos.x"], row["pos.y"], row["pos.z"]])
    rot = R.from_quat([row["orient.x"], row["orient.y"], row["orient.z"], row["orient.w"]])
    return pos, rot


def get_pose_at_interpolated(df: pd.DataFrame, ts: float, max_latency: float | None = None):
    """
    Look up the trajectory pose at timestamp `ts` by interpolating between
    the two bracketing rows closest in time (linear interpolation on
    position, SLERP on rotation), instead of snapping to the single
    nearest row like `get_pose_at()`.

    `df` must already be sorted by `timestamp` ascending (`load_trajectory()`
    guarantees this).

    Args:
        df: trajectory dataframe (canonical schema).
        ts: query timestamp, in the same units as `df["timestamp"]` (seconds).
        max_latency: if given, return None when `ts` falls outside this many
            seconds of trajectory coverage (before the first or after the
            last sample, or in a gap wider than `2 * max_latency`).

    Returns:
        (cam_pos, cam_rot), or None if `ts` has no coverage within
        `max_latency`.
    """
    timestamps = df["timestamp"].values
    n = len(timestamps)
    if n == 0:
        return None

    idx = int(np.searchsorted(timestamps, ts))

    if idx <= 0:
        if max_latency is not None and abs(ts - timestamps[0]) > max_latency:
            return None
        return _row_pose(df, 0)

    if idx >= n:
        if max_latency is not None and abs(ts - timestamps[-1]) > max_latency:
            return None
        return _row_pose(df, n - 1)

    t_before = timestamps[idx - 1]
    t_after = timestamps[idx]

    if ts == t_before:
        return _row_pose(df, idx - 1)
    if ts == t_after:
        return _row_pose(df, idx)

    if (
        max_latency is not None
        and (ts - t_before) > max_latency
        and (t_after - ts) > max_latency
    ):
        return None

    if t_after == t_before:
        return _row_pose(df, idx - 1)

    fraction = (ts - t_before) / (t_after - t_before)
    fraction = min(1.0, max(0.0, fraction))

    pos_before, rot_before = _row_pose(df, idx - 1)
    pos_after, rot_after = _row_pose(df, idx)

    pos = pos_before * (1.0 - fraction) + pos_after * fraction

    rotations = R.from_quat(np.stack([rot_before.as_quat(), rot_after.as_quat()]))
    slerp = Slerp([0.0, 1.0], rotations)
    rot = slerp([fraction])[0]

    return pos, rot


def pose_to_matrix(pos: np.ndarray, rot: R) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a (pos, rot) trajectory pose."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot.as_matrix()
    transform[:3, 3] = pos
    return transform


def parse_stem_timestamp(stem: str, offset: float = 0.0) -> float:
    """
    Convert a '{sec}-{nanosec}' filename stem into a float timestamp in seconds.
    Falls back to treating the whole stem as a single float if no '-' separator.
    """
    try:
        parts = stem.split("-")
        sec = float(parts[0])
        nsec = float(parts[1])
        return sec + nsec * 1e-9 + offset
    except (IndexError, ValueError):
        return float(stem) + offset
