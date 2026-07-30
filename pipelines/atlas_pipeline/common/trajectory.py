"""
colormesh3d.common.trajectory
==============================
Shared trajectory-loading and column-normalization utilities.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

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

    # ── Timestamp normalization ──────────────────────────────────────────
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
            )  # FIX: was missing closing )

    # ── Position normalization ───────────────────────────────────────────
    if "pos.x" not in df.columns and "pos_x" in df.columns:
        df.rename(columns={"pos_x": "pos.x", "pos_y": "pos.y", "pos_z": "pos.z"},
                  inplace=True)

    # ── Orientation normalization ────────────────────────────────────────
    if "orient.x" not in df.columns and "orient_x" in df.columns:
        df.rename(
            columns={
                "orient_x": "orient.x", "orient_y": "orient.y",
                "orient_z": "orient.z", "orient_w": "orient.w",
            },
            inplace=True,
        )  # FIX: was missing closing )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory CSV missing required columns after "
                         f"normalization: {sorted(missing)}")

    return df


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
    )  # FIX: was missing closing )
    return cam_pos, cam_rot


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
