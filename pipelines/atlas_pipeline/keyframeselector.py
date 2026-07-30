"""
pipelines.atlas_pipeline.keyframeselector
==============================================
Keyframe selection for the atlas-bake pipeline.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R

from .common.trajectory import load_trajectory, parse_stem_timestamp
from ..shared.ros_io import TYPESTORE, convert_ros_image, get_odom_transform


class KeyframeSelector:
    def __init__(self, image_folder, trajectory_path, min_movement_m, min_rotation_deg):
        self.image_folder = Path(image_folder)
        self.min_move = min_movement_m
        self.min_rot = np.deg2rad(min_rotation_deg)

        self.traj_df = load_trajectory(trajectory_path)
        self.images = (
            sorted(list(self.image_folder.glob("*.png")))
            + sorted(list(self.image_folder.glob("*.jpg")))
        )
        self.total_images = len(self.images)
        logging.info(f"Found {self.total_images} images")

    @classmethod
    def for_bag(cls, min_movement_m: float, min_rotation_deg: float) -> "KeyframeSelector":
        """
        Construct a bag-native selector without requiring a pre-populated
        image folder or trajectory CSV up front (both are unnecessary for
        select_keyframes_from_bag, which streams the bag directly).
        """
        instance = cls.__new__(cls)
        instance.image_folder = None
        instance.min_move = min_movement_m
        instance.min_rot = np.deg2rad(min_rotation_deg)
        instance.traj_df = None
        instance.images = []
        instance.total_images = 0
        return instance

    def _image_timestamp(self, image_path: Path) -> float:
        return parse_stem_timestamp(image_path.stem)

    def _get_pose(self, ts):
        idx = (self.traj_df["timestamp"] - ts).abs().idxmin()
        row = self.traj_df.loc[idx]
        pos = np.array([row["pos.x"], row["pos.y"], row["pos.z"]])
        rot = R.from_quat([row["orient.x"], row["orient.y"], row["orient.z"], row["orient.w"]])
        return pos, rot

    def _compute_sharpness(self, image_path: Path) -> float:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def select_keyframes(self):
        """
        Folder-based keyframe selection.

        Walks self.images (sorted png/jpg files already discovered in
        __init__), looks up each image's pose via the trajectory dataframe,
        and keeps an image as a keyframe only if it has moved at least
        min_move metres or rotated at least min_rot radians since the last
        accepted keyframe. Mirrors the bag-streaming logic in
        select_keyframes_from_bag(), but reads pre-extracted images from
        disk and a pre-loaded trajectory CSV instead of streaming the bag.

        Returns:
            List of (image_path, timestamp_seconds, sharpness) tuples for
            every accepted keyframe, in chronological order.
        """
        if self.traj_df is None:
            raise RuntimeError(
                "select_keyframes() requires a trajectory CSV; this "
                "instance was constructed via for_bag(). Use "
                "select_keyframes_from_bag() instead."
            )

        if not self.images:
            logging.warning("No images found in folder; returning no keyframes.")
            return []

        keyframes = []
        pos_last = rot_last = None

        for image_path in self.images:
            try:
                ts = self._image_timestamp(image_path)
            except Exception as exc:
                logging.warning(f"Could not parse timestamp for {image_path.name}: {exc}")
                continue

            try:
                pos, rot = self._get_pose(ts)
            except Exception as exc:
                logging.warning(f"Could not find pose for {image_path.name}: {exc}")
                continue

            if pos_last is None:
                keep = True
            else:
                move = np.linalg.norm(pos - pos_last)
                rot_diff = (rot_last.inv() * rot).magnitude()
                keep = move >= self.min_move or rot_diff >= self.min_rot

            if not keep:
                continue

            sharpness = self._compute_sharpness(image_path)
            keyframes.append((image_path, ts, sharpness))
            pos_last, rot_last = pos, rot

        if len(keyframes) < 10:
            logging.warning(f"Only {len(keyframes)} keyframes selected from folder.")

        return keyframes

    def save_keyframes(self, keyframes, path):
        with open(path, "w") as f:
            for img, ts, sharp in keyframes:
                f.write(f"{img.name},{ts},{sharp:.2f}\n")

    def select_keyframes_from_bag(
        self, bag_path, camera_topic, odom_topic,
        image_output_dir, timestamp_offset=0.0,
    ):
        """Stream camera_topic + odom_topic once; keep only frames that
        pass the movement/rotation test, writing ONLY those to disk."""
        image_output_dir.mkdir(parents=True, exist_ok=True)
        keyframes = []
        pos_last = rot_last = None
        odom_poses: list[tuple[int, np.ndarray, R]] = []

        with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
            connections = [c for c in reader.connections
                           if c.topic in (camera_topic, odom_topic)]
            for connection, ts_ns, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, connection.msgtype)

                if connection.topic == odom_topic:
                    transform = get_odom_transform(msg)
                    if transform is not None:
                        rot = R.from_matrix(transform[:3, :3])
                        odom_poses.append((ts_ns, transform[:3, 3], rot))
                    continue

                if connection.topic == camera_topic and odom_poses:
                    ts_sec = ts_ns * 1e-9 + timestamp_offset
                    _, pos, rot = min(odom_poses, key=lambda p: abs(p[0] * 1e-9 - ts_sec))

                    if pos_last is None:
                        keep = True
                    else:
                        move = np.linalg.norm(pos - pos_last)
                        rot_diff = (rot_last.inv() * rot).magnitude()
                        keep = move >= self.min_move or rot_diff >= self.min_rot

                    if keep:
                        image = convert_ros_image(msg)
                        if image is not None:
                            out_path = image_output_dir / f"{ts_ns}-0.png"
                            image.save(out_path)
                            sharpness = cv2.Laplacian(
                                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY),
                                cv2.CV_64F,
                            ).var()
                            keyframes.append((out_path, ts_sec, sharpness))
                            pos_last, rot_last = pos, rot

        if len(keyframes) < 10:
            logging.warning(f"Only {len(keyframes)} keyframes selected from bag.")
        return keyframes
