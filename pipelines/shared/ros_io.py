#!/usr/bin/env python3
"""ros_io.py - Shared ROS 2 message deserialization helpers."""
import bisect
import logging
import math
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, UnidentifiedImageError
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
_POINTFIELD_TO_DTYPE = {7: np.float32, 8: np.float64}
_GPS_NO_FIX = -1

try:
    from tf_transformations import quaternion_matrix as _qm
    quaternion_matrix = _qm
except ImportError:
    def quaternion_matrix(q):
        x, y, z, w = q
        n = (x * x + y * y + z * z + w * w) ** 0.5
        x, y, z, w = x / n, y / n, z / n, w / n
        m = np.zeros((4, 4))
        m[3, 3] = 1.0
        m[0, 0] = 1 - 2 * (y * y + z * z); m[0, 1] = 2 * (x * y - z * w); m[0, 2] = 2 * (x * z + y * w)
        m[1, 0] = 2 * (x * y + z * w); m[1, 1] = 1 - 2 * (x * x + z * z); m[1, 2] = 2 * (y * z - x * w)
        m[2, 0] = 2 * (x * z - y * w); m[2, 1] = 2 * (y * z + x * w); m[2, 2] = 1 - 2 * (x * x + y * y)
        return m


def convert_ros_pc2_to_o3d(msg):
    """Convert a PointCloud2 message to an Open3D point cloud.

    Returns None if the message is malformed or has too few finite points.
    Logs a warning (not silent) on any conversion failure so callers can
    trace dropped frames back to a root cause.
    """
    try:
        fields = {f.name: (int(f.offset), int(f.datatype)) for f in msg.fields}
    except AttributeError as exc:
        logger.warning("PointCloud2 message missing 'fields' attribute: %s", exc)
        return None

    if not all(k in fields for k in ("x", "y", "z")):
        logger.warning("PointCloud2 message missing x/y/z fields; got: %s", list(fields))
        return None

    xo, xd = fields["x"]; yo, yd = fields["y"]; zo, zd = fields["z"]
    if xd not in _POINTFIELD_TO_DTYPE or xd != yd or yd != zd:
        logger.warning(
            "Unsupported or mismatched PointField datatypes (x=%s y=%s z=%s); "
            "expected one of %s for all axes.",
            xd, yd, zd, list(_POINTFIELD_TO_DTYPE),
        )
        return None

    try:
        dt = _POINTFIELD_TO_DTYPE[xd]
        n = int(msg.width) * int(msg.height)
        dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": [dt, dt, dt],
            "offsets": [xo, yo, zo],
            "itemsize": int(msg.point_step),
        })
        arr = np.frombuffer(msg.data, dtype=dtype, count=n)
        pts = np.empty((n, 3), np.float64)
        pts[:, 0] = arr["x"]; pts[:, 1] = arr["y"]; pts[:, 2] = arr["z"]
    except (ValueError, TypeError) as exc:
        # e.g. buffer size mismatch between msg.data and computed dtype/count
        logger.warning("Failed to parse PointCloud2 buffer into structured array: %s", exc)
        return None

    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 10:
        logger.warning("PointCloud2 message yielded only %d finite points (<10); skipping.", len(pts))
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def pointcloud2_to_numpy(msg):
    """Convert a PointCloud2 message to an (N, 3) numpy array.

    Tries the robotdatapy fast-path first, then falls back to a manual
    per-field parse. Both paths log the specific failure reason instead
    of failing silently.
    """
    try:
        from robotdatapy.pointcloud.pointcloud_conversions import pointcloud2_to_xyz_array
        return pointcloud2_to_xyz_array(msg)
    except ImportError:
        logger.debug("robotdatapy not installed; using manual PointCloud2 parser.")
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("robotdatapy PointCloud2 conversion failed (%s); falling back to manual parser.", exc)

    try:
        n = msg.height * msg.width
        if n == 0:
            return np.array([])

        xo = yo = zo = None
        for f in msg.fields:
            if f.name == "x":
                xo = f.offset
            elif f.name == "y":
                yo = f.offset
            elif f.name == "z":
                zo = f.offset

        if None in (xo, yo, zo):
            logger.warning("PointCloud2 message missing one of x/y/z field offsets.")
            return np.array([])

        data = bytes(msg.data)
        step = msg.point_step
        pts = []
        for i in range(n):
            o = i * step
            x = np.frombuffer(data[o + xo:o + xo + 4], np.float32)[0]
            y = np.frombuffer(data[o + yo:o + yo + 4], np.float32)[0]
            z = np.frombuffer(data[o + zo:o + zo + 4], np.float32)[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                pts.append((x, y, z))
        return np.array(pts, np.float32) if pts else np.array([])
    except (ValueError, TypeError, struct.error) as exc:
        logger.warning("Manual PointCloud2 parsing failed: %s", exc)
        return np.array([])


def get_odom_transform(msg):
    """Build a 4x4 transform from an Odometry message's pose."""
    try:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [p.x, p.y, p.z]
        return T
    except AttributeError as exc:
        logger.warning("Odometry message missing expected pose fields: %s", exc)
        return None
    except ValueError as exc:
        # e.g. non-unit / degenerate quaternion rejected by scipy
        logger.warning("Invalid quaternion in odometry message: %s", exc)
        return None


def get_odom_transform_matrix(msg):
    """Build a 4x4 transform from an Odometry message using quaternion_matrix()."""
    try:
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        m = quaternion_matrix([o.x, o.y, o.z, o.w])
        m[0, 3] = p.x; m[1, 3] = p.y; m[2, 3] = p.z
        return m
    except AttributeError as exc:
        logger.warning("Odometry message missing expected pose fields: %s", exc)
        return None
    except (ValueError, ZeroDivisionError) as exc:
        logger.warning("Invalid quaternion in odometry message: %s", exc)
        return None


def get_closest_timestamp(ts, sorted_keys):
    if not sorted_keys:
        return None
    i = bisect.bisect_left(sorted_keys, ts)
    if i == 0:
        return sorted_keys[0]
    if i == len(sorted_keys):
        return sorted_keys[-1]
    b, a = sorted_keys[i - 1], sorted_keys[i]
    return b if (ts - b) <= (a - ts) else a


def convert_ros_image(msg):
    """Decode a ROS Image or CompressedImage message to a PIL RGB image."""
    try:
        if hasattr(msg, "format"):
            return Image.open(BytesIO(bytes(msg.data))).convert("RGB")

        enc = getattr(msg, "encoding", "bgr8")
        data = bytes(msg.data)
        h, w = int(msg.height), int(msg.width)

        if enc in ("bgr8", "rgb8", "8UC3"):
            arr = np.frombuffer(data, np.uint8).reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[:, :, ::-1]
            return Image.fromarray(arr, "RGB")

        if enc in ("mono8", "8UC1"):
            return Image.fromarray(np.frombuffer(data, np.uint8).reshape(h, w), "L").convert("RGB")

        logger.debug("Unrecognized image encoding '%s'; attempting generic decode.", enc)
        return Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, ValueError) as exc:
        logger.warning("Failed to decode image message: %s", exc)
        return None
    except AttributeError as exc:
        logger.warning("Image message missing expected fields: %s", exc)
        return None


def intrinsics_from_camera_info(msg):
    """Extract (fx, fy, cx, cy, width, height) from a CameraInfo message."""
    try:
        K = list(msg.k)
        return float(K[0]), float(K[4]), float(K[2]), float(K[5]), int(msg.width), int(msg.height)
    except AttributeError as exc:
        logger.warning("CameraInfo message missing 'k' matrix or width/height: %s", exc)
        return None
    except (IndexError, ValueError, TypeError) as exc:
        logger.warning("CameraInfo message has malformed intrinsics matrix: %s", exc)
        return None


def parse_gps_fixes(bag_path, gps_topic):
    """Read all valid NavSatFix messages on gps_topic and return the mean (lat, lon, alt)."""
    bag_path = Path(bag_path)
    fixes = []
    skipped = 0

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic == gps_topic]
        if not conns:
            raise RuntimeError(f"GPS topic '{gps_topic}' not found in bag.")

        for conn, _ts, raw in reader.messages(connections=conns):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                if int(msg.status.status) == _GPS_NO_FIX:
                    continue
                lat, lon, alt = float(msg.latitude), float(msg.longitude), float(msg.altitude)
                if all(math.isfinite(v) for v in (lat, lon, alt)):
                    fixes.append((lat, lon, alt))
                else:
                    skipped += 1
            except (AttributeError, ValueError, TypeError) as exc:
                skipped += 1
                logger.debug("Skipping malformed GPS message: %s", exc)
                continue

    if skipped:
        logger.warning("Skipped %d malformed/invalid GPS messages on '%s'.", skipped, gps_topic)

    if not fixes:
        raise RuntimeError(f"No valid GPS fixes found on topic '{gps_topic}'.")

    n = len(fixes)
    lat0 = sum(f[0] for f in fixes) / n
    lon0 = sum(f[1] for f in fixes) / n
    alt0 = sum(f[2] for f in fixes) / n
    logger.info("GPS: %d fix(es) averaged -> lat=%.7f lon=%.7f alt=%.3f m", n, lat0, lon0, alt0)
    return lat0, lon0, alt0
