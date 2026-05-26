#!/usr/bin/env python3
"""preflight.py – Pre-flight bag topic check."""
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

def check_topics(bag_path, required_topics: list) -> list:
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    try:
        with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
            present = {c.topic for c in reader.connections}
    except Exception as e:
        raise RuntimeError(f"Could not open bag: {e}") from e
    return [t for t in required_topics if t not in present]
