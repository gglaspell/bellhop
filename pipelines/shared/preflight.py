#!/usr/bin/env python3
"""preflight.py – Pre-flight bag topic check + topic/type discovery for GUI autofill."""
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


def list_topics_by_type(bag_path) -> dict[str, list[str]]:
    """Return ``{msgtype: [topic, ...]}`` for every connection in the bag.

    Backs the GUI's "Autofill Topics" button (``_run_autofill_topics`` /
    ``_apply_autofill`` in ``gui.py``). Reads each connection's ``.topic`` and
    ``.msgtype`` straight off the bag's connection table via the same
    ``AnyReader`` used by ``check_topics`` -- no messages are deserialized,
    so this is cheap even on large bags.
    """
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    by_type: dict[str, list[str]] = {}
    try:
        with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
            for connection in reader.connections:
                by_type.setdefault(connection.msgtype, []).append(connection.topic)
    except Exception as e:
        raise RuntimeError(f"Could not open bag: {e}") from e
    return {msgtype: sorted(set(topics)) for msgtype, topics in by_type.items()}
