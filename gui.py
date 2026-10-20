#!/usr/bin/env python3
"""
gui.py - Bellhop graphical launcher.

The GUI runs on the host. Pre-flight checks and all pipelines run
inside the configured Bellhop Docker image.

PATCH NOTE (topic autofill, ported from bag2ATAK's gui.py):
Added an "Autofill Topics" button (see `_run_autofill_topics()` /
`_apply_autofill()`), placed in its own row directly under the
"PARAMETERS" header since it populates fields in that section. It scans
the bag in Docker (same pattern as `_run_preflight`) via
`list_topics_by_type()` in `pipelines/shared/preflight.py`, which reads
each connection's topic + message type off the bag's connection table
(no message deserialization). Every topic-picker field (`pc_topic`,
`odom_topic`, `camera_topic`, `camera_info_topic`) is now a
"topic_combobox": an EDITABLE combobox whose dropdown is populated with
matching topics after a scan, per `_TOPIC_TYPE_HINTS`. A field with
exactly one matching topic is auto-filled when it is blank OR holds a
value that does not exist in the bag (e.g. a preset such as "/points"
or "/odom" on a bag that names the topic differently). A value
that DOES exist in the bag is never overwritten, so presets that already
match stay put and hand-typed valid topics are respected; with several
matches only the dropdown is populated. Typing a topic by hand still
works exactly as before. The Docker image must be rebuilt so it contains
the new `list_topics_by_type()`.

PATCH NOTE (odom-anchored registration + frame-awareness wiring):
See odom-anchored-registration-fix-prompt.md and
pointcloud-frame-check-prompt-2.md. `mesh`, `color_mesh`, and `gazebo_world`
were migrated to odom-anchored registration with optional/gated ICP refinement and REP-105 frame
detection. `og_map` and `texture_baking` were separately given REP-105
frame detection (with odom-anchored local-frame point transformation, but
no ICP/loop-closure -- neither ever had a registration step to begin
with). This file wires the relevant new flags through to all five
profiles so the CLI commands the GUI builds actually use them end to end.

PATCH NOTE (motion-gated frame selection + odometry health check wiring):
Ported from `mesh.py`/`color_mesh.py`/`gazebo_world.py` (see `shared/registration.py` for the full design).
`--frame_stride` has been removed from `_COMMON_REGISTRATION` -- it no
longer exists on any of the three pipelines that share that param group,
so the GUI could never have sent it without the pipeline rejecting it as
an unrecognized argument. In its place, `_COMMON_REGISTRATION` now
includes `min_move_distance`/`min_rotation_angle_deg` (motion-gated frame
selection -- a frame is kept only if it moved/turned relative to the last
KEPT frame's odometry pose) and `disable_odom_health_check`/
`odom_loss_speed_multiplier` (the automatic odometry tracking-loss/
teleport check, enabled by default). `og_map`'s own inline `frame_stride`
field is UNCHANGED and intentionally NOT touched here -- `og_map.py` has
its own independent, index-based `--frame_stride`/`--max_frames` CLI
surface that never routed through `shared/registration.py` and was never
part of this migration (see `registration.py`'s own docstring: "NOT used
by: og_map"). `texture_baking` never had a `frame_stride` field either
and is unaffected.

PATCH NOTE (mesh/color_mesh PLY-only outputs):
`mesh` and `color_mesh` no longer produce `.obj` mesh files -- both now
write only a point-cloud `.ply` and a mesh `.ply`. `mesh`'s optional
height false-color export was also converted from a UV-textured
OBJ+MTL+PNG bundle to per-vertex colors baked directly into a PLY, so
`--height_texture_size` no longer does anything and has been removed
from the Mesh profile's parameter form.

PATCH NOTE (OG Map OcTree insertion perf flags):
`og_map.py` step [2/6] (OcTree build) gained three new CLI flags --
`--octree_max_range`, `--octree_lazy_eval`, and `--octree_discretize` --
to address OcTree-insertion becoming the dominant runtime cost on dense
LiDAR bags (see the PERF NOTE in og_map.py's module docstring). Those
flags existed in the CLI but were missing from this GUI's OG Map
profile, meaning the GUI could never pass them through to `docker run`.
They are now added directly after the existing `octree_res`/`grid_res`
fields, matching og_map.py's own defaults (`octree_max_range=40.0`,
`octree_lazy_eval=True`, `octree_discretize=False`).

PATCH NOTE (OG Map --octree_discretize default flipped to OFF -- upstream
pyoctomap crash):
The "OcTree discretize scan before insertion" checkbox previously
defaulted to checked (`True`), matching og_map.py's old default. Running
with it enabled crashes on every frame inside the pinned pyoctomap
build's `insertPointCloud(discretize=True)` path with
`AttributeError: 'OcTreeKey' object has no attribute 'thisptr'` -- a bug
in pyoctomap's own Cython `OcTreeKey` construction, not in this project's
code. og_map.py's own default was flipped to `False` for the same
reason (see that file's module docstring, "BUGFIX (--octree_discretize
now defaults OFF...)"). This checkbox is updated to match so a GUI run
with no changes and a bare CLI run with no flags keep producing
identical, working `docker run` commands. The checkbox (and the
underlying flag) are left in place, unchecked by default, in case a
future pyoctomap release fixes the bug and an operator wants to
re-enable the speedup.

BUGFIX (Texture Baking profile sent unrecognized arguments):
The Texture Baking profile previously reused `_COMMON_COLOR` (built for
the camera-*projection*-coloring pipeline `color_mesh`:
`max_time_diff`, `color_min_depth`, `color_max_depth`,
`gray_filter_radius`) and included explicit `remesh`/
`remesh_smooth_iterations` fields. `texture_baking.py`'s actual
`build_parser()` never registers ANY of those six flags -- it has no
per-point camera-projection step at all (it bakes a UV atlas from
keyframe view assignment instead), and no isotropic-remesh CLI flag
either. Sending them produced:
    unrecognized arguments: --max_time_diff ... --remesh ...
and the pipeline exited with code 2 before doing any work. Fixed by
replacing the profile's param list with one that matches
`texture_baking.py`'s argparse exactly: `camera_topic`/`camera_info_topic`
(required, but with no projection-specific fields), plus the two flags
that pipeline DOES accept but the GUI was missing entirely --
`--min_frame_points` and `--overwrite`.

PATCH NOTE (Color Mesh remeshing option removed):
`color_mesh.py` no longer accepts `--remesh`/`--remesh_smooth_iterations`
at all -- isotropic remeshing rebuilds the mesh from bare vertex/face
matrices with no color channel, so it silently destroyed every per-vertex
camera color that pipeline exists to produce (see color_mesh.py's own
PATCH NOTE). `_COMMON_RECONSTRUCTION` has been split into a base group
(no remesh) plus a separate `_REMESH_OPTIONS` group. The "Mesh" profile
(plain, uncolored mesh, where remeshing is safe -- its optional height
false-color pass runs as a *separate* post-process on the already-
remeshed output file) still includes `_REMESH_OPTIONS`. The "Color Mesh"
profile no longer does, so the GUI can never send `--remesh` /
`--remesh_smooth_iterations` to a pipeline that would reject them as
unrecognized arguments.

PATCH NOTE (sidebar pipeline order):
The `PROFILES` dict's insertion order drives the sidebar button order
(see `_build_layout`'s `for profile_name in PROFILES:` loop) and the
"first profile selected on launch" default (`list(PROFILES.keys())[0]`
in `BellhopGUI.__init__`). Order is now: OG Map, Mesh, Gazebo World,
Color Mesh, Texture Baking. This is purely a display/
default-selection change -- no profile's `params`, `pipeline` name, or
`required_topics_fields` were altered.

PATCH NOTE (3D Tiles pipelines removed):
The `tiles_3d` and `color_tiles_3d` pipelines were removed from Bellhop.
The "3D Tiles" and "Color Tiles" profiles, the `_GPS_TOPIC` param group,
and the `gps_topic` entry in `_TOPIC_TYPE_HINTS` (nothing else used a
NavSatFix topic) were removed with them. The GUI now has five profiles.

PATCH NOTE (ATAK references removed from Texture Baking):
`texture_baking.py` no longer packages its output into an ATAK-compatible
zip -- it now stops after writing the baked OBJ + PNG texture (see that
module's own PATCH NOTE). The "Texture Baking" profile's description
string here previously said "Keyframe-baked textured mesh (ATAK zip)";
updated to "...(OBJ + texture)" to match. No `params` changed -- this
profile never had an ATAK-specific field to begin with (packaging was
unconditional in the old pipeline, not a flag).
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, scrolledtext, ttk

logger = logging.getLogger(__name__)

# Maps each topic-picker argument to the ROS message type string(s) that make
# a bag topic a candidate for it. Both ROS 2 ("nav_msgs/msg/Odometry") and
# legacy ROS 1-style ("nav_msgs/Odometry") type strings are listed since
# `rosbags` can surface either depending on the bag/typestore.
_TOPIC_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "pc_topic": ("sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2"),
    "odom_topic": ("nav_msgs/msg/Odometry", "nav_msgs/Odometry"),
    "camera_topic": (
        "sensor_msgs/msg/Image",
        "sensor_msgs/Image",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/CompressedImage",
    ),
    "camera_info_topic": ("sensor_msgs/msg/CameraInfo", "sensor_msgs/CameraInfo"),
}


def _topic_type_matches(msgtype: str, hints: tuple[str, ...]) -> bool:
    """True if `msgtype` exactly matches, or contains, one of `hints`."""
    return any(hint == msgtype or hint in msgtype for hint in hints)

# ---------------------------------------------------------------------------
# Shared param groups
# ---------------------------------------------------------------------------

# PATCH: `frame_stride` removed (no longer exists on any of the three
# pipelines that share this group -- see the PATCH NOTE at the top of this
# file). `min_move_distance`/`min_rotation_angle_deg` (motion-gated frame
# selection) and `disable_odom_health_check`/`odom_loss_speed_multiplier`
# (automatic odometry tracking-loss truncation) take its place.
_COMMON_REGISTRATION = [
    ("voxel_size", "Voxel size (m)", "entry", "0.05", {}),
    ("odom_max_latency", "Odom max latency (s)", "entry", "0.5", {}),
    (
        "min_move_distance",
        "Min move distance (m)",
        "entry",
        "0.50",
        {},
    ),
    (
        "min_rotation_angle_deg",
        "Min rotation angle (deg)",
        "entry",
        "15.0",
        {},
    ),
    ("enable_icp_refinement", "Enable ICP refinement (optional, gated)", "check", False, {}),
    ("icp_dist_thresh", "ICP max correspondence (m)", "entry", "0.2", {}),
    ("icp_fitness_thresh", "ICP fitness threshold", "entry", "0.7", {}),
    ("max_icp_translation_correction", "Max ICP translation correction (m)", "entry", "0.3", {}),
    ("max_icp_rotation_correction_deg", "Max ICP rotation correction (deg)", "entry", "15.0", {}),
    ("enable_loop_closure", "Enable loop closure", "check", False, {}),
    ("loop_closure_radius", "Loop closure radius (m)", "entry", "3.0", {}),
    ("loop_closure_fitness_thresh", "LC fitness threshold", "entry", "0.7", {}),
    ("loop_closure_search_interval", "LC search interval", "entry", "10", {}),
    (
        "loop_closure_temporal_window",
        "LC temporal window (frames)",
        "spinbox",
        "100",
        {"from_": 1, "to": 10000},
    ),
    ("workers", "Worker threads", "spinbox", "1", {"from_": 1, "to": 32}),
    (
        "max_registration_frames",
        "Maximum registration frames (0 = all)",
        "entry",
        "0",
        {},
    ),
    (
        "merge_chunk_frames",
        "Merge chunk frames",
        "spinbox",
        "16",
        {"from_": 1, "to": 500},
    ),
    (
        "disable_odom_health_check",
        "Disable odometry health check",
        "check",
        False,
        {},
    ),
    (
        "odom_loss_speed_multiplier",
        "Odom loss sensitivity (x baseline)",
        "entry",
        "8.0",
        {},
    ),
]

# Used by every pipeline that has REP-105 frame detection wired up:
# mesh, color_mesh, gazebo_world (odom-anchored
# registration + frame detection); og_map and texture_baking (frame
# detection + odom-anchored local-frame point transform, no ICP/loop
# closure in either).
_PC_FRAME_MODE = [
    (
        "pc_frame_mode",
        "Point cloud frame mode",
        "combobox",
        "auto",
        {"values": ["auto", "global", "local"]},
    ),
]

# PATCH: split out of the old _COMMON_RECONSTRUCTION -- these fields are
# safe for both the plain "Mesh" pipeline and the colored "Color Mesh"
# pipeline. Remeshing is intentionally NOT in this group anymore; see
# _REMESH_OPTIONS below and the PATCH NOTE at the top of this file.
_COMMON_RECONSTRUCTION = [
    (
        "poisson_depth",
        "Poisson depth",
        "combobox",
        "Auto",
        {"values": ["Auto", "6", "7", "8", "9", "10", "11"]},
    ),
    ("min_density_percentile", "Min density percentile", "entry", "1.0", {}),
    ("distance_multiplier", "Distance trim multiplier", "entry", "3.0", {}),
    ("max_vertex_distance", "Distance trim hard cap (m)", "entry", "", {}),
    ("decimate_target", "Decimate target (ratio/>1)", "entry", "", {}),
    ("curvature_percentile", "Protect curvature percentile", "entry", "80.0", {}),
    (
        "curvature_protect_rings",
        "Curvature protection rings",
        "spinbox",
        "1",
        {"from_": 0, "to": 5},
    ),
    ("level_floor", "Level floor", "check", False, {}),
]

# PATCH: isotropic remeshing (reconstruction.remesh_isotropic) rebuilds
# the mesh from bare vertex/face matrices with no color channel, so it
# silently destroys per-vertex color. Only pipelines whose mesh has no
# per-vertex color baked in before this step -- i.e. only "Mesh" -- may
# include this group. `color_mesh.py` no longer has a `--remesh` /
# `--remesh_smooth_iterations` CLI flag at all, so "Color Mesh" MUST NOT
# include this group, or the GUI would send unrecognized arguments.
_REMESH_OPTIONS = [
    ("remesh", "Remesh + smooth", "check", True, {}),
    (
        "remesh_smooth_iterations",
        "Remesh smooth iterations",
        "spinbox",
        "5",
        {"from_": 0, "to": 50},
    ),
]

# PATCH: height false-color export is now baked as per-vertex PLY colors
# (see mesh_utils.apply_height_colormap), not a UV-textured OBJ/MTL/PNG
# bundle, so the "height_texture_size" LUT-resolution field has been
# removed -- there is no texture/LUT to size anymore.
_HEIGHT_FALSE_COLOR = [
    (
        "height_colormap",
        "Height false-color (per-vertex PLY)",
        "combobox",
        "gray",
        {"values": ["jet", "hot", "cool", "gray"]},
    ),
]

_COMMON_TOPICS = [
    ("pc_topic", "PointCloud2 topic", "topic_combobox", "/points", {}),
    ("odom_topic", "Odometry topic", "topic_combobox", "", {}),
]

# Used only by the camera-PROJECTION-coloring pipeline (color_mesh),
# which color individual points by projecting a camera
# image onto them per-frame. texture_baking does NOT use this group -- it
# has no per-point projection step (it bakes a UV atlas from keyframe view
# assignment instead), so none of these four fields apply there. See the
# BUGFIX note above.
_COMMON_COLOR = [
    ("camera_topic", "Camera topic", "topic_combobox", "", {}),
    ("camera_info_topic", "CameraInfo topic", "topic_combobox", "", {}),
    ("max_time_diff", "Max time difference (s)", "entry", "0.1", {}),
    ("color_min_depth", "Color minimum depth (m)", "entry", "0.1", {}),
    ("color_max_depth", "Color maximum depth (m)", "entry", "", {}),
    ("gray_filter_radius", "Gray filter radius (m)", "entry", "0.05", {}),
]

# PATCH: OcTree insertion perf flags for OG Map (see PATCH NOTE above).
# Defaults match og_map.py's argparse defaults exactly. `octree_discretize`
# defaults to unchecked (False) -- see the "PATCH NOTE (OG Map
# --octree_discretize default flipped to OFF...)" note at the top of this
# file for why.
_OCTREE_PERF = [
    (
        "octree_max_range",
        "OcTree max ray range (m, -1 = unlimited)",
        "entry",
        "40",
        {},
    ),
    ("octree_lazy_eval", "OcTree lazy-eval inner-node update", "check", True, {}),
    (
        "octree_discretize",
        "OcTree discretize scan before insertion ",
        "check",
        True,
        {},
    ),
]

# ---------------------------------------------------------------------------
# Pipeline profiles
#
# PATCH: insertion order here drives both the sidebar button order and
# the default-selected profile on launch (see PATCH NOTE at the top of
# this file). Order: OG Map, Mesh, 3D Tiles, Gazebo World, Color Mesh,
# Color Tiles, Texture Baking.
# ---------------------------------------------------------------------------
PROFILES = {
    "OG Map": {
        "pipeline": "og_map",
        "description": "2D Nav2 occupancy grid -> .pgm + .yaml",
        "required_topics_fields": ["pc_topic", "odom_topic"],
        "params": (
            [
                ("pc_topic", "PointCloud2 topic", "topic_combobox", "/points", {}),
                ("odom_topic", "Odometry topic", "topic_combobox", "/odom", {}),
            ]
            + _PC_FRAME_MODE
            + [
                ("octree_res", "OcTree resolution (m)", "entry", "0.10", {}),
                ("grid_res", "Grid resolution (m)", "entry", "0.10", {}),
            ]
            + _OCTREE_PERF
            + [
                ("slope_deg", "Ground slope (deg)", "entry", "15.0", {}),
                ("normal_radius", "Normal radius (m)", "entry", "0.2", {}),
                ("z_min", "Obstacle Z minimum (m)", "entry", "0.1", {}),
                ("z_max", "Obstacle Z maximum (m)", "entry", "2.0", {}),
                ("voxel_size", "Voxel size (m)", "entry", "0.10", {}),
                ("odom_max_latency", "Odom max latency (s)", "entry", "0.5", {}),
                # NOTE: og_map.py has its OWN independent, index-based
                # --frame_stride/--max_frames CLI surface -- it never routed
                # through shared/registration.py and was never part of the
                # motion-gated-selection/odometry-health-check migration
                # (see the PATCH NOTE at the top of this file). This field
                # is intentionally unchanged.
                (
                    "frame_stride",
                    "Frame stride",
                    "spinbox",
                    "2",
                    {"from_": 2, "to": 100},
                ),
                ("max_frames", "Maximum frames (0 = all)", "entry", "0", {}),
                (
                    "min_cluster_size",
                    "Minimum cluster size",
                    "spinbox",
                    "20",
                    {"from_": 0, "to": 500},
                ),
                (
                    "closing_iters",
                    "Closing iterations",
                    "spinbox",
                    "1",
                    {"from_": 0, "to": 10},
                ),
                (
                    "workers",
                    "Worker threads",
                    "spinbox",
                    "4",
                    {"from_": 1, "to": 32},
                ),
            ]
        ),
    },
    "Mesh": {
        "pipeline": "mesh",
        "description": (
            "Poisson surface mesh -> point-cloud .ply + mesh .ply; "
            "optional per-vertex height-colored PLY"
        ),
        "required_topics_fields": ["pc_topic"],
        "params": (
            _COMMON_TOPICS
            + _PC_FRAME_MODE
            + _COMMON_REGISTRATION
            + _COMMON_RECONSTRUCTION
            + _REMESH_OPTIONS
            + _HEIGHT_FALSE_COLOR
        ),
    },
    "Gazebo World": {
        "pipeline": "gazebo_world",
        "description": "Gazebo simulation world -> .stl + .sdf + .world",
        "required_topics_fields": ["pc_topic"],
        "params": (
            [
                ("pc_topic", "PointCloud2 topic", "topic_combobox", "/points", {}),
                ("odom_topic", "Odometry topic", "topic_combobox", "", {}),
            ]
            + _PC_FRAME_MODE
            + [
                ("model_name", "Model name", "entry", "bag_environment", {}),
                (
                    "gazebo_material",
                    "Gazebo material",
                    "combobox",
                    "Gazebo/Grey",
                    {
                        "values": [
                            "Gazebo/Grey",
                            "Gazebo/White",
                            "Gazebo/Wood",
                            "Gazebo/Black",
                            "Gazebo/Green",
                        ]
                    },
                ),
                # level_floor removed here -- already provided by _COMMON_RECONSTRUCTION
            ]
            + _COMMON_REGISTRATION
            + _COMMON_RECONSTRUCTION
            + _REMESH_OPTIONS
        ),
    },
    "Color Mesh": {
        "pipeline": "color_mesh",
        "description": "Camera-coloured Poisson mesh -> point-cloud .ply + mesh .ply",
        "required_topics_fields": [
            "pc_topic",
            "camera_topic",
            "camera_info_topic",
        ],
        # PATCH: no _REMESH_OPTIONS here -- color_mesh.py has no
        # --remesh/--remesh_smooth_iterations CLI flag at all anymore
        # (isotropic remeshing destroyed the per-vertex camera color this
        # pipeline exists to produce). See the PATCH NOTE at the top of
        # this file.
        "params": (
            _COMMON_TOPICS
            + _PC_FRAME_MODE
            + _COMMON_COLOR
            + _COMMON_REGISTRATION
            + _COMMON_RECONSTRUCTION
        ),
    },
    "Texture Baking": {
        "pipeline": "texture_baking",
        # PATCH: was "Keyframe-baked textured mesh (ATAK zip)" -- the
        # pipeline no longer produces an ATAK zip at all; it now stops
        # after writing the baked OBJ + PNG texture. See texture_baking.py's
        # own PATCH NOTE.
        "description": "Keyframe-baked textured mesh (OBJ + texture)",
        "required_topics_fields": [
            "pc_topic",
            "camera_topic",
            "camera_info_topic",
            "odom_topic",
        ],
        # BUGFIX: this param list now matches texture_baking.py's actual
        # argparse exactly -- no _COMMON_COLOR (that group's four fields
        # are for camera-PROJECTION coloring, which this pipeline doesn't
        # do), no remesh/remesh_smooth_iterations (no such CLI flag
        # exists here), and it now includes --min_frame_points and
        # --overwrite, which the pipeline accepts but the GUI previously
        # never exposed at all.
        "params": (
            [
                ("pc_topic", "PointCloud2 topic", "topic_combobox", "points", {}),
                ("camera_topic", "Camera topic", "topic_combobox", "", {}),
                ("camera_info_topic", "CameraInfo topic", "topic_combobox", "", {}),
                ("odom_topic", "Odometry topic", "topic_combobox", "", {}),
            ]
            + _PC_FRAME_MODE
            + [
                ("odom_max_latency", "Odom max latency (s, local frame only)", "entry", "0.5", {}),
                (
                    "min_frame_points",
                    "Min frame points",
                    "spinbox",
                    "100",
                    {"from_": 0, "to": 100000},
                ),
                ("voxel_size", "Voxel size (m)", "entry", "0.05", {}),
                ("ror_radius", "ROR radius (m, 0=off)", "entry", "0.0", {}),
                (
                    "ror_min_neighbors",
                    "ROR min neighbors",
                    "spinbox",
                    "10",
                    {"from_": 1, "to": 200},
                ),
                (
                    "sor_neighbors",
                    "SOR neighbors",
                    "spinbox",
                    "20",
                    {"from_": 1, "to": 200},
                ),
                ("sor_std_ratio", "SOR std ratio", "entry", "2.0", {}),
                ("min_movement_m", "Keyframe min movement (m)", "entry", "0.5", {}),
                (
                    "min_rotation_deg",
                    "Keyframe min rotation (deg)",
                    "entry",
                    "15.0",
                    {},
                ),
                (
                    "poisson_depth",
                    "Poisson depth",
                    "spinbox",
                    "9",
                    {"from_": 4, "to": 14},
                ),
                (
                    "poisson_max_distance",
                    "Poisson max distance (m)",
                    "entry",
                    "0.3",
                    {},
                ),
                (
                    "smooth_method",
                    "Smoothing method",
                    "combobox",
                    "taubin",
                    {"values": ["taubin", "laplacian"]},
                ),
                (
                    "smooth_iterations",
                    "Smoothing iterations",
                    "spinbox",
                    "3",
                    {"from_": 0, "to": 50},
                ),
                ("smooth_lambda", "Smoothing lambda", "entry", "0.5", {}),
                ("cull_min_angle", "Cull min angle (deg)", "entry", "65.0", {}),
                (
                    "target_faces",
                    "Target face count (blank = off)",
                    "entry",
                    "",
                    {},
                ),
                ("assign_min_angle", "Assign min angle (deg)", "entry", "70.0", {}),
                ("max_bake_distance", "Max bake distance (m)", "entry", "10.0", {}),
                ("min_bake_distance", "Min bake distance (m)", "entry", "1.0", {}),
                (
                    "assignment_smooth_iterations",
                    "Assignment smoothing iterations",
                    "spinbox",
                    "6",
                    {"from_": 0, "to": 20},
                ),
                (
                    "atlas_size",
                    "Atlas texture size (px)",
                    "combobox",
                    "8192",
                    {"values": ["2048", "4096", "8192", "16384"]},
                ),
                ("overwrite", "Overwrite existing output workspace", "check", True, {}),
            ]
        ),
    },
}

LIGHT = {
    "bg": "#f7f6f2",
    "surface": "#f9f8f5",
    "surface2": "#ffffff",
    "border": "#d4d1ca",
    "text": "#28251d",
    "text_muted": "#7a7974",
    "primary": "#01696f",
    "primary_fg": "#ffffff",
    "sidebar_sel": "#cedcd8",
    "sidebar_bg": "#f3f0ec",
    "error": "#a12c7b",
    "success": "#437a22",
    "warning": "#964219",
    "log_bg": "#1c1b19",
    "log_fg": "#cdccca",
}

DARK = {
    "bg": "#171614",
    "surface": "#1c1b19",
    "surface2": "#201f1d",
    "border": "#393836",
    "text": "#cdccca",
    "text_muted": "#797876",
    "primary": "#4f98a3",
    "primary_fg": "#171614",
    "sidebar_sel": "#313b3b",
    "sidebar_bg": "#1d1c1a",
    "error": "#d163a7",
    "success": "#6daa45",
    "warning": "#bb653b",
    "log_bg": "#0f0e0d",
    "log_fg": "#cdccca",
}


class BellhopGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("bellhop")
        self.root.minsize(920, 640)
        self.root.geometry("1100x720")

        self.dark = False
        self.palette = LIGHT
        self.current_profile = list(PROFILES.keys())[0]
        self.param_vars: dict[str, tk.Variable] = {}
        self._topic_comboboxes: dict[str, ttk.Combobox] = {}
        self.sidebar_buttons: dict[str, tk.Button] = {}
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.running = False
        self.docker_image = tk.StringVar(value="bellhop:latest")

        self._build_fonts()
        self._build_layout()
        self._select_profile(self.current_profile)
        self._apply_theme()
        self._poll_log()

    def _build_fonts(self) -> None:
        self.font_body = font.Font(family="Helvetica Neue", size=12)
        self.font_small = font.Font(family="Helvetica Neue", size=11)
        self.font_mono = font.Font(family="Courier", size=11)
        self.font_title = font.Font(
            family="Helvetica Neue",
            size=15,
            weight="bold",
        )
        self.font_label = font.Font(family="Helvetica Neue", size=11)
        self.font_sidebar = font.Font(family="Helvetica Neue", size=12)

    def _build_layout(self) -> None:
        self.header = tk.Frame(self.root, height=48)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)

        self.title_label = tk.Label(
            self.header,
            text="\u2b21 bellhop",
            font=self.font_title,
            padx=16,
        )
        self.title_label.pack(side="left", fill="y")

        self.theme_button = tk.Button(
            self.header,
            text="\u25d1",
            font=self.font_body,
            relief="flat",
            padx=12,
            cursor="hand2",
            command=self._toggle_theme,
        )
        self.theme_button.pack(side="right", padx=8)

        self.main_pane = tk.PanedWindow(
            self.root,
            orient="horizontal",
            sashwidth=1,
        )
        self.main_pane.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(self.main_pane, width=180)
        self.sidebar.pack_propagate(False)
        self.main_pane.add(self.sidebar, minsize=150)

        tk.Label(
            self.sidebar,
            text="PIPELINE",
            font=self.font_small,
            padx=12,
            pady=10,
            anchor="w",
        ).pack(fill="x")

        for profile_name in PROFILES:
            button = tk.Button(
                self.sidebar,
                text=profile_name,
                font=self.font_sidebar,
                relief="flat",
                anchor="w",
                padx=14,
                pady=8,
                cursor="hand2",
                command=lambda name=profile_name: self._select_profile(name),
            )
            button.pack(fill="x")
            self.sidebar_buttons[profile_name] = button

        self.right_panel = tk.Frame(self.main_pane)
        self.main_pane.add(self.right_panel, minsize=600)

        self.vertical_pane = tk.PanedWindow(
            self.right_panel,
            orient="vertical",
            sashwidth=1,
        )
        self.vertical_pane.pack(fill="both", expand=True)

        self.form_outer = tk.Frame(self.vertical_pane)
        self.vertical_pane.add(self.form_outer, minsize=340)

        self.canvas = tk.Canvas(self.form_outer, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self.form_outer,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.form_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.form_frame,
            anchor="nw",
        )
        self.form_frame.bind("<Configure>", self._on_form_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.log_frame = tk.Frame(self.vertical_pane)
        self.vertical_pane.add(self.log_frame, minsize=150)

        log_header = tk.Frame(self.log_frame)
        log_header.pack(fill="x")

        tk.Label(
            log_header,
            text="Output",
            font=self.font_small,
            padx=8,
            pady=4,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        tk.Button(
            log_header,
            text="Clear",
            font=self.font_small,
            relief="flat",
            padx=8,
            cursor="hand2",
            command=self._clear_log,
        ).pack(side="right", padx=4)

        self.log = scrolledtext.ScrolledText(
            self.log_frame,
            font=self.font_mono,
            wrap="word",
            relief="flat",
            borderwidth=0,
            state="disabled",
        )
        self.log.pack(fill="both", expand=True)

    def _on_form_configure(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _select_profile(self, profile_name: str) -> None:
        self.current_profile = profile_name
        self._rebuild_form()
        self._update_sidebar_highlight()

    def _update_sidebar_highlight(self) -> None:
        colors = self.palette

        for name, button in self.sidebar_buttons.items():
            if name == self.current_profile:
                button.configure(
                    bg=colors["sidebar_sel"],
                    fg=colors["primary"],
                    font=font.Font(
                        family="Helvetica Neue",
                        size=12,
                        weight="bold",
                    ),
                )
            else:
                button.configure(
                    bg=colors["sidebar_bg"],
                    fg=colors["text"],
                    font=self.font_sidebar,
                )

    def _rebuild_form(self) -> None:
        for widget in self.form_frame.winfo_children():
            widget.destroy()

        self.param_vars.clear()
        self._topic_comboboxes.clear()
        profile = PROFILES[self.current_profile]

        tk.Label(
            self.form_frame,
            text=profile["description"],
            font=self.font_small,
            anchor="w",
            pady=6,
            padx=16,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        self._build_path_row(1, "Bag path", "bag_path", "bag")
        self._build_path_row(2, "Output directory", "output_path", "dir")
        self._build_docker_image_row(3)

        self._section_label(4, "Parameters")

        # "Autofill Topics" lives directly under the Parameters header (not
        # the Pre-flight row) since it populates fields IN that section --
        # see `_run_autofill_topics` / `_apply_autofill`.
        autofill_frame = tk.Frame(self.form_frame)
        autofill_frame.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=16,
            pady=(0, 6),
        )
        self.autofill_button = tk.Button(
            autofill_frame,
            text="Autofill Topics",
            font=self.font_body,
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._run_autofill_topics,
        )
        self.autofill_button.pack(side="left")
        tk.Label(
            autofill_frame,
            text="Scans the bag and fills in the topic fields below.",
            font=self.font_small,
            padx=8,
        ).pack(side="left")

        row = 6

        for argument, label, widget_type, default, extra in profile["params"]:
            row = self._build_param_row(
                row,
                argument,
                label,
                widget_type,
                default,
                extra,
            )

        self._section_label(row, "Pre-flight")
        row += 1

        preflight_frame = tk.Frame(self.form_frame)
        preflight_frame.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=16,
            pady=4,
        )
        row += 1

        self.check_button = tk.Button(
            preflight_frame,
            text="Check Topics",
            font=self.font_body,
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._run_preflight,
        )
        self.check_button.pack(side="left")

        self.preflight_label = tk.Label(
            preflight_frame,
            text="",
            font=self.font_small,
            padx=12,
        )
        self.preflight_label.pack(side="left", fill="x", expand=True)

        self.run_button = tk.Button(
            self.form_frame,
            text="\u25b6 Run Pipeline",
            font=font.Font(
                family="Helvetica Neue",
                size=13,
                weight="bold",
            ),
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._run_pipeline,
        )
        self.run_button.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=16,
            pady=14,
        )

        self.form_frame.columnconfigure(1, weight=1)
        self._apply_theme()

    def _section_label(self, row: int, text: str) -> None:
        frame = tk.Frame(self.form_frame)
        frame.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=16,
            pady=(12, 2),
        )
        tk.Label(
            frame,
            text=text.upper(),
            font=self.font_small,
            anchor="w",
        ).pack(side="left")

    def _build_path_row(
        self,
        row: int,
        label: str,
        variable_key: str,
        kind: str,
    ) -> None:
        variable = tk.StringVar()
        self.param_vars[variable_key] = variable

        tk.Label(
            self.form_frame,
            text=label,
            font=self.font_label,
            anchor="w",
            padx=16,
        ).grid(row=row, column=0, sticky="w", pady=3)

        tk.Entry(
            self.form_frame,
            textvariable=variable,
            font=self.font_body,
            relief="flat",
            bd=1,
        ).grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=3)

        def browse() -> None:
            title = (
                "Select ROS 2 bag directory"
                if kind == "bag"
                else "Select output directory"
            )
            selected = filedialog.askdirectory(title=title)
            if selected:
                variable.set(selected)

        tk.Button(
            self.form_frame,
            text="Browse",
            font=self.font_small,
            relief="flat",
            padx=8,
            cursor="hand2",
            command=browse,
        ).grid(row=row, column=2, padx=(0, 16), pady=3)

    def _build_docker_image_row(self, row: int) -> None:
        tk.Label(
            self.form_frame,
            text="Docker image",
            font=self.font_label,
            anchor="w",
            padx=16,
        ).grid(row=row, column=0, sticky="w", pady=3)

        tk.Entry(
            self.form_frame,
            textvariable=self.docker_image,
            font=self.font_body,
            relief="flat",
            bd=1,
        ).grid(
            row=row,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=(0, 16),
            pady=3,
        )

    def _build_param_row(
        self,
        row: int,
        argument: str,
        label: str,
        widget_type: str,
        default,
        extra: dict,
    ) -> int:
        tk.Label(
            self.form_frame,
            text=label,
            font=self.font_label,
            anchor="w",
            padx=16,
        ).grid(row=row, column=0, sticky="w", pady=2)

        if widget_type == "check":
            variable = tk.BooleanVar(value=bool(default))
            tk.Checkbutton(
                self.form_frame,
                variable=variable,
                relief="flat",
                cursor="hand2",
            ).grid(row=row, column=1, sticky="w", pady=2)

        elif widget_type == "spinbox":
            variable = tk.StringVar(value=str(default))
            tk.Spinbox(
                self.form_frame,
                textvariable=variable,
                font=self.font_body,
                relief="flat",
                bd=1,
                from_=extra.get("from_", 0),
                to=extra.get("to", 9999),
                width=10,
            ).grid(row=row, column=1, sticky="w", pady=2)

        elif widget_type == "combobox":
            variable = tk.StringVar(value=str(default))
            ttk.Combobox(
                self.form_frame,
                textvariable=variable,
                values=extra.get("values", []),
                font=self.font_body,
                state="readonly",
                width=24,
            ).grid(row=row, column=1, sticky="w", pady=2)

        elif widget_type == "topic_combobox":
            # Editable (state="normal", NOT "readonly") so a topic typed by
            # hand -- before scanning, or one the scan did not recognize --
            # is still accepted. Dropdown values are filled in by
            # `_apply_autofill` after "Autofill Topics" scans the bag.
            variable = tk.StringVar(value="" if default is None else str(default))
            combobox = ttk.Combobox(
                self.form_frame,
                textvariable=variable,
                values=(),
                font=self.font_body,
                state="normal",
                width=24,
            )
            combobox.grid(
                row=row,
                column=1,
                columnspan=2,
                sticky="ew",
                padx=(0, 16),
                pady=2,
            )
            self._topic_comboboxes[argument] = combobox

        else:
            variable = tk.StringVar(
                value=str(default) if default is not None else ""
            )
            tk.Entry(
                self.form_frame,
                textvariable=variable,
                font=self.font_body,
                relief="flat",
                bd=1,
            ).grid(
                row=row,
                column=1,
                columnspan=2,
                sticky="ew",
                padx=(0, 16),
                pady=2,
            )

        self.param_vars[argument] = variable
        return row + 1

    def _run_preflight(self) -> None:
        bag_path = self.param_vars.get(
            "bag_path",
            tk.StringVar(),
        ).get().strip()

        if not bag_path:
            self._set_preflight("\u26a0 No bag path set.", "warning")
            return

        bag_dir = Path(bag_path).expanduser().resolve()

        if not bag_dir.is_dir():
            self._set_preflight("\u2717 Bag directory does not exist.", "error")
            return

        profile = PROFILES[self.current_profile]
        required_topics = []

        for field in profile["required_topics_fields"]:
            value = self.param_vars.get(
                field,
                tk.StringVar(),
            ).get().strip()
            if value:
                required_topics.append(value)

        if not required_topics:
            self._set_preflight(
                "\u26a0 No required topics configured.",
                "warning",
            )
            return

        self._set_preflight("Checking in Docker...", "muted")

        image = self.docker_image.get().strip() or "bellhop:latest"
        topic_literal = repr(required_topics)

        check_code = (
            "from pathlib import Path; "
            "from pipelines.shared.preflight import check_topics; "
            f"missing = check_topics(Path('/data/bag'), {topic_literal}); "
            "print('MISSING: ' + ', '.join(missing) if missing else 'OK')"
        )

        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{bag_dir}:/data/bag:ro",
            "--entrypoint",
            "python",
            image,
            "-c",
            check_code,
        ]

        def worker() -> None:
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                output = completed.stdout.strip()

                if completed.returncode != 0:
                    self.log_queue.put((
                        "preflight_error",
                        output or "Docker pre-flight command failed.",
                    ))
                elif output.startswith("MISSING:"):
                    self.log_queue.put(("preflight_error", output))
                else:
                    self.log_queue.put(("preflight_ok", output or "OK"))

            except FileNotFoundError:
                self.log_queue.put((
                    "preflight_error",
                    "Docker was not found on PATH.",
                ))
            except (subprocess.SubprocessError, OSError) as exc:
                # subprocess.run() can raise OSError (e.g. permission
                # denied executing docker) or SubprocessError subclasses.
                # Anything else is an unexpected bug and should propagate.
                self.log_queue.put((
                    "preflight_error",
                    f"Pre-flight failed: {exc}",
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _run_autofill_topics(self) -> None:
        """Scan the bag (in Docker) and populate the topic comboboxes.

        Uses `list_topics_by_type()` (pipelines/shared/preflight.py) to get
        {msgtype: [topic, ...]}, then matches each topic field against
        `_TOPIC_TYPE_HINTS`. Dropdowns are always populated; a field is
        auto-filled when exactly one topic matches and the field is blank or
        holds a topic that does not exist in the bag. Values that exist in
        the bag are kept.
        """
        bag_path = self.param_vars.get("bag_path", tk.StringVar()).get().strip()

        if not bag_path:
            self._set_preflight("\u26a0 No bag path set.", "warning")
            return

        bag_dir = Path(bag_path).expanduser().resolve()

        if not bag_dir.is_dir():
            self._set_preflight("\u2717 Bag directory does not exist.", "error")
            return

        self._set_preflight("Scanning bag topics...", "muted")
        try:
            self.autofill_button.configure(state="disabled")
        except tk.TclError as exc:
            logger.debug("Could not disable autofill button: %s", exc)

        image = self.docker_image.get().strip() or "bellhop:latest"
        scan_code = (
            "import json\n"
            "from pathlib import Path\n"
            "from pipelines.shared.preflight import list_topics_by_type\n"
            "print(json.dumps(list_topics_by_type(Path('/data/bag'))))"
        )

        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{bag_dir}:/data/bag:ro",
            "--entrypoint",
            "python",
            image,
            "-c",
            scan_code,
        ]

        def worker() -> None:
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                output = completed.stdout.strip()

                if completed.returncode != 0:
                    self.log_queue.put((
                        "autofill_error",
                        output or "Docker topic scan failed.",
                    ))
                    return
                if not output:
                    self.log_queue.put((
                        "autofill_error",
                        "Topic scan produced no output.",
                    ))
                    return
                # The JSON is expected to be the last line of stdout; any
                # incidental warnings printed before it are harmless noise.
                self.log_queue.put(("autofill_ok", output.splitlines()[-1]))

            except FileNotFoundError:
                self.log_queue.put((
                    "autofill_error",
                    "Docker was not found on PATH.",
                ))
            except (subprocess.SubprocessError, OSError) as exc:
                self.log_queue.put((
                    "autofill_error",
                    f"Topic scan failed: {exc}",
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_autofill(self, topics_by_type: dict[str, list[str]]) -> None:
        scanned: list[str] = []
        filled: list[str] = []
        all_topics = {t for topics in topics_by_type.values() for t in topics}

        for argument, hints in _TOPIC_TYPE_HINTS.items():
            combobox = self._topic_comboboxes.get(argument)
            if combobox is None:
                continue  # field not present in the current profile

            matches = sorted(
                {
                    topic
                    for msgtype, topics in topics_by_type.items()
                    if _topic_type_matches(msgtype, hints)
                    for topic in topics
                }
            )
            if not matches:
                continue

            try:
                combobox["values"] = matches
            except tk.TclError as exc:
                logger.debug("Combobox for %s is gone: %s", argument, exc)
                continue

            scanned.append(argument)
            variable = self.param_vars.get(argument)
            if variable is None or len(matches) != 1:
                continue
            current = str(variable.get()).strip()
            # Fill when blank, or when the current value (e.g. a profile
            # preset such as "/points") is not a topic that exists in this
            # bag at all. A value that IS present in the bag is never touched.
            if not current or current not in all_topics:
                variable.set(matches[0])
                filled.append(argument)

        if scanned:
            message = f"Found topics for: {', '.join(scanned)}"
            if filled:
                message += f" (auto-filled: {', '.join(filled)})"
            self._set_preflight(f"\u2713 {message}", "ok")
            self._log(message + "\n", "ok")
        else:
            self._set_preflight(
                "\u26a0 No recognizable topics found in this bag.",
                "warning",
            )
            self._log("Autofill scan found no recognizable topic types.\n", "warning")

    def _finish_autofill_button(self) -> None:
        try:
            self.autofill_button.configure(state="normal")
        except (tk.TclError, AttributeError) as exc:
            logger.debug("Could not re-enable autofill button: %s", exc)

    def _set_preflight(self, message: str, style: str) -> None:
        colors = self.palette

        color = {
            "ok": colors["success"],
            "error": colors["error"],
            "warning": colors["warning"],
            "muted": colors["text_muted"],
        }.get(style, colors["text"])

        try:
            self.preflight_label.configure(text=message, fg=color)
        except tk.TclError as exc:
            # The label widget may have been destroyed mid-rebuild (e.g.
            # the user switched profiles while a background preflight
            # thread was still delivering its result).
            logger.debug("Could not update preflight label (widget gone?): %s", exc)

    def _build_command(self) -> list[str] | None:
        bag_path = self.param_vars.get(
            "bag_path",
            tk.StringVar(),
        ).get().strip()

        output_path = self.param_vars.get(
            "output_path",
            tk.StringVar(),
        ).get().strip()

        if not bag_path:
            self._log("ERROR: No bag path specified.\n", "error")
            return None

        if not output_path:
            self._log("ERROR: No output directory specified.\n", "error")
            return None

        bag_dir = Path(bag_path).expanduser().resolve()
        output_dir = Path(output_path).expanduser().resolve()

        if not bag_dir.is_dir():
            self._log(
                f"ERROR: Bag directory does not exist: {bag_dir}\n",
                "error",
            )
            return None

        output_dir.mkdir(parents=True, exist_ok=True)

        profile = PROFILES[self.current_profile]
        image = self.docker_image.get().strip() or "bellhop:latest"

        container_output = "/data/output"

        # OG Map expects a base filename rather than an output directory.
        if profile["pipeline"] == "og_map":
            container_output = "/data/output/occupancy_map"

        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{bag_dir}:/data/bag:ro",
            "-v",
            f"{output_dir}:/data/output",
            image,
            profile["pipeline"],
            "/data/bag",
            container_output,
        ]

        for argument, _label, widget_type, _default, _extra in profile["params"]:
            variable = self.param_vars.get(argument)

            if variable is None:
                continue

            value = variable.get()

            if widget_type == "check":
                if value:
                    command.append(f"--{argument}")
                elif argument == "remesh":
                    command.append("--no-remesh")
                elif argument == "octree_lazy_eval":
                    # PATCH: og_map.py's --octree_lazy_eval defaults to True
                    # via action="store_true", default=True, with a paired
                    # --no-octree_lazy_eval flag to turn it off. Mirror that
                    # here: an unchecked box must explicitly disable it.
                    command.append("--no-octree_lazy_eval")
                # NOTE: no elif branch is needed for "octree_discretize".
                # Its GUI default and og_map.py's argparse default are both
                # now False/off (see the "PATCH NOTE (OG Map
                # --octree_discretize default flipped to OFF...)" note at
                # the top of this file), so an unchecked box already
                # matches the pipeline's own default and needs no explicit
                # --no-octree_discretize flag. Checking the box appends
                # --octree_discretize via the `if value:` branch above.
                continue

            value = str(value).strip()

            # Omit "Auto" so reconstruction selects adaptive depth.
            if argument == "poisson_depth" and value.lower() == "auto":
                continue

            if value:
                command.extend([f"--{argument}", value])

        return command

    def _run_pipeline(self) -> None:
        if self.running:
            self._log(
                "A pipeline is already running. Please wait.\n",
                "warning",
            )
            return

        command = self._build_command()

        if command is None:
            return

        self._log(f"$ {' '.join(command)}\n\n", "muted")
        self.running = True
        self.run_button.configure(state="disabled", text="\u23f3 Running...")

        def worker() -> None:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                if process.stdout is not None:
                    for line in process.stdout:
                        self.log_queue.put(("log", line))

                process.wait()

                if process.returncode == 0:
                    self.log_queue.put((
                        "done_ok",
                        "Pipeline finished successfully.\n",
                    ))
                else:
                    self.log_queue.put((
                        "done_error",
                        f"Pipeline exited with code {process.returncode}.\n",
                    ))

            except FileNotFoundError:
                self.log_queue.put((
                    "done_error",
                    "Docker was not found on PATH.\n",
                ))
            except OSError as exc:
                # Popen can raise OSError for permission issues or other
                # OS-level failures starting the subprocess.
                self.log_queue.put((
                    "done_error",
                    f"Failed to start pipeline: {exc}\n",
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_log(self) -> None:
        try:
            while True:
                kind, text = self.log_queue.get_nowait()

                if kind == "log":
                    self._log(text)

                elif kind == "preflight_ok":
                    self._set_preflight(f"\u2713 {text}", "ok")
                    self._log(f"Pre-flight OK: {text}\n", "ok")

                elif kind == "preflight_error":
                    self._set_preflight(f"\u2717 {text}", "error")
                    self._log(f"Pre-flight FAIL: {text}\n", "error")

                elif kind == "autofill_ok":
                    self._finish_autofill_button()
                    try:
                        topics_by_type = json.loads(text)
                        if not isinstance(topics_by_type, dict):
                            raise ValueError("expected a JSON object")
                    except ValueError as exc:  # JSONDecodeError subclasses ValueError
                        self._set_preflight(
                            "\u2717 Could not parse topic scan output.",
                            "error",
                        )
                        self._log(
                            f"Autofill FAIL: could not parse scan output ({exc}): "
                            f"{text}\n",
                            "error",
                        )
                    else:
                        self._apply_autofill(topics_by_type)

                elif kind == "autofill_error":
                    self._finish_autofill_button()
                    self._set_preflight(f"\u2717 {text}", "error")
                    self._log(f"Autofill FAIL: {text}\n", "error")

                elif kind == "done_ok":
                    self._log(text, "ok")
                    self.running = False
                    self.run_button.configure(
                        state="normal",
                        text="\u25b6 Run Pipeline",
                    )

                elif kind == "done_error":
                    self._log(text, "error")
                    self.running = False
                    self.run_button.configure(
                        state="normal",
                        text="\u25b6 Run Pipeline",
                    )

        except queue.Empty:
            pass

        self.root.after(100, self._poll_log)

    def _log(self, text: str, style: str = "normal") -> None:
        colors = self.palette

        color = {
            "normal": colors["log_fg"],
            "muted": colors["text_muted"],
            "ok": colors["success"],
            "error": colors["error"],
            "warning": colors["warning"],
        }.get(style, colors["log_fg"])

        self.log.configure(state="normal")

        tag_name = f"tag_{style}"
        self.log.tag_configure(tag_name, foreground=color)
        self.log.insert("end", text, tag_name)
        self.log.see("end")

        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _toggle_theme(self) -> None:
        self.dark = not self.dark
        self.palette = DARK if self.dark else LIGHT
        self._apply_theme()

    def _apply_theme(self) -> None:
        colors = self.palette

        def apply_to_children(widget) -> None:
            widget_class = widget.winfo_class()

            try:
                if widget_class in ("Frame", "Canvas"):
                    widget.configure(bg=colors["bg"])

                elif widget_class == "Label":
                    widget.configure(bg=colors["bg"], fg=colors["text"])

                elif widget_class == "Button":
                    widget.configure(
                        bg=colors["surface2"],
                        fg=colors["text"],
                        activebackground=colors["sidebar_sel"],
                        activeforeground=colors["text"],
                    )

                elif widget_class == "Entry":
                    widget.configure(
                        bg=colors["surface2"],
                        fg=colors["text"],
                        insertbackground=colors["text"],
                        relief="flat",
                        highlightthickness=1,
                        highlightbackground=colors["border"],
                        highlightcolor=colors["primary"],
                    )

                elif widget_class == "Spinbox":
                    widget.configure(
                        bg=colors["surface2"],
                        fg=colors["text"],
                        buttonbackground=colors["surface2"],
                    )

                elif widget_class == "Checkbutton":
                    widget.configure(
                        bg=colors["bg"],
                        fg=colors["text"],
                        selectcolor=colors["surface2"],
                        activebackground=colors["bg"],
                    )

                elif widget_class == "Text":
                    widget.configure(
                        bg=colors["log_bg"],
                        fg=colors["log_fg"],
                    )

            except tk.TclError as exc:
                # A widget on this platform/Tk build may not support one
                # of the options passed to configure() for its class
                # (rare). Log it instead of hiding it; a KeyError from a
                # missing palette entry will still propagate as a bug.
                logger.debug(
                    "Theme option unsupported for widget class '%s': %s",
                    widget_class, exc,
                )

            for child in widget.winfo_children():
                apply_to_children(child)

        self.root.configure(bg=colors["bg"])
        apply_to_children(self.root)

        self.header.configure(bg=colors["surface"])
        self.title_label.configure(
            bg=colors["surface"],
            fg=colors["primary"],
        )
        self.theme_button.configure(
            bg=colors["surface"],
            fg=colors["text_muted"],
            activebackground=colors["surface"],
        )

        self.sidebar.configure(bg=colors["sidebar_bg"])

        for widget in self.sidebar.winfo_children():
            if isinstance(widget, tk.Label):
                widget.configure(
                    bg=colors["sidebar_bg"],
                    fg=colors["text_muted"],
                )

        try:
            self.run_button.configure(
                bg=colors["primary"],
                fg=colors["primary_fg"],
                activebackground=colors["primary"],
                activeforeground=colors["primary_fg"],
            )
        except tk.TclError as exc:
            # run_button may not exist yet on the very first call from
            # __init__ before _build_layout() has finished, or may be
            # mid-destruction during a profile switch.
            logger.debug("Could not theme run_button: %s", exc)

        self._update_sidebar_highlight()


def main() -> None:
    root = tk.Tk()

    try:
        root.tk.call("tk", "scaling", 1.3)
    except tk.TclError as exc:
        # Some minimal Tk builds do not support the "scaling"
        # subcommand; the GUI still works at default DPI scaling.
        logger.warning("Could not set Tk DPI scaling to 1.3: %s", exc)

    BellhopGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
