#!/usr/bin/env python3
"""
gui.py - Bellhop graphical launcher.

The GUI runs on the host. Pre-flight checks and all pipelines run
inside the configured Bellhop Docker image.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, scrolledtext, ttk

logger = logging.getLogger(__name__)

_COMMON_REGISTRATION = [
    ("voxel_size", "Voxel size (m)", "entry", "0.05", {}),
    ("icp_dist_thresh", "ICP max correspondence (m)", "entry", "0.2", {}),
    ("icp_fitness_thresh", "ICP fitness threshold", "entry", "0.6", {}),
    ("odom_max_latency", "Odom max latency (s)", "entry", "0.5", {}),
    ("enable_loop_closure", "Enable loop closure", "check", False, {}),
    ("loop_closure_radius", "Loop closure radius (m)", "entry", "10.0", {}),
    ("loop_closure_fitness_thresh", "LC fitness threshold", "entry", "0.3", {}),
    ("loop_closure_search_interval", "LC search interval", "entry", "10", {}),
    ("workers", "Worker threads", "spinbox", "1", {"from_": 1, "to": 32}),
    (
        "frame_stride",
        "Registration frame stride",
        "spinbox",
        "4",
        {"from_": 1, "to": 100},
    ),
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
]

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
    ("remesh", "Remesh + smooth", "check", True, {}),
    (
        "remesh_smooth_iterations",
        "Remesh smooth iterations",
        "spinbox",
        "5",
        {"from_": 0, "to": 50},
    ),
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

_HEIGHT_FALSE_COLOR = [
    (
        "height_colormap",
        "Height false-color",
        "combobox",
        "gray",
        {"values": ["jet", "hot", "cool", "gray"]},
    ),
    (
        "height_texture_size",
        "Height texture size (px)",
        "combobox",
        "1024",
        {"values": ["256", "512", "1024", "2048", "4096"]},
    ),
]

_COMMON_TOPICS = [
    ("pc_topic", "PointCloud2 topic", "entry", "points", {}),
    ("odom_topic", "Odometry topic", "entry", "", {}),
]

_COMMON_COLOR = [
    ("camera_topic", "Camera topic", "entry", "", {}),
    ("camera_info_topic", "CameraInfo topic", "entry", "", {}),
    ("max_time_diff", "Max time difference (s)", "entry", "0.1", {}),
    ("color_min_depth", "Color minimum depth (m)", "entry", "0.1", {}),
    ("color_max_depth", "Color maximum depth (m)", "entry", "", {}),
    ("gray_filter_radius", "Gray filter radius (m)", "entry", "0.05", {}),
]

_GPS_TOPIC = [
    ("gps_topic", "GPS/NavSatFix topic", "entry", "/gps/fix", {}),
]

PROFILES = {
    "OG Map": {
        "pipeline": "og_map",
        "description": "2D Nav2 occupancy grid -> .pgm + .yaml",
        "required_topics_fields": ["pc_topic", "odom_topic"],
        "params": [
            (
                "pc_topic",
                "PointCloud2 topic",
                "entry",
                "/dlio/odom_node/pointcloud/deskewed",
                {},
            ),
            (
                "odom_topic",
                "Odometry topic",
                "entry",
                "/dlio/odom_node/odom",
                {},
            ),
            ("octree_res", "OcTree resolution (m)", "entry", "0.1", {}),
            ("grid_res", "Grid resolution (m)", "entry", "0.10", {}),
            ("slope_deg", "Ground slope (deg)", "entry", "15.0", {}),
            ("normal_radius", "Normal radius (m)", "entry", "0.2", {}),
            ("z_min", "Obstacle Z minimum (m)", "entry", "0.1", {}),
            ("z_max", "Obstacle Z maximum (m)", "entry", "2.0", {}),
            ("voxel_size", "Voxel size (m)", "entry", "0.05", {}),
            ("odom_max_latency", "Odom max latency (s)", "entry", "0.5", {}),
            (
                "frame_stride",
                "Frame stride",
                "spinbox",
                "1",
                {"from_": 1, "to": 100},
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
        ],
    },
    "Mesh": {
        "pipeline": "mesh",
        "description": (
            "Poisson surface mesh -> .ply + .obj; "
            "optional height-coloured OBJ + MTL + PNG"
        ),
        "required_topics_fields": ["pc_topic"],
        "params": (
            _COMMON_TOPICS
            + _COMMON_REGISTRATION
            + _COMMON_RECONSTRUCTION
            + _HEIGHT_FALSE_COLOR
        ),
    },
    "Color Mesh": {
        "pipeline": "color_mesh",
        "description": "Camera-coloured Poisson mesh -> .ply + .obj",
        "required_topics_fields": [
            "pc_topic",
            "camera_topic",
            "camera_info_topic",
        ],
        "params": (
            _COMMON_TOPICS
            + _COMMON_COLOR
            + _COMMON_REGISTRATION
            + _COMMON_RECONSTRUCTION
        ),
    },
    "Texture Baking": {
        "pipeline": "texture_baking",
        "description": "Keyframe-baked textured mesh (ATAK zip)",
        "required_topics_fields": [
            "pc_topic",
            "camera_topic",
            "camera_info_topic",
            "odom_topic",
        ],
        "params": (
            _COMMON_TOPICS
            + _COMMON_COLOR
            + [
                ("min_movement_m", "Keyframe min movement (m)", "entry", "0.5", {}),
                (
                    "min_rotation_deg",
                    "Keyframe min rotation (deg)",
                    "entry",
                    "15.0",
                    {},
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
                (
                    "poisson_depth",
                    "Poisson depth",
                    "spinbox",
                    "8",
                    {"from_": 4, "to": 14},
                ),
                (
                    "poisson_max_distance",
                    "Poisson max distance (m)",
                    "entry",
                    "0.5",
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
                    "5",
                    {"from_": 0, "to": 50},
                ),
                ("smooth_lambda", "Smoothing lambda", "entry", "0.5", {}),
                ("cull_min_angle", "Cull min angle (deg)", "entry", "75.0", {}),
                (
                    "target_faces",
                    "Target face count (blank = off)",
                    "entry",
                    "",
                    {},
                ),
                ("assign_min_angle", "Assign min angle (deg)", "entry", "75.0", {}),
                ("max_bake_distance", "Max bake distance (m)", "entry", "4.0", {}),
                ("min_bake_distance", "Min bake distance (m)", "entry", "0.4", {}),
                (
                    "assignment_smooth_iterations",
                    "Assignment smoothing iterations",
                    "spinbox",
                    "3",
                    {"from_": 0, "to": 20},
                ),
                (
                    "atlas_size",
                    "Atlas texture size (px)",
                    "combobox",
                    "8192",
                    {"values": ["2048", "4096", "8192", "16384"]},
                ),
            ]
        ),
    },
    "Gazebo World": {
        "pipeline": "gazebo_world",
        "description": "Gazebo simulation world -> .stl + .sdf + .world",
        "required_topics_fields": ["pc_topic"],
        "params": (
            [
                ("pc_topic", "PointCloud2 topic", "entry", "points", {}),
                ("odom_topic", "Odometry topic", "entry", "", {}),
                ("model_name", "Model name", "entry", "bag_environment", {}),
                (
                    "gazebo_material",
                    "Gazebo material",
                    "combobox",
                    "Gazebo/Grey",
                    {"values": [...]},
                ),
                # level_floor removed here — already provided by _COMMON_RECONSTRUCTION
            ]
            + _COMMON_REGISTRATION
            + _COMMON_RECONSTRUCTION
        ),
    },
    "3D Tiles": {
        "pipeline": "tiles_3d",
        "description": "Georeferenced Cesium 3D Tiles -> tileset.json",
        "required_topics_fields": ["pc_topic", "gps_topic"],
        "params": _COMMON_TOPICS + _GPS_TOPIC + _COMMON_REGISTRATION,
    },
    "Color Tiles": {
        "pipeline": "color_tiles_3d",
        "description": "Coloured georeferenced Cesium 3D Tiles -> tileset.json",
        "required_topics_fields": ["pc_topic", "gps_topic"],
        "params": (
            _COMMON_TOPICS
            + _GPS_TOPIC
            + _COMMON_COLOR
            + _COMMON_REGISTRATION
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
        row = 5

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
                continue

            value = str(value).strip()

            # Omit Auto so reconstruction selects adaptive depth.
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
