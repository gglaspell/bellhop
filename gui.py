#!/usr/bin/env python3
"""
gui.py – bellhop graphical launcher (Tkinter).

Layout
------
┌─────────────────────────────────────────────────────────────────┐
│  header bar                                            [◑ theme]│
├──────────────┬──────────────────────────────────────────────────┤
│              │  ┌── Bag file ─────────────────────────────────┐ │
│  OG Map      │  │ /path/to/bag          [Browse]              │ │
│  Mesh        │  └─────────────────────────────────────────────┘ │
│  Color Mesh  │  ┌── Output directory ─────────────────────────┐ │
│  Gazebo      │  │ /path/to/output       [Browse]              │ │
│  3D Tiles    │  └─────────────────────────────────────────────┘ │
│  Color Tiles │  ┌── Parameters ───────────────────────────────┐ │
│              │  │  (profile-specific fields)                  │ │
│              │  └─────────────────────────────────────────────┘ │
│              │  ┌── Pre-flight ───────────────────────────────┐ │
│              │  │  [Check Topics]   status line               │ │
│              │  └─────────────────────────────────────────────┘ │
│              │  [  Run Pipeline  ]                              │ │
├──────────────┴──────────────────────────────────────────────────┤
│  log / output console                                           │
└─────────────────────────────────────────────────────────────────┘
"""

import importlib
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, scrolledtext, ttk

# ---------------------------------------------------------------------------
# Parameter definitions per profile
# Each entry: (arg_name, label, widget_type, default, extra)
# widget_type: "entry" | "check" | "spinbox" | "combobox"
# extra: dict with options (e.g. {"values": [...]} for combobox,
#                                 {"from_": 0, "to": 20} for spinbox)
# ---------------------------------------------------------------------------

_COMMON_REGISTRATION = [
    ("voxel_size",           "Voxel size (m)",            "entry",   "0.05", {}),
    ("icp_dist_thresh",      "ICP max correspondence (m)", "entry",   "0.2",  {}),
    ("icp_fitness_thresh",   "ICP fitness threshold",      "entry",   "0.6",  {}),
    ("odom_max_latency",     "Odom max latency (s)",       "entry",   "0.5",  {}),
    ("enable_loop_closure",  "Enable loop closure",        "check",   False,  {}),
    ("loop_closure_radius",  "Loop closure radius (m)",    "entry",   "10.0", {}),
    ("loop_closure_fitness_thresh",   "LC fitness thresh", "entry",   "0.3",  {}),
    ("loop_closure_search_interval",  "LC search interval","entry",   "10",   {}),
    ("workers",              "Worker threads",             "spinbox", "4",
     {"from_": 1, "to": 32}),
]

_COMMON_RECONSTRUCTION = [
    ("poisson_depth",          "Poisson depth",              "spinbox", "9",
     {"from_": 5, "to": 14}),
    ("min_density_percentile", "Min density percentile",     "entry",   "1.0", {}),
    ("max_vertex_distance",    "Max vertex distance (m)",    "entry",   "0.15",{}),
    ("decimate_target",        "Decimate target (ratio/>1)", "entry",   "",    {}),
    ("level_floor",            "Level floor",                "check",   False, {}),
]

_COMMON_TOPICS = [
    ("pc_topic",   "PointCloud2 topic",  "entry", "points",    {}),
    ("odom_topic", "Odometry topic",     "entry", "",          {}),
]

_COMMON_COLOR = [
    ("camera_topic",      "Camera topic",       "entry", "",    {}),
    ("camera_info_topic", "CameraInfo topic",   "entry", "",    {}),
    ("max_time_diff",     "Max time diff (s)",  "entry", "0.1", {}),
    ("color_min_depth",   "Color min depth (m)","entry", "0.1", {}),
    ("color_max_depth",   "Color max depth (m)","entry", "",    {}),
    ("gray_filter_radius","Gray filter radius",  "entry", "0.05",{}),
]

_GPS_TOPIC = [
    ("gps_topic", "GPS/NavSatFix topic", "entry", "/gps/fix", {}),
]

PROFILES = {
    "OG Map": {
        "pipeline": "og_map",
        "description": "2D Nav2 occupancy grid  →  .pgm + .yaml",
        "required_topics_fields": ["pc_topic", "odom_topic"],
        "params": [
            ("pc_topic",         "PointCloud2 topic",   "entry",   "/dlio/odom_node/pointcloud/deskewed", {}),
            ("odom_topic",       "Odometry topic",      "entry",   "/dlio/odom_node/odom",               {}),
            ("octree_res",       "OcTree resolution (m)","entry",  "0.1",  {}),
            ("grid_res",         "Grid resolution (m)", "entry",   "0.05", {}),
            ("slope_deg",        "Ground slope (deg)",  "entry",   "15.0", {}),
            ("normal_radius",    "Normal radius (m)",   "entry",   "0.2",  {}),
            ("z_min",            "Obstacle Z min (m)",  "entry",   "0.1",  {}),
            ("z_max",            "Obstacle Z max (m)",  "entry",   "2.0",  {}),
            ("voxel_size",       "Voxel size (m)",      "entry",   "0.05", {}),
            ("odom_max_latency", "Odom max latency (s)","entry",   "0.5",  {}),
            ("min_cluster_size", "Min cluster size",    "spinbox", "20",   {"from_": 0, "to": 500}),
            ("closing_iters",    "Closing iterations",  "spinbox", "1",    {"from_": 0, "to": 10}),
            ("workers",          "Worker threads",      "spinbox", "4",    {"from_": 1, "to": 32}),
        ],
    },
    "Mesh": {
        "pipeline": "mesh",
        "description": "Poisson surface mesh  →  .ply + .obj",
        "required_topics_fields": ["pc_topic"],
        "params": _COMMON_TOPICS + _COMMON_REGISTRATION + _COMMON_RECONSTRUCTION,
    },
    "Color Mesh": {
        "pipeline": "color_mesh",
        "description": "Camera-colored Poisson mesh  →  .ply + .obj",
        "required_topics_fields": ["pc_topic"],
        "params": _COMMON_TOPICS + _COMMON_COLOR + _COMMON_REGISTRATION + _COMMON_RECONSTRUCTION,
    },
    "Gazebo World": {
        "pipeline": "gazebo_world",
        "description": "Gazebo simulation world  →  .stl + .sdf + .world",
        "required_topics_fields": ["pc_topic"],
        "params": [
            ("pc_topic",        "PointCloud2 topic",  "entry", "points",           {}),
            ("odom_topic",      "Odometry topic",     "entry", "",                 {}),
            ("model_name",      "Model name",         "entry", "bag_environment",  {}),
            ("gazebo_material", "Gazebo material",    "combobox", "Gazebo/Grey",
             {"values": ["Gazebo/Grey","Gazebo/White","Gazebo/Black",
                         "Gazebo/Wood","Gazebo/Bricks","Gazebo/Grass"]}),
        ] + _COMMON_REGISTRATION + _COMMON_RECONSTRUCTION,
    },
    "3D Tiles": {
        "pipeline": "tiles_3d",
        "description": "Georeferenced Cesium 3D Tiles  →  tileset.json",
        "required_topics_fields": ["pc_topic", "gps_topic"],
        "params": _COMMON_TOPICS + _GPS_TOPIC + _COMMON_REGISTRATION,
    },
    "Color Tiles": {
        "pipeline": "color_tiles_3d",
        "description": "Colored georeferenced Cesium 3D Tiles  →  tileset.json",
        "required_topics_fields": ["pc_topic", "gps_topic"],
        "params": _COMMON_TOPICS + _GPS_TOPIC + _COMMON_COLOR + _COMMON_REGISTRATION,
    },
}

# ---------------------------------------------------------------------------
# Color palette (light / dark)
# ---------------------------------------------------------------------------
LIGHT = {
    "bg":           "#f7f6f2",
    "surface":      "#f9f8f5",
    "surface2":     "#ffffff",
    "border":       "#d4d1ca",
    "divider":      "#dcd9d5",
    "text":         "#28251d",
    "text_muted":   "#7a7974",
    "primary":      "#01696f",
    "primary_fg":   "#ffffff",
    "sidebar_sel":  "#cedcd8",
    "sidebar_bg":   "#f3f0ec",
    "error":        "#a12c7b",
    "success":      "#437a22",
    "warning":      "#964219",
    "log_bg":       "#1c1b19",
    "log_fg":       "#cdccca",
}
DARK = {
    "bg":           "#171614",
    "surface":      "#1c1b19",
    "surface2":     "#201f1d",
    "border":       "#393836",
    "divider":      "#262523",
    "text":         "#cdccca",
    "text_muted":   "#797876",
    "primary":      "#4f98a3",
    "primary_fg":   "#171614",
    "sidebar_sel":  "#313b3b",
    "sidebar_bg":   "#1d1c1a",
    "error":        "#d163a7",
    "success":      "#6daa45",
    "warning":      "#bb653b",
    "log_bg":       "#0f0e0d",
    "log_fg":       "#cdccca",
}


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class BellhopGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("bellhop")
        self.root.minsize(920, 640)
        self.root.geometry("1100x720")

        self._dark = False
        self._palette = LIGHT
        self._current_profile = list(PROFILES.keys())[0]
        self._param_vars: dict[str, tk.Variable] = {}
        self._sidebar_buttons: dict[str, tk.Button] = {}
        self._log_queue: queue.Queue = queue.Queue()
        self._running = False

        self._build_fonts()
        self._build_layout()
        self._select_profile(self._current_profile, init=True)
        self._apply_theme()
        self._poll_log()

    # ── Fonts ──────────────────────────────────────────────────────────────
    def _build_fonts(self) -> None:
        self.font_body  = font.Font(family="Helvetica Neue", size=12)
        self.font_small = font.Font(family="Helvetica Neue", size=11)
        self.font_mono  = font.Font(family="Courier",        size=11)
        self.font_head  = font.Font(family="Helvetica Neue", size=13, weight="bold")
        self.font_title = font.Font(family="Helvetica Neue", size=15, weight="bold")
        self.font_label = font.Font(family="Helvetica Neue", size=11)
        self.font_sidebar = font.Font(family="Helvetica Neue", size=12)

    # ── Top-level layout ───────────────────────────────────────────────────
    def _build_layout(self) -> None:
        C = self._palette

        # Header bar
        self.header = tk.Frame(self.root, height=48)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)

        self.lbl_title = tk.Label(
            self.header, text="⬡  bellhop",
            font=self.font_title, padx=16,
        )
        self.lbl_title.pack(side="left", fill="y")

        self.btn_theme = tk.Button(
            self.header, text="◑", font=self.font_body,
            relief="flat", padx=12, cursor="hand2",
            command=self._toggle_theme,
        )
        self.btn_theme.pack(side="right", padx=8)

        # Main pane
        self.pane = tk.PanedWindow(
            self.root, orient="horizontal", sashwidth=1,
        )
        self.pane.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar = tk.Frame(self.pane, width=170)
        self.sidebar.pack_propagate(False)
        self.pane.add(self.sidebar, minsize=150)

        tk.Label(
            self.sidebar, text="PIPELINE", font=self.font_small,
            padx=12, pady=10, anchor="w",
        ).pack(fill="x")

        for name in PROFILES:
            btn = tk.Button(
                self.sidebar, text=name, font=self.font_sidebar,
                relief="flat", anchor="w", padx=14, pady=8,
                cursor="hand2",
                command=lambda n=name: self._select_profile(n),
            )
            btn.pack(fill="x")
            self._sidebar_buttons[name] = btn

        # Right panel
        self.right = tk.Frame(self.pane)
        self.pane.add(self.right, minsize=600)

        # Right: form area + log split
        self.vpane = tk.PanedWindow(self.right, orient="vertical", sashwidth=1)
        self.vpane.pack(fill="both", expand=True)

        # Form scroll canvas
        self.form_outer = tk.Frame(self.vpane)
        self.vpane.add(self.form_outer, minsize=340)

        self.canvas = tk.Canvas(self.form_outer, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self.form_outer, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.form_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.form_frame, anchor="nw"
        )
        self.form_frame.bind("<Configure>", self._on_form_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel scroll
        self.canvas.bind_all("<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # Log
        self.log_frame = tk.Frame(self.vpane)
        self.vpane.add(self.log_frame, minsize=140)

        log_header = tk.Frame(self.log_frame)
        log_header.pack(fill="x")
        tk.Label(log_header, text="Output", font=self.font_small, padx=8, pady=4,
                 anchor="w").pack(side="left", fill="x")
        tk.Button(log_header, text="Clear", font=self.font_small,
                  relief="flat", padx=8, cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=4)

        self.log = scrolledtext.ScrolledText(
            self.log_frame, font=self.font_mono, wrap="word",
            relief="flat", borderwidth=0, state="disabled",
        )
        self.log.pack(fill="both", expand=True)

    # ── Form canvas helpers ────────────────────────────────────────────────
    def _on_form_configure(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    # ── Profile selection ─────────────────────────────────────────────────
    def _select_profile(self, name: str, init: bool = False) -> None:
        self._current_profile = name
        self._rebuild_form()
        self._update_sidebar_highlight()

    def _update_sidebar_highlight(self) -> None:
        C = self._palette
        for name, btn in self._sidebar_buttons.items():
            if name == self._current_profile:
                btn.configure(bg=C["sidebar_sel"], fg=C["primary"],
                              font=font.Font(family="Helvetica Neue", size=12, weight="bold"))
            else:
                btn.configure(bg=C["sidebar_bg"], fg=C["text"],
                              font=self.font_sidebar)

    # ── Form builder ──────────────────────────────────────────────────────
    def _rebuild_form(self) -> None:
        """Destroy and rebuild the parameter form for the current profile."""
        for widget in self.form_frame.winfo_children():
            widget.destroy()
        self._param_vars.clear()

        profile = PROFILES[self._current_profile]
        C = self._palette

        # ── Description label
        tk.Label(
            self.form_frame, text=profile["description"],
            font=self.font_small, anchor="w", pady=6, padx=16,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        tk.Frame(self.form_frame, height=1).grid(
            row=1, column=0, columnspan=3, sticky="ew", padx=16, pady=(0, 8)
        )

        # ── Bag file
        self._build_path_row(2, "Bag path", "bag_path", kind="bag")
        # ── Output dir
        self._build_path_row(3, "Output directory", "output_path", kind="dir")

        # ── Section: Parameters
        self._section_label(4, "Parameters")

        row = 5
        for (arg, label, wtype, default, extra) in profile["params"]:
            row = self._build_param_row(row, arg, label, wtype, default, extra)

        # ── Section: Pre-flight check
        self._section_label(row, "Pre-flight")
        row += 1

        pf_frame = tk.Frame(self.form_frame)
        pf_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=16, pady=4)
        row += 1

        self.btn_check = tk.Button(
            pf_frame, text="Check Topics", font=self.font_body,
            relief="flat", padx=12, pady=4, cursor="hand2",
            command=self._run_preflight,
        )
        self.btn_check.pack(side="left")

        self.lbl_preflight = tk.Label(
            pf_frame, text="", font=self.font_small, padx=12,
        )
        self.lbl_preflight.pack(side="left", fill="x", expand=True)

        # ── Run button
        tk.Frame(self.form_frame, height=1).grid(
            row=row, column=0, columnspan=3, sticky="ew", padx=16, pady=(8, 0)
        )
        row += 1

        self.btn_run = tk.Button(
            self.form_frame, text="▶  Run Pipeline",
            font=font.Font(family="Helvetica Neue", size=13, weight="bold"),
            relief="flat", padx=20, pady=10, cursor="hand2",
            command=self._run_pipeline,
        )
        self.btn_run.grid(row=row, column=0, columnspan=3,
                          sticky="ew", padx=16, pady=12)

        self._apply_theme()

    def _section_label(self, row: int, text: str) -> None:
        C = self._palette
        frm = tk.Frame(self.form_frame)
        frm.grid(row=row, column=0, columnspan=3, sticky="ew", padx=16, pady=(12, 2))
        tk.Label(frm, text=text.upper(),
                 font=self.font_small, anchor="w").pack(side="left")
        tk.Frame(frm, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _build_path_row(self, row: int, label: str, var_key: str, kind: str) -> None:
        var = tk.StringVar()
        self._param_vars[var_key] = var

        tk.Label(self.form_frame, text=label, font=self.font_label,
                 anchor="w", padx=16).grid(row=row, column=0, sticky="w", pady=3)

        entry = tk.Entry(self.form_frame, textvariable=var, font=self.font_body,
                         relief="flat", bd=1)
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=3)

        def _browse(k=kind, v=var):
            if k == "bag":
                path = filedialog.askdirectory(title="Select ROS 2 bag directory")
            else:
                path = filedialog.askdirectory(title="Select output directory")
            if path:
                v.set(path)

        btn = tk.Button(self.form_frame, text="Browse", font=self.font_small,
                        relief="flat", padx=8, cursor="hand2", command=_browse)
        btn.grid(row=row, column=2, padx=(0, 16), pady=3)

        self.form_frame.columnconfigure(1, weight=1)

    def _build_param_row(self, row: int, arg: str, label: str,
                         wtype: str, default, extra: dict) -> int:
        tk.Label(self.form_frame, text=label, font=self.font_label,
                 anchor="w", padx=16).grid(row=row, column=0, sticky="w", pady=2)

        if wtype == "check":
            var = tk.BooleanVar(value=bool(default))
            widget = tk.Checkbutton(self.form_frame, variable=var,
                                    relief="flat", cursor="hand2")
            widget.grid(row=row, column=1, sticky="w", pady=2)
        elif wtype == "spinbox":
            var = tk.StringVar(value=str(default))
            widget = tk.Spinbox(
                self.form_frame, textvariable=var,
                font=self.font_body, relief="flat", bd=1,
                from_=extra.get("from_", 0), to=extra.get("to", 9999),
                width=8,
            )
            widget.grid(row=row, column=1, sticky="w", pady=2)
        elif wtype == "combobox":
            var = tk.StringVar(value=str(default))
            widget = ttk.Combobox(
                self.form_frame, textvariable=var,
                values=extra.get("values", []),
                font=self.font_body, state="readonly", width=24,
            )
            widget.grid(row=row, column=1, sticky="w", pady=2)
        else:  # entry
            var = tk.StringVar(value=str(default) if default else "")
            widget = tk.Entry(self.form_frame, textvariable=var,
                              font=self.font_body, relief="flat", bd=1)
            widget.grid(row=row, column=1, sticky="ew", padx=(0, 16), pady=2)

        self._param_vars[arg] = var
        return row + 1

    # ── Pre-flight ─────────────────────────────────────────────────────────
    def _run_preflight(self) -> None:
        bag_path = self._param_vars.get("bag_path", tk.StringVar()).get().strip()
        if not bag_path:
            self._set_preflight("⚠  No bag path set.", "warning")
            return

        profile = PROFILES[self._current_profile]
        required_keys = profile.get("required_topics_fields", [])
        required_topics = []
        for key in required_keys:
            val = self._param_vars.get(key, tk.StringVar()).get().strip()
            if val:
                required_topics.append(val)

        if not required_topics:
            self._set_preflight("⚠  No required topics configured.", "warning")
            return

        self._set_preflight("Checking…", "muted")
        self.root.update_idletasks()

        def _check():
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from pipelines.shared.preflight import check_topics
                missing = check_topics(bag_path, required_topics)
                if missing:
                    self._log_queue.put(("preflight_error",
                        f"Missing topics: {missing}"))
                else:
                    self._log_queue.put(("preflight_ok",
                        f"All topics found: {required_topics}"))
            except Exception as e:
                self._log_queue.put(("preflight_error", f"Check failed: {e}"))

        threading.Thread(target=_check, daemon=True).start()

    def _set_preflight(self, msg: str, style: str = "ok") -> None:
        C = self._palette
        color = {
            "ok":      C["success"],
            "error":   C["error"],
            "warning": C["warning"],
            "muted":   C["text_muted"],
        }.get(style, C["text"])
        try:
            self.lbl_preflight.configure(text=msg, fg=color)
        except Exception:
            pass

    # ── Pipeline runner ────────────────────────────────────────────────────
    def _build_command(self) -> list[str] | None:
        """Assemble the cli.py subprocess command from current form state."""
        bag_path    = self._param_vars.get("bag_path",    tk.StringVar()).get().strip()
        output_path = self._param_vars.get("output_path", tk.StringVar()).get().strip()

        if not bag_path:
            self._log("ERROR: No bag path specified.\n", "error")
            return None
        if not output_path:
            self._log("ERROR: No output directory specified.\n", "error")
            return None

        profile  = PROFILES[self._current_profile]
        pipeline = profile["pipeline"]

        cli_path = str(Path(__file__).parent / "cli.py")
        cmd = [sys.executable, cli_path, pipeline, bag_path, output_path]

        for (arg, _label, wtype, _default, _extra) in profile["params"]:
            var = self._param_vars.get(arg)
            if var is None:
                continue
            val = var.get()
            if wtype == "check":
                if val:
                    cmd.append(f"--{arg}")
            else:
                val = str(val).strip()
                if val:
                    cmd += [f"--{arg}", val]

        return cmd

    def _run_pipeline(self) -> None:
        if self._running:
            self._log("A pipeline is already running. Please wait.\n", "warning")
            return

        cmd = self._build_command()
        if cmd is None:
            return

        self._log(f"$ {' '.join(cmd)}\n\n", "muted")
        self._running = True
        self.btn_run.configure(state="disabled", text="⏳  Running…")

        def _worker():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    self._log_queue.put(("log", line))
                proc.wait()
                if proc.returncode == 0:
                    self._log_queue.put(("done_ok",  "Pipeline finished successfully.\n"))
                else:
                    self._log_queue.put(("done_err",
                        f"Pipeline exited with code {proc.returncode}.\n"))
            except Exception as e:
                self._log_queue.put(("done_err", f"Failed to start pipeline: {e}\n"))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Log helpers ────────────────────────────────────────────────────────
    def _poll_log(self) -> None:
        """Drain the log queue every 100 ms so the UI stays responsive."""
        C = self._palette
        try:
            while True:
                kind, text = self._log_queue.get_nowait()
                if kind == "log":
                    self._log(text)
                elif kind == "preflight_ok":
                    self._set_preflight(f"✓  {text}", "ok")
                    self._log(f"Pre-flight OK: {text}\n", "ok")
                elif kind == "preflight_error":
                    self._set_preflight(f"✗  {text}", "error")
                    self._log(f"Pre-flight FAIL: {text}\n", "error")
                elif kind == "done_ok":
                    self._log(text, "ok")
                    self._running = False
                    self.btn_run.configure(state="normal", text="▶  Run Pipeline")
                elif kind == "done_err":
                    self._log(text, "error")
                    self._running = False
                    self.btn_run.configure(state="normal", text="▶  Run Pipeline")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _log(self, text: str, style: str = "normal") -> None:
        C = self._palette
        color_map = {
            "normal":  C["log_fg"],
            "muted":   C["text_muted"],
            "ok":      C["success"],
            "error":   C["error"],
            "warning": C["warning"],
        }
        color = color_map.get(style, C["log_fg"])
        self.log.configure(state="normal")
        tag = f"tag_{style}"
        self.log.tag_configure(tag, foreground=color)
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ── Theme ─────────────────────────────────────────────────────────────
    def _toggle_theme(self) -> None:
        self._dark = not self._dark
        self._palette = DARK if self._dark else LIGHT
        self._apply_theme()

    def _apply_theme(self) -> None:
        C = self._palette

        def _walk(widget):
            cls = widget.winfo_class()
            try:
                if cls in ("Frame", "Canvas"):
                    widget.configure(bg=C["bg"])
                elif cls == "Label":
                    widget.configure(bg=C["bg"], fg=C["text"])
                elif cls == "Button":
                    widget.configure(bg=C["surface2"], fg=C["text"],
                                     activebackground=C["sidebar_sel"],
                                     activeforeground=C["text"])
                elif cls == "Entry":
                    widget.configure(bg=C["surface2"], fg=C["text"],
                                     insertbackground=C["text"],
                                     relief="flat",
                                     highlightthickness=1,
                                     highlightbackground=C["border"],
                                     highlightcolor=C["primary"])
                elif cls == "Spinbox":
                    widget.configure(bg=C["surface2"], fg=C["text"],
                                     buttonbackground=C["surface2"])
                elif cls == "Checkbutton":
                    widget.configure(bg=C["bg"], fg=C["text"],
                                     selectcolor=C["surface2"],
                                     activebackground=C["bg"])
                elif cls == "Text":
                    widget.configure(bg=C["log_bg"], fg=C["log_fg"])
            except Exception:
                pass
            for child in widget.winfo_children():
                _walk(child)

        self.root.configure(bg=C["bg"])
        _walk(self.root)

        # Header
        self.header.configure(bg=C["surface"])
        self.lbl_title.configure(bg=C["surface"], fg=C["primary"])
        self.btn_theme.configure(bg=C["surface"], fg=C["text_muted"],
                                 activebackground=C["surface"])

        # Sidebar
        self.sidebar.configure(bg=C["sidebar_bg"])
        for w in self.sidebar.winfo_children():
            if isinstance(w, tk.Label):
                w.configure(bg=C["sidebar_bg"], fg=C["text_muted"])

        # Run button
        try:
            self.btn_run.configure(
                bg=C["primary"], fg=C["primary_fg"],
                activebackground=C["primary"], activeforeground=C["primary_fg"],
            )
        except Exception:
            pass

        # Check button
        try:
            self.btn_check.configure(
                bg=C["surface2"], fg=C["text"],
                activebackground=C["sidebar_sel"],
            )
        except Exception:
            pass

        self._update_sidebar_highlight()


def main() -> None:
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.3)
    except Exception:
        pass
    app = BellhopGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
