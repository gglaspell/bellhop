#!/usr/bin/env python3
"""
cli.py – bellhop unified CLI entry point.

Usage examples
--------------
# Run a pipeline directly:
  python cli.py og_map       /data/bag ./output --grid_res 0.05
  python cli.py mesh         /data/bag ./output --voxel_size 0.05
  python cli.py color_mesh   /data/bag ./output --camera_topic /camera/image_raw
  python cli.py gazebo_world /data/bag ./output --model_name my_env
  python cli.py tiles_3d     /data/bag ./output --gps_topic /gps/fix
  python cli.py color_tiles_3d /data/bag ./output --camera_topic /camera/image_raw

# Launch the GUI:
  python cli.py --gui

# Docker equivalent:
  docker run --rm -v /host/data:/data bellhop mesh /data/bag /data/output
  docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix bellhop --gui
"""

import argparse
import sys


def _launch_gui() -> None:
    """Import and start the Tk GUI (deferred so CLI works without Tk installed)."""
    try:
        import gui  # noqa: F401 – gui.py lives next to cli.py
        gui.main()
    except ImportError as e:
        sys.exit(
            f"Error: could not import gui.py ({e}).\n"
            "Make sure gui.py is in the same directory as cli.py and tkinter is available."
        )


def main() -> None:
    # ── Top-level parser ──────────────────────────────────────────────────
    root = argparse.ArgumentParser(
        prog="bellhop",
        description=(
            "bellhop – Convert ROS 2 bag files to spatial outputs.\n\n"
            "Available pipelines:\n"
            "  og_map          2D Nav2 occupancy grid (.pgm + .yaml)\n"
            "  mesh            Poisson surface mesh (.ply + .obj)\n"
            "  color_mesh      Camera-colored Poisson mesh (.ply + .obj)\n"
            "  gazebo_world    Gazebo simulation world (.stl + .sdf + .world)\n"
            "  tiles_3d        Georeferenced Cesium 3D Tiles (tileset.json)\n"
            "  color_tiles_3d  Colored georeferenced Cesium 3D Tiles\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Launch the graphical user interface instead of running a pipeline.",
    )

    # ── Sub-command registration ──────────────────────────────────────────
    sub = root.add_subparsers(dest="pipeline", metavar="PIPELINE")

    # Import each pipeline module and register its parser.
    # Imports are deferred here so that a missing optional dependency in one
    # pipeline does not prevent the others from loading.
    pipeline_modules = [
        ("og_map",          "pipelines.og_map"),
        ("mesh",            "pipelines.mesh"),
        ("color_mesh",      "pipelines.color_mesh"),
        ("gazebo_world",    "pipelines.gazebo_world"),
        ("tiles_3d",        "pipelines.tiles_3d"),
        ("color_tiles_3d",  "pipelines.color_tiles_3d"),
    ]

    import importlib
    parsers: dict = {}
    for name, module_path in pipeline_modules:
        try:
            mod = importlib.import_module(module_path)
            parsers[name] = mod.build_parser(sub)
        except ImportError as e:
            # Register a stub that explains the missing dependency
            stub = sub.add_parser(name, help=f"[unavailable – missing dependency: {e}]")
            stub.set_defaults(
                func=lambda _a, _e=e, _n=name: sys.exit(
                    f"Pipeline '{_n}' is unavailable: {_e}\n"
                    "Install the required dependency and retry."
                )
            )

    # ── Parse ─────────────────────────────────────────────────────────────
    args = root.parse_args()

    if args.gui:
        _launch_gui()
        return

    if args.pipeline is None:
        root.print_help()
        sys.exit(0)

    if not hasattr(args, "func"):
        root.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
