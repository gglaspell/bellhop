import argparse
import sys


def _launch_gui() -> None:
    """Import and start the Tk GUI (deferred so CLI works without Tk installed)."""
    try:
        import gui  # noqa: F401 - gui.py lives next to cli.py
        gui.main()
    except ImportError as e:
        sys.exit(
            f"Error: could not import gui.py ({e}).\n"
            "Make sure gui.py is in the same directory as cli.py and tkinter is available."
        )


def _make_missing_pipeline_stub(name: str, error: ImportError):
    """Build a func= callback that preserves the original import traceback."""
    def _missing(_args) -> None:
        raise ImportError(
            f"Pipeline '{name}' is unavailable: {error}\n"
            "Install the required dependency and retry."
        ) from error
    return _missing


def main() -> None:
    root = argparse.ArgumentParser(
        prog="bellhop",
        description=(
            "bellhop - Convert ROS 2 bag files to spatial outputs.\n\n"
            "Available pipelines:\n"
            "  og_map          2D Nav2 occupancy grid (.pgm + .yaml)\n"
            "  mesh            Poisson surface mesh (.ply cloud + .ply mesh)\n"
            "  color_mesh      Camera-colored Poisson mesh (.ply cloud + .ply mesh)\n"
            "  gazebo_world    Gazebo simulation world (.stl + .sdf + .world)\n"
            "  tiles_3d        Georeferenced Cesium 3D Tiles (tileset.json)\n"
            "  color_tiles_3d  Colored georeferenced Cesium 3D Tiles\n"
            "  texture_baking  Keyframe-baked textured mesh (OBJ + texture)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Launch the graphical user interface instead of running a pipeline.",
    )

    sub = root.add_subparsers(dest="pipeline", metavar="PIPELINE")

    pipeline_modules = [
        ("og_map", "pipelines.og_map"),
        ("mesh", "pipelines.mesh"),
        ("color_mesh", "pipelines.color_mesh"),
        ("gazebo_world", "pipelines.gazebo_world"),
        ("tiles_3d", "pipelines.tiles_3d"),
        ("color_tiles_3d", "pipelines.color_tiles_3d"),
        ("texture_baking", "pipelines.texture_baking"),
    ]

    import importlib
    parsers: dict = {}
    for name, module_path in pipeline_modules:
        try:
            mod = importlib.import_module(module_path)
            parsers[name] = mod.build_parser(sub)
        except ImportError as e:
            stub = sub.add_parser(name, help=f"[unavailable - missing dependency: {e}]")
            stub.set_defaults(func=_make_missing_pipeline_stub(name, e))

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
