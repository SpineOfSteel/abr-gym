# abrGym/runners/saber_runner.py
from __future__ import annotations

import subprocess
import sys


def run_sabre(args, meta, paths) -> None:
    plugin_path = str(paths.root / meta["plugin"])
    network = args.network or str(paths.default_network)
    movie = args.movie or str(paths.default_movie)
    chunk_folder = args.chunk_folder or str(paths.default_chunk_folder)
    chunk_log = args.chunk_log or meta.get("default_chunk_log", "log.txt")

    cmd = [
        sys.executable,
        str(paths.sabre_entry),
        "--plugin",
        plugin_path,
        "-a",
        meta["backend_name"],
        "-n",
        network,
        "-m",
        movie,
        "--chunk-log",
        chunk_log,
        "--chunk-folder",
        chunk_folder,
    ]

    if getattr(args, "debug_p", False):
        cmd.append("--debug_p")

    if getattr(args, "verbose", False):
        cmd.append("-v")

    subprocess.run(cmd, check=True)