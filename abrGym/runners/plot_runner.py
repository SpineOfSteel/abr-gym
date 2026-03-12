# abrGym/runners/plot_runner.py
from __future__ import annotations

import subprocess
import sys


def run_plot(args, paths) -> None:
    source = args.source or str(paths.default_plot_source)
    output_dir = args.output_dir or str(paths.default_plot_output_dir)

    cmd = [
        sys.executable,
        str(paths.plot_entry),
        "--source",
        source,
        "--output-dir",
        output_dir,
    ]

    if args.algo:
        cmd += ["--algo", *args.algo]

    if args.group:
        cmd += ["--group", *args.group]

    if args.plot:
        cmd += ["--plot", *args.plot]

    if args.include_all:
        cmd.append("--include-all")

    if args.video_len is not None:
        cmd += ["--video-len", str(args.video_len)]

    if args.recursive:
        cmd.append("--recursive")

    subprocess.run(cmd, check=True)