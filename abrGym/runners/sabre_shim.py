# abrGym/runners/sabre_shim.py
from __future__ import annotations

import socket
import subprocess
import sys
import time


def wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for server on {host}:{port}")


def build_server_cmd(args, meta, paths, port: int, movie: str) -> list[str]:
    server_script = str(paths.root / meta["server_script"])
    backend = meta["backend_name"]

    host = getattr(args, "server_host", None) or "localhost"
    log_prefix = getattr(args, "server_log_prefix", None) or ""

    cmd = [
        sys.executable,
        server_script,
        "--port",
        str(port),
        "--host",
        host,
        "--movie",
        args.server_movie or movie,
    ]

    if log_prefix:
        cmd += ["--log-prefix", log_prefix]
    else:
        # harmless for servers that accept it and keeps interface uniform
        cmd += ["--log-prefix", ""]

    if getattr(args, "server_debug", False):
        cmd.append("--debug")

    if getattr(args, "verbose", False) or getattr(args, "server_verbose", False):
        cmd.append("--verbose")

    if backend == "pensieve":
        if args.server_model:
            cmd += ["--actor", args.server_model]

    elif backend == "ppo":
        if args.server_model:
            cmd += ["--model", args.server_model]

    elif backend == "dqn":
        if args.server_model:
            cmd += ["--model", args.server_model]
        if getattr(args, "server_epsilon", None) is not None:
            cmd += ["--epsilon", str(args.server_epsilon)]

    elif backend in {"fastmpc", "robustmpc"}:
        pass

    if args.server_extra:
        cmd += list(args.server_extra)

    return cmd


def run_sabre_shim(args, meta, paths) -> None:
    port = args.port or meta["port"]
    host = getattr(args, "server_host", None) or "localhost"

    plugin_path = str(paths.root / meta["plugin"])
    network = args.network or str(paths.default_network)
    movie = args.movie or str(paths.default_movie)
    chunk_folder = args.chunk_folder or str(paths.default_chunk_folder)
    chunk_log = args.chunk_log or meta.get("default_chunk_log", "log.txt")

    server_cmd = build_server_cmd(args, meta, paths, port, movie)
    server_proc = subprocess.Popen(server_cmd)

    try:
        wait_for_port(host, port, args.startup_timeout)

        sab_cmd = [
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
            "--shim",
            str(port),
            "--chunk-log",
            chunk_log,
            "--chunk-folder",
            chunk_folder,
        ]

        if getattr(args, "debug_p", False):
            sab_cmd.append("--debug_p")

        if getattr(args, "verbose", False):
            sab_cmd.append("--verbose")

        subprocess.run(sab_cmd, check=True)

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()