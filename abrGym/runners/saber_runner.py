# abr/runners/saber_runner.py
import subprocess
import sys


def run_saber(cfg, backend_name: str) -> None:
    cmd = [
        sys.executable,
        str(cfg.paths.saber_entry),
        "-n", cfg.network,
        "-nm", str(cfg.network_multiplier),
        "-m", cfg.movie,
        "-a", backend_name,
        "-ma", cfg.moving_average,
        "-r", cfg.replace,
        "-b", str(cfg.max_buffer),
        "--chunk-log", cfg.chunk_log,
        "--chunk-folder", cfg.chunk_folder,
        "--shim", str(cfg.shim),
        "--timeout_s", str(cfg.timeout_s),
    ]

    for p in cfg.plugin:
        cmd += ["--plugin", p]

    if cfg.movie_length is not None:
        cmd += ["-ml", str(cfg.movie_length)]
    if cfg.seek is not None:
        cmd += ["-s", str(cfg.seek[0]), str(cfg.seek[1])]
    if cfg.window_size:
        cmd += ["-ws", *map(str, cfg.window_size)]
    if cfg.half_life:
        cmd += ["-hl", *map(str, cfg.half_life)]
    if cfg.rampup_threshold is not None:
        cmd += ["-rmp", str(cfg.rampup_threshold)]
    if cfg.gamma_p is not None:
        cmd += ["-gp", str(cfg.gamma_p)]
    if cfg.chunk_log_start_ts is not None:
        cmd += ["--chunk-log-start-ts", str(cfg.chunk_log_start_ts)]

    if cfg.no_abandon:
        cmd.append("-noa")
    if cfg.no_insufficient_buffer_rule:
        cmd.append("-noibr")
    if cfg.debug_p:
        cmd.append("--debug_p")
    if cfg.ping_on_start:
        cmd.append("--ping_on_start")
    if cfg.verbose:
        cmd.append("-v")

    subprocess.run(cmd, check=True)