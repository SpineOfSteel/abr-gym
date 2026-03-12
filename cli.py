# abr/cli.py
from __future__ import annotations

import argparse

from .config import CommonConfig
from .registry import ALGORITHMS, get_algorithm
from .runners.saber_runner import run_saber


def add_shared_sim_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()), help="Algorithm name")
    p.add_argument("--plugin", action="append", default=[], help="Load plugin .py")

    p.add_argument("-n", "--network", default=None)
    p.add_argument("-nm", "--network-multiplier", type=float, default=1.0)

    p.add_argument("-m", "--movie", default=None)
    p.add_argument("-ml", "--movie-length", type=float, default=None)

    p.add_argument("-ma", "--moving-average", default="ewma")
    p.add_argument("-ws", "--window-size", nargs="+", type=int, default=[3])
    p.add_argument("-hl", "--half-life", nargs="+", type=float, default=[3.0, 8.0])

    p.add_argument("-s", "--seek", nargs=2, type=float, default=None, metavar=("WHEN", "SEEK"))
    p.add_argument("-r", "--replace", choices=["none", "left", "right"], default="none")
    p.add_argument("-b", "--max-buffer", type=float, default=25.0)
    p.add_argument("-noa", "--no-abandon", action="store_true")
    p.add_argument("-rmp", "--rampup-threshold", type=int, default=None)
    p.add_argument("-gp", "--gamma-p", type=float, default=5.0)
    p.add_argument("-noibr", "--no-insufficient-buffer-rule", action="store_true")

    p.add_argument("--chunk-log", default="log.txt")
    p.add_argument("--chunk-folder", default="")
    p.add_argument("--chunk-log-start-ts", type=float, default=1608418125.0)

    p.add_argument("--shim", type=int, default=8333)
    p.add_argument("--timeout_s", type=float, default=1.0)
    p.add_argument("--debug_p", action="store_true")
    p.add_argument("--ping_on_start", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", default=True)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=100003)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abr",
        description="ABR toolkit with simulators, RL methods, and LLM-based algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sim_p = sub.add_parser("simulate", help="Simulate one ABR session")
    add_shared_sim_args(sim_p)

    train_p = sub.add_parser("train", help="Train an ABR algorithm")
    train_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    train_p.add_argument("--device", default=None)
    train_p.add_argument("--seed", type=int, default=100003)
    train_p.add_argument("--num-epochs", type=int, default=80)
    train_p.add_argument("--lr", type=float, default=1e-4)

    test_p = sub.add_parser("test", help="Evaluate an ABR algorithm")
    test_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    test_p.add_argument("--device", default=None)
    test_p.add_argument("--seed", type=int, default=100003)

    info_p = sub.add_parser("info", help="Show algorithm info")
    info_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))

    sub.add_parser("list", help="List available algorithms")

    return parser


def cmd_list() -> None:
    print("Available algorithms:")
    for name, meta in sorted(ALGORITHMS.items()):
        print(f"  {name:10} | {meta['family']:7} | default simulator={meta['simulator']}")


def cmd_info(algorithm: str) -> None:
    meta = get_algorithm(algorithm)
    print(f"Algorithm:         {algorithm}")
    print(f"Family:            {meta['family']}")
    print(f"Default simulator: {meta['simulator']}")
    print(f"Backend name:      {meta['backend_name']}")
    print(f"Description:       {meta['description']}")


def cmd_sim(args) -> None:
    meta = get_algorithm(args.algorithm)

    cfg = CommonConfig(
        algorithm=args.algorithm,
        simulator=meta["simulator"],
        network=args.network,
        network_multiplier=args.network_multiplier,
        movie=args.movie,
        movie_length=args.movie_length,
        plugin=args.plugin,
        moving_average=args.moving_average,
        window_size=args.window_size,
        half_life=args.half_life,
        seek=args.seek,
        replace=args.replace,
        max_buffer=args.max_buffer,
        no_abandon=args.no_abandon,
        rampup_threshold=args.rampup_threshold,
        gamma_p=args.gamma_p,
        no_insufficient_buffer_rule=args.no_insufficient_buffer_rule,
        chunk_log=args.chunk_log,
        chunk_folder=args.chunk_folder,
        chunk_log_start_ts=args.chunk_log_start_ts,
        shim=args.shim,
        timeout_s=args.timeout_s,
        debug_p=args.debug_p,
        ping_on_start=args.ping_on_start,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
    )
    cfg.finalize_defaults()

    if meta["simulator"] == "saber":
        run_saber(cfg, meta["backend_name"])
    else:
        raise NotImplementedError(f"Simulator runner not implemented yet: {meta['simulator']}")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        cmd_list()
    elif args.command == "info":
        cmd_info(args.algorithm)
    elif args.command == "simulate":
        cmd_sim(args)
    elif args.command == "train":
        print(f"Train not implemented yet for {args.algorithm}")
    elif args.command == "test":
        print(f"Test not implemented yet for {args.algorithm}")
    else:
        parser.error("Unknown command")