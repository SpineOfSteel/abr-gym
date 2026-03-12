# abrGym/cli.py
# ============================================================
# Algorithm registry
# ============================================================

ALGORITHMS = {
    "rb": {
        "family": "classic",
        "runner": "sabre",
        "plugin": "ALGO/rb.py",
        "backend_name": "rb",
        "description": "Rate-based ABR",
        "default_chunk_log": "log_RB_driving_4g.txt",
    },
    "bb": {
        "family": "classic",
        "runner": "sabre",
        "plugin": "ALGO/bb.py",
        "backend_name": "bb",
        "description": "Buffer-based ABR",
        "default_chunk_log": "log_BB_driving_4g.txt",
    },
    "bola": {
        "family": "classic",
        "runner": "sabre",
        "plugin": "ALGO/bola.py",
        "backend_name": "bola",
        "description": "BOLA",
        "default_chunk_log": "log_BOLA_driving_4g.txt",
    },
    "bola_throughput": {
        "family": "classic",
        "runner": "sabre",
        "plugin": "ALGO/bola-d.py",
        "backend_name": "throughput",
        "description": "BOLA-D throughput mode",
        "default_chunk_log": "log_BOLAT_driving_4g.txt",
    },
    "bola_dynamic": {
        "family": "classic",
        "runner": "sabre",
        "plugin": "ALGO/bola-d.py",
        "backend_name": "dynamic",
        "description": "BOLA-D dynamic mode",
        "default_chunk_log": "log_BOLAD_dynamic_driving_4g.txt",
    },
    "ppo": {
        "family": "rl",
        "runner": "rl",
        "rl_backend": "ppo",
        "description": "Proximal Policy Optimization ABR",
    },
    "dqn": {
        "family": "rl",
        "runner": "rl",
        "rl_backend": "dqn",
        "description": "Deep Q-Network ABR",
    },
    "a3c": {
        "family": "rl",
        "runner": "rl",
        "rl_backend": "a3c",
        "description": "Asynchronous Advantage Actor-Critic ABR",
    },
    "pensieve": {
        "family": "rl",
        "runner": "rl",
        "rl_backend": "pensieve",
        "description": "Pensieve-style RL ABR",
    },
    "llama": {
        "family": "llm",
        "runner": "plm",
        "plm_type": "llama",
        "description": "LLaMA-based PLM ABR",
    },
    "gpt2": {
        "family": "llm",
        "runner": "plm",
        "plm_type": "gpt2",
        "description": "GPT-2 based PLM ABR",
    },
    "mistral": {
        "family": "llm",
        "runner": "plm",
        "plm_type": "mistral",
        "description": "Mistral-based PLM ABR",
    },
    "opt": {
        "family": "llm",
        "runner": "plm",
        "plm_type": "opt",
        "description": "OPT-based PLM ABR",
    },
    "t5-lm": {
        "family": "llm",
        "runner": "plm",
        "plm_type": "t5-lm",
        "description": "T5-LM based PLM ABR",
    }
}

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import subprocess
import sys


# ============================================================
# Paths
# ============================================================

@dataclass
class PathConfig:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def sabre_entry(self) -> Path:
        return self.root / "SIMULATOR" / "SABRE" / "sab.py"

    @property
    def llm_entry(self) -> Path:
        return self.root / "ALGO" / "llm" / "run_plm.py"

    @property
    def default_network(self) -> Path:
        return self.root / "DATASET" / "NETWORK" / "4Glogs_lum" / "logs.parquet"

    @property
    def default_movie(self) -> Path:
        return self.root / "DATASET" / "MOVIE" / "movie_4g.json"

    @property
    def default_chunk_folder(self) -> Path:
        return self.root / "DATASET" / "artifacts" / "tmp"





# ============================================================
# Runners
# ============================================================

def run_sabre(args, meta, paths: PathConfig) -> None:
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

    if args.verbose:
        cmd.append("-v")

    subprocess.run(cmd, check=True)


def run_plm_test(args, meta, paths: PathConfig) -> None:
    if not args.model_dir:
        raise ValueError("--model-dir is required for PLM test")
    if not args.exp_pool_path:
        raise ValueError("--exp-pool-path is required for PLM test")

    cmd = [
        sys.executable,
        str(paths.llm_entry),
        "--test",
        "--plm-type",
        meta["plm_type"],
        "--plm-size",
        args.plm_size,
        "--rank",
        str(args.rank),
        "--device",
        args.device,
        "--model-dir",
        args.model_dir,
        "--exp-pool-path",
        args.exp_pool_path,
    ]

    subprocess.run(cmd, check=True)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abrGym",
        description="ABR toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    simulate_p = sub.add_parser("simulate", help="Run SABRE simulation")
    simulate_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    simulate_p.add_argument("-n", "--network", default=None)
    simulate_p.add_argument("-m", "--movie", default=None)
    simulate_p.add_argument("--chunk-log", default=None)
    simulate_p.add_argument("--chunk-folder", default=None)
    simulate_p.add_argument("-v", "--verbose", action="store_true")

    test_p = sub.add_parser("test", help="Test RL/PLM algorithm")
    test_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    test_p.add_argument("--device", default="cpu")
    test_p.add_argument("--model-dir", default=None)
    test_p.add_argument("--exp-pool-path", default=None)
    test_p.add_argument("--plm-size", default="base")
    test_p.add_argument("--rank", type=int, default=128)

    info_p = sub.add_parser("info", help="Show algorithm info")
    info_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))

    sub.add_parser("list", help="List algorithms")

    return parser


def cmd_list() -> None:
    for name, meta in sorted(ALGORITHMS.items()):
        print(f"{name:18} {meta['family']:8} {meta['runner']:8} {meta['description']}")


def cmd_info(algorithm: str) -> None:
    meta = ALGORITHMS[algorithm]
    print(f"algorithm:   {algorithm}")
    for key, value in meta.items():
        print(f"{key:12} {value}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = PathConfig()

    if args.command == "list":
        cmd_list()
        return

    if args.command == "info":
        cmd_info(args.algorithm)
        return

    meta = ALGORITHMS[args.algorithm]

    if args.command == "simulate":
        if meta["runner"] != "sabre":
            raise ValueError(f"{args.algorithm} is not a SABRE simulation algorithm")
        run_sabre(args, meta, paths)
        return

    if args.command == "test":
        if meta["runner"] != "plm":
            raise ValueError(f"{args.algorithm} is not a PLM test algorithm")
        run_plm_test(args, meta, paths)
        return


if __name__ == "__main__":
    main()