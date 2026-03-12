# abrGym/cli.py
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


from .runners.saber_runner import run_sabre
from .runners.sabre_shim import run_sabre_shim
from .runners.plm_runner import run_plm_test
from .runners.plot_runner import run_plot

ALGORITHMS = {
    "rb": {
        "family": "classic",
        "runner": "sabre_local",
        "plugin": "ALGO/rb.py",
        "backend_name": "rb",
        "description": "Rate-based ABR",
        "default_chunk_log": "log_RB_driving_4g.txt",
    },
    "bb": {
        "family": "classic",
        "runner": "sabre_local",
        "plugin": "ALGO/bb.py",
        "backend_name": "bb",
        "description": "Buffer-based ABR",
        "default_chunk_log": "log_BB_driving_4g.txt",
    },
    "bola": {
        "family": "classic",
        "runner": "sabre_local",
        "plugin": "ALGO/bola.py",
        "backend_name": "bola",
        "description": "BOLA",
        "default_chunk_log": "log_BOLA_driving_4g.txt",
    },
    "bola_throughput": {
        "family": "classic",
        "runner": "sabre_local",
        "plugin": "ALGO/bola-d.py",
        "backend_name": "throughput",
        "description": "BOLA-D throughput mode",
        "default_chunk_log": "log_BOLAT_driving_4g.txt",
    },
    "bola_dynamic": {
        "family": "classic",
        "runner": "sabre_local",
        "plugin": "ALGO/bola-d.py",
        "backend_name": "dynamic",
        "description": "BOLA-D dynamic mode",
        "default_chunk_log": "log_BOLAD_dynamic_driving_4g.txt",
    },
    "pensieve": {
        "family": "rl",
        "runner": "sabre_shim",
        "plugin": "ALGO/pensieve.py",
        "backend_name": "pensieve",
        "server_script": "ALGO/algo-server/pensieve/pensieve_server.py",
        "port": 8605,
        "description": "Pensieve A3C via shim server",
        "default_chunk_log": "log_PENSIEVE_driving_4g.txt",
    },
    "ppo": {
        "family": "rl",
        "runner": "sabre_shim",
        "plugin": "ALGO/ppo.py",
        "backend_name": "ppo",
        "server_script": "ALGO/algo-server/ppo/ppo_server.py",
        "port": 8607,
        "description": "PPO via shim server",
        "default_chunk_log": "log_PPO_driving_4g.txt",
    },
    "dqn": {
        "family": "rl",
        "runner": "sabre_shim",
        "plugin": "ALGO/dqn.py",
        "backend_name": "dqn",
        "server_script": "ALGO/algo-server/dqn/dqn_server.py",
        "port": 8606,
        "description": "DQN via shim server",
        "default_chunk_log": "log_DQN_driving_4g.txt",
    },
    "fastmpc": {
        "family": "rl",
        "runner": "sabre_shim",
        "plugin": "ALGO/fastmpc.py",
        "backend_name": "fastmpc",
        "server_script": "ALGO/algo-server/fastmpc_server.py",
        "port": 8395,
        "description": "FastMPC via shim server",
        "default_chunk_log": "log_FASTMPC_driving_4g.txt",
    },
    "robustmpc": {
        "family": "rl",
        "runner": "sabre_shim",
        "plugin": "ALGO/robustmpc.py",
        "backend_name": "robustmpc",
        "server_script": "ALGO/algo-server/robustmpc_server.py",
        "port": 8390,
        "description": "RobustMPC via shim server",
        "default_chunk_log": "log_ROBUSTMPC_driving_4g.txt",
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
    },
    "llava": {
        "family": "llm",
        "runner": "plm",
        "plm_type": "llava",
        "description": "LLaVA-based PLM ABR",
    },
}

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
    
    @property
    def plot_entry(self) -> Path:
        return self.root / "PLOT" / "plot_grouped.py"

    @property
    def default_plot_source(self) -> Path:
        return self.root / "DATASET" / "artifacts" / "norway" / "results.all.parquet"

    @property
    def default_plot_output_dir(self) -> Path:
        return self.root / "graphs"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abrGym",
        description="ABR toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    
    simulate_p = sub.add_parser("simulate", help="Run SABRE simulation")
    simulate_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    simulate_p.add_argument("-n", "--network", default=None, help="Path to network trace/parquet")
    simulate_p.add_argument("-m", "--movie", default=None, help="Path to movie json")
    simulate_p.add_argument("--chunk-log", default=None, help="Chunk log filename")
    simulate_p.add_argument("--chunk-folder", default=None, help="Folder for chunk logs")
    simulate_p.add_argument("--port", type=int, default=None, help="Override shim server port")
    simulate_p.add_argument("--server-model", default=None, help="Server model path, e.g. actor/model checkpoint")
    simulate_p.add_argument("--server-movie", default=None, help="Movie path passed to server when needed")
    simulate_p.add_argument("--server-extra", nargs="*", default=[], help="Extra args forwarded to server")
    simulate_p.add_argument("--startup-timeout", type=float, default=15.0, help="Seconds to wait for server port")
    simulate_p.add_argument("--debug_p", action="store_true", help="Enable SABRE debug_p")
    simulate_p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    test_p = sub.add_parser("test", help="Test PLM/LLM algorithm")
    test_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    test_p.add_argument("--device", default="cpu")
    test_p.add_argument("--model-dir", default=None)
    test_p.add_argument("--exp-pool-path", default=None)
    test_p.add_argument("--plm-size", default="base")
    test_p.add_argument("--rank", type=int, default=128)

    info_p = sub.add_parser("info", help="Show algorithm info")
    info_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))

    sub.add_parser("list", help="List algorithms")

    plot_p = sub.add_parser("plot", help="Generate grouped evaluation plots")
    plot_p.add_argument(
        "--source",
        default=None,
        help="Folder of txt logs or parquet file; if omitted, default plot source is used",
    )
    plot_p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots",
    )
    plot_p.add_argument(
        "--algo",
        nargs="+",
        default=None,
        help="Algorithms to include; if omitted, all default algorithms are used",
    )
    plot_p.add_argument(
        "--group",
        nargs="+",
        default=None,
        help="Transport groups to include; if omitted, all default groups are used",
    )
    plot_p.add_argument(
        "--plot",
        nargs="+",
        default=["all"],
        choices=["all", "tradeoff", "smoothness", "bitrate", "stall", "qoe"],
        help="Plot types to generate",
    )
    plot_p.add_argument(
        "--include-all",
        action="store_true",
        help="Also generate aggregate plots across all groups with suffix 'all'",
    )
    plot_p.add_argument(
        "--video-len",
        type=float,
        default=48.0,
        help="Video length used for txt-log derived stall calculations",
    )
    plot_p.add_argument(
        "--recursive",
        action="store_true",
        help="Search source folder recursively for txt logs",
    )

    return parser


def cmd_list() -> None:
    for name, meta in sorted(ALGORITHMS.items()):
        print(f"{name:18} {meta['family']:8} {meta['runner']:12} {meta['description']}")


def cmd_info(algorithm: str) -> None:
    meta = ALGORITHMS[algorithm]
    print(f"algorithm:   {algorithm}")
    for key, value in meta.items():
        print(f"{key:12} {value}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = PathConfig()

    if args.command == "list":
        cmd_list()
        return

    if args.command == "info":
        cmd_info(args.algorithm)
        return

    if args.command == "plot":
        run_plot(args, paths)
        return

    meta = ALGORITHMS[args.algorithm]

    if args.command == "simulate":
        if meta["runner"] == "sabre_local":
            run_sabre(args, meta, paths)
            return
        if meta["runner"] == "sabre_shim":
            run_sabre_shim(args, meta, paths)
            return
        raise ValueError(f"{args.algorithm} does not support simulate")

    if args.command == "test":
        if meta["runner"] != "plm":
            raise ValueError(f"{args.algorithm} is not a PLM test algorithm")
        run_plm_test(args, meta, paths)
        return


if __name__ == "__main__":
    main()