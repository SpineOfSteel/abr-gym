# abrGym/cli.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .config import PathConfig
from .runners.saber_runner import run_sabre
from .runners.sabre_shim import run_sabre_shim
from .runners.plm_runner import run_plm_test
from .runners.plot_runner import run_plot
from .runners.rl_runner import run_rl_train, run_rl_test, run_rl_rollout


ALGORITHMS = {
    # ---------------------------
    # Classic / SABRE local
    # ---------------------------
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

    # ---------------------------
    # Shim/server RL algorithms
    # ---------------------------
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

    # ---------------------------
    # PLM / LLM
    # ---------------------------
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

    # ---------------------------
    # Gym / SB3 / SB3-Contrib RL
    # ---------------------------
    "gym_dqn": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "dqn",
        "policy": "MlpPolicy",
        "feature_extractor": "abrGym.algo.rl.dqn_feature:DQNFeatureExtractor",
        "action_mode": "discrete",
        "description": "AbrGym over SB3 DQN training/evaluation",
    },
    "gym_qrdqn": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "qrdqn",
        "policy": "MlpPolicy",
        "feature_extractor": "abrGym.algo.rl.qrdqn_features:QRDQNExtractor",
        "action_mode": "discrete",
        "description": "AbrGym over SB3-Contrib QR-DQN training/evaluation",
    },
    "gym_ppo": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "ppo",
        "policy": "abrGym.algo.rl.ppo_feature:CustomPPOPolicy",
        "action_mode": "discrete",
        "description": "AbrGym over SB3 PPO training/evaluation",
    },
    "gym_a2c": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "a2c",
        "policy": "abrGym.algo.rl.a2c_feature:CustomA2CPolicy",
        "action_mode": "discrete",
        "description": "AbrGym over SB3 A2C training/evaluation",
    },
    "gym_recurrent_ppo": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "recurrent_ppo",
        "policy": "MlpLstmPolicy",
        "feature_extractor": "abrGym.algo.rl.recurrent_features:RecurrentAbrExtractor",
        "action_mode": "discrete",
        "description": "AbrGym over SB3-Contrib Recurrent PPO training/evaluation",
    },
    "gym_trpo": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "trpo",
        "policy": "MlpPolicy",
        "feature_extractor": "abrGym.algo.rl.trpo_features:TRPOAbrExtractor",
        "action_mode": "discrete",
        "description": "AbrGym over SB3-Contrib TRPO training/evaluation",
    },
    "gym_ars": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "ars",
        "policy": "MlpPolicy",
        "action_mode": "continuous",
        "description": "AbrGym over SB3-Contrib ARS training/evaluation",
    },
    "gym_sac": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "sac",
        "policy": "MlpPolicy",
        "feature_extractor": "abrGym.algo.rl.continuous_features:ContinuousAbrExtractor",
        "action_mode": "continuous",
        "description": "AbrGym over SB3 SAC training/evaluation",
    },
    "gym_td3": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "td3",
        "policy": "MlpPolicy",
        "feature_extractor": "abrGym.algo.rl.continuous_features:ContinuousAbrExtractor",
        "action_mode": "continuous",
        "description": "AbrGym over SB3 TD3 training/evaluation",
    },
    "gym_ddpg": {
        "family": "rl_train",
        "runner": "rl_runner",
        "algo_name": "ddpg",
        "policy": "MlpPolicy",
        "feature_extractor": "abrGym.algo.rl.continuous_features:ContinuousAbrExtractor",
        "action_mode": "continuous",
        "description": "AbrGym over SB3 DDPG training/evaluation",
    },
}


def default_rl_model_path(paths: PathConfig, algorithm: str) -> str:
    return str(paths.rl_models_dir / f"{algorithm}_abr_gym.zip")


def build_parser() -> argparse.ArgumentParser:
    paths = PathConfig()

    parser = argparse.ArgumentParser(
        prog="abrGym",
        description="ABR toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------------------
    # simulate
    # ---------------------------
    simulate_p = sub.add_parser("simulate", help="Run SABRE simulation")
    simulate_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    simulate_p.add_argument("-n", "--network", default=str(paths.network), help="Path to network trace/parquet")
    simulate_p.add_argument("-m", "--movie", default=str(paths.movie), help="Path to movie json")
    simulate_p.add_argument("--chunk-log", default=None, help="Chunk log filename")
    simulate_p.add_argument("--chunk-folder", default=str(paths.chunk_folder), help="Folder for chunk logs")
    simulate_p.add_argument("--port", type=int, default=None, help="Override shim server port")
    simulate_p.add_argument("--server-model", default=None, help="Server model path, e.g. actor/model checkpoint")
    simulate_p.add_argument("--server-movie", default=None, help="Movie path passed to server when needed")
    simulate_p.add_argument("--server-extra", nargs="*", default=[], help="Extra args forwarded to server")
    simulate_p.add_argument("--startup-timeout", type=float, default=15.0, help="Seconds to wait for server port")
    simulate_p.add_argument("--debug_p", action="store_true", help="Enable SABRE debug_p")
    simulate_p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    # ---------------------------
    # test (PLM + gym RL)
    # ---------------------------
    test_p = sub.add_parser("test", help="Test PLM or RL algorithm")
    test_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    test_p.add_argument("--device", default="cpu")

    # PLM-specific
    test_p.add_argument("--model-dir", default=None)
    test_p.add_argument("--exp-pool-path", default=None)
    test_p.add_argument("--plm-size", default="base")
    test_p.add_argument("--rank", type=int, default=128)

    # RL-specific
    test_p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    test_p.add_argument("--train-trace", default=str(paths.train_trace))
    test_p.add_argument("--valid-trace", default=str(paths.valid_trace))
    test_p.add_argument("--test-trace", default=str(paths.test_trace))
    test_p.add_argument("--video-meta", default=str(paths.video_meta))
    test_p.add_argument("--model-path", default=None)
    test_p.add_argument("--n-eval-episodes", type=int, default=10)
    test_p.add_argument("--seed", type=int, default=42)
    test_p.add_argument("--continuous-map", default="nearest", choices=["nearest", "threshold"])

    # ---------------------------
    # train (gym RL)
    # ---------------------------
    train_p = sub.add_parser("train", help="Train RL algorithm with Gym/SB3 runner")
    train_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    train_p.add_argument("--train-trace", default=str(paths.train_trace))
    train_p.add_argument("--video-meta", default=str(paths.video_meta))
    train_p.add_argument("--model-path", default=None)
    train_p.add_argument("--tensorboard-log", default=str(paths.tb_log_dir))
    train_p.add_argument("--total-timesteps", type=int, default=50_000)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--continuous-map", default="nearest", choices=["nearest", "threshold"])

    # ---------------------------
    # rollout (gym RL)
    # ---------------------------
    rollout_p = sub.add_parser("rollout", help="Roll out a trained RL model")
    rollout_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))
    rollout_p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    rollout_p.add_argument("--train-trace", default=str(paths.train_trace))
    rollout_p.add_argument("--valid-trace", default=str(paths.valid_trace))
    rollout_p.add_argument("--test-trace", default=str(paths.test_trace))
    rollout_p.add_argument("--video-meta", default=str(paths.video_meta))
    rollout_p.add_argument("--model-path", default=None)
    rollout_p.add_argument("--max-steps", type=int, default=1000)
    rollout_p.add_argument("--seed", type=int, default=42)
    rollout_p.add_argument("--device", default="auto")
    rollout_p.add_argument("--continuous-map", default="nearest", choices=["nearest", "threshold"])

    # ---------------------------
    # info / list
    # ---------------------------
    info_p = sub.add_parser("info", help="Show algorithm info")
    info_p.add_argument("algorithm", choices=sorted(ALGORITHMS.keys()))

    sub.add_parser("list", help="List algorithms")

    # ---------------------------
    # plot
    # ---------------------------
    plot_p = sub.add_parser("plot", help="Generate grouped evaluation plots")
    plot_p.add_argument(
        "--source",
        default=str(paths.plot_source),
        help="Folder of txt logs or parquet file",
    )
    plot_p.add_argument(
        "--output-dir",
        default=str(paths.plot_output_dir),
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
        print(f"{name:18} {meta['family']:10} {meta['runner']:12} {meta['description']}")


def cmd_info(algorithm: str) -> None:
    meta = ALGORITHMS[algorithm]
    print(f"algorithm:   {algorithm}")
    for key, value in meta.items():
        print(f"{key:16} {value}")


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

    # Fill algorithm-specific default RL model path
    if getattr(args, "model_path", None) is None and meta.get("runner") == "rl_runner":
        args.model_path = default_rl_model_path(paths, args.algorithm)

    if args.command == "simulate":
        if meta["runner"] == "sabre_local":
            run_sabre(args, meta, paths)
            return
        if meta["runner"] == "sabre_shim":
            run_sabre_shim(args, meta, paths)
            return
        raise ValueError(f"{args.algorithm} does not support simulate")

    if args.command == "test":
        if meta["runner"] == "rl_runner":
            run_rl_test(args, meta, paths)
            return
        if meta["runner"] == "plm":
            run_plm_test(args, meta, paths)
            return
        raise ValueError(f"{args.algorithm} does not support test")

    if args.command == "train":
        if meta["runner"] != "rl_runner":
            raise ValueError(f"{args.algorithm} is not an RL training algorithm")
        run_rl_train(args, meta, paths)
        return

    if args.command == "rollout":
        if meta["runner"] != "rl_runner":
            raise ValueError(f"{args.algorithm} is not an RL training algorithm")
        run_rl_rollout(args, meta, paths)
        return


if __name__ == "__main__":
    main()