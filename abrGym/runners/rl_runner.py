from __future__ import annotations

import subprocess
import sys


def _append_if(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _require(value, msg: str) -> None:
    if value in (None, "", False):
        raise ValueError(msg)


def _resolve_split_path(args, paths, split: str) -> str:
    if split == "train":
        return args.train_trace or str(paths.default_train_trace)
    if split == "valid":
        return args.valid_trace or str(paths.default_valid_trace)
    if split == "test":
        return args.test_trace or str(paths.default_test_trace)
    raise ValueError(f"Unsupported split: {split}")


def _base_rl_cmd(args, meta, paths, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        str(paths.rl_entry),
        "--mode",
        mode,
        "--algo",
        meta["algo_name"],
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]

    policy = meta.get("policy")
    if policy:
        cmd.extend(["--policy", policy])

    feature_extractor = meta.get("feature_extractor")
    if feature_extractor:
        cmd.extend(["--feature-extractor", feature_extractor])

    action_mode = meta.get("action_mode")
    if action_mode:
        cmd.extend(["--action-mode", action_mode])

    _append_if(cmd, "--continuous-map", getattr(args, "continuous_map", None))
    _append_if(cmd, "--video-meta", args.video_meta or str(paths.default_video_meta))
    _append_if(cmd, "--model-path", args.model_path)
    _append_if(cmd, "--tensorboard-log", getattr(args, "tensorboard_log", None))

    return cmd


def run_rl_train(args, meta, paths) -> None:
    train_trace = args.train_trace or str(paths.default_train_trace)

    cmd = _base_rl_cmd(args, meta, paths, mode="train")
    cmd.extend([
        "--train-trace", train_trace,
        "--total-timesteps", str(args.total_timesteps),
    ])

    subprocess.run(cmd, check=True)


def run_rl_test(args, meta, paths) -> None:
    split = args.split
    split_trace = _resolve_split_path(args, paths, split)
    model_path = args.model_path or str(paths.rl_models_dir / f"{meta['algo_name']}.zip")

    cmd = _base_rl_cmd(args, meta, paths, mode="test")
    cmd.extend([
        "--split", split,
        "--eval-trace", split_trace,
        "--model-path", model_path,
        "--n-eval-episodes", str(args.n_eval_episodes),
    ])

    subprocess.run(cmd, check=True)


def run_rl_rollout(args, meta, paths) -> None:
    split = args.split
    split_trace = _resolve_split_path(args, paths, split)
    model_path = args.model_path or str(paths.rl_models_dir / f"{meta['algo_name']}.zip")

    cmd = _base_rl_cmd(args, meta, paths, mode="rollout")
    cmd.extend([
        "--split", split,
        "--eval-trace", split_trace,
        "--model-path", model_path,
        "--max-steps", str(args.max_steps),
    ])

    subprocess.run(cmd, check=True)