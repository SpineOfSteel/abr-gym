from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from sb3_contrib import ARS, QRDQN, RecurrentPPO, TRPO

from envs.abr_sb3_env import AbrStreamingEnv, EnvConfig


ALGO_REGISTRY = {
    "dqn": DQN,
    "qrdqn": QRDQN,
    "ppo": PPO,
    "a2c": A2C,
    "recurrent_ppo": RecurrentPPO,
    "trpo": TRPO,
    "ars": ARS,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


def import_from_string(spec: Optional[str]):
    """
    Supports:
      - None
      - 'MlpPolicy' (returned as-is)
      - 'abrGym.algo.rl.ppo_feature:CustomPPOPolicy'
    """
    if spec is None:
        return None
    if ":" not in spec:
        return spec
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def parse_unknown_overrides(unknown: list[str]) -> Dict[str, Any]:
    """
    Convert:
      --algo-kw learning_rate=1e-4
      --algo-kw gamma=0.99
    style trailing args into a dict.

    Expected format in unknown:
      --algo-kw key=value
      --policy-kw key=value
      --learn-kw key=value
    """
    out = {
        "algo_kwargs": {},
        "policy_kwargs": {},
        "learn_kwargs": {},
    }

    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token not in ("--algo-kw", "--policy-kw", "--learn-kw"):
            raise ValueError(f"Unknown extra argument: {token}")
        if i + 1 >= len(unknown):
            raise ValueError(f"Missing value after {token}")

        raw = unknown[i + 1]
        if "=" not in raw:
            raise ValueError(f"Expected key=value after {token}, got: {raw}")

        key, value = raw.split("=", 1)
        parsed = auto_parse(value)

        if token == "--algo-kw":
            out["algo_kwargs"][key] = parsed
        elif token == "--policy-kw":
            out["policy_kwargs"][key] = parsed
        elif token == "--learn-kw":
            out["learn_kwargs"][key] = parsed

        i += 2

    return out


def auto_parse(value: str) -> Any:
    v = value.strip()

    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "none":
        return None

    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def build_env(
    trace_path: str,
    video_meta: str,
    seed: int,
    action_mode: str,
    continuous_map: str,
    fixed_start: bool,
    default_quality: int = 1,
    rebuf_penalty: float = 4.3,
    smooth_penalty: float = 1.0,
):
    env = AbrStreamingEnv(
        trace_path=trace_path,
        video_metadata_file=video_meta,
        random_seed=seed,
        default_quality=default_quality,
        rebuf_penalty=rebuf_penalty,
        smooth_penalty=smooth_penalty,
        env_config=EnvConfig(
            fixed_start=fixed_start,
            random_seed=seed,
        ),
        action_mode=action_mode,
        continuous_map=continuous_map,
    )
    return Monitor(env)


def build_policy_kwargs(
    env,
    feature_extractor_spec: Optional[str],
    base_policy_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    policy_kwargs = dict(base_policy_kwargs)

    feature_extractor_cls = import_from_string(feature_extractor_spec)
    if feature_extractor_cls is not None and "features_extractor_class" not in policy_kwargs:
        policy_kwargs["features_extractor_class"] = feature_extractor_cls
        policy_kwargs.setdefault(
            "features_extractor_kwargs",
            {
                "a_dim": env.unwrapped.a_dim,
                "feature_num": 128,
                "features_dim": 128,
            },
        )

    return policy_kwargs


def maybe_add_action_noise(algo_name: str, env, algo_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    algo_kwargs = dict(algo_kwargs)

    if algo_name in {"td3", "ddpg"} and "action_noise" not in algo_kwargs:
        n_actions = env.action_space.shape[-1]
        algo_kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
    return algo_kwargs


def build_model(
    algo_name: str,
    policy_spec: str,
    env,
    device: str,
    feature_extractor_spec: Optional[str],
    algo_kwargs: Dict[str, Any],
    policy_kwargs: Dict[str, Any],
):
    algo_cls = ALGO_REGISTRY[algo_name]
    policy = import_from_string(policy_spec)

    merged_policy_kwargs = build_policy_kwargs(env, feature_extractor_spec, policy_kwargs)
    merged_algo_kwargs = dict(algo_kwargs)
    merged_algo_kwargs["policy_kwargs"] = merged_policy_kwargs
    merged_algo_kwargs["device"] = device

    merged_algo_kwargs = maybe_add_action_noise(algo_name, env, merged_algo_kwargs)

    model = algo_cls(policy, env, **merged_algo_kwargs)
    return model


def resolve_model_path(model_path: str) -> str:
    p = Path(model_path)
    if p.suffix == "":
        return str(p.with_suffix(".zip"))
    return str(p)


def run_train(args, extra):
    env = build_env(
        trace_path=args.train_trace,
        video_meta=args.video_meta,
        seed=args.seed,
        action_mode=args.action_mode,
        continuous_map=args.continuous_map,
        fixed_start=False,
    )

    model = build_model(
        algo_name=args.algo,
        policy_spec=args.policy,
        env=env,
        device=args.device,
        feature_extractor_spec=args.feature_extractor,
        algo_kwargs=extra["algo_kwargs"],
        policy_kwargs=extra["policy_kwargs"],
    )

    learn_kwargs = dict(extra["learn_kwargs"])
    learn_kwargs.setdefault("total_timesteps", args.total_timesteps)

    model.learn(**learn_kwargs)
    model.save(resolve_model_path(args.model_path))
    print(f"[train] saved model to {resolve_model_path(args.model_path)}")


def run_test(args, extra):
    del extra  # reserved for future use

    env = build_env(
        trace_path=args.eval_trace,
        video_meta=args.video_meta,
        seed=args.seed,
        action_mode=args.action_mode,
        continuous_map=args.continuous_map,
        fixed_start=True,
    )

    algo_cls = ALGO_REGISTRY[args.algo]
    model = algo_cls.load(resolve_model_path(args.model_path), env=env, device=args.device)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    print(
        f"[test] algo={args.algo} split={args.split} "
        f"mean_reward={mean_reward:.3f} +/- {std_reward:.3f}"
    )


def run_rollout(args, extra):
    del extra  # reserved for future use

    env = build_env(
        trace_path=args.eval_trace,
        video_meta=args.video_meta,
        seed=args.seed,
        action_mode=args.action_mode,
        continuous_map=args.continuous_map,
        fixed_start=True,
    )

    algo_cls = ALGO_REGISTRY[args.algo]
    model = algo_cls.load(resolve_model_path(args.model_path), env=env, device=args.device)

    obs, info = env.reset()
    total_reward = 0.0
    step_idx = 0

    # RecurrentPPO needs hidden state handling
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while step_idx < args.max_steps:
        if args.algo == "recurrent_ppo":
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_idx += 1

        print(
            f"[rollout] step={step_idx:04d} "
            f"action={action} reward={reward:.3f} "
            f"bitrate={info.get('bitrate_kbps', -1):.0f} "
            f"buffer={info.get('buffer_s', -1):.3f}s "
            f"rebuf={info.get('rebuffer_s', -1):.3f}s"
        )

        if args.algo == "recurrent_ppo":
            episode_starts = np.array([done], dtype=bool)

        if done:
            break

    print(f"[rollout] total_reward={total_reward:.3f}")


def build_parser():
    p = argparse.ArgumentParser(description="ABR RL entry script")

    p.add_argument("--mode", required=True, choices=["train", "test", "rollout"])
    p.add_argument("--algo", required=True, choices=sorted(ALGO_REGISTRY.keys()))
    p.add_argument("--policy", required=True)
    p.add_argument("--feature-extractor", default=None)

    p.add_argument("--action-mode", required=True, choices=["discrete", "continuous"])
    p.add_argument("--continuous-map", default="nearest", choices=["nearest", "threshold"])

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    p.add_argument("--video-meta", required=True)
    p.add_argument("--model-path", required=True)

    # train
    p.add_argument("--train-trace", default=None)
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--tensorboard-log", default=None)

    # test / rollout
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--eval-trace", default=None)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1000)

    return p


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    extra = parse_unknown_overrides(unknown)

    # Inject tensorboard log into algo kwargs if provided
    if args.tensorboard_log:
        extra["algo_kwargs"].setdefault("tensorboard_log", args.tensorboard_log)
    extra["algo_kwargs"].setdefault("seed", args.seed)

    if args.mode == "train":
        if not args.train_trace:
            raise ValueError("--train-trace is required for train")
        run_train(args, extra)
        return

    if args.mode in {"test", "rollout"}:
        if not args.eval_trace:
            raise ValueError("--eval-trace is required for test/rollout")

    if args.mode == "test":
        run_test(args, extra)
        return

    if args.mode == "rollout":
        run_rollout(args, extra)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()