#JUST DEBUG
import multiprocessing as mp
import os
from typing import List, Tuple

import numpy as np
import torch

from SERVER.EnvAbr import ABREnv
import SERVER.pensieve.a3c as network  # expects ActorNetwork, CriticNetwork, compute_gradients


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -----------------------------
# Paths / inputs
# -----------------------------

# trace json path, video file with bitrate ladder, model path, log path and test script
TRACE_JSON_PATH = "DATASET\\NETWORK\\network.json"  # trace JSON [{duration_ms, bandwidth_kbps, latency_ms}, ...]
VIDEO_PATH = "DATASET\\MOVIE\\movie_4g.json"
SUMMARY_DIR = "DATASET\\MODELS"
os.makedirs(SUMMARY_DIR, exist_ok=True)

ACTOR_INIT_MODEL = "a3c_actor.pth"   # optional preload
CRITIC_INIT_MODEL = "a3c_critic.pth" # optional preload

# -----------------------------
# Training config
# -----------------------------
S_DIM = [6, 8]
A_DIM = 6

ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3

NUM_AGENTS = 1
TRAIN_SEQ_LEN = 1000
TRAIN_EPOCH = 2000
MODEL_SAVE_INTERVAL = 300
RANDOM_SEED = 42





# -----------------------------
# Coordinator
# -----------------------------
def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    torch.set_num_threads(1)
    np.random.seed(RANDOM_SEED)

    actor = network.ActorNetwork(
        state_dim=S_DIM,
        action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE,
    )
    critic = network.CriticNetwork(
        state_dim=S_DIM,
        learning_rate=CRITIC_LR_RATE,
        action_dim=A_DIM,
    )

    # Optional preload
    if ACTOR_INIT_MODEL and os.path.exists(ACTOR_INIT_MODEL):
        _load_model_compat(actor, ACTOR_INIT_MODEL)
        print(f"[CENTRAL] Loaded actor from {ACTOR_INIT_MODEL}")
    else:
        print("[CENTRAL] Actor training from scratch")

    if CRITIC_INIT_MODEL and os.path.exists(CRITIC_INIT_MODEL):
        _load_model_compat(critic, CRITIC_INIT_MODEL)
        print(f"[CENTRAL] Loaded critic from {CRITIC_INIT_MODEL}")
    else:
        print("[CENTRAL] Critic training from scratch")

    for epoch in range(TRAIN_EPOCH):
        # 1) push latest params to workers
        _sync_to_workers(net_params_queues, actor, critic)

        # 2) collect worker gradients
        actor_grads_all, critic_grads_all, rewards, entropies, steps = _collect_worker_payloads(exp_queues)

        # 3) aggregate + apply
        actor_grad_mean = _mean_grads(actor_grads_all)
        critic_grad_mean = _mean_grads(critic_grads_all)

        actor.apply_gradients(actor_grad_mean)
        critic.apply_gradients(critic_grad_mean)

        # 4) logging
        if epoch % 10 == 0:
            mean_reward = float(np.mean(rewards)) if rewards else float("nan")
            mean_entropy = float(np.mean(entropies)) if entropies else float("nan")
            mean_steps = float(np.mean(steps)) if steps else 0.0
            print(
                f"[CENTRAL] epoch={epoch}/{TRAIN_EPOCH} "
                f"reward_mean={mean_reward:.3f} entropy_mean={mean_entropy:.4f} steps_mean={mean_steps:.1f}"
            )

        # 5) checkpoint
        if epoch % MODEL_SAVE_INTERVAL == 0:
            actor_path = os.path.join(SUMMARY_DIR, f"a3c_actor_ep_{epoch}.pth")
            critic_path = os.path.join(SUMMARY_DIR, f"a3c_critic_ep_{epoch}.pth")
            _save_model_compat(actor, actor_path)
            _save_model_compat(critic, critic_path)
            print(f"[CENTRAL] Saved checkpoints: {actor_path}, {critic_path}")


# -----------------------------
# Worker
# -----------------------------
def agent(agent_id: int, net_params_queue, exp_queue):
    torch.set_num_threads(1)
    seed = RANDOM_SEED + agent_id
    np.random.seed(seed)

    env = ABREnv(
        trace_json_path=TRACE_JSON_PATH,
        video_path=VIDEO_PATH,
        random_seed=seed,
    )

    actor = network.ActorNetwork(
        state_dim=S_DIM,
        action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE,
    )
    critic = network.CriticNetwork(
        state_dim=S_DIM,
        learning_rate=CRITIC_LR_RATE,
        action_dim=A_DIM,
    )

    # initial sync
    actor_params, critic_params = net_params_queue.get()
    actor.set_network_params(actor_params)
    critic.set_network_params(critic_params)

    rng = np.random.RandomState(RANDOM_SEED + 1000 + agent_id)

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()

        s_batch, a_batch, r_batch = [], [], []
        entropy_vals = []
        reward_sum = 0.0
        done = False

        for _ in range(TRAIN_SEQ_LEN):
            s_batch.append(obs.copy())

            probs = _predict_probs(actor, obs)
            entropy_vals.append(_entropy(probs))

            # Gumbel sampling (same idea as PPO worker)
            noise = rng.gumbel(size=probs.shape[0])
            bit_rate = int(np.argmax(np.log(probs) + noise))

            obs, rew, done, _info = env.step(bit_rate)

            a = np.zeros(A_DIM, dtype=np.float32)
            a[bit_rate] = 1.0

            a_batch.append(a)
            r_batch.append(float(rew))
            reward_sum += float(rew)

            if done:
                break

        # Compute worker-side grads using local actor+critic
        actor_grads, critic_grads, td_batch = network.compute_gradients(
            s_batch=s_batch,
            a_batch=a_batch,
            r_batch=r_batch,
            terminal=done,
            actor=actor,
            critic=critic,
        )

        # Send gradients + stats
        entropy_mean = float(np.mean(entropy_vals)) if entropy_vals else 0.0
        exp_queue.put((actor_grads, critic_grads, reward_sum, entropy_mean, len(r_batch)))

        # Sync latest params for next rollout
        actor_params, critic_params = net_params_queue.get()
        actor.set_network_params(actor_params)
        critic.set_network_params(critic_params)

        if epoch % 50 == 0:
            td_abs = float(np.mean(np.abs(td_batch))) if len(td_batch) else 0.0
            print(
                f"[AGENT {agent_id}] epoch={epoch}/{TRAIN_EPOCH} "
                f"steps={len(r_batch)} reward_sum={reward_sum:.3f} td_abs_mean={td_abs:.4f}"
            )


# -----------------------------
# Entry
# -----------------------------
def main():
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)

    # safer with PyTorch multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    net_params_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]
    exp_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]

    coordinator = mp.Process(target=central_agent, args=(net_params_queues, exp_queues))
    coordinator.start()

    workers = [
        mp.Process(target=agent, args=(i, net_params_queues[i], exp_queues[i]))
        for i in range(NUM_AGENTS)
    ]
    for p in workers:
        p.start()

    coordinator.join()
    for p in workers:
        p.join()


if __name__ == "__main__":
    main()
    

# -----------------------------
# Helpers
# -----------------------------
def _to_np32(x):
    return np.asarray(x, dtype=np.float32)


def _predict_probs(actor, obs: np.ndarray) -> np.ndarray:
    """Returns shape [A_DIM] float32, clipped for stability."""
    probs = actor.predict(obs.reshape(1, S_DIM[0], S_DIM[1]))
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 2:
        probs = probs[0]
    probs = np.clip(probs, 1e-8, 1.0)
    s = float(probs.sum())
    if s <= 0:
        probs = np.ones_like(probs, dtype=np.float32) / float(len(probs))
    else:
        probs = probs / s
    return probs.astype(np.float32)


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _mean_grads(grad_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Average gradients across workers.
    Supports `None` entries (for params with no grad).
    """
    if not grad_list:
        return []

    n_params = len(grad_list[0])
    out = []
    for i in range(n_params):
        parts = [g[i] for g in grad_list if g[i] is not None]
        if not parts:
            out.append(None)
            continue
        arr = np.stack([np.asarray(p, dtype=np.float32) for p in parts], axis=0)
        out.append(arr.mean(axis=0))
    return out


def _save_model_compat(model, path: str):
    """Works with either save(...) or save_model(...)."""
    if hasattr(model, "save_model"):
        model.save_model(path)
    elif hasattr(model, "save"):
        model.save(path)
    else:
        raise AttributeError(f"{type(model).__name__} has no save/save_model method")


def _load_model_compat(model, path: str):
    """Works with either load(...) or load_model(...)."""
    if hasattr(model, "load_model"):
        model.load_model(path)
    elif hasattr(model, "load"):
        model.load(path)
    else:
        raise AttributeError(f"{type(model).__name__} has no load/load_model method")


def _sync_to_workers(net_param_queues, actor, critic):
    actor_params = actor.get_network_params()
    critic_params = critic.get_network_params()
    payload = (actor_params, critic_params)
    for q in net_param_queues:
        q.put(payload)


def _collect_worker_payloads(exp_queues):
    """
    Each worker sends:
      (actor_grads, critic_grads, reward_sum, entropy_mean, n_steps)
    """
    actor_grads_all = []
    critic_grads_all = []
    rewards = []
    entropies = []
    steps = []

    for q in exp_queues:
        actor_g, critic_g, reward_sum, entropy_mean, n_steps = q.get()
        actor_grads_all.append(actor_g)
        critic_grads_all.append(critic_g)
        rewards.append(float(reward_sum))
        entropies.append(float(entropy_mean))
        steps.append(int(n_steps))

    return actor_grads_all, critic_grads_all, rewards, entropies, steps
