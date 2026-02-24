import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from typing import List, Tuple

import numpy as np
import torch

from server.ppo.train_env import ABREnv
import ppo2 as network

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -----------------------------
# Training config
# -----------------------------
S_DIM = [6, 8]
A_DIM = 6

ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 10
TRAIN_SEQ_LEN = 1000
TRAIN_EPOCH = 2000
MODEL_SAVE_INTERVAL = 300
RANDOM_SEED = 42

# New env2.py expects trace json path
TRACE_JSON_PATH = "network.json"  # your trace-style JSON [{duration_ms, bandwidth_kbps, latency_ms}, ...]

# file with bitrate ladder
VIDEO_PATH = "movie_4g.json"

SUMMARY_DIR = "server\\models"
LOG_FILE = os.path.join(SUMMARY_DIR, "log")
NN_MODEL = "ppo_model2.pth"  # optional preload
TEST_SCRIPT = "test.py"     # or your updated test script path

os.makedirs(SUMMARY_DIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def _to_np_float32(x):
    return np.asarray(x, dtype=np.float32)


def compute_bootstrap_returns(actor: network.Network, s_batch, r_batch, terminal: bool):
    """
    Computes R_t for PPO target (bootstrapped returns).
    This is a train.py-side replacement for ppo2.compute_v(...) to avoid tensor type issues.
    Returns shape [N, 1] float32.
    """
    s_arr = _to_np_float32(s_batch)               # [N, S_INFO, S_LEN]
    r_arr = _to_np_float32(r_batch).reshape(-1)   # [N]

    n = len(r_arr)
    R = np.zeros(n, dtype=np.float32)

    if n == 0:
        return R.reshape(-1, 1)

    if terminal:
        R[-1] = r_arr[-1]
    else:
        with torch.no_grad():
            s_t = torch.from_numpy(s_arr).to(torch.float32)
            v = actor.critic(s_t)  # [N,1]
            R[-1] = float(v[-1, 0].item())

    for t in range(n - 2, -1, -1):
        R[t] = r_arr[t] + network.GAMMA * R[t + 1]

    return R.reshape(-1, 1)


def testing(epoch: int, nn_model: str, log_file):
    test_log_folder = SUMMARY_DIR +  "/test_ppo/"

    # clean test folder (portable)
    if os.path.isdir(test_log_folder):
        shutil.rmtree(test_log_folder)
    os.makedirs(test_log_folder, exist_ok=True)

    # run test script
    # If your test script writes logs to ./test_results/, keep as-is
    try:
        subprocess.run([sys.executable, TEST_SCRIPT, nn_model], check=True)
    except Exception as e:
        print(f"[TEST] Warning: test script failed: {e}")
        return float("nan"), float("nan")

    rewards, entropies = [], []
    for fname in os.listdir(test_log_folder):
        fpath = os.path.join(test_log_folder, fname)
        if not os.path.isfile(fpath):
            continue

        reward_vals, entropy_vals = [], []
        with open(fpath, "rb") as f:
            for line in f:
                parse = line.split()
                try:
                    entropy_vals.append(float(parse[-2]))
                    reward_vals.append(float(parse[-1]))
                except Exception:
                    break

        if len(reward_vals) > 1:
            rewards.append(np.mean(reward_vals[1:]))
        elif len(reward_vals) == 1:
            rewards.append(reward_vals[0])

        if len(entropy_vals) > 1:
            entropies.append(np.mean(entropy_vals[1:]))
        elif len(entropy_vals) == 1:
            entropies.append(entropy_vals[0])

    if len(rewards) == 0:
        return float("nan"), float("nan")

    rewards = np.asarray(rewards, dtype=np.float32)
    rewards_min = float(np.min(rewards))
    rewards_5per = float(np.percentile(rewards, 5))
    rewards_mean = float(np.mean(rewards))
    rewards_median = float(np.percentile(rewards, 50))
    rewards_95per = float(np.percentile(rewards, 95))
    rewards_max = float(np.max(rewards))

    log_file.write(
        f"{epoch}\t{rewards_min}\t{rewards_5per}\t{rewards_mean}\t"
        f"{rewards_median}\t{rewards_95per}\t{rewards_max}\n"
    )
    log_file.flush()

    avg_entropy = float(np.mean(entropies)) if len(entropies) else float("nan")
    return rewards_mean, avg_entropy


# -----------------------------
# Coordinator
# -----------------------------
def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    with open(LOG_FILE + "_test.txt", "w") as test_log_file:
        actor = network.Network(
            state_dim=S_DIM,
            action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE,
        )

        # restore model if it exists
        if NN_MODEL and os.path.exists(NN_MODEL):
            actor.load_model(NN_MODEL)
            print(f"[CENTRAL] Model restored from {NN_MODEL}")
        elif NN_MODEL:
            print(f"[CENTRAL] No preload model found at {NN_MODEL}, training from scratch")

        for epoch in range(TRAIN_EPOCH):
            # sync params to agents
            actor_net_params = actor.get_network_params()
            for q in net_params_queues:
                q.put(actor_net_params)

            # collect experience
            s_parts, a_parts, p_parts, v_parts = [], [], [], []
            for q in exp_queues:
                s_, a_, p_, v_ = q.get()
                s_parts.append(_to_np_float32(s_))
                a_parts.append(_to_np_float32(a_))
                p_parts.append(_to_np_float32(p_))
                v_parts.append(_to_np_float32(v_))

            s_batch = np.concatenate(s_parts, axis=0)  # [N,6,8]
            a_batch = np.concatenate(a_parts, axis=0)  # [N,6]
            p_batch = np.concatenate(p_parts, axis=0)  # [N,6]
            v_batch = np.concatenate(v_parts, axis=0)  # [N,1]

            actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                model_path = os.path.join(SUMMARY_DIR, f"nn_model_ep_{epoch}.pth")
                actor.save_model(model_path)

                #avg_reward, avg_entropy = testing(epoch, model_path, test_log_file)

                print(f"[CENTRAL] SKIPPIN TEST epoch={epoch} entropy_weight={actor._entropy_weight:.6f}")
                #print(f"[CENTRAL] epoch={epoch} reward={avg_reward}")
                #print(f"[CENTRAL] epoch={epoch} entropy={avg_entropy}")


# -----------------------------
# Worker
# -----------------------------
def agent(agent_id: int, net_params_queue, exp_queue):
    
    # Different seed per worker for diversity
    env = ABREnv(
        trace_json_path=TRACE_JSON_PATH,
        video_path=VIDEO_PATH,
        random_seed=RANDOM_SEED + agent_id,
    )

    actor = network.Network(
        state_dim=S_DIM,
        action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE,
    )

    # initial sync
    actor_net_params = net_params_queue.get()
    actor.set_network_params(actor_net_params)

    rng = np.random.RandomState(RANDOM_SEED + 1000 + agent_id)

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()

        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        if epoch%10==0: print(f'Epoch {epoch}/{TRAIN_EPOCH}  Agent:{agent_id}')
    
        for _ in range(TRAIN_SEQ_LEN):
            
            s_batch.append(obs.copy())

            action_prob = actor.predict(np.reshape(obs, (1, S_DIM[0], S_DIM[1])))  # shape [A_DIM]

            # gumbel sampling (stable)
            probs = np.clip(action_prob, 1e-8, 1.0)
            noise = rng.gumbel(size=probs.shape[0])
            bit_rate = int(np.argmax(np.log(probs) + noise))

            obs, rew, done, _info = env.step(bit_rate)

            action_vec = np.zeros(A_DIM, dtype=np.float32)
            action_vec[bit_rate] = 1.0

            a_batch.append(action_vec)
            p_batch.append(probs.astype(np.float32))
            r_batch.append(float(rew))

            if done:
                break

        # PPO target values / returns (train.py-side bootstrap)
        v_batch = compute_bootstrap_returns(actor, s_batch, r_batch, done)

        # Send compact numpy arrays across process boundary
        exp_queue.put(
            (
                np.asarray(s_batch, dtype=np.float32),  # [T,6,8]
                np.asarray(a_batch, dtype=np.float32),  # [T,6]
                np.asarray(p_batch, dtype=np.float32),  # [T,6]
                np.asarray(v_batch, dtype=np.float32),  # [T,1]
            )
        )

        # sync updated params
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)


# -----------------------------
# Entry
# -----------------------------
def main():
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)

    # mp start method (safe with PyTorch)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    net_params_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]
    exp_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]

    coordinator = mp.Process(target=central_agent, args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = [
        mp.Process(target=agent, args=(i, net_params_queues[i], exp_queues[i]))
        for i in range(NUM_AGENTS)
    ]
    for p in agents:
        p.start()

    coordinator.join()
    for p in agents:
        p.join()


if __name__ == "__main__":
    main()