# ppo_server_core.py
# Core logic for PPO ABR server (no HTTP here)

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import server.ppo.ppo2 as network  # keep ppo2.py unchanged


# ---------- PPO/RL constants ----------
S_INFO = 6
S_LEN = 8
A_DIM = 6

ACTOR_LR_RATE = 0.0001
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0

REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0
DEFAULT_QUALITY = 1
RANDOM_SEED = 42

SUMMARY_DIR = "output"
LOG_FILE = "output/log_PPO_"

DEFAULT_MODEL_PATH = os.environ.get("TORCH_PPO_MODEL", "../rl_server/results/ppo_model.pth")


# ---------- Utility / logging ----------
def make_printers(debug: bool, verbose: bool):
    def dprint(*args):
        if debug:
            print(*args)

    def vprint(*args):
        if verbose:
            print(*args)

    return dprint, vprint


# ---------- Movie loading ----------
def load_movie_json(path: str, debug: bool = False) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        movie = json.load(f)

    for k in ("segment_duration_ms", "bitrates_kbps", "segment_sizes_bits"):
        if k not in movie:
            raise ValueError(f"movie.json missing {k}")

    chunk_duration_s = float(movie["segment_duration_ms"]) / 1000.0
    video_bit_rate_kbps = [int(x) for x in movie["bitrates_kbps"]]

    if len(video_bit_rate_kbps) != A_DIM:
        raise ValueError(f"Expected exactly {A_DIM} qualities, got {len(video_bit_rate_kbps)}")

    seg_bytes_quality: List[List[int]] = []
    for seg_idx, arr in enumerate(movie["segment_sizes_bits"]):
        if len(arr) != len(video_bit_rate_kbps):
            raise ValueError(
                f"segment_sizes_bits[{seg_idx}] has {len(arr)} entries, expected {len(video_bit_rate_kbps)}"
            )
        seg_bytes_quality.append([(int(x) + 7) // 8 for x in arr])  # bits -> bytes (ceil)

    total_video_chunks = int(movie.get("total_video_chunks", len(seg_bytes_quality) - 1))

    if debug:
        print("\n[MOVIE] Loaded movie.json")
        print(f"[MOVIE] path={path}")
        print(f"[MOVIE] movie_id={movie.get('movie_id', os.path.basename(path))}")
        print(f"[MOVIE] segment_duration_ms={movie['segment_duration_ms']} -> chunk_duration_s={chunk_duration_s}")
        print(f"[MOVIE] qualities={len(video_bit_rate_kbps)} bitrates_kbps={video_bit_rate_kbps}")
        print(f"[MOVIE] segments_in_json={len(seg_bytes_quality)} => total_video_chunks={total_video_chunks}")
        if seg_bytes_quality:
            print(f"[MOVIE] first_segment_sizes_bytes(q0..q5)={seg_bytes_quality[0]}")
        exp_bytes_q0 = int(video_bit_rate_kbps[0] * 1000 * chunk_duration_s / 8)
        print(f"[MOVIE] sanity: expected bytes/segment at q0≈{exp_bytes_q0}\n")

    return {
        "movie_id": movie.get("movie_id", os.path.basename(path)),
        "chunk_duration_s": chunk_duration_s,
        "video_bit_rate_kbps": video_bit_rate_kbps,   # [A_DIM]
        "seg_bytes_quality": seg_bytes_quality,        # [segment_idx][quality] -> bytes
        "total_video_chunks": total_video_chunks,      # max valid segment index
    }


# ---------- PPO model loading ----------
def build_ppo_network(n_q: int):
    return network.Network(
        state_dim=[S_INFO, S_LEN],
        action_dim=n_q,
        learning_rate=ACTOR_LR_RATE,
    )


def load_ppo_checkpoint(ppo, model_path: Optional[str], debug: bool = False) -> None:
    if not model_path:
        return

    if not os.path.exists(model_path):
        print(f"[BOOT] Warning: model file not found: {model_path}")
        print("[BOOT] Continuing with randomly initialized weights.")
        return

    try:
        ppo.load_model(model_path)
        print(f"[BOOT] PPO model restored from {model_path}")
    except Exception as e:
        print(f"[BOOT] Warning: failed to load PPO model: {model_path}")
        if debug:
            print(f"[BOOT][DEBUG] {e}")
        else:
            print(f"[BOOT] {e}")
        print("[BOOT] Continuing with randomly initialized weights.")


# ---------- Server state ----------
def make_initial_server_state(movie: Dict, log_file, startup_time: float) -> Dict:
    return {
        "log_file": log_file,
        "last_bit_rate_kbps": movie["video_bit_rate_kbps"][DEFAULT_QUALITY],
        "last_total_rebuf_ms": 0.0,
        "video_chunk_count": 0,
        "s_batch": [np.zeros((S_INFO, S_LEN), dtype=np.float32)],
        "req_id": 0,
        "startup_time": startup_time,
        "last_entropy": 0.0,
        "episode_count": 0,
    }


def reset_episode_state(state: Dict, movie: Dict) -> None:
    state["last_total_rebuf_ms"] = 0.0
    state["last_bit_rate_kbps"] = movie["video_bit_rate_kbps"][DEFAULT_QUALITY]
    state["video_chunk_count"] = 0
    state["s_batch"] = [np.zeros((S_INFO, S_LEN), dtype=np.float32)]
    state["last_entropy"] = 0.0
    state["episode_count"] += 1


# ---------- Request parsing / validation ----------
def extract_client_metrics(post_data: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        metrics = {
            "last_quality": int(post_data["lastquality"]),
            "last_index": int(post_data["lastRequest"]),
            "buffer_s": float(post_data["buffer"]),
            "total_rebuf_ms": float(post_data["RebufferTime"]),
            "fetch_time_ms": float(post_data["lastChunkFinishTime"] - post_data["lastChunkStartTime"]),
            "chunk_size_bytes": float(post_data["lastChunkSize"]),
        }
        return metrics, None
    except KeyError as e:
        return None, f"MISSING_FIELD:{e}"
    except Exception as e:
        return None, f"BAD_FIELD:{e}"


# ---------- ABR helpers ----------
def get_chunk_size(chunk_sizes: List[List[int]], quality: int, index: int, n_q: int, dprint) -> int:
    if index < 0 or index >= len(chunk_sizes):
        dprint(f"[WARN] get_chunk_size out-of-range index={index} len={len(chunk_sizes)}")
        return 0
    if quality < 0 or quality >= n_q:
        dprint(f"[WARN] get_chunk_size bad quality q={quality} nQ={n_q}")
        return 0
    return chunk_sizes[index][quality]


def compute_rebuffer_delta_ms(total_rebuf_ms: float, last_total_rebuf_ms: float, dprint, rid: int) -> float:
    delta = float(total_rebuf_ms - last_total_rebuf_ms)
    if delta < 0:
        dprint(f"[WARN] rid={rid} negative rebuf delta={delta}; clamp to 0")
        delta = 0.0
    return delta


def compute_qoe_reward(
    last_quality: int,
    rebuffer_time_ms: float,
    last_bit_rate_kbps: float,
    video_bitrates_kbps: List[int],
) -> float:
    rebuffer_time_s = rebuffer_time_ms / M_IN_K
    return (
        video_bitrates_kbps[last_quality] / M_IN_K
        - REBUF_PENALTY * rebuffer_time_s
        - SMOOTH_PENALTY * abs(video_bitrates_kbps[last_quality] - last_bit_rate_kbps) / M_IN_K
    )


def build_next_observation(
    state: Dict,
    metrics: Dict,
    movie: Dict,
    dprint,
) -> Tuple[np.ndarray, List[int], int]:
    """
    Returns:
      obs, next_video_chunk_sizes, video_chunk_remain
    """
    video_bitrates = movie["video_bit_rate_kbps"]
    chunk_sizes = movie["seg_bytes_quality"]
    total_video_chunks = movie["total_video_chunks"]
    n_q = len(video_bitrates)

    s_batch: List[np.ndarray] = state["s_batch"]
    obs = np.zeros((S_INFO, S_LEN), dtype=np.float32) if not s_batch else np.array(s_batch[-1], copy=True)

    video_chunk_remain = total_video_chunks - state["video_chunk_count"]
    state["video_chunk_count"] += 1

    obs = np.roll(obs, -1, axis=1)

    next_video_chunk_sizes = [
        get_chunk_size(chunk_sizes, i, state["video_chunk_count"], n_q, dprint) for i in range(n_q)
    ]

    last_quality = metrics["last_quality"]
    buffer_s = metrics["buffer_s"]
    fetch_time_ms = metrics["fetch_time_ms"]
    chunk_size_bytes = metrics["chunk_size_bytes"]

    obs[0, -1] = video_bitrates[last_quality] / float(np.max(video_bitrates))
    obs[1, -1] = buffer_s / BUFFER_NORM_FACTOR
    obs[2, -1] = float(chunk_size_bytes) / max(float(fetch_time_ms), 1e-6) / M_IN_K
    obs[3, -1] = float(fetch_time_ms) / M_IN_K / BUFFER_NORM_FACTOR
    obs[4, :] = 0.0
    obs[4, :n_q] = np.array(next_video_chunk_sizes, dtype=np.float32) / M_IN_K / M_IN_K
    obs[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

    return obs, next_video_chunk_sizes, video_chunk_remain


def normalize_action_probs(action_prob: np.ndarray, n_q: int, dprint, rid: int) -> np.ndarray:
    p = np.asarray(action_prob, dtype=np.float32).reshape(-1)

    if p.size != n_q:
        dprint(f"[WARN] rid={rid} bad policy output size={p.size}, expected={n_q}; using uniform")
        return np.ones(n_q, dtype=np.float32) / float(n_q)

    p = np.clip(p, 1e-6, 1.0)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0:
        dprint(f"[WARN] rid={rid} invalid prob sum={s}; using uniform")
        return np.ones(n_q, dtype=np.float32) / float(n_q)

    return p / s


def sample_action_gumbel(action_prob: np.ndarray) -> int:
    noise = np.random.gumbel(size=action_prob.shape[0])
    return int(np.argmax(np.log(action_prob) + noise))


def compute_entropy(action_prob: np.ndarray) -> float:
    p = np.clip(np.asarray(action_prob, dtype=np.float32), 1e-6, 1.0)
    p /= float(np.sum(p))
    return float(-np.sum(p * np.log(p)))


def write_step_log(state: Dict, video_bitrate_kbps: int, metrics: Dict, rebuffer_time_ms: float, reward: float) -> None:
    rebuffer_time_s = rebuffer_time_ms / M_IN_K
    state["log_file"].write(
        f"{time.time()}\t{video_bitrate_kbps}\t{metrics['buffer_s']}\t"
        f"{rebuffer_time_s}\t{int(metrics['chunk_size_bytes'])}\t{metrics['fetch_time_ms']}\t"
        f"{state['last_entropy']}\t{reward}\n"
    )
    state["log_file"].flush()


# ---------- High-level step processor ----------
def process_decision_step(
    state: Dict,
    movie: Dict,
    ppo,
    post_data: Dict,
    rid: int,
    dprint,
    vprint,
) -> Tuple[str, Optional[str]]:
    """
    Returns:
      (response_text, error_text)
      - response_text is "0", "REFRESH", or bitrate index string
      - error_text is None on success, else error string for HTTP 400
    """
    if "pastThroughput" in post_data:
        dprint(f"[POST] rid={rid} summary payload ignored")
        return "0", None

    metrics, metrics_err = extract_client_metrics(post_data)
    if metrics_err:
        return "", metrics_err

    video_bitrates = movie["video_bit_rate_kbps"]
    total_video_chunks = movie["total_video_chunks"]
    n_q = len(video_bitrates)

    if not (0 <= metrics["last_quality"] < n_q):
        return "", f"BAD_FIELD:lastquality={metrics['last_quality']}"

    rebuffer_time_ms = compute_rebuffer_delta_ms(
        metrics["total_rebuf_ms"], state["last_total_rebuf_ms"], dprint, rid
    )

    reward = compute_qoe_reward(
        last_quality=metrics["last_quality"],
        rebuffer_time_ms=rebuffer_time_ms,
        last_bit_rate_kbps=state["last_bit_rate_kbps"],
        video_bitrates_kbps=video_bitrates,
    )

    state["last_bit_rate_kbps"] = video_bitrates[metrics["last_quality"]]
    state["last_total_rebuf_ms"] = metrics["total_rebuf_ms"]

    obs, next_video_chunk_sizes, _ = build_next_observation(state, metrics, movie, dprint)

    write_step_log(
        state=state,
        video_bitrate_kbps=video_bitrates[metrics["last_quality"]],
        metrics=metrics,
        rebuffer_time_ms=rebuffer_time_ms,
        reward=reward,
    )

    raw_probs = ppo.predict(np.reshape(obs, (1, S_INFO, S_LEN)))
    action_prob = normalize_action_probs(raw_probs, n_q, dprint, rid)
    bit_rate = sample_action_gumbel(action_prob)

    entropy_ = compute_entropy(action_prob)
    state["last_entropy"] = entropy_

    dprint(
        f"[POST] rid={rid} lastReq={metrics['last_index']}/{total_video_chunks} q={metrics['last_quality']} "
        f"buf={metrics['buffer_s']:.3f}s rebufΔ={rebuffer_time_ms/M_IN_K:.3f}s "
        f"fetch={metrics['fetch_time_ms']/1000.0:.3f}s sz={int(metrics['chunk_size_bytes'])}B "
        f"reward={reward:.3f} -> next_q={bit_rate}"
    )
    vprint(
        f"[POST][DBG] rid={rid} probs={np.round(action_prob, 4).tolist()} "
        f"H={entropy_:.4f} next_sizes={next_video_chunk_sizes}"
    )

    end_of_video = (metrics["last_index"] == total_video_chunks)
    if end_of_video:
        dprint(f"[EP] rid={rid} END_OF_VIDEO -> REFRESH + reset")
        reset_episode_state(state, movie)
        state["log_file"].write("\n")
        state["log_file"].flush()
        return "REFRESH", None

    state["s_batch"].append(obs)
    return str(bit_rate), None