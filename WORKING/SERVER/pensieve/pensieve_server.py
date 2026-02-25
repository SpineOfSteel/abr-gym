# python pensieve_server.py --port 8605 --debug --verbose --actor ..\models_a3c\a3c_actor_ep_600.pth

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import a3c_min as a3c
# ---------- RL constants ----------
S_INFO = 6  # bitrate, buffer, throughput, dl_time, next_chunk_sizes, chunks_left
S_LEN = 8

M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0

DEFAULT_QUALITY = 0
REBUF_PENALTY = 20
SMOOTH_PENALTY = 1

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RANDOM_SEED = 42
RAND_RANGE = 1000

SUMMARY_DIR = "..//SERVER_LOGS"
LOG_FILE = SUMMARY_DIR + "//log_RL"
print("Current Working Directory:", os.getcwd())
DEFAULT_MODEL_PATH = os.environ.get("TORCH_PENSIEVE_MODEL", "..//..//DATASET//MODELS//a3c_actor_ep_1800.pth")


# ---------- Movie loader (exact movie.json schema) ----------
def load_movie_json(path: str, debug: bool = False) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        movie = json.load(f)

    for key in ("segment_duration_ms", "bitrates_kbps", "segment_sizes_bits"):
        if key not in movie:
            raise ValueError(f"movie.json missing {key}")

    chunk_duration_s = float(movie["segment_duration_ms"]) / 1000.0
    video_bit_rate_kbps = [int(x) for x in movie["bitrates_kbps"]]
    if len(video_bit_rate_kbps) != 6:
        raise ValueError(f"Expected exactly 6 qualities, got {len(video_bit_rate_kbps)}")

    seg_bytes_quality: List[List[int]] = []
    for seg_idx, arr in enumerate(movie["segment_sizes_bits"]):
        if len(arr) != len(video_bit_rate_kbps):
            raise ValueError(
                f"segment_sizes_bits[{seg_idx}] has {len(arr)} entries, expected {len(video_bit_rate_kbps)}"
            )
        seg_bytes_quality.append([(int(x) + 7) // 8 for x in arr])  # bits -> bytes

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

    return {
        "movie_id": movie.get("movie_id", os.path.basename(path)),
        "chunk_duration_s": chunk_duration_s,
        "video_bit_rate_kbps": video_bit_rate_kbps,
        "seg_bytes_quality": seg_bytes_quality,  # [segment_idx][quality] -> bytes
        "total_video_chunks": total_video_chunks,  # max valid segment index
    }


def cors_headers(h: BaseHTTPRequestHandler) -> None:
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")


def _dprint(debug: bool, *args):
    if debug:
        print(*args)


def _vprint(verbose: bool, *args):
    if verbose:
        print(*args)


def _safe_prob_vector(prob, a_dim: int, debug=False) -> np.ndarray:
    arr = np.asarray(prob, dtype=np.float32).reshape(-1)
    if arr.size != a_dim:
        if arr.size == 0:
            arr = np.ones(a_dim, dtype=np.float32) / float(a_dim)
        else:
            _dprint(debug, f"[WARN] policy size mismatch: got {arr.size}, expected {a_dim}; resizing")
            arr = np.resize(arr, a_dim).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 1e-8, None)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0:
        arr[:] = 1.0 / float(a_dim)
    else:
        arr /= s
    return arr


def _sample_action(prob: np.ndarray, debug=False) -> int:
    cdf = np.cumsum(prob)
    if cdf[-1] <= 0:
        return 0
    cdf[-1] = 1.0  # avoid edge due to float rounding
    r = np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
    action = int(np.searchsorted(cdf, r, side="right"))
    action = min(max(action, 0), len(prob) - 1)
    _dprint(debug, f"[POLICY] prob={np.array2string(prob, precision=4)} cdf={np.array2string(cdf, precision=4)} r={r:.3f} -> a={action}")
    return action


def _policy_entropy(prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _compute_reward(last_quality: int, last_bitrate_kbps: float, video_bitrates: List[int], rebuffer_delta_ms: float) -> float:
    return (
        video_bitrates[last_quality] / M_IN_K
        - REBUF_PENALTY * (rebuffer_delta_ms / M_IN_K)
        - SMOOTH_PENALTY * abs(video_bitrates[last_quality] - last_bitrate_kbps) / M_IN_K
    )


def _reset_episode_state(state: Dict, video_bitrates: List[int]) -> None:
    state["last_total_rebuf_ms"] = 0.0
    state["last_bit_rate_kbps"] = float(video_bitrates[DEFAULT_QUALITY])
    state["video_chunk_count"] = 0
    state["s_batch"] = [np.zeros((S_INFO, S_LEN), dtype=np.float32)]
    state["episode_id"] = state.get("episode_id", 0) + 1


def _update_observation(prev_obs: np.ndarray,
                        last_quality: int,
                        buffer_s: float,
                        chunk_size_bytes: float,
                        fetch_time_ms: float,
                        next_video_chunk_sizes: List[int],
                        video_chunk_remain: int,
                        video_bitrates: List[int],
                        chunk_til_video_end_cap: float,
                        a_dim: int,
                        debug=False) -> np.ndarray:
    obs = np.array(prev_obs, copy=True)
    obs = np.roll(obs, -1, axis=1)

    safe_fetch_ms = max(fetch_time_ms, 1e-6)
    obs[0, -1] = float(video_bitrates[last_quality]) / float(np.max(video_bitrates))
    obs[1, -1] = float(buffer_s) / BUFFER_NORM_FACTOR
    obs[2, -1] = (float(chunk_size_bytes) / safe_fetch_ms) / M_IN_K   # KB/ms
    obs[3, -1] = (safe_fetch_ms / M_IN_K) / BUFFER_NORM_FACTOR         # sec/10
    obs[4, :] = 0.0
    obs[4, :a_dim] = np.asarray(next_video_chunk_sizes, dtype=np.float32) / M_IN_K / M_IN_K  # MB
    cap = max(float(chunk_til_video_end_cap), 1.0)
    obs[5, -1] = min(float(video_chunk_remain), cap) / cap

    if not np.isfinite(obs).all():
        _dprint(debug, "[WARN] Non-finite obs detected; zeroing invalid values")
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs.astype(np.float32, copy=False)


def _load_policy_checkpoint(policy: a3c.Network, model_path: Optional[str], debug=False) -> None:
    if not model_path:
        _dprint(debug, "[BOOT] No model path provided; random init")
        return
    if not os.path.exists(model_path):
        print(f"[BOOT] Warning: model file not found: {model_path}")
        print("[BOOT] Continuing with randomly initialized weights.")
        return

    # a3c_torch2.Network.load_model now supports actor-only or actor+critic checkpoints.
    policy.load_model(model_path)
    print(f"[BOOT] Model restored from {model_path}")


def make_handler(
    state: Dict,
    movie: Dict,
    policy: a3c.Network,
    debug: bool = False,
    verbose: bool = False,
):
    VIDEO_BIT_RATE = movie["video_bit_rate_kbps"]        # kbps
    CHUNK_SIZES = movie["seg_bytes_quality"]             # [seg][q] in bytes
    TOTAL_VIDEO_CHUNKS = movie["total_video_chunks"]     # max valid seg index
    CHUNK_TIL_VIDEO_END_CAP = float(TOTAL_VIDEO_CHUNKS) if TOTAL_VIDEO_CHUNKS > 0 else 1.0
    A_DIM = len(VIDEO_BIT_RATE)

    def get_chunk_size(quality: int, index: int) -> int:
        if index < 0 or index >= len(CHUNK_SIZES):
            _dprint(debug, f"[WARN] get_chunk_size out-of-range: index={index}, len={len(CHUNK_SIZES)}")
            return 0
        if quality < 0 or quality >= A_DIM:
            _dprint(debug, f"[WARN] get_chunk_size bad quality: q={quality}, nQ={A_DIM}")
            return 0
        return CHUNK_SIZES[index][quality]

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            return

        def _reply_text(self, text: str, status: int = 200):
            body = text.encode("utf-8")
            self.send_response(status)
            cors_headers(self)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            _dprint(debug, f"[HTTP] -> {status} reply='{text}' ({len(body)} bytes)")

        def do_OPTIONS(self):
            self.send_response(204)
            cors_headers(self)
            self.send_header("Content-Length", "0")
            self.end_headers()
            _dprint(debug, f"[HTTP] OPTIONS from {self.client_address}")

        def do_GET(self):
            body = f"console.log('pensieve_server2: ok ({movie['movie_id']})');\n"
            data = body.encode("utf-8")
            self.send_response(200)
            cors_headers(self)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "max-age=3000")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            _dprint(debug, f"[HTTP] GET from {self.client_address} -> 200 ({len(data)} bytes)")

        def do_POST(self):
            state["req_id"] += 1
            rid = state["req_id"]

            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            try:
                post_data = json.loads(raw.decode("utf-8"))
            except Exception as e:
                _dprint(debug, f"[HTTP] POST rid={rid} BAD_JSON err={e}")
                return self._reply_text("BAD_JSON", status=400)

            if "pastThroughput" in post_data:
                _dprint(debug, f"[POST] rid={rid} summary payload ignored keys={list(post_data.keys())}")
                return self._reply_text("0")

            try:
                last_quality = int(post_data["lastquality"])
                last_index = int(post_data["lastRequest"])
                buffer_s = float(post_data["buffer"])
                total_rebuf_ms = float(post_data["RebufferTime"])
                fetch_time_ms = float(post_data["lastChunkFinishTime"]) - float(post_data["lastChunkStartTime"])
                chunk_size_bytes = float(post_data["lastChunkSize"])
            except KeyError as e:
                return self._reply_text(f"MISSING_FIELD:{e}", status=400)
            except Exception as e:
                return self._reply_text(f"BAD_FIELD:{e}", status=400)

            if not (0 <= last_quality < A_DIM):
                return self._reply_text(f"BAD_LASTQUALITY:{last_quality}", status=400)

            if fetch_time_ms <= 0:
                _dprint(debug, f"[WARN] rid={rid} non-positive fetch_time_ms={fetch_time_ms}; clamping")
                fetch_time_ms = 1e-6

            rebuffer_delta_ms = max(0.0, total_rebuf_ms - state["last_total_rebuf_ms"])
            reward = _compute_reward(
                last_quality=last_quality,
                last_bitrate_kbps=state["last_bit_rate_kbps"],
                video_bitrates=VIDEO_BIT_RATE,
                rebuffer_delta_ms=rebuffer_delta_ms,
            )

            # Update running state first (current chunk was just played/downloaded)
            state["last_bit_rate_kbps"] = float(VIDEO_BIT_RATE[last_quality])
            state["last_total_rebuf_ms"] = float(total_rebuf_ms)

            prev_obs = state["s_batch"][-1] if state["s_batch"] else np.zeros((S_INFO, S_LEN), dtype=np.float32)
            video_chunk_remain = TOTAL_VIDEO_CHUNKS - state["video_chunk_count"]
            state["video_chunk_count"] += 1
            next_video_chunk_sizes = [get_chunk_size(i, state["video_chunk_count"]) for i in range(A_DIM)]

            obs = _update_observation(
                prev_obs=prev_obs,
                last_quality=last_quality,
                buffer_s=buffer_s,
                chunk_size_bytes=chunk_size_bytes,
                fetch_time_ms=fetch_time_ms,
                next_video_chunk_sizes=next_video_chunk_sizes,
                video_chunk_remain=video_chunk_remain,
                video_bitrates=VIDEO_BIT_RATE,
                chunk_til_video_end_cap=CHUNK_TIL_VIDEO_END_CAP,
                a_dim=A_DIM,
                debug=debug,
            )

            # Keep log format stable (same 7 columns) for downstream scripts.
            state["log_file"].write(
                f"{time.time()}\t{VIDEO_BIT_RATE[last_quality]}\t{buffer_s}\t"
                f"{rebuffer_delta_ms / M_IN_K}\t{int(chunk_size_bytes)}\t{fetch_time_ms}\t{reward}\n"
            )
            state["log_file"].flush()

            raw_prob = policy.predict(np.reshape(obs, (1, S_INFO, S_LEN)))
            action_prob = _safe_prob_vector(raw_prob, A_DIM, debug=debug)
            entropy = _policy_entropy(action_prob)
            next_bitrate_idx = _sample_action(action_prob, debug=debug)
            state["last_entropy"] = entropy

            _dprint(
                debug,
                f"[POST] rid={rid} ep={state.get('episode_id', 0)} lastReq={last_index}/{TOTAL_VIDEO_CHUNKS} "
                f"q={last_quality} buf={buffer_s:.3f}s rebufΔ={rebuffer_delta_ms/1000.0:.3f}s "
                f"fetch={fetch_time_ms/1000.0:.3f}s sz={int(chunk_size_bytes)}B reward={reward:.3f} "
                f"H={entropy:.3f} -> next_q={next_bitrate_idx}",
            )
            _vprint(verbose, f"[POST][RAW] rid={rid} keys={list(post_data.keys())} raw_prob={np.asarray(raw_prob).tolist()}")

            end_of_video = (last_index == TOTAL_VIDEO_CHUNKS)
            if end_of_video:
                send_data = "REFRESH"
                _dprint(debug, f"[EP] rid={rid} END_OF_VIDEO -> REFRESH + reset")
                _reset_episode_state(state, VIDEO_BIT_RATE)
            else:
                send_data = str(next_bitrate_idx)
                state["s_batch"].append(obs)

            return self._reply_text(send_data)

    return RequestHandler


def run(
    port: int,
    movie_path: str,
    model_path: Optional[str] = None,
    log_prefix: str = "",
    host: str = "0.0.0.0",
    debug: bool = False,
    verbose: bool = False,
):
    np.random.seed(RANDOM_SEED)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    movie = load_movie_json(movie_path, debug=debug)
    video_bitrates = movie["video_bit_rate_kbps"]
    n_q = len(video_bitrates)

    startup_time = time.time()
    log_file_path = f"{LOG_FILE}{log_prefix}_PT_RL_{movie['movie_id']}"

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        policy = a3c.Network(
            state_dim=[S_INFO, S_LEN],
            action_dim=n_q,
            learning_rate=ACTOR_LR_RATE,
            critic_learning_rate=CRITIC_LR_RATE,
            verbose=verbose,
        )

        model_path = model_path if model_path is not None else DEFAULT_MODEL_PATH
        _load_policy_checkpoint(policy, model_path, debug=debug)

        state = {
            "log_file": log_file,
            "last_bit_rate_kbps": float(video_bitrates[DEFAULT_QUALITY]),
            "last_total_rebuf_ms": 0.0,
            "video_chunk_count": 0,
            "s_batch": [np.zeros((S_INFO, S_LEN), dtype=np.float32)],
            "req_id": 0,
            "startup_time": startup_time,
            "episode_id": 0,
            "last_entropy": 0.0,
        }

        handler_cls = make_handler(state=state, movie=movie, policy=policy, debug=debug, verbose=verbose)
        server = ThreadingHTTPServer((host, int(port)), handler_cls)

        print(f"[BOOT] Loaded movie: {movie['movie_id']}")
        print(
            f"[BOOT] Qualities: {n_q} | TOTAL_VIDEO_CHUNKS(max idx): {movie['total_video_chunks']} "
            f"| seg_dur: {movie['chunk_duration_s']}s"
        )
        print(f"[BOOT] Log: {log_file_path}")
        print(f"[BOOT] Listening on http://{host}:{port}  (CORS enabled)")
        if debug:
            print("[BOOT] Debug enabled")
            if verbose:
                print("[BOOT] Verbose enabled")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[BOOT] KeyboardInterrupt -> shutting down")
        finally:
            server.server_close()
            print("[BOOT] Server closed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="..//..//DATASET//MOVIE//movie_4g.json", help="Path to movie.json")
    ap.add_argument("--port", type=int, default=8605)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--log-prefix", default="")
    ap.add_argument("--actor", default=DEFAULT_MODEL_PATH, help="Path to actor model (.pth), actor-only ")
    ap.add_argument("--debug", action="store_true", help="Print debug logs")
    ap.add_argument("--verbose", action="store_true", help="Extra verbose logs (raw probs / keys)")
    args = ap.parse_args()

    run(
        port=int(args.port),
        movie_path=args.movie,
        model_path=args.actor,
        log_prefix=args.log_prefix,
        host=args.host,
        debug=args.debug,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
