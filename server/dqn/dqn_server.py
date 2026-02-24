# dqn_server.py
# python dqn_server.py --movie ../movie_4g.json --port 8606 --model dqn_ep_1000.pth --debug

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import dqn_torch as dqn  # your Torch DQN module


# ---------- RL / state constants ----------
S_INFO = 6   # bitrate, buffer, throughput, dl_time, next_chunk_sizes, chunks_left
S_LEN = 8

M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0

DEFAULT_QUALITY = 0
REBUF_PENALTY = 20
SMOOTH_PENALTY = 1

DQN_LR_RATE = 1e-4         # only used to instantiate the network object
RANDOM_SEED = 42
SUMMARY_DIR = "output"
LOG_FILE = "output/log_DQN_"

DEFAULT_MODEL_PATH = os.environ.get("TORCH_DQN_MODEL", "dqn_model.pth")


# ---------- Movie loader ----------
def load_movie_json(path: str, debug: bool = False) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        movie = json.load(f)

    for k in ("segment_duration_ms", "bitrates_kbps", "segment_sizes_bits"):
        if k not in movie:
            raise ValueError(f"movie.json missing {k}")

    chunk_duration_s = float(movie["segment_duration_ms"]) / 1000.0
    video_bit_rate_kbps = [int(x) for x in movie["bitrates_kbps"]]

    seg_bytes_quality: List[List[int]] = []
    for seg_idx, arr in enumerate(movie["segment_sizes_bits"]):
        if len(arr) != len(video_bit_rate_kbps):
            raise ValueError(
                f"segment_sizes_bits[{seg_idx}] has {len(arr)} entries, expected {len(video_bit_rate_kbps)}"
            )
        seg_bytes_quality.append([(int(x) + 7) // 8 for x in arr])  # bits -> bytes (ceil)

    total_video_chunks = int(movie.get("total_video_chunks", len(seg_bytes_quality) - 1))

    if debug:
        print("[MOVIE] Loaded", path)
        print(f"[MOVIE] movie_id={movie.get('movie_id', os.path.basename(path))}")
        print(f"[MOVIE] segment_duration_ms={movie['segment_duration_ms']} ({chunk_duration_s:.3f}s)")
        print(f"[MOVIE] bitrates_kbps={video_bit_rate_kbps}")
        print(f"[MOVIE] total_video_chunks(max_idx)={total_video_chunks}")
        if seg_bytes_quality:
            print(f"[MOVIE] first segment bytes={seg_bytes_quality[0]}")

    return {
        "movie_id": movie.get("movie_id", os.path.basename(path)),
        "chunk_duration_s": chunk_duration_s,
        "video_bit_rate_kbps": video_bit_rate_kbps,   # [A_DIM]
        "seg_bytes_quality": seg_bytes_quality,        # [seg_idx][q] bytes
        "total_video_chunks": total_video_chunks,      # max valid segment index
    }


# ---------- Helpers ----------
def cors_headers(h: BaseHTTPRequestHandler) -> None:
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")


def _safe_q_values(q_values, a_dim: int) -> np.ndarray:
    q = np.asarray(q_values, dtype=np.float32)
    if q.ndim == 2:
        q = q[0]
    if q.ndim != 1 or q.shape[0] != a_dim:
        q = np.zeros((a_dim,), dtype=np.float32)
    q = np.nan_to_num(q, nan=-1e9, posinf=1e9, neginf=-1e9)
    return q


def _load_policy_checkpoint(policy, model_path: Optional[str]) -> None:
    if model_path and os.path.exists(model_path):
        policy.load_model(model_path)
        print(f"[BOOT] DQN model restored from {model_path}")
    elif model_path:
        print(f"[BOOT] Warning: model not found: {model_path}")
        print("[BOOT] Continuing with randomly initialized weights.")


def _reset_episode_state(state: Dict, default_bitrate_kbps: int):
    state["last_total_rebuf"] = 0.0
    state["last_bit_rate"] = default_bitrate_kbps
    state["video_chunk_count"] = 0
    state["s_batch"] = [np.zeros((S_INFO, S_LEN), dtype=np.float32)]


def _update_observation(
    prev_obs: np.ndarray,
    last_quality: int,
    buffer_s: float,
    fetch_time_ms: float,
    chunk_size_bytes: float,
    next_video_chunk_sizes: List[int],
    video_chunk_remain: int,
    video_bitrates: List[int],
    chunks_till_end_cap: float,
) -> np.ndarray:
    obs = np.roll(np.array(prev_obs, copy=True, dtype=np.float32), -1, axis=1)
    max_br = float(max(video_bitrates))

    fetch_time_ms = max(float(fetch_time_ms), 1e-6)

    obs[0, -1] = video_bitrates[last_quality] / max_br
    obs[1, -1] = float(buffer_s) / BUFFER_NORM_FACTOR
    obs[2, -1] = (float(chunk_size_bytes) / fetch_time_ms) / M_IN_K           # KB/ms
    obs[3, -1] = (fetch_time_ms / M_IN_K) / BUFFER_NORM_FACTOR                # sec / 10
    obs[4, :] = 0.0
    obs[4, :len(video_bitrates)] = np.asarray(next_video_chunk_sizes, dtype=np.float32) / M_IN_K / M_IN_K  # MB
    obs[5, -1] = min(float(video_chunk_remain), chunks_till_end_cap) / chunks_till_end_cap

    if not np.isfinite(obs).all():
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return obs


def make_handler(state: Dict, movie: Dict, policy, debug: bool = False, verbose: bool = False, epsilon: float = 0.0):
    VIDEO_BIT_RATE = movie["video_bit_rate_kbps"]        # kbps
    CHUNK_SIZES = movie["seg_bytes_quality"]             # [seg][q] bytes
    TOTAL_VIDEO_CHUNKS = movie["total_video_chunks"]     # max valid seg index
    CHUNK_TIL_VIDEO_END_CAP = float(TOTAL_VIDEO_CHUNKS) if TOTAL_VIDEO_CHUNKS > 0 else 1.0
    A_DIM = len(VIDEO_BIT_RATE)

    def dprint(*args):
        if debug:
            print(*args)

    def vprint(*args):
        if verbose:
            print(*args)

    def get_chunk_size(quality: int, index: int) -> int:
        if index < 0 or index >= len(CHUNK_SIZES):
            dprint(f"[WARN] get_chunk_size out-of-range index={index}, len={len(CHUNK_SIZES)}")
            return 0
        if quality < 0 or quality >= A_DIM:
            dprint(f"[WARN] get_chunk_size bad quality q={quality}, A_DIM={A_DIM}")
            return 0
        return CHUNK_SIZES[index][quality]

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            return

        def _reply_text(self, text: str, status: int = 200):
            body = text.encode("utf-8")
            self.send_response(status)
            cors_headers(self)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            dprint(f"[HTTP] -> {status} '{text}'")

        def do_OPTIONS(self):
            self.send_response(204)
            cors_headers(self)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            body = f"console.log('dqn_server: ok ({movie['movie_id']})');\n"
            data = body.encode("utf-8")
            self.send_response(200)
            cors_headers(self)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "max-age=3000")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            state["req_id"] += 1
            rid = state["req_id"]

            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"

            try:
                post_data = json.loads(raw.decode("utf-8"))
            except Exception as e:
                dprint(f"[HTTP] rid={rid} BAD_JSON {e}")
                return self._reply_text("BAD_JSON", status=400)

            # dash.js sometimes sends summary payloads; ignore and respond dummy bitrate
            if "pastThroughput" in post_data:
                dprint(f"[POST] rid={rid} summary payload ignored")
                return self._reply_text("0")

            try:
                last_quality = int(post_data["lastquality"])
                last_index = int(post_data["lastRequest"])
                buffer_s = float(post_data["buffer"])
                total_rebuf = float(post_data["RebufferTime"])  # ms
                fetch_time_ms = float(post_data["lastChunkFinishTime"] - post_data["lastChunkStartTime"])
                chunk_size_bytes = float(post_data["lastChunkSize"])
            except KeyError as e:
                return self._reply_text(f"MISSING_FIELD:{e}", status=400)
            except Exception as e:
                return self._reply_text(f"BAD_FIELD:{e}", status=400)

            fetch_time_ms = max(fetch_time_ms, 1e-6)
            rebuffer_time_ms = max(float(total_rebuf - state["last_total_rebuf"]), 0.0)

            # Log reward (for analysis; DQN inference doesn't use reward online)
            reward = (
                VIDEO_BIT_RATE[last_quality] / M_IN_K
                - REBUF_PENALTY * (rebuffer_time_ms / M_IN_K)
                - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[last_quality] - state["last_bit_rate"]) / M_IN_K
            )

            state["last_bit_rate"] = VIDEO_BIT_RATE[last_quality]
            state["last_total_rebuf"] = total_rebuf

            prev_obs = state["s_batch"][-1] if state["s_batch"] else np.zeros((S_INFO, S_LEN), dtype=np.float32)

            video_chunk_remain = TOTAL_VIDEO_CHUNKS - state["video_chunk_count"]
            state["video_chunk_count"] += 1

            next_video_chunk_sizes = [get_chunk_size(i, state["video_chunk_count"]) for i in range(A_DIM)]

            obs = _update_observation(
                prev_obs=prev_obs,
                last_quality=last_quality,
                buffer_s=buffer_s,
                fetch_time_ms=fetch_time_ms,
                chunk_size_bytes=chunk_size_bytes,
                next_video_chunk_sizes=next_video_chunk_sizes,
                video_chunk_remain=video_chunk_remain,
                video_bitrates=VIDEO_BIT_RATE,
                chunks_till_end_cap=CHUNK_TIL_VIDEO_END_CAP,
            )

            state["log_file"].write(
                f"{time.time()}\t{VIDEO_BIT_RATE[last_quality]}\t{buffer_s}\t"
                f"{rebuffer_time_ms / M_IN_K}\t{int(chunk_size_bytes)}\t{fetch_time_ms}\t{reward}\n"
            )
            state["log_file"].flush()

            q_values = _safe_q_values(policy.predict(obs.reshape(1, S_INFO, S_LEN)), A_DIM)

            if epsilon > 0.0 and np.random.rand() < epsilon:
                bit_rate = int(np.random.randint(A_DIM))
                mode = "eps"
            else:
                bit_rate = int(np.argmax(q_values))
                mode = "greedy"

            dprint(
                f"[POST] rid={rid} q={last_quality} idx={last_index}/{TOTAL_VIDEO_CHUNKS} "
                f"buf={buffer_s:.3f}s rebufΔ={rebuffer_time_ms/1000.0:.3f}s fetch={fetch_time_ms/1000.0:.3f}s "
                f"reward={reward:.3f} -> next_q={bit_rate} ({mode})"
            )
            vprint(f"[Q] rid={rid} q_values={np.array2string(q_values, precision=3, suppress_small=True)}")

            end_of_video = (last_index == TOTAL_VIDEO_CHUNKS)
            if end_of_video:
                dprint(f"[EP] rid={rid} END_OF_VIDEO -> REFRESH")
                _reset_episode_state(state, VIDEO_BIT_RATE[DEFAULT_QUALITY])
                return self._reply_text("REFRESH")

            state["s_batch"].append(obs)
            return self._reply_text(str(bit_rate))

    return RequestHandler


def run(
    port: int,
    movie_path: str,
    model_path: Optional[str] = None,
    log_prefix: str = "",
    host: str = "0.0.0.0",
    debug: bool = False,
    verbose: bool = False,
    epsilon: float = 0.0,
):
    np.random.seed(RANDOM_SEED)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    movie = load_movie_json(movie_path, debug=debug)
    n_q = len(movie["video_bit_rate_kbps"])
    log_file_path = f"{LOG_FILE}{log_prefix}_PT_DQN_{movie['movie_id']}"

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        policy = dqn.Network(
            state_dim=[S_INFO, S_LEN],
            action_dim=n_q,
            learning_rate=DQN_LR_RATE,
        )

        model_path = model_path if model_path is not None else DEFAULT_MODEL_PATH
        _load_policy_checkpoint(policy, model_path)

        state = {
            "log_file": log_file,
            "last_bit_rate": movie["video_bit_rate_kbps"][DEFAULT_QUALITY],
            "last_total_rebuf": 0.0,
            "video_chunk_count": 0,
            "s_batch": [np.zeros((S_INFO, S_LEN), dtype=np.float32)],
            "req_id": 0,
        }

        handler_cls = make_handler(
            state=state,
            movie=movie,
            policy=policy,
            debug=debug,
            verbose=verbose,
            epsilon=float(max(0.0, min(1.0, epsilon))),
        )
        server = ThreadingHTTPServer((host, int(port)), handler_cls)

        print(f"[BOOT] Loaded movie: {movie['movie_id']}")
        print(f"[BOOT] Qualities: {n_q} | TOTAL_VIDEO_CHUNKS(max idx): {movie['total_video_chunks']}")
        print(f"[BOOT] Log: {log_file_path}")
        print(f"[BOOT] Listening on http://{host}:{port} (CORS enabled)")
        if epsilon > 0:
            print(f"[BOOT] Epsilon-greedy enabled: epsilon={epsilon:.3f}")
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
    ap.add_argument("--movie", default="../../movie_4g.json", help="Path to movie.json")
    ap.add_argument("--port", type=int, default=8606)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--log-prefix", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to DQN checkpoint (.pth)")
    ap.add_argument("--epsilon", type=float, default=0.0, help="Exploration prob (0 = greedy inference)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run(
        port=int(args.port),
        movie_path=args.movie,
        model_path=args.model,
        log_prefix=args.log_prefix,
        host=args.host,
        debug=args.debug,
        verbose=args.verbose,
        epsilon=float(args.epsilon),
    )


if __name__ == "__main__":
    main()