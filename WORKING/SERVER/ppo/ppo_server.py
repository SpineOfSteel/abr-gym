#python ppo_server.py --verbose --debug  --port 8607  -model ../models/nn_model_ep_1800.pth --movie ../../movie_4g.json
# DIRECT python ppo_server.py --port 8605 --debug --verbose --model 'models/ppo_model.pth' --movie '../movie_4g.json'

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import ppo2 as network  # keep ppo2.py unchanged

SUMMARY_DIR = "..//SERVER_LOGS"
LOG_FILE = SUMMARY_DIR + "//log_PPO"
print("Current Working Directory:", os.getcwd())
DEFAULT_MODEL_PATH = os.environ.get("TORCH_PPO_MODEL", "..//..//DATASET//MODELS//ppo_model.pth")

# ---------- PPO/RL constants (aligned to attached PPO test script) ----------
S_INFO = 6
S_LEN = 8
A_DIM = 6

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001  # ppo2.Network uses one optimizer; kept for clarity / parity
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0  # same as test script
M_IN_K = 1000.0

# PPO test script uses 4.3 (not 20 like Pensieve/A3C variant)
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0
DEFAULT_QUALITY = 1  # same as attached PPO test script
RANDOM_SEED = 42


# ---------- Movie loader (movie.json schema) ----------
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


def cors_headers(h: BaseHTTPRequestHandler) -> None:
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")


def make_handler(
    state: Dict,
    movie: Dict,
    ppo: network.Network,
    debug: bool = False,
    verbose: bool = False,
):
    VIDEO_BIT_RATE = movie["video_bit_rate_kbps"]        # kbps
    CHUNK_SIZES = movie["seg_bytes_quality"]             # [seg][q] bytes
    TOTAL_VIDEO_CHUNKS = movie["total_video_chunks"]     # max valid seg idx
    NQ = len(VIDEO_BIT_RATE)

    def dprint(*args):
        if debug:
            print(*args)

    def vprint(*args):
        if verbose:
            print(*args)

    def get_chunk_size(quality: int, index: int) -> int:
        if index < 0 or index >= len(CHUNK_SIZES):
            dprint(f"[WARN] get_chunk_size out-of-range index={index} len={len(CHUNK_SIZES)}")
            return 0
        if quality < 0 or quality >= NQ:
            dprint(f"[WARN] get_chunk_size bad quality q={quality} nQ={NQ}")
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
            dprint(f"[HTTP] -> {status} '{text}'")

        def do_OPTIONS(self):
            self.send_response(204)
            cors_headers(self)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            body = f"console.log('ppo_server: ok ({movie['movie_id']})');\n"
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
                dprint(f"[HTTP] rid={rid} BAD_JSON err={e}")
                return self._reply_text("BAD_JSON", status=400)

            # Ignore summary payloads (same behavior as earlier server)
            if "pastThroughput" in post_data:
                dprint(f"[POST] rid={rid} summary payload ignored")
                return self._reply_text("0")

            try:
                last_quality = int(post_data["lastquality"])
                last_index = int(post_data["lastRequest"])
                buffer_s = float(post_data["buffer"])
                total_rebuf_ms = float(post_data["RebufferTime"])
                fetch_time_ms = float(post_data["lastChunkFinishTime"] - post_data["lastChunkStartTime"])
                chunk_size_bytes = float(post_data["lastChunkSize"])
            except KeyError as e:
                return self._reply_text(f"MISSING_FIELD:{e}", status=400)
            except Exception as e:
                return self._reply_text(f"BAD_FIELD:{e}", status=400)

            if not (0 <= last_quality < NQ):
                return self._reply_text(f"BAD_FIELD:lastquality={last_quality}", status=400)

            # Rebuffer delta (incoming total is cumulative ms)
            rebuffer_time_ms = float(total_rebuf_ms - state["last_total_rebuf_ms"])
            if rebuffer_time_ms < 0:
                dprint(f"[WARN] rid={rid} negative rebuf delta={rebuffer_time_ms}; clamp to 0")
                rebuffer_time_ms = 0.0
            rebuffer_time_s = rebuffer_time_ms / M_IN_K

            # PPO test reward style:
            # reward = bitrate(Mbps-ish using /1000) - 4.3*rebuf(sec) - smooth_penalty
            reward = (
                VIDEO_BIT_RATE[last_quality] / M_IN_K
                - REBUF_PENALTY * rebuffer_time_s
                - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[last_quality] - state["last_bit_rate_kbps"]) / M_IN_K
            )

            state["last_bit_rate_kbps"] = VIDEO_BIT_RATE[last_quality]
            state["last_total_rebuf_ms"] = total_rebuf_ms

            s_batch: List[np.ndarray] = state["s_batch"]
            obs = np.zeros((S_INFO, S_LEN), dtype=np.float32) if not s_batch else np.array(s_batch[-1], copy=True)

            # same style as previous server: internal chunk counter tracks "next chunk" feature
            video_chunk_remain = TOTAL_VIDEO_CHUNKS - state["video_chunk_count"]
            state["video_chunk_count"] += 1

            # shift history
            obs = np.roll(obs, -1, axis=1)

            next_video_chunk_sizes = [get_chunk_size(i, state["video_chunk_count"]) for i in range(NQ)]

            # State features (same layout as PPO test code)
            obs[0, -1] = VIDEO_BIT_RATE[last_quality] / float(np.max(VIDEO_BIT_RATE))          # last quality (normalized)
            obs[1, -1] = buffer_s / BUFFER_NORM_FACTOR                                         # buffer/10s
            obs[2, -1] = float(chunk_size_bytes) / max(float(fetch_time_ms), 1e-6) / M_IN_K   # KB/ms
            obs[3, -1] = float(fetch_time_ms) / M_IN_K / BUFFER_NORM_FACTOR                    # dl time / 10s
            obs[4, :] = 0.0
            obs[4, :NQ] = np.array(next_video_chunk_sizes, dtype=np.float32) / M_IN_K / M_IN_K # MB
            obs[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            # Log format mirrors attached test script (+ fetch_time_ms retained)
            state["log_file"].write(
                f"{time.time()}\t{VIDEO_BIT_RATE[last_quality]}\t{buffer_s}\t"
                f"{rebuffer_time_s}\t{int(chunk_size_bytes)}\t{fetch_time_ms}\t"
                f"{state['last_entropy']}\t{reward}\n"
            )
            state["log_file"].flush()

            # ppo2.Network.predict returns 1D pi for single state (shape [A_DIM]) in your ppo2.py
            action_prob = np.asarray(ppo.predict(np.reshape(obs, (1, S_INFO, S_LEN))), dtype=np.float32).reshape(-1)

            if action_prob.size != NQ:
                dprint(f"[WARN] rid={rid} bad policy output size={action_prob.size}, expected={NQ}; using uniform")
                action_prob = np.ones(NQ, dtype=np.float32) / float(NQ)
            else:
                action_prob = np.clip(action_prob, 1e-6, 1.0)
                s = float(np.sum(action_prob))
                if not np.isfinite(s) or s <= 0:
                    dprint(f"[WARN] rid={rid} invalid prob sum={s}; using uniform")
                    action_prob = np.ones(NQ, dtype=np.float32) / float(NQ)
                else:
                    action_prob /= s

            # Attached test script uses Gumbel sampling over log(pi)
            noise = np.random.gumbel(size=action_prob.shape[0])
            bit_rate = int(np.argmax(np.log(action_prob) + noise))

            entropy_ = float(-np.sum(action_prob * np.log(action_prob)))
            state["last_entropy"] = entropy_

            dprint(
                f"[POST] rid={rid} lastReq={last_index}/{TOTAL_VIDEO_CHUNKS} q={last_quality} "
                f"buf={buffer_s:.3f}s rebufΔ={rebuffer_time_s:.3f}s fetch={fetch_time_ms/1000.0:.3f}s "
                f"sz={int(chunk_size_bytes)}B reward={reward:.3f} -> next_q={bit_rate}"
            )
            vprint(
                f"[POST][DBG] rid={rid} probs={np.round(action_prob, 4).tolist()} "
                f"H={entropy_:.4f} next_sizes={next_video_chunk_sizes}"
            )

            end_of_video = (last_index == TOTAL_VIDEO_CHUNKS)
            if end_of_video:
                send_data = "REFRESH"
                dprint(f"[EP] rid={rid} END_OF_VIDEO -> REFRESH + reset")
                state["last_total_rebuf_ms"] = 0.0
                state["last_bit_rate_kbps"] = VIDEO_BIT_RATE[DEFAULT_QUALITY]
                state["video_chunk_count"] = 0
                state["s_batch"] = [np.zeros((S_INFO, S_LEN), dtype=np.float32)]
                state["last_entropy"] = 0.0
                state["episode_count"] += 1
                state["log_file"].write("\n")
                state["log_file"].flush()
            else:
                send_data = str(bit_rate)
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
    n_q = len(movie["video_bit_rate_kbps"])

    startup_time = time.time()
    log_file_path = f"{LOG_FILE}{log_prefix}_PT_PPO_{movie['movie_id']}"

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        # ppo2.Network expects combined actor+critic checkpoint for load_model()
        ppo = network.Network(
            state_dim=[S_INFO, S_LEN],
            action_dim=n_q,
            learning_rate=ACTOR_LR_RATE,
        )

        model_path = model_path if model_path is not None else DEFAULT_MODEL_PATH
        if model_path:
            if os.path.exists(model_path):
                try:
                    ppo.load_model(model_path)
                    print(f"[BOOT] PPO model restored from {model_path}")
                except Exception as e:
                    print(f"[BOOT] Warning: failed to load PPO model: {model_path}")
                    print(f"[BOOT] {e}")
                    print("[BOOT] Continuing with randomly initialized weights.")
            else:
                print(f"[BOOT] Warning: model file not found: {model_path}")
                print("[BOOT] Continuing with randomly initialized weights.")

        state = {
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

        handler_cls = make_handler(state=state, movie=movie, ppo=ppo, debug=debug, verbose=verbose)
        server = ThreadingHTTPServer((host, int(port)), handler_cls)

        print(f"[BOOT] Loaded movie: {movie['movie_id']}")
        print(
            f"[BOOT] Qualities: {n_q} | TOTAL_VIDEO_CHUNKS(max idx): {movie['total_video_chunks']} "
            f"| seg_dur: {movie['chunk_duration_s']}s"
        )
        print(f"[BOOT] Log: {log_file_path}")
        print(f"[BOOT] Listening on http://{host}:{port}  (CORS enabled)")
        print("[BOOT] Using ppo2.py Network API (ppo2.py unchanged)")
        if debug:
            print("[BOOT] Debug enabled")
            if verbose:
                print("[BOOT] Verbose debug enabled")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[BOOT] KeyboardInterrupt -> shutting down")
        finally:
            server.server_close()
            print("[BOOT] Server closed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="..//..//DATASET\MOVIE//movie_4g.json", help="Path to movie.json")
    ap.add_argument("--port", type=int, default=8605)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--log-prefix", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to PPO .pth (combined actor+critic)")
    ap.add_argument("--debug", action="store_true", help="Print informative debug lines")
    ap.add_argument("--verbose", action="store_true", help="More verbose debug (probs/entropy/next sizes)")
    args = ap.parse_args()

    
    run(
        port=int(args.port),
        movie_path=args.movie,
        model_path=args.model,
        log_prefix=args.log_prefix,
        host=args.host,
        debug=args.debug,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()