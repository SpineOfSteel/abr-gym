#!/usr/bin/env python3
#python fastmpc_server.py --verbose --debug --port 8395

import argparse
import itertools
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Tuple, Optional

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---------- MPC constants (not movie-specific) ----------
S_INFO = 5
S_LEN = 8
MPC_FUTURE_CHUNK_COUNT = 5

M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0

DEFAULT_QUALITY = 0
REBUF_PENALTY = 20
SMOOTH_PENALTY = 1

RANDOM_SEED = 42
SUMMARY_DIR = "SERVER_LOGS"
LOG_FILE = SUMMARY_DIR + "//log1"
LOG_BW = SUMMARY_DIR + "//log2"
print("Current Working Directory:", os.getcwd())


# ---------- Movie loader ----------
def load_movie_json(path: str, debug: bool = False) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        movie = json.load(f)

    if "segment_duration_ms" not in movie:
        raise ValueError("movie.json missing segment_duration_ms")
    if "bitrates_kbps" not in movie:
        raise ValueError("movie.json missing bitrates_kbps")
    if "segment_sizes_bits" not in movie:
        raise ValueError("movie.json missing segment_sizes_bits")

    # segment_duration_ms -> seconds
    chunk_duration_s = float(movie["segment_duration_ms"]) / 1000.0

    # bitrates in kbps
    video_bit_rate_kbps = [int(x) for x in movie["bitrates_kbps"]]
    if len(video_bit_rate_kbps) != 6:
        raise ValueError(f"Expected exactly 6 qualities, got {len(video_bit_rate_kbps)}")

    # segment_sizes_bits is assumed to be [segments][qualities] in *bits*
    # Convert bits -> bytes (ceil)
    seg_bytes_quality: List[List[int]] = []
    for seg_idx, arr in enumerate(movie["segment_sizes_bits"]):
        if len(arr) != len(video_bit_rate_kbps):
            raise ValueError(
                f"segment_sizes_bits[{seg_idx}] has {len(arr)} entries, expected {len(video_bit_rate_kbps)}"
            )
        seg_bytes_quality.append([(int(x) + 7) // 8 for x in arr])

    # IMPORTANT: total_video_chunks should be the MAX valid segment index.
    # If there are N segments in JSON, valid indices are 0..N-1, so TOTAL = N-1.
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
        # quick sanity check for q0
        exp_bytes_q0 = int(video_bit_rate_kbps[0] * 1000 * chunk_duration_s / 8)
        print(f"[MOVIE] sanity: expected bytes/segment at q0≈{exp_bytes_q0} (from bitrate & duration)\n")

    return {
        "movie_id": movie.get("movie_id", os.path.basename(path)),
        "chunk_duration_s": chunk_duration_s,
        "video_bit_rate_kbps": video_bit_rate_kbps,
        # shape: [segment_index][quality] -> bytes
        "seg_bytes_quality": seg_bytes_quality,
        "total_video_chunks": total_video_chunks,
    }


def build_chunk_combo_options(n_qualities: int) -> List[Tuple[int, ...]]:
    return list(itertools.product(range(n_qualities), repeat=MPC_FUTURE_CHUNK_COUNT))


def cors_headers(h: BaseHTTPRequestHandler) -> None:
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")


def make_handler(
    state: Dict,
    movie: Dict,
    chunk_combo_options: List[Tuple[int, ...]],
    debug: bool = False,
    verbose: bool = False,
):
    VIDEO_BIT_RATE = movie["video_bit_rate_kbps"]        # kbps
    CHUNK_SIZES = movie["seg_bytes_quality"]             # [seg][q] in bytes
    TOTAL_VIDEO_CHUNKS = movie["total_video_chunks"]     # max valid seg index
    CHUNK_DURATION_S = float(movie["chunk_duration_s"])
    CHUNK_TIL_VIDEO_END_CAP = float(TOTAL_VIDEO_CHUNKS)

    def dprint(*args):
        if debug:
            print(*args)

    def vprint(*args):
        if verbose:
            print(*args)

    def get_chunk_size(quality: int, index: int) -> int:
        # Safe bounds
        if index < 0 or index >= len(CHUNK_SIZES):
            if debug:
                print(f"[WARN] get_chunk_size out-of-range: index={index}, len={len(CHUNK_SIZES)}")
            return 0
        if quality < 0 or quality >= len(VIDEO_BIT_RATE):
            if debug:
                print(f"[WARN] get_chunk_size bad quality: q={quality}, nQ={len(VIDEO_BIT_RATE)}")
            return 0
        return CHUNK_SIZES[index][quality]  # bytes

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            return

        def do_OPTIONS(self):
            self.send_response(204)
            cors_headers(self)
            self.send_header("Content-Length", "0")
            self.end_headers()
            dprint(f"[HTTP] OPTIONS from {self.client_address}")

        def do_GET(self):
            body = f"console.log('fast_mpc_server: ok ({movie['movie_id']})');\n"
            data = body.encode("utf-8")
            self.send_response(200)
            cors_headers(self)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "max-age=3000")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            dprint(f"[HTTP] GET from {self.client_address} -> 200 ({len(data)} bytes)")

        def do_POST(self):
            state["req_id"] += 1
            rid = state["req_id"]

            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            try:
                post_data = json.loads(raw.decode("utf-8"))
            except Exception as e:
                dprint(f"[HTTP] POST rid={rid} BAD_JSON err={e}")
                return self._reply_text("BAD_JSON", status=400)

            if "pastThroughput" in post_data:
                dprint(f"[POST] rid={rid} Summary received (ignoring decision). keys={list(post_data.keys())}")
                return self._reply_text("0")

            # Pull fields
            last_quality = int(post_data["lastquality"])
            last_index = int(post_data["lastRequest"])
            buffer_s = float(post_data["buffer"])
            total_rebuf = float(post_data["RebufferTime"])
            rebuffer_time = float(total_rebuf - state["last_total_rebuf"])

            fetch_time = float(post_data["lastChunkFinishTime"] - post_data["lastChunkStartTime"])
            chunk_size = float(post_data["lastChunkSize"])  # bytes (from client)

            # Reward (same as your code)
            reward = (
                VIDEO_BIT_RATE[last_quality] / M_IN_K
                - REBUF_PENALTY * rebuffer_time / M_IN_K
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[last_quality] - state["last_bit_rate"]) / M_IN_K
            )

            # Update running state
            state["last_bit_rate"] = VIDEO_BIT_RATE[last_quality]
            state["last_total_rebuf"] = total_rebuf

            # Retrieve previous observation
            s_batch: List[np.ndarray] = state["s_batch"]
            obs = np.zeros((S_INFO, S_LEN), dtype=np.float32) if not s_batch else np.array(s_batch[-1], copy=True)

            video_chunk_remain = TOTAL_VIDEO_CHUNKS - state["video_chunk_count"]
            state["video_chunk_count"] += 1

            # Update obs with new sample
            obs = np.roll(obs, -1, axis=1)
            try:
                obs[0, -1] = VIDEO_BIT_RATE[last_quality] / float(np.max(VIDEO_BIT_RATE))
                obs[1, -1] = buffer_s / BUFFER_NORM_FACTOR
                obs[2, -1] = rebuffer_time / M_IN_K
                # protect against fetch_time==0
                if fetch_time > 0:
                    obs[3, -1] = (chunk_size / fetch_time) / M_IN_K
                else:
                    obs[3, -1] = 0.0
                    dprint(f"[WARN] rid={rid} fetch_time<=0 (fetch_time={fetch_time})")
                obs[4, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
            except ZeroDivisionError:
                obs = np.zeros((S_INFO, S_LEN), dtype=np.float32) if not s_batch else np.array(s_batch[-1], copy=True)
                dprint(f"[WARN] rid={rid} ZeroDivision in obs update; rolled back")

            # Log line
            state["log_file"].write(
                f"{time.time()}\t{VIDEO_BIT_RATE[last_quality]}\t{buffer_s}\t"
                f"{rebuffer_time / M_IN_K}\t{int(chunk_size)}\t{fetch_time}\t{reward}\n"
            )
            state["log_file"].flush()

            # harmonic mean bandwidth from last up to 5 samples
            past_bw = obs[3, -5:]
            while len(past_bw) > 0 and past_bw[0] == 0.0:
                past_bw = past_bw[1:]

            if len(past_bw) == 0:
                future_bw = 0.0
            else:
                inv = [1.0 / float(x) for x in past_bw]
                future_bw = 1.0 / (sum(inv) / len(inv))

            state["bw_file"].write(f"{time.time()}\t{future_bw}\n")
            state["bw_file"].flush()

            remaining = TOTAL_VIDEO_CHUNKS - last_index
            future_chunk_length = min(MPC_FUTURE_CHUNK_COUNT, max(0, remaining))

            # Debug summary line (one per POST, not too spammy)
            dprint(
                f"[POST] rid={rid} lastReq={last_index}/{TOTAL_VIDEO_CHUNKS} q={last_quality} "
                f"buf={buffer_s:.3f}s rebufΔ={rebuffer_time:.3f}s fetch={fetch_time:.3f}s "
                f"sz={int(chunk_size)}B bw_est={future_bw:.3f} "
                f"lookahead={future_chunk_length} reward={reward:.3f}"
            )
            vprint(f"[POST][RAW] rid={rid} keys={list(post_data.keys())}")

            # MPC pick
            max_reward = -1e18
            best_combo: Optional[Tuple[int, ...]] = None
            start_buffer = buffer_s

            if future_bw <= 0 or future_chunk_length == 0:
                best_combo = (0,)
                dprint(f"[MPC] rid={rid} fallback -> q=0 (future_bw={future_bw:.3f}, lookahead={future_chunk_length})")
            else:
                t0 = time.time()
                for full_combo in chunk_combo_options:
                    combo = full_combo[:future_chunk_length]

                    curr_rebuf = 0.0
                    curr_buf = start_buffer
                    bitrate_sum = 0.0
                    smooth = 0.0
                    last_q = last_quality

                    for pos, q in enumerate(combo):
                        idx = last_index + pos + 1
                        seg_bytes = get_chunk_size(q, idx)
                        dl_time = (seg_bytes / 1_000_000.0) / future_bw  # seconds

                        if curr_buf < dl_time:
                            curr_rebuf += (dl_time - curr_buf)
                            curr_buf = 0.0
                        else:
                            curr_buf -= dl_time

                        curr_buf += CHUNK_DURATION_S

                        bitrate_sum += VIDEO_BIT_RATE[q]
                        smooth += abs(VIDEO_BIT_RATE[q] - VIDEO_BIT_RATE[last_q])
                        last_q = q

                    combo_reward = (bitrate_sum / 1000.0) - (REBUF_PENALTY * curr_rebuf) - (smooth / 1000.0)
                    if combo_reward > max_reward:
                        max_reward = combo_reward
                        best_combo = combo

                t1 = time.time()
                dprint(
                    f"[MPC] rid={rid} chose q={best_combo[0] if best_combo else 0} "
                    f"best_combo={best_combo} max_reward={max_reward:.3f} "
                    f"search_time={(t1 - t0)*1000:.1f}ms"
                )

            end_of_video = (last_index == TOTAL_VIDEO_CHUNKS)
            if end_of_video:
                send_data = "REFRESH"
                dprint(f"[EP] rid={rid} END_OF_VIDEO -> REFRESH + reset state")
                state["last_total_rebuf"] = 0.0
                state["last_bit_rate"] = VIDEO_BIT_RATE[DEFAULT_QUALITY]
                state["video_chunk_count"] = 0
                state["s_batch"] = [np.zeros((S_INFO, S_LEN), dtype=np.float32)]
            else:
                send_data = str(best_combo[0] if best_combo else 0)
                state["s_batch"].append(obs)

            return self._reply_text(send_data)

        def _reply_text(self, text: str, status: int = 200):
            body = text.encode("utf-8")
            self.send_response(status)
            cors_headers(self)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            dprint(f"[HTTP] -> {status} reply='{text}' ({len(body)} bytes)")

    return RequestHandler

def run(port: int, movie_path: str, log_prefix: str = "", host: str = "0.0.0.0", debug: bool = False, verbose: bool = False):
    np.random.seed(RANDOM_SEED)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_BW), exist_ok=True)

    movie = load_movie_json(movie_path, debug=debug)
    n_q = len(movie["video_bit_rate_kbps"])
    chunk_combo_options = build_chunk_combo_options(n_q)
    video_bitrates = movie["video_bit_rate_kbps"]  # <-- FIX

    startup_time = time.time()
    log_file_path = f"{LOG_FILE}{log_prefix}_MPC_{movie['movie_id']}"
    bw_file_path = f"{LOG_BW}{log_prefix}_MPC_{movie['movie_id']}"

    with open(log_file_path, "w", encoding="utf-8") as log_file, open(bw_file_path, "w", encoding="utf-8") as bw_file:
        bw_file.write(f"{startup_time}\n")
        bw_file.flush()

        state = {
            "log_file": log_file,
            "bw_file": bw_file,
            "last_bit_rate": video_bitrates[DEFAULT_QUALITY],  # <-- FIXED
            "last_total_rebuf": 0.0,
            "video_chunk_count": 0,
            "s_batch": [np.zeros((S_INFO, S_LEN), dtype=np.float32)],
            "req_id": 0,
        }

        handler_cls = make_handler(state, movie, chunk_combo_options, debug=debug, verbose=verbose)
        server = ThreadingHTTPServer((host, port), handler_cls)

        print(f"[BOOT] Loaded movie: {movie['movie_id']}")
        print(f"[BOOT] Qualities: {n_q} | TOTAL_VIDEO_CHUNKS(max idx): {movie['total_video_chunks']} | seg_dur: {movie['chunk_duration_s']}s")
        print(f"[BOOT] Logs: {log_file_path} , {bw_file_path}")
        print(f"[BOOT] Listening on http://{host}:{port}  (CORS enabled)")
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
    ap.add_argument("--movie", default="..//DATASET//MOVIE//movie_4g.json", help="Path to movie.json")
    ap.add_argument("--port", type=int, default=8333)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--log-prefix", default="")
    ap.add_argument("--debug", action="store_true", help="Print informative debug lines")
    ap.add_argument("--verbose", action="store_true", help="More verbose debug (adds raw/key dumps)")
    args = ap.parse_args()

    run(
        port= int(args.port),
        movie_path=args.movie,
        log_prefix=args.log_prefix,
        host=args.host,
        debug=args.debug,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
