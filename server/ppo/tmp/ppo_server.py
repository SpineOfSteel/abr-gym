# ppo_server_http.py
# HTTP wrapper for PPO ABR server

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple

import numpy as np

from server.ppo.tmp.ppo_core import (
    DEFAULT_MODEL_PATH,
    LOG_FILE,
    RANDOM_SEED,
    SUMMARY_DIR,
    build_ppo_network,
    load_movie_json,
    load_ppo_checkpoint,
    make_initial_server_state,
    make_printers,
    process_decision_step,
)


def cors_headers(h: BaseHTTPRequestHandler) -> None:
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")


def parse_post_json(handler: BaseHTTPRequestHandler) -> Tuple[Optional[Dict], Optional[str]]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length) if content_length else b"{}"
    try:
        return json.loads(raw.decode("utf-8")), None
    except Exception as e:
        return None, f"BAD_JSON:{e}"


def make_handler(
    state: Dict,
    movie: Dict,
    ppo,
    debug: bool = False,
    verbose: bool = False,
):
    dprint, _ = make_printers(debug, verbose)

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

            post_data, parse_err = parse_post_json(self)
            if parse_err:
                dprint(f"[HTTP] rid={rid} {parse_err}")
                return self._reply_text("BAD_JSON", status=400)

            dprint_fn, vprint_fn = make_printers(debug, verbose)
            response_text, step_err = process_decision_step(
                state=state,
                movie=movie,
                ppo=ppo,
                post_data=post_data,
                rid=rid,
                dprint=dprint_fn,
                vprint=vprint_fn,
            )

            if step_err:
                return self._reply_text(step_err, status=400)

            return self._reply_text(response_text)

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
        ppo = build_ppo_network(n_q=n_q)
        model_path = model_path if model_path is not None else DEFAULT_MODEL_PATH
        load_ppo_checkpoint(ppo, model_path, debug=debug)

        state = make_initial_server_state(movie=movie, log_file=log_file, startup_time=startup_time)

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
    ap.add_argument("--movie", default="../movie_4g.json", help="Path to movie.json")
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