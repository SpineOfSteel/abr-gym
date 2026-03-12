# mpcshim_base.py
import json
import urllib.request
import urllib.error

try:
    from __main__ import AbrBase, AbrDecision, AbrContext
except ImportError:
    from sab import AbrBase, AbrDecision, AbrContext


class HttpShimBase(AbrBase):
    """
    Generic HTTP shim for MPC-style servers.
    Subclasses only need to set defaults (URL/name), registration happens in wrapper files.
    """

    DEFAULT_SHIM_URL = "http://127.0.0.1:8333/"
    SHIM_NAME = "mpc"

    def __init__(self, cfg: dict, ctx: AbrContext):
        super().__init__(cfg, ctx)
        cfg = dict(cfg or {})

        self.server_url = self.DEFAULT_SHIM_URL
        print('BASE-URL', self.server_url)
        
        self.timeout_s = float(cfg.get("timeout_s", 1.0))
        self._debug = bool(cfg.get("debug_p", False))
        self._ping_on_start = bool(cfg.get("ping_on_start", True))

        self._pending_post = None
        self._synthetic_clock_ms = 0.0
        self._last_server_q = 0

        self._log(f"{self.SHIM_NAME} init server_url={self.server_url} timeout={self.timeout_s}s")

    # ---------- Helpers ----------
    def _log(self, *args):
        if self._debug:
            print(f"[{self.SHIM_NAME.upper()}_HTTP]", *args)

    def _rb_fallback(self, ctx: AbrContext) -> int:
        """
        throughput_est is bits/ms. bitrates are kbps.
        Numerically bits/ms == kbps, so direct comparison is valid.
        """
        tput = ctx.throughput_est
        if tput is None or tput <= 0:
            return 0

        bitrates = ctx.manifest.bitrates_kbps
        q = 0
        while q + 1 < len(bitrates) and bitrates[q + 1] <= tput:
            q += 1
        return q

    def _post_mpc(self, payload: dict) -> int:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.server_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8").strip()
        except (urllib.error.URLError, TimeoutError, Exception) as e:
            self._log(f"POST failed -> fallback ({e}) payload={payload}")
            raise

        if body == "REFRESH":
            self._log("Server returned REFRESH")
            self._last_server_q = 0
            return 0

        q = int(body)
        self._last_server_q = q
        return q

    # ---------- ABR API ----------
    def first_quality(self, ctx: AbrContext) -> int:
        self._pending_post = None
        self._synthetic_clock_ms = 0.0
        self._last_server_q = 0

        if self._ping_on_start:
            try:
                with urllib.request.urlopen(self.server_url, timeout=0.5) as r:
                    self._log(f"ping ok status={r.status}")
            except Exception as e:
                self._log(f"ping failed: {e}")

        return 0

    def on_download(self, ctx: AbrContext, dp, is_replacement: bool) -> None:
        """
        Uses:
          ctx.buffer.level_ms()
          ctx.buffer.stats.rebuffer_time_ms
          ctx.manifest.segments_bits
        """
        if is_replacement:
            self._log(f"Skipping replacement segment idx={dp.index}")
            return

        if dp.abandon_to_quality is not None and dp.downloaded_bits < dp.size_bits:
            self._log(
                f"Skipping partial/abandoned idx={dp.index} "
                f"(downloaded_bits={dp.downloaded_bits}, size_bits={dp.size_bits}, "
                f"abandon_to_quality={dp.abandon_to_quality})"
            )
            return

        seg_idx = int(dp.index)
        q = int(dp.quality)

        if seg_idx < 0 or seg_idx >= len(ctx.manifest.segments_bits):
            self._log(f"Bad seg_idx={seg_idx}; manifest has {len(ctx.manifest.segments_bits)} segments")
            return
        if q < 0 or q >= len(ctx.manifest.bitrates_kbps):
            self._log(f"Bad quality={q}; manifest has {len(ctx.manifest.bitrates_kbps)} qualities")
            return

        size_bits = float(dp.size_bits)
        if size_bits <= 0:
            size_bits = float(ctx.manifest.segments_bits[seg_idx][q])
            self._log(f"dp.size_bits<=0, fallback to manifest size_bits={size_bits}")

        size_bytes = int((size_bits + 7) // 8)

        dl_time_ms = max(0.0, float(dp.time_ms))
        t_start_ms = self._synthetic_clock_ms
        t_end_ms = t_start_ms + dl_time_ms
        self._synthetic_clock_ms = t_end_ms

        buffer_ms = max(0.0, float(ctx.buffer.level_ms()))
        buffer_s = buffer_ms / 1000.0

        try:
            total_rebuf_ms = float(ctx.buffer.stats.rebuffer_time_ms)
        except Exception:
            total_rebuf_ms = 0.0

        self._pending_post = {
            "lastquality": q,                 # quality index
            "RebufferTime": total_rebuf_ms,   # cumulative ms
            "buffer": buffer_s,               # seconds
            "lastChunkStartTime": t_start_ms, # ms
            "lastChunkFinishTime": t_end_ms,  # ms
            "lastChunkSize": size_bytes,      # bytes
            "lastRequest": seg_idx,           # just-finished segment index
        }

        self._log(
            f"Prepared seg={seg_idx} q={q} "
            f"size={size_bytes}B ({size_bits:.0f} bits) "
            f"dl={dl_time_ms:.1f}ms ttfb={float(dp.time_to_first_bit_ms):.1f}ms "
            f"buf={buffer_s:.3f}s rebuf_total={total_rebuf_ms:.1f}ms"
        )

    def select(self, ctx: AbrContext, seg_idx: int) -> AbrDecision:
        if self._pending_post is None:
            q = self._rb_fallback(ctx)
            self._log(f"No payload yet -> fallback q={q} for seg={seg_idx}")
            return AbrDecision(q, 0.0)

        expected_last_req = seg_idx - 1
        if self._pending_post["lastRequest"] != expected_last_req:
            self._log(
                f"Warning: pending lastRequest={self._pending_post['lastRequest']} "
                f"but next seg_idx={seg_idx} (expected {expected_last_req})"
            )

        try:
            q = self._post_mpc(self._pending_post)
            q = max(0, min(q, len(ctx.manifest.bitrates_kbps) - 1))
            self._log(f"MPC chose q={q} for next seg={seg_idx}")
            return AbrDecision(q, 0.0)
        except Exception:
            q = self._rb_fallback(ctx)
            self._log(f"Server unavailable -> fallback q={q} for seg={seg_idx}")
            return AbrDecision(q, 0.0)
