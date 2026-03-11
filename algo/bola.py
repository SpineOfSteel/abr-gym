# CustomAbr.py
# Simple throughput-based ABR plugin for refactored sab.py

# When sab.py is executed as a script, plugin symbols live in __main__.
# Fallback import supports running this in other contexts/tests.
try:
    from __main__ import register_abr, AbrBase, AbrDecision, AbrContext
except ImportError:
    from sab import register_abr, AbrBase, AbrDecision, AbrContext  # optional fallback
import math

class BolaBase(AbrBase):
    def __init__(self, cfg: Dict, ctx: AbrContext):
        super().__init__(cfg, ctx)
        m = ctx.manifest

        self.utility_mode = cfg.get("utility_mode", "zero")  # "zero" or "one"
        if self.utility_mode == "one":
            u0 = 1 - math.log(m.bitrates_kbps[0])
        else:
            u0 = -math.log(m.bitrates_kbps[0])
        self.utilities = [math.log(b) + u0 for b in m.bitrates_kbps]

        self.buffer_size_ms = float(cfg["buffer_size_ms"])
        self.gp = float(cfg["gp"])
        self.abr_basic = bool(cfg.get("abr_basic", False))
        self.abr_osc = bool(cfg.get("abr_osc", False))

        self.Vp = (self.buffer_size_ms - m.segment_time_ms) / (self.utilities[-1] + self.gp)

    def score_quality(self, ctx: AbrContext, q: int, level_ms: float) -> float:
        m = ctx.manifest
        return (self.Vp * (self.utilities[q] + self.gp) - level_ms) / m.bitrates_kbps[q]

    def quality_from_level(self, ctx: AbrContext, level_ms: Optional[float] = None) -> int:
        m = ctx.manifest
        level_ms = ctx.buffer.level_ms() if level_ms is None else level_ms
        best_q = 0
        best_s = None
        for q in range(len(m.bitrates_kbps)):
            s = self.score_quality(ctx, q, level_ms)
            if best_s is None or s > best_s:
                best_q, best_s = q, s
        return best_q

    def max_level_for(self, q: int) -> float:
        return self.Vp * (self.utilities[q] + self.gp)


@register_abr("bola")
class Bola(BolaBase):
    def __init__(self, cfg: Dict, ctx: AbrContext):
        cfg = dict(cfg)
        cfg["utility_mode"] = "zero"
        super().__init__(cfg, ctx)
        self.last_seek_idx = 0
        self.last_q = 0

    def select(self, ctx: AbrContext, seg_idx: int) -> AbrDecision:
        m = ctx.manifest
        tput = ctx.throughput_est
        if tput is None or tput <= 0:
            return AbrDecision(self.last_q, 0.0)

        if not self.abr_basic:
            t = min(seg_idx - self.last_seek_idx, len(m.segments_bits) - seg_idx)
            t = max(t / 2, 3)
            buf_ms = min(self.buffer_size_ms, t * m.segment_time_ms)
            self.Vp = (buf_ms - m.segment_time_ms) / (self.utilities[-1] + self.gp)

        q = self.quality_from_level(ctx)
        delay = 0.0

        if q > self.last_q:
            q_t = self.quality_from_throughput(ctx, tput)
            if q <= q_t:
                delay = 0.0
            elif self.last_q > q_t:
                q = self.last_q
                delay = 0.0
            else:
                if not self.abr_osc:
                    q = q_t + 1
                    delay = 0.0
                else:
                    q = q_t
                    lmax = self.max_level_for(q)
                    delay = max(0.0, ctx.buffer.level_ms() - lmax)
                    if q == len(m.bitrates_kbps) - 1:
                        delay = 0.0

        self.last_q = q
        return AbrDecision(q, delay)

    def on_seek(self, ctx: AbrContext, where_ms: float) -> None:
        self.last_seek_idx = int(where_ms // ctx.manifest.segment_time_ms)

    def check_abandon(self, ctx: AbrContext, progress: DownloadProgress, buffer_level_ms: float) -> Optional[int]:
        if self.abr_basic:
            return None
        m = ctx.manifest
        remain = progress.size_bits - progress.downloaded_bits
        if progress.downloaded_bits <= 0 or remain <= 0:
            return None

        best = None
        score = (self.Vp * (self.gp + self.utilities[progress.quality]) - buffer_level_ms) / remain
        if score < 0:
            return None

        for q in range(progress.quality):
            other_size = progress.size_bits * m.bitrates_kbps[q] / m.bitrates_kbps[progress.quality]
            other_score = (self.Vp * (self.gp + self.utilities[q]) - buffer_level_ms) / other_size
            if other_size < remain and other_score > score:
                score = other_score
                best = q
        if best is not None:
            self.last_q = best
        return best