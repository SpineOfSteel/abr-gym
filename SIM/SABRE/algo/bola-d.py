# CustomAbr.py
# Simple throughput-based ABR plugin for refactored sab.py

# When sab.py is executed as a script, plugin symbols live in __main__.
# Fallback import supports running this in other contexts/tests.
try:
    from __main__ import register_abr, AbrBase, AbrDecision, AbrContext, DownloadProgress
except ImportError:
    from sab import register_abr, AbrBase, AbrDecision, AbrContext, DownloadProgress  # optional fallback
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



@register_abr("throughput")
class ThroughputRule(AbrBase):
    safety_factor = 0.9
    low_buffer_safety_factor = 0.5
    low_buffer_safety_factor_init = 0.9
    abandon_multiplier = 1.8
    abandon_grace_ms = 500

    def __init__(self, cfg: Dict, ctx: AbrContext):
        super().__init__(cfg, ctx)
        self.no_ibr = bool(cfg.get("no_ibr", False))
        self.ibr_safety = self.low_buffer_safety_factor_init

    def select(self, ctx: AbrContext, seg_idx: int) -> AbrDecision:
        if ctx.throughput_est is None or ctx.throughput_est <= 0:
            return AbrDecision(0, 0.0)

        q = self.quality_from_throughput(ctx, ctx.throughput_est * self.safety_factor)

        if not self.no_ibr and ctx.latency_est is not None:
            safe_bits = self.ibr_safety * (max(0.0, ctx.buffer.level_ms() - ctx.latency_est)) * ctx.throughput_est
            self.ibr_safety *= self.low_buffer_safety_factor_init
            self.ibr_safety = max(self.ibr_safety, self.low_buffer_safety_factor)
            m = ctx.manifest
            for qq in range(q):
                if m.bitrates_kbps[qq + 1] * m.segment_time_ms > safe_bits:
                    q = qq
                    break

        return AbrDecision(q, 0.0)

    def check_abandon(self, ctx: AbrContext, progress: DownloadProgress, buffer_level_ms: float) -> Optional[int]:
        m = ctx.manifest
        dl_ms = progress.time_ms - progress.time_to_first_bit_ms
        if progress.time_ms < self.abandon_grace_ms or dl_ms <= 0:
            return None

        tput = progress.downloaded_bits / dl_ms
        size_left = progress.size_bits - progress.downloaded_bits
        if tput <= 0 or size_left <= 0:
            return None

        est_left = size_left / tput
        if progress.time_ms + est_left <= self.abandon_multiplier * m.segment_time_ms:
            return None

        q = self.quality_from_throughput(ctx, tput * self.safety_factor)
        est_size = progress.size_bits * m.bitrates_kbps[q] / m.bitrates_kbps[progress.quality]
        if q >= progress.quality or est_size >= size_left:
            return None
        return q


@register_abr("dynamic")
class Dynamic(AbrBase):
    low_buffer_threshold_ms = 10_000

    def __init__(self, cfg: Dict, ctx: AbrContext):
        super().__init__(cfg, ctx)
        self.bola = Bola(cfg, ctx)
        self.tput = ThroughputRule(cfg, ctx)
        self.is_bola = False

    def select(self, ctx: AbrContext, seg_idx: int) -> AbrDecision:
        level = ctx.buffer.level_ms()
        b = self.bola.select(ctx, seg_idx)
        t = self.tput.select(ctx, seg_idx)

        if self.is_bola:
            if level < self.low_buffer_threshold_ms and b.quality < t.quality:
                self.is_bola = False
        else:
            if level > self.low_buffer_threshold_ms and b.quality >= t.quality:
                self.is_bola = True
        return b if self.is_bola else t

    def first_quality(self, ctx: AbrContext) -> int:
        return self.bola.first_quality(ctx) if self.is_bola else self.tput.first_quality(ctx)

    def on_delay(self, ctx: AbrContext, delay_ms: float) -> None:
        self.bola.on_delay(ctx, delay_ms)
        self.tput.on_delay(ctx, delay_ms)

    def on_download(self, ctx: AbrContext, dp: DownloadProgress, is_replacement: bool) -> None:
        self.bola.on_download(ctx, dp, is_replacement)
        self.tput.on_download(ctx, dp, is_replacement)
        if is_replacement:
            self.is_bola = False

    def check_abandon(self, ctx: AbrContext, progress: DownloadProgress, buffer_level_ms: float) -> Optional[int]:
        return self.tput.check_abandon(ctx, progress, buffer_level_ms)