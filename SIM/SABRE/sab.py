# MARK COPYRIGHTS

from __future__ import annotations

import argparse
import importlib.util
import math
import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import  Dict,  Optional, Type

from DataModel import AbrContext, Buffer, DownloadProgress, Manifest, NetworkPeriod, ReactionTracker, SimStats
from NetworkModel import NetworkModel
from ChunkLogger import ChunkLogger
from AbrInterface import AbrBase, AbrDecision


# ----------------------------- registries -----------------------------

ABR_REGISTRY: Dict[str, Type["AbrBase"]] = {}
AVG_REGISTRY: Dict[str, Type["ThroughputHistory"]] = {}

def register_abr(name: str):
    def deco(cls: Type["AbrBase"]):
        ABR_REGISTRY[name] = cls
        cls.NAME = name
        return cls
    return deco

def register_avg(name: str):
    def deco(cls: Type["ThroughputHistory"]):
        AVG_REGISTRY[name] = cls
        cls.NAME = name
        return cls
    return deco

def load_plugin(path: str) -> None:
    p = Path(path)
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load plugin: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # plugin is expected to call @register_abr / @register_avg


# ----------------------- throughput

# ----------------------------- throughput estimators -----------------------------

class ThroughputHistory:
    NAME = "base"
    def __init__(self, *_args, **_kwargs): ...
    def push(self, download_time_ms: float, tput_bits_per_ms: float, latency_ms: float) -> Tuple[float, float]:
        raise NotImplementedError


@register_avg("sliding")
class SlidingWindow(ThroughputHistory):
    max_store = 20
    def __init__(self, window_size: List[int]):
        self.window_size = window_size or [3]
        self.tputs: List[float] = []
        self.lats: List[float] = []

    def push(self, dt_ms: float, tput: float, lat: float) -> Tuple[float, float]:
        self.tputs = (self.tputs + [tput])[-self.max_store:]
        self.lats = (self.lats + [lat])[-self.max_store:]

        est_t = None
        est_l = None
        for ws in self.window_size:
            ws = max(1, ws)
            sample_t = self.tputs[-ws:]
            sample_l = self.lats[-ws:]
            t = sum(sample_t) / len(sample_t)
            l = sum(sample_l) / len(sample_l)
            est_t = t if est_t is None else min(est_t, t)
            est_l = l if est_l is None else max(est_l, l)
        return float(est_t), float(est_l)


@register_avg("ewma")
class Ewma(ThroughputHistory):
    def __init__(self, half_life_s: List[float], seg_time_ms: float):
        if seg_time_ms <= 0:
            raise ValueError("seg_time_ms must be > 0")
        self.half_life_ms = [(h * 1000.0) for h in (half_life_s or [3, 8])]
        self.lat_half_life = [hl / seg_time_ms for hl in self.half_life_ms]
        self.t_est = [0.0] * len(self.half_life_ms)
        self.l_est = [0.0] * len(self.half_life_ms)
        self.w_t = 0.0
        self.w_l = 0.0

    def push(self, dt_ms: float, tput: float, lat: float) -> Tuple[float, float]:
        for i, hl in enumerate(self.half_life_ms):
            a = math.pow(0.5, dt_ms / hl)
            self.t_est[i] = a * self.t_est[i] + (1 - a) * tput

            a = math.pow(0.5, 1.0 / self.lat_half_life[i])
            self.l_est[i] = a * self.l_est[i] + (1 - a) * lat

        self.w_t += dt_ms
        self.w_l += 1.0

        est_t = None
        est_l = None
        for i, hl in enumerate(self.half_life_ms):
            zt = 1 - math.pow(0.5, self.w_t / hl)
            zl = 1 - math.pow(0.5, self.w_l / self.lat_half_life[i])
            t = self.t_est[i] / zt
            l = self.l_est[i] / zl
            est_t = t if est_t is None else min(est_t, t)
            est_l = l if est_l is None else max(est_l, l)
        return float(est_t), float(est_l)

# ----------------------------- Replace strategies -----------------------------

class ReplaceStrategy:
    def check_replace(self, _ctx: AbrContext, _quality: int) -> Optional[int]:
        return None

    # FIX: include ctx so we can avoid globals
    def check_abandon(self, _ctx: AbrContext, _progress: DownloadProgress, _buffer_level_ms: float) -> Optional[int]:
        return None

class NoReplace(ReplaceStrategy):
    pass

class Replace(ReplaceStrategy):
    """strategy=0 scans left->right; strategy=1 scans right->left."""
    def __init__(self, strategy: int):
        self.strategy = strategy
        self.replacing: Optional[int] = None

    def check_replace(self, ctx: AbrContext, quality: int) -> Optional[int]:
        self.replacing = None
        buf = ctx.buffer.contents
        if not buf:
            return None

        skip = math.ceil(1.5 + ctx.buffer.fcc_ms / ctx.manifest.segment_time_ms)
        if self.strategy == 0:
            rng = range(skip, len(buf))
        else:
            rng = range(len(buf) - 1, skip - 1, -1)

        for i in rng:
            if buf[i] < quality:
                self.replacing = i - len(buf)  # negative offset
                break
        return self.replacing

    def check_abandon(self, ctx: AbrContext, _progress: DownloadProgress, buffer_level_ms: float) -> Optional[int]:
        if self.replacing is None:
            return None
        # too late to replace => abandon
        if buffer_level_ms + ctx.manifest.segment_time_ms * self.replacing <= 0:
            return -1
        return None


# ----------------------------- ABR implementations -----------------------------

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


# ----------------------------- simulator core -----------------------------

def deplete_buffer(ctx: AbrContext, tracker: ReactionTracker, dt_ms: float) -> None:
    m = ctx.manifest
    buf = ctx.buffer
    st = ctx.stats

    if not buf.contents:
        st.rebuffer_time_ms += dt_ms
        st.total_play_time_ms += dt_ms
        return

    if buf.fcc_ms > 0:
        if dt_ms + buf.fcc_ms < m.segment_time_ms:
            buf.fcc_ms += dt_ms
            st.total_play_time_ms += dt_ms
            return
        dt_ms -= (m.segment_time_ms - buf.fcc_ms)
        st.total_play_time_ms += (m.segment_time_ms - buf.fcc_ms)
        buf.contents.pop(0)
        buf.fcc_ms = 0.0

    while dt_ms > 0 and buf.contents:
        q = buf.contents[0]
        st.played_utility += m.utilities[q]
        st.played_bitrate += m.bitrates_kbps[q]

        if st.last_played is not None and q != st.last_played:
            st.total_bitrate_change += abs(m.bitrates_kbps[q] - m.bitrates_kbps[st.last_played])
            st.total_log_bitrate_change += abs(math.log(m.bitrates_kbps[q] / m.bitrates_kbps[st.last_played]))
        st.last_played = q

        if st.rampup_time_ms is None:
            rt = ctx.sustainable_quality if ctx.rampup_threshold is None else ctx.rampup_threshold
            if rt is not None and q >= rt:
                st.rampup_time_ms = st.total_play_time_ms - st.rampup_origin_ms

        tracker.on_played_quality(st.total_play_time_ms, q)

        if dt_ms >= m.segment_time_ms:
            buf.contents.pop(0)
            st.total_play_time_ms += m.segment_time_ms
            dt_ms -= m.segment_time_ms
        else:
            buf.fcc_ms = dt_ms
            st.total_play_time_ms += dt_ms
            dt_ms = 0.0

    if dt_ms > 0:
        st.rebuffer_time_ms += dt_ms
        st.total_play_time_ms += dt_ms
        st.rebuffer_event_count += 1

    tracker.on_time_advanced(st.total_play_time_ms)

def playout_buffer(ctx: AbrContext, tracker: ReactionTracker) -> None:
    deplete_buffer(ctx, tracker, ctx.buffer.level_ms())
    ctx.buffer.clear()

def bits_per_ms(downloaded_bits: float, download_ms: float) -> float:
    return downloaded_bits / download_ms if download_ms > 0 else 0.0


def run_session(args: argparse.Namespace) -> None:
    manifest = Manifest.from_json(args.movie, args.movie_length)
    trace = NetworkPeriod.from_json(args.network, 0, args.network_multiplier)
    chunk_logger =  None
    if args.chunk_folder!='' and args.chunk_log!='':
        chunk_logger = ChunkLogger(args.chunk_folder, args.chunk_log, args.chunk_log_start_ts)
        
    stats = SimStats(rampup_origin_ms=0.0)
    buffer = Buffer(manifest=manifest)
    tracker = ReactionTracker(max_buffer_ms=args.max_buffer * 1000.0)

    # throughput estimator
    if args.moving_average == "ewma":
        avg = Ewma(args.half_life, manifest.segment_time_ms)
    else:
        avg = AVG_REGISTRY[args.moving_average](args.window_size)

    abr_cfg = dict(
        buffer_size_ms=args.max_buffer * 1000.0,
        gp=args.gamma_p,
        abr_basic=args.abr_basic,
        abr_osc=args.abr_osc,
        no_ibr=args.no_insufficient_buffer_rule,
        shim = args.shim,
        timeout_s = args.timeout_s,
        debug_p = args.debug_p,
        ping_on_start = args.ping_on_start
    )

    ctx = AbrContext(
        manifest=manifest,
        buffer=buffer,
        max_buffer_ms=args.max_buffer * 1000.0,
        throughput_est=None,
        latency_est=None,
        sustainable_quality=None,
        rampup_threshold=args.rampup_threshold,
        stats=stats,
    )

    net = NetworkModel(trace, manifest, args.verbose)

    if args.replace == "left":
        replacer: ReplaceStrategy = Replace(0)
    elif args.replace == "right":
        replacer = Replace(1)
    else:
        replacer = NoReplace()

    abr = ABR_REGISTRY[args.abr](abr_cfg, ctx)

    # --- download first segment ---
    rebuf_before = stats.rebuffer_time_ms
    
    q0 = abr.first_quality(ctx)
    bits0 = manifest.segments_bits[0][q0]
    dp0 = net.download(bits0, 0, q0, 0.0, None)
    dl0 = dp0.time_ms - dp0.time_to_first_bit_ms
    startup_time_ms = dl0  # kept for parity (not printed separately)
    buffer.contents.append(dp0.quality)

    t = bits_per_ms(dp0.size_bits, dl0)
    l = dp0.time_to_first_bit_ms
    if dl0 > 0:
        ctx.throughput_est, ctx.latency_est = avg.push(dl0, t, l)

    stats.total_play_time_ms += dp0.time_ms
    
    prev_sust = ctx.sustainable_quality
    ctx.sustainable_quality = net.sustainable_quality
    if ctx.sustainable_quality is not None:
        tracker.on_network_quality_change(stats.total_play_time_ms, ctx.sustainable_quality, prev_sust, buffer.contents)
    
    stall_ms = stats.rebuffer_time_ms - rebuf_before  # likely 0 on startup unless you model it
    if chunk_logger and dp0.abandon_to_quality is None:
        chunk_logger.log(
            sim_time_ms=stats.total_play_time_ms,
            bitrate_kbps=manifest.bitrates_kbps[dp0.quality],
            buffer_ms=buffer.level_ms(),            # after append
            stall_ms=stall_ms,
            chunk_bits=dp0.size_bits,
            download_ms=dp0.time_ms,                # includes latency
        )
    # --- main loop ---
    next_seg = 1
    pending_abandon_to: Optional[int] = None
    
    while next_seg < len(manifest.segments_bits):
        if args.seek is not None:
            when_s, where_s = args.seek
            if next_seg * manifest.segment_time_ms >= 1000.0 * when_s:
                next_seg = int((1000.0 * where_s) // manifest.segment_time_ms)
                buffer.clear()
                abr.on_seek(ctx, 1000.0 * where_s)
                args.seek = None
                stats.rampup_origin_ms = stats.total_play_time_ms
                stats.rampup_time_ms = None
        
        
        
        full_delay = buffer.level_ms() + manifest.segment_time_ms - (args.max_buffer * 1000.0)
        if full_delay > 0:
            deplete_buffer(ctx, tracker, full_delay)
            net.delay(full_delay)
            abr.on_delay(ctx, full_delay)

        if pending_abandon_to is None:
            dec = abr.select(ctx, next_seg)
            q = dec.quality
            replace = replacer.check_replace(ctx, q)
            delay = dec.delay_ms
        else:
            q, delay, replace = pending_abandon_to, 0.0, None
            pending_abandon_to = None

        if replace is not None:
            delay = 0.0
            cur_seg = next_seg + replace
            check_abandon = (lambda dp, bl: replacer.check_abandon(ctx, dp, bl))
        else:
            cur_seg = next_seg
            check_abandon = (lambda dp, bl: abr.check_abandon(ctx, dp, bl))

        if args.no_abandon:
            check_abandon = None

        if delay > 0:
            deplete_buffer(ctx, tracker, delay)
            net.delay(delay)
            abr.on_delay(ctx, delay)
        
        rebuf_before = stats.rebuffer_time_ms

        bits = manifest.segments_bits[cur_seg][q]
        dp = net.download(bits, cur_seg, q, buffer.level_ms(), check_abandon)
        deplete_buffer(ctx, tracker, dp.time_ms)
        
         
        if replace is None:
            if dp.abandon_to_quality is None:
                buffer.contents.append(q)
                next_seg += 1
            else:
                pending_abandon_to = dp.abandon_to_quality
        else:
            if dp.abandon_to_quality is None:
                if buffer.level_ms() + manifest.segment_time_ms * replace >= 0:
                    buffer.contents[replace] = q

        ###Only log when a chunk is actually "delivered" (not abandoned mid-download)
        stall_ms = stats.rebuffer_time_ms - rebuf_before


        if chunk_logger and dp.abandon_to_quality is None:
            # Determine the bitrate we actually ended up delivering:
            delivered_q = dp.quality  # not q ? (because abandon_to_quality is None)
            chunk_logger.log(
                sim_time_ms=stats.total_play_time_ms,
                bitrate_kbps=manifest.bitrates_kbps[delivered_q],
                buffer_ms=buffer.level_ms(),     # after append/replace so it matches typical logs
                stall_ms=stall_ms,
                chunk_bits=dp.size_bits,
                download_ms=dp.time_ms,
            )
        ###
        
        abr.on_download(ctx, dp, is_replacement=(replace is not None))

        prev_sust = ctx.sustainable_quality
        ctx.sustainable_quality = net.sustainable_quality
        if ctx.sustainable_quality is not None:
            tracker.on_network_quality_change(stats.total_play_time_ms, ctx.sustainable_quality, prev_sust, buffer.contents)

        dl_ms = dp.time_ms - dp.time_to_first_bit_ms
        
        actual_t = None
        actual_l = None

        # Count estimate accuracy even if abandoned (original behavior)
        if dl_ms > 0 and dp.downloaded_bits > 0:
            actual_t = bits_per_ms(dp.downloaded_bits, dl_ms)
            actual_l = dp.time_to_first_bit_ms

            if ctx.throughput_est is not None:
                if ctx.throughput_est > actual_t:
                    stats.overestimate_count += 1
                    stats.overestimate_avg += (ctx.throughput_est - actual_t - stats.overestimate_avg) / stats.overestimate_count
                else:
                    stats.goodestimate_count += 1
                    stats.goodestimate_avg += (actual_t - ctx.throughput_est - stats.goodestimate_avg) / stats.goodestimate_count

                stats.estimate_avg += ((ctx.throughput_est - actual_t - stats.estimate_avg) /
                                    (stats.overestimate_count + stats.goodestimate_count))

        # Only update throughput history if NOT abandoned (original behavior)
        if dp.abandon_to_quality is None and actual_t is not None and actual_l is not None:
            ctx.throughput_est, ctx.latency_est = avg.push(dl_ms, actual_t, actual_l)
    
    
    
    playout_buffer(ctx, tracker)
    if chunk_logger: chunk_logger.close()

    to_time_avg = 1.0 / (stats.total_play_time_ms / manifest.segment_time_ms)

    print(f"buffer size: {args.max_buffer * 1000.0:.0f}")
    print(f"total played utility: {stats.played_utility:f}")
    print(f"time average played utility: {stats.played_utility * to_time_avg:f}")

    print(f"total played bitrate: {stats.played_bitrate:f}")
    print(f"time average played bitrate: {stats.played_bitrate * to_time_avg:f}")

    print(f"total play time: {stats.total_play_time_ms / 1000:f}")
    print(f"total play time chunks: {stats.total_play_time_ms / manifest.segment_time_ms:f}")

    print(f"total rebuffer: {stats.rebuffer_time_ms / 1000:f}")
    print(f"rebuffer ratio: {stats.rebuffer_time_ms / stats.total_play_time_ms:f}")
    print(f"time average rebuffer: {(stats.rebuffer_time_ms / 1000) * to_time_avg:f}")

    print(f"total rebuffer events: {stats.rebuffer_event_count:f}")
    print(f"time average rebuffer events: {stats.rebuffer_event_count * to_time_avg:f}")

    print(f"total bitrate change: {stats.total_bitrate_change:f}")
    print(f"time average bitrate change: {stats.total_bitrate_change * to_time_avg:f}")

    print(f"total log bitrate change: {stats.total_log_bitrate_change:f}")
    print(f"time average log bitrate change: {stats.total_log_bitrate_change * to_time_avg:f}")

    print(f"time average score: {to_time_avg * (stats.played_utility - args.gamma_p * (stats.rebuffer_time_ms / manifest.segment_time_ms)):f}")

    if stats.overestimate_count == 0:
        print("over estimate count: 0")
        print("over estimate: 0")
    else:
        print(f"over estimate count: {stats.overestimate_count}")
        print(f"over estimate: {stats.overestimate_avg:f}")

    if stats.goodestimate_count == 0:
        print("leq estimate count: 0")
        print("leq estimate: 0")
    else:
        print(f"leq estimate count: {stats.goodestimate_count}")
        print(f"leq estimate: {stats.goodestimate_avg:f}")

    print(f"estimate: {stats.estimate_avg:f}")

    if stats.rampup_time_ms is None:
        print(f"rampup time: {(len(manifest.segments_bits) * manifest.segment_time_ms) / 1000:f}")
    else:
        print(f"rampup time: {stats.rampup_time_ms / 1000:f}")

    print(f"total reaction time: {tracker.total_reaction_ms / 1000:f}")


# ----------------------------- CLI -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Simulate an ABR session (refactored).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--plugin", action="append", default=[], help="Load a plugin .py file that registers ABR/AVG.")

    p.add_argument("-n", "--network", default="..//..//DATASET//NETWORK//4Glogs_lum//4g_trace_driving_50015_dr.json")
    p.add_argument("-nm", "--network-multiplier", type=float, default=1.0)

    p.add_argument("-m", "--movie", default="..//..//DATASET//MOVIE//movie_4g.json", help="Path to movie.json")
    p.add_argument("-ml", "--movie-length", type=float, default=None, help="Movie length in seconds (None=full).")

    p.add_argument("-a", "--abr", default="bola", help="ABR algorithm name (registry).")
    p.add_argument("-ab", "--abr-basic", action="store_true")
    p.add_argument("-ao", "--abr-osc", action="store_true")
    p.add_argument("-gp", "--gamma-p", type=float, default=5.0)
    p.add_argument("-noibr", "--no-insufficient-buffer-rule", action="store_true")

    p.add_argument("-ma", "--moving-average", default="ewma", help="Throughput estimator name (registry).")
    p.add_argument("-ws", "--window-size", nargs="+", type=int, default=[3])
    p.add_argument("-hl", "--half-life", nargs="+", type=float, default=[3, 8])

    p.add_argument("-s", "--seek", nargs=2, type=float, default=None, metavar=("WHEN", "SEEK"))

    p.add_argument("-r", "--replace", choices=["none", "left", "right"], default="none")
    p.add_argument("-b", "--max-buffer", type=float, default=25.0, help="seconds")
    p.add_argument("-noa", "--no-abandon", action="store_true")

    p.add_argument("-rmp", "--rampup-threshold", type=int, default=None)
    p.add_argument("-v", "--verbose", default=True, action="store_true")

    p.add_argument("--list", action="store_true", help="List available ABRs and averages.")
    p.add_argument("--chunk-log", default='.txt', help="Write per-chunk trace ")
    p.add_argument("--chunk-folder", default='', help="folder for logs")
    p.add_argument("--chunk-log-start-ts", type=float, default='1608418125', help="If set, timestamp column = start_ts + simulated_time_s; else uses simulated_time_s.")

    p.add_argument("--shim", type=int, default=8333, help="List available ABRs and averages.")
    p.add_argument("--timeout_s", type=float, default=1.0, help="List available ABRs and averages.")
    p.add_argument("--debug_p", action="store_true",  help="List available ABRs and averages.")
    p.add_argument("--ping_on_start", action="store_true", help="List available ABRs and averages.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    for pl in args.plugin:
        load_plugin(pl)

    if args.list:
        print("ABR:", ", ".join(sorted(ABR_REGISTRY.keys())))
        print("AVG:", ", ".join(sorted(AVG_REGISTRY.keys())))
        return

    if args.abr not in ABR_REGISTRY:
        raise SystemExit(f"Unknown ABR '{args.abr}'. Use --list.")
    if args.moving_average not in AVG_REGISTRY:
        raise SystemExit(f"Unknown average '{args.moving_average}'. Use --list.")

    run_session(args)


if __name__ == "__main__":
    main()
