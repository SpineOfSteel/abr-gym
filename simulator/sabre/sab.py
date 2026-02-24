from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type


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


# ----------------------------- models -----------------------------

@dataclass(frozen=True)
class Manifest:
    segment_time_ms: float
    bitrates_kbps: List[float]
    utilities: List[float]
    segments_bits: List[List[int]]  # segments_bits[seg_idx][q] -> bits

    @staticmethod
    def from_json(path: str, movie_length_s: Optional[float] = None) -> "Manifest":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))

        seg_time = float(obj["segment_duration_ms"])
        if seg_time <= 0:
            raise ValueError(f"Invalid segment_duration_ms={seg_time} in {path}")

        bitrates = list(map(float, obj["bitrates_kbps"]))
        if not bitrates or any(b <= 0 for b in bitrates):
            raise ValueError(f"Invalid bitrates_kbps in {path}: {bitrates}")

        u0 = -math.log(bitrates[0])
        utilities = [math.log(b) + u0 for b in bitrates]

        segs = obj["segment_sizes_bits"]  # list[list[int]]
        if not segs:
            raise ValueError(f"No segment_sizes_bits in {path}")

        if movie_length_s is not None:
            l1 = len(segs)
            l2 = math.ceil(movie_length_s * 1000 / seg_time)
            rep = math.ceil(l2 / l1)
            segs = (segs * rep)[:l2]

        return Manifest(segment_time_ms=seg_time, bitrates_kbps=bitrates, utilities=utilities, segments_bits=segs)


@dataclass(frozen=True)
class NetworkPeriod:
    duration_ms: float
    bandwidth_kbps: float
    latency_ms: float

    @staticmethod
    def from_json(path: str, multiplier: float = 1.0) -> List["NetworkPeriod"]:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        out: List[NetworkPeriod] = []
        for p in obj:
            dur = float(p["duration_ms"])
            if dur <= 0:
                # Drop invalid periods so the simulator always advances time.
                continue

            bw = float(p["bandwidth_kbps"]) * multiplier
            lat = float(p["latency_ms"])

            if bw < 0 or lat < 0:
                raise ValueError(f"Invalid network period (negative bw/lat): {p}")

            out.append(NetworkPeriod(duration_ms=dur, bandwidth_kbps=bw, latency_ms=lat))

        if not out:
            raise ValueError(
                f"Network trace '{path}' became empty after filtering duration_ms<=0. "
                f"Fix the trace (duration_ms must be > 0)."
            )
        return out


@dataclass(frozen=True)
class DownloadProgress:
    index: int
    quality: int
    size_bits: float
    downloaded_bits: float
    time_ms: float
    time_to_first_bit_ms: float
    abandon_to_quality: Optional[int] = None


@dataclass
class Buffer:
    """Stores qualities; fcc_ms is 'first chunk consumed' inside first segment."""
    manifest: Manifest
    contents: List[int] = field(default_factory=list)
    fcc_ms: float = 0.0

    def level_ms(self) -> float:
        return self.manifest.segment_time_ms * len(self.contents) - self.fcc_ms

    def clear(self) -> None:
        self.contents.clear()
        self.fcc_ms = 0.0


@dataclass
class ReactionTracker:
    """Tracks 'reaction time' to upward sustainable network quality."""
    max_buffer_ms: float
    pending: List[List[float]] = field(default_factory=list)  # [time_ms, target_q, (optional) reached_time_ms]
    total_reaction_ms: float = 0.0

    def _process_matured(self, now_ms: float) -> None:
        cutoff = now_ms - self.max_buffer_ms
        while self.pending and self.pending[0][0] < cutoff:
            p = self.pending.pop(0)
            reaction = self.max_buffer_ms if len(p) == 2 else min(self.max_buffer_ms, p[2] - p[0])
            self.total_reaction_ms += reaction

    def on_network_quality_change(self, now_ms: float, new_q: int, prev_q: Optional[int], buffer_contents: List[int]) -> None:
        self._process_matured(now_ms)
        if prev_q is None or new_q <= prev_q:
            return

        # mark any pending switch-up done if quality drops below its target
        for p in self.pending:
            if len(p) == 2 and int(p[1]) > new_q:
                p.append(now_ms)

        # ignore if buffer already has >= new_q, or if already pending
        if any(new_q <= q for q in buffer_contents):
            return
        if any(len(p) >= 2 and new_q <= int(p[1]) for p in self.pending):
            return

        self.pending.append([now_ms, float(new_q)])

    def on_played_quality(self, now_ms: float, q: int) -> None:
        for p in self.pending:
            if len(p) == 2 and q >= int(p[1]):
                p.append(now_ms)

    def on_time_advanced(self, now_ms: float) -> None:
        self._process_matured(now_ms)


@dataclass
class SimStats:
    rebuffer_event_count: int = 0
    rebuffer_time_ms: float = 0.0
    played_utility: float = 0.0
    played_bitrate: float = 0.0
    total_play_time_ms: float = 0.0
    total_bitrate_change: float = 0.0
    total_log_bitrate_change: float = 0.0
    last_played: Optional[int] = None

    overestimate_count: int = 0
    overestimate_avg: float = 0.0
    goodestimate_count: int = 0
    goodestimate_avg: float = 0.0
    estimate_avg: float = 0.0

    rampup_origin_ms: float = 0.0
    rampup_time_ms: Optional[float] = None


@dataclass
class AbrContext:
    manifest: Manifest
    buffer: Buffer
    max_buffer_ms: float
    throughput_est: Optional[float]  # bits/ms
    latency_est: Optional[float]     # ms
    sustainable_quality: Optional[int]
    rampup_threshold: Optional[int]
    stats: SimStats


# ----------------------------- network model -----------------------------

class NetworkModel:
    """
    Time is advanced only by NetworkModel (delay/download); simulator then depletes playback buffer for same duration.
    """
    min_progress_size_bits = 12000
    min_progress_time_ms = 50

    def __init__(self, trace: List[NetworkPeriod], manifest: Manifest, verbose: bool):
        self.trace = trace
        self.manifest = manifest
        self.verbose = verbose

        self.idx = -1
        self.time_to_next_ms = 0.0
        self.time_ms = 0.0
        self.sustainable_quality: Optional[int] = None

        self._next_period()

    def _ensure_period_ready(self) -> None:
        """
        Critical fix: time_to_next_ms can become exactly 0 after consuming a period boundary.
        If we don't advance to the next period, loops can stall forever (b==0,t==0 cases).
        """
        guard = 0
        while self.time_to_next_ms <= 0.0:
            self._next_period()
            guard += 1
            if guard > len(self.trace) + 1:
                raise RuntimeError("NetworkModel stuck: non-positive period durations or invalid trace.")

    def _next_period(self) -> None:
        self.idx = (self.idx + 1) % len(self.trace)
        self.time_to_next_ms = float(self.trace[self.idx].duration_ms)

        # clamp latency factor to [0,1] to avoid negative effective bandwidth
        lat = self.trace[self.idx].latency_ms
        lat_factor = 1.0 - (lat / self.manifest.segment_time_ms)
        if lat_factor < 0.0:
            lat_factor = 0.0
        eff_bw = self.trace[self.idx].bandwidth_kbps * lat_factor

        prev = self.sustainable_quality
        q = 0
        for i in range(1, len(self.manifest.bitrates_kbps)):
            if self.manifest.bitrates_kbps[i] > eff_bw:
                break
            q = i
        self.sustainable_quality = q

        if self.verbose:
            dur = self.trace[self.idx].duration_ms
            print(
                f"[{round(self.time_ms)}] Network: bw={self.trace[self.idx].bandwidth_kbps:.0f},"
                f"lat={lat:.0f},dur={dur:.0f}  (q={q}: bitrate={self.manifest.bitrates_kbps[q]:.0f})"
            )

    def _advance_periods(self, dt_ms: float) -> None:
        """Advance network time by dt_ms, crossing one or more network periods."""
        while dt_ms > 0:
            # If we're exactly on a boundary, move forward so time always progresses.
            if self.time_to_next_ms <= 0:
                self._next_period()
                continue

            if dt_ms < self.time_to_next_ms:
                self.time_to_next_ms -= dt_ms
                self.time_ms += dt_ms
                return

            # Consume remainder of the current period, then advance.
            dt_ms -= self.time_to_next_ms
            self.time_ms += self.time_to_next_ms
            self._next_period()


    def delay(self, dt_ms: float) -> None:
        self._advance_periods(dt_ms)

    def _latency_delay(self, units: float) -> float:
        total = 0.0
        while units > 0:
            self._ensure_period_ready()
            lat = self.trace[self.idx].latency_ms
            t = units * lat
            if t <= self.time_to_next_ms:
                total += t
                self._advance_periods(t)
                units = 0
            else:
                total += self.time_to_next_ms
                if lat > 0:
                    units -= self.time_to_next_ms / lat
                self._advance_periods(self.time_to_next_ms)
        return total

    def _download_bits(self, bits: float) -> float:
        total = 0.0
        while bits > 0:
            self._ensure_period_ready()
            bw = self.trace[self.idx].bandwidth_kbps  # (kbit/s) == (bit/ms) numerically
            if bw <= 0:
                # No capacity in this period: advance to next period in time.
                total += self.time_to_next_ms
                self._advance_periods(self.time_to_next_ms)
                continue

            can = self.time_to_next_ms * bw
            if bits <= can:
                t = bits / bw
                total += t
                self._advance_periods(t)
                bits = 0
            else:
                total += self.time_to_next_ms
                bits -= can
                self._advance_periods(self.time_to_next_ms)
        return total

    def _minimal_latency_delay(self, units: float, min_time_ms: float) -> Tuple[float, float]:
        got_units = 0.0
        got_time = 0.0
        while units > 0 and min_time_ms > 0:
            self._ensure_period_ready()
            lat = self.trace[self.idx].latency_ms
            t = units * lat
            if t <= min_time_ms and t <= self.time_to_next_ms:
                u = units
                self._advance_periods(t)
            elif min_time_ms <= self.time_to_next_ms:
                t = min_time_ms
                u = (t / lat) if lat > 0 else units  # if lat==0, consume all units instantly
                self._advance_periods(t)
            else:
                t = self.time_to_next_ms
                u = (t / lat) if lat > 0 else units  # if lat==0, consume all units instantly
                self._advance_periods(t)

            got_units += u
            got_time += t
            units -= u
            min_time_ms -= t

        return got_units, got_time

    def _minimal_download(self, bits_left: float, min_bits: float, min_time_ms: float) -> Tuple[float, float]:
        got_bits = 0.0
        got_time = 0.0

        while bits_left > 0 and (min_bits > 0 or min_time_ms > 0):
            self._ensure_period_ready()
            bw = self.trace[self.idx].bandwidth_kbps

            if bw > 0:
                need = max(min_bits, min_time_ms * bw)
                to_next = self.time_to_next_ms * bw

                # If to_next is 0 (boundary), ensure_period_ready() should have prevented it,
                # but keep a hard guard anyway.
                if to_next <= 0:
                    self._ensure_period_ready()
                    continue

                if bits_left <= need and bits_left <= to_next:
                    b = bits_left
                    t = b / bw
                    self._advance_periods(t)
                elif need <= to_next:
                    b = need
                    t = b / bw
                    min_bits = 0
                    min_time_ms = 0
                    self._advance_periods(t)
                else:
                    b = to_next
                    t = self.time_to_next_ms
                    self._advance_periods(t)
            else:
                # bw == 0: consume time to next period (or min_time if smaller), no bits delivered.
                b = 0.0
                t = min(self.time_to_next_ms, max(0.0, min_time_ms)) if min_time_ms > 0 else self.time_to_next_ms
                self._advance_periods(t)

            # progress accounting
            got_bits += b
            got_time += t
            bits_left -= b
            min_bits -= b
            min_time_ms -= t

            # Absolute deadlock guard: if neither bits nor time advanced, force a period step
            if b == 0.0 and t == 0.0:
                self.time_to_next_ms = 0.0
                self._ensure_period_ready()

        return got_bits, got_time

    def download(
        self,
        bits: float,
        seg_idx: int,
        quality: int,
        buffer_level_ms: float,
        check_abandon: Optional[Callable[[DownloadProgress, float], Optional[int]]] = None,
    ) -> DownloadProgress:
        if bits <= 0:
            return DownloadProgress(seg_idx, quality, 0, 0, 0, 0, None)

        if not check_abandon or (self.min_progress_time_ms <= 0 and self.min_progress_size_bits <= 0):
            ttfb = self._latency_delay(1)
            t = ttfb + self._download_bits(bits)
            return DownloadProgress(seg_idx, quality, bits, bits, t, ttfb, None)

        total_t = 0.0
        total_b = 0.0
        min_t = self.min_progress_time_ms
        min_b = self.min_progress_size_bits

        if self.min_progress_size_bits > 0:
            ttfb = self._latency_delay(1)
            total_t += ttfb
            min_t -= total_t
            delay_units = 0.0
        else:
            ttfb = None
            delay_units = 1.0

        abandon_to: Optional[int] = None

        # stall guard
        guard_iters = 0

        while total_b < bits and abandon_to is None:
            guard_iters += 1
            if guard_iters > 2_000_000:
                raise RuntimeError("Network download appears stuck (too many iterations). Check trace durations/bandwidth.")

            prev_b, prev_t = total_b, total_t

            if delay_units > 0:
                units, t = self._minimal_latency_delay(delay_units, min_t)
                total_t += t
                delay_units -= units
                min_t -= t
                if delay_units <= 0:
                    ttfb = total_t

            if delay_units <= 0:
                b, t = self._minimal_download(bits - total_b, min_b, min_t)
                total_t += t
                total_b += b

            if total_b == prev_b and total_t == prev_t:
                # hard escape: force advance
                self.time_to_next_ms = 0.0
                self._ensure_period_ready()

            dp = DownloadProgress(seg_idx, quality, bits, total_b, total_t, (ttfb or 0.0), None)
            if total_b < bits:
                abandon_to = check_abandon(dp, max(0.0, buffer_level_ms - total_t))
                min_t = self.min_progress_time_ms
                min_b = self.min_progress_size_bits

        return DownloadProgress(seg_idx, quality, bits, total_b, total_t, (ttfb or 0.0), abandon_to)


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


# ----------------------------- ABR interface -----------------------------

@dataclass(frozen=True)
class AbrDecision:
    quality: int
    delay_ms: float = 0.0

class AbrBase:
    NAME = "base"
    def __init__(self, cfg: Dict, ctx: AbrContext):
        self.cfg = cfg

    def first_quality(self, ctx: AbrContext) -> int:
        return 0

    def select(self, ctx: AbrContext, seg_idx: int) -> AbrDecision:
        raise NotImplementedError

    def on_delay(self, ctx: AbrContext, delay_ms: float) -> None:
        pass

    def on_download(self, ctx: AbrContext, dp: DownloadProgress, is_replacement: bool) -> None:
        pass

    def on_seek(self, ctx: AbrContext, where_ms: float) -> None:
        pass

    def check_abandon(self, ctx: AbrContext, progress: DownloadProgress, buffer_level_ms: float) -> Optional[int]:
        return None

    def quality_from_throughput(self, ctx: AbrContext, tput_bits_per_ms: float) -> int:
        if tput_bits_per_ms <= 0:
            return 0
        p = ctx.manifest.segment_time_ms
        lat = ctx.latency_est or 0.0
        q = 0
        while (q + 1 < len(ctx.manifest.bitrates_kbps) and
               lat + p * ctx.manifest.bitrates_kbps[q + 1] / tput_bits_per_ms <= p):
            q += 1
        return q
    
    def quality_from_throughput2(self, ctx: AbrContext, seg_idx: int, tput_bits_per_ms: float) -> int:
        if tput_bits_per_ms <= 0:
            return 0

        lat_ms = float(ctx.latency_est or 0.0)          # ms
        buf_ms = float(ctx.buffer.level_ms())           # ms

        GUARD_MS = 200.0                                 # small safety margin
        budget_ms = max(0.0, 0.85 * buf_ms - GUARD_MS)

        q = 0
        while q + 1 < len(ctx.manifest.bitrates_kbps):
            seg_bits = float(ctx.manifest.segments_bits[seg_idx][q + 1])
            dl_ms = seg_bits / tput_bits_per_ms  # end-to-end estimate
            if dl_ms > budget_ms:
                break
            q += 1
        return q



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
    trace = NetworkPeriod.from_json(args.network, args.network_multiplier)
    chunk_logger = ChunkLogger(f'output\\log_{args.abr}_{args.chunk_log}', args.chunk_log_start_ts) if args.chunk_log else None

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

    p.add_argument("-n", "--network", default="network.json")
    p.add_argument("-nm", "--network-multiplier", type=float, default=1.0)

    p.add_argument("-m", "--movie", default="movie.json")
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


M_IN_K = 1000.0
REBUF_PENALTY = 32.0  #160.0
SMOOTH_PENALTY = 1.0

def compute_qoe_reward(bitrate_kbps: float, stall_s: float, last_bitrate_kbps: float | None) -> float:
    bitrate_mbps = bitrate_kbps / M_IN_K
    last_mbps = (last_bitrate_kbps / M_IN_K) if last_bitrate_kbps is not None else 0.0
    return bitrate_mbps - REBUF_PENALTY * stall_s - SMOOTH_PENALTY * abs(bitrate_mbps - last_mbps)


class ChunkLogger:
    def __init__(self, path: str, start_ts: Optional[float]):
        self.f = open(path, "w", encoding="utf-8")
        self.start_ts = start_ts
        self.last_bitrate_kbps: Optional[float] = None

    def close(self) -> None:
        self.f.close()

    def log(self, sim_time_ms: float, bitrate_kbps: float, buffer_ms: float, stall_ms: float,
            chunk_bits: float, download_ms: float) -> None:
        ts_s = (self.start_ts + sim_time_ms / 1000.0) if self.start_ts is not None else (sim_time_ms / 1000.0)
        stall_s = stall_ms / 1000.0
        reward = compute_qoe_reward(bitrate_kbps, stall_s, self.last_bitrate_kbps)
        self.last_bitrate_kbps = bitrate_kbps

        chunk_bytes = chunk_bits / 8.0
        buf_s = buffer_ms / 1000.0

        self.f.write(
            f"{ts_s:.2f}\t{bitrate_kbps:.0f}\t{buf_s:.6f}\t{stall_s:.3f}\t"
            f"{chunk_bytes:.0f}\t{download_ms:.0f}\t{reward:.12g}\n"
        )

if __name__ == "__main__":
    main()
