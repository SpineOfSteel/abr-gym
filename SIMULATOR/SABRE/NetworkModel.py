
from DataModel import DownloadProgress, NetworkPeriod, Manifest
from typing import List, Optional, Tuple, Callable

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

