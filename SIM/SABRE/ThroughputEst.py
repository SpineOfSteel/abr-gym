import math
from typing import List, Tuple
from sab import register_avg
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

