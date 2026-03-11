
from typing import Dict, Optional
from DataModel import AbrContext, DownloadProgress
from dataclasses import dataclass, field

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

