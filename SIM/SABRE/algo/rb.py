# CustomAbr.py
# Simple throughput-based ABR plugin for refactored sab.py

# When sab.py is executed as a script, plugin symbols live in __main__.
# Fallback import supports running this in other contexts/tests.
try:
    from __main__ import register_abr, AbrBase, AbrDecision, AbrContext
except ImportError:
    from sab import register_abr, AbrBase, AbrDecision, AbrContext  # optional fallback


@register_abr("rb")
class RateBasedAbr(AbrBase):
    """
    Matches old behavior:
    - Pick the highest quality whose bitrate <= estimated throughput
    - No extra delay
    """

    def first_quality(self, ctx: AbrContext) -> int:
        return 0

    def select(self, ctx: AbrContext, seg_idx: int) -> AbrDecision:
        tput = ctx.throughput_est
        if tput is None or tput <= 0:
            return AbrDecision(0, 0.0)

        bitrates = ctx.manifest.bitrates_kbps
        q = 0
        while q + 1 < len(bitrates) and bitrates[q + 1] <= tput:
            q += 1

        return AbrDecision(q, 0.0)

    # Optional hooks (keep defaults if not needed):
    # def on_delay(self, ctx: AbrContext, delay_ms: float) -> None:
    #     pass
    #
    # def on_download(self, ctx: AbrContext, dp, is_replacement: bool) -> None:
    #     pass
    #
    # def check_abandon(self, ctx: AbrContext, progress, buffer_level_ms: float):
    #     return None
