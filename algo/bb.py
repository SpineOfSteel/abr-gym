# bb.py  (buffer-based ABR plugin)
# python sab.py --plugin bb.py -a bb -n 4Glogs_lum\4g_trace_driving_50015_dr.json -m movie.json --chunk-log-start-ts 1608418123 --chunk-log log_BB_driving_4g -v
# BBA-style buffer-based rule:
# - If buffer <= reservoir: pick lowest quality
# - If buffer >= reservoir + cushion: pick highest quality
# - Else: linearly map buffer level to a quality index
#
# Optional: cap by throughput_est for safety (can disable by setting SAFETY_CAP = None)

try:
    from __main__ import register_abr, AbrBase, AbrDecision, AbrContext
except ImportError:
    from sab import register_abr, AbrBase, AbrDecision, AbrContext  # optional fallback


@register_abr("bb")
class BufferBasedAbr(AbrBase):
    """
    Buffer-based ABR (BBA-like).

    Defaults (derived from max buffer):
      reservoir = 20% of max_buffer
      cushion   = 60% of max_buffer

    Behavior:
      - Uses ctx.buffer.level_ms() only (plus optional safety cap).
      - No delay.
    """

    # Set to a float like 0.90 to cap by throughput_est * SAFETY_CAP
    # Set to None to disable throughput capping.
    SAFETY_CAP = 0.85

    def __init__(self, cfg: dict, ctx: AbrContext):
        super().__init__(cfg, ctx)

        # max buffer in ms is provided by sab.py in cfg as buffer_size_ms
        buf_ms = float( ctx.max_buffer_ms)
        self.reservoir_ms = 5000.0 #float( 0.20 * buf_ms)
        self.cushion_ms = 6500.0 #float( 0.60 * buf_ms)

    def first_quality(self, ctx: AbrContext) -> int:
        return 1

    def select(self, ctx: AbrContext, seg_idx: int) -> AbrDecision:
        

    
        level_ms = ctx.buffer.level_ms()
        nQ = len(ctx.manifest.bitrates_kbps)
        if nQ <= 1:
            return AbrDecision(0, 0.0)

        # BBA mapping
        if level_ms <= self.reservoir_ms:
            q = 0
        else:
            hi = self.reservoir_ms + self.cushion_ms
            if level_ms >= hi:
                q = nQ - 1
            else:
                frac = (level_ms - self.reservoir_ms) / max(1e-9, self.cushion_ms)
                q = int(frac * (nQ - 1))
                q = max(0, min(nQ - 1, q))

        # optional safety cap using throughput estimate (prevents crazy overshoots)
        if self.SAFETY_CAP is not None:
            tput = ctx.throughput_est
            if tput is not None and tput > 0:
                cap_q = self.quality_from_throughput(ctx,  tput * float(self.SAFETY_CAP))
                q = min(q, cap_q)

        if seg_idx < 10:
            print("buf_ms", ctx.buffer.level_ms(),
                "tput", ctx.throughput_est,
                "lat", ctx.latency_est,
                "q_bba", q,
                "cap_q", cap_q if self.SAFETY_CAP is not None else None)
        return AbrDecision(q, 0.0)
