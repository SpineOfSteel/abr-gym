RB (Rate-Based)
===============

Plugin name: ``rb``

RB is a throughput-only baseline:

- If throughput estimate is missing or non-positive → choose quality 0
- Else choose the highest quality whose bitrate is <= throughput estimate
- Adds no artificial delay

Intuition
---------

RB is maximally reactive. It can achieve high utilization but is sensitive to:

- noisy/optimistic throughput estimates
- latency spikes not captured in throughput
- short segments (higher variance in measured throughput)

Behavior (high level)
---------------------

1. Read ``ctx.throughput_est``
2. Scan bitrate ladder (ascending)
3. Pick the largest quality index ``q`` where ``bitrates[q] <= tput``

Pros / Cons
-----------

Pros:
- extremely simple
- good “ceiling” baseline to test estimator quality

Cons:
- can oscillate heavily
- stall risk if estimator overshoots
- ignores buffer entirely


BB (Buffer-Based / BBA-like)
============================

Plugin name: ``bb``

BB maps buffer occupancy to a target quality:

- If buffer <= reservoir → choose lowest quality
- If buffer >= reservoir + cushion → choose highest quality
- Else linearly interpolate buffer fraction → quality index

This is a classic “BBA-like” approach.

Key parameters
--------------

The plugin uses:

- ``reservoir_ms``: lower buffer threshold (defaults hardcoded to 5000ms)
- ``cushion_ms``: range above reservoir that ramps quality (defaults hardcoded to 6500ms)

These two values define the ramp window:

- below reservoir: conservative
- above reservoir+cushion: aggressive (max quality)

Optional throughput safety cap
------------------------------

BB can optionally cap the chosen quality by throughput:

- ``SAFETY_CAP = 0.85`` (set to ``None`` to disable)
- cap quality so that selected bitrate does not exceed ``throughput_est * SAFETY_CAP``

This guardrail prevents “buffer is high → pick top quality” overshoots when
throughput cannot actually sustain the chosen bitrate.

When to use BB
--------------

BB is a good baseline when you want:

- stability
- stall avoidance
- controlled ramp-up behavior

Tradeoffs:

- can be slow to climb quality if reservoir/cushion are conservative
- depends on reasonable buffer sizing for your content and segment duration