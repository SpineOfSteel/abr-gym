Dynamic (BOLA-D Hybrid)
=======================

Plugin name: ``dynamic``

Dynamic is a hybrid ABR policy that combines:

- **BOLA** for stable, buffer-aware decisions
- a **Throughput rule** for responsiveness when buffer is low

It switches between the two based on buffer level and which policy suggests a
better quality.

Components
----------

Throughput rule
~~~~~~~~~~~~~~~

The throughput component:

- chooses quality by throughput estimate multiplied by a safety factor
- optionally uses latency-aware logic (IBR-like) when latency is available
- can abandon downloads if remaining time is likely to exceed a threshold

BOLA component
~~~~~~~~~~~~~~

The BOLA component is the same as the ``bola`` plugin.

Switching rule
~~~~~~~~~~~~~~

Dynamic maintains an internal mode:

- When in BOLA mode:
  - if buffer drops below a low threshold and throughput-rule suggests a higher
    quality than BOLA, switch to throughput-rule mode

- When in throughput-rule mode:
  - if buffer is above threshold and BOLA suggests >= throughput quality, switch
    back to BOLA mode

Intuition
---------

- throughput mode helps you escape “low buffer traps” and ramps up early
- BOLA mode stabilizes steady-state behavior and reduces oscillation
- the hybrid often performs well in practice but introduces more parameters