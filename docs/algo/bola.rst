BOLA
====

Plugin name: ``bola``

BOLA (Buffer Occupancy based Lyapunov Algorithm) chooses quality by maximizing a
utility-per-bit score that balances:

- utility of quality (e.g., log bitrate)
- buffer occupancy (avoid stalls)
- bitrate cost

Key concepts
------------

Utility function
~~~~~~~~~~~~~~~~

BOLA uses a concave utility per quality:

- ``u(q) = log(bitrate_q) + u0``

Where ``u0`` shifts utilities so the lowest bitrate has a convenient baseline.
The plugin supports modes like ``utility_mode="zero"`` (default).

Gain parameter (gp) and Vp
~~~~~~~~~~~~~~~~~~~~~~~~~~

BOLA uses:

- ``gp``: tuning parameter controlling aggressiveness
- ``Vp``: derived control knob based on buffer size, segment time, utility range

A simplified form used by the plugin:

- ``Vp = (buffer_size_ms - segment_time_ms) / (u_max + gp)``

Decision rule (score)
~~~~~~~~~~~~~~~~~~~~~

For each quality ``q`` compute a score:

.. math::

   score(q) = \frac{Vp \cdot (u(q) + gp) - buffer\_level}{bitrate(q)}

Pick the quality with the maximum score.

Delay (oscillation control)
---------------------------

The plugin can optionally add a delay (sleep) when buffer is “too full” for the
selected quality:

- computes a max buffer level threshold for that quality
- adds delay if current buffer exceeds that threshold
- helps prevent oscillations at the top end

Abandonment (optional)
----------------------

BOLA can optionally abandon a partially downloaded segment and restart at a lower
quality if it becomes clearly suboptimal (based on remaining bits and score
comparisons).

Practical guidance
------------------

- BOLA is more stable than purely throughput-based rules.
- It’s sensitive to correct assumptions about buffer size and segment duration.
- Use it as a strong baseline against RL methods.