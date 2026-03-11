BOLA
====

Plugin name: ``bola``

BOLA stands for **Buffer Occupancy based Lyapunov Algorithm**. It is a buffer-aware
ABR method that selects the next bitrate by maximizing a utility-per-bit score
that balances:

- video quality utility
- current buffer occupancy
- bitrate / chunk cost

Compared with simpler heuristics such as RB or BB, BOLA is both practically strong
and theoretically motivated.

Quickstart
----------

Run BOLA:

.. code-block:: bash

   !python sab.py --plugin algo/bola.py -a bola --chunk-log log_BOLA_driving_4g.txt -v

Run throughput mode from the hybrid BOLA-D file:

.. code-block:: bash

   !python sab.py --plugin algo/bola-d.py -a throughput --chunk-log log_BOLA-TPT_driving_4g.txt -v

Run Dynamic / BOLA-D:

.. code-block:: bash

   !python sab.py --plugin algo/bola-d.py -a dynamic --chunk-log log_BOLA-D_driving_4g.txt -v

Core intuition
--------------

BOLA treats bitrate choice as a utility-optimization problem under finite buffer.

The central idea is:

- high quality should be rewarded
- large buffer gives freedom to be more ambitious
- low buffer should push decisions toward safer choices
- the score should normalize by bitrate cost

Utility model
-------------

BOLA uses a concave utility function. In your implementation, a typical form is
log-bitrate utility:

.. math::

   u(q) = \log(R_q) + u_0

where:

- :math:`R_q` is the bitrate for quality level :math:`q`
- :math:`u_0` is a shift term used to normalize the utility baseline

A concave utility is important because the perceptual gain from increasing bitrate
usually has diminishing returns.

Gain parameter and control term
-------------------------------

Your implementation uses a gain parameter ``gp`` and a derived control quantity ``Vp``.

A simplified form is:

.. math::

   V_p = \frac{B_{\max} - p}{u_{\max} + \gamma_p}

where:

- :math:`B_{\max}` is the maximum buffer size
- :math:`p` is the segment duration
- :math:`u_{\max}` is the maximum utility value over all qualities
- :math:`\gamma_p` is the gain parameter exposed as ``--gamma-p``

This control term scales how aggressively the algorithm values utility versus
buffer safety.

BOLA score
----------

For each quality level :math:`q`, BOLA computes a score of the form:

.. math::

   \mathrm{score}(q) =
   \frac{V_p \cdot (u(q) + \gamma_p) - B}{R_q}

where:

- :math:`B` is the current buffer occupancy
- :math:`R_q` is the bitrate of representation :math:`q`

The selected quality is:

.. math::

   q^\* = \arg\max_q \mathrm{score}(q)

This creates an intuitive tradeoff:

- larger utility increases the score
- larger buffer decreases the urgency to download aggressively
- larger bitrate in the denominator penalizes expensive choices

Practical interpretation
------------------------

When the buffer is low, the numerator becomes smaller, which tends to reduce the
score of high-bitrate options. When the buffer is healthy, higher-utility qualities
become more competitive.

That is why BOLA is often more stable than purely throughput-based methods.

Delay / oscillation control
---------------------------

Your implementation includes practical oscillation control by inserting delay when
the buffer is too full for the chosen quality.

Conceptually, if the current buffer exceeds a quality-dependent threshold
:math:`B_{\max}(q)`, then delay is added:

.. math::

   d = \max(0, B - B_{\max}(q))

This helps avoid needless top-end oscillation when the system is already ahead.

Abandonment
-----------

BOLA can optionally abandon a partially downloaded chunk and restart at a lower
quality if the remaining download appears suboptimal.

The idea is to compare the score of continuing versus switching, taking into account
the remaining bits and the current network situation.

This is especially useful when the bandwidth estimate changes suddenly during the
download of a high-quality chunk.

Connection to the original BOLA formulation
-------------------------------------------

The original BOLA analysis frames the decision as a Lyapunov optimization problem
for fixed-duration chunks. At each decision point, BOLA chooses the bitrate index
that maximizes a ratio similar to:

.. math::

   \frac{V \, v_m + V \gamma p - Q(t_k)}{S_m}

where:

- :math:`v_m` is utility for level :math:`m`
- :math:`S_m` is chunk size
- :math:`Q(t_k)` is the buffer level at decision time
- :math:`V` is the Lyapunov control parameter
- :math:`\gamma` is the rebuffer-avoidance term
- :math:`p` is chunk duration

Your implementation mirrors this structure using bitrate- and utility-based scoring.

Strengths
---------

- strong theoretical motivation
- naturally buffer-aware
- does not require explicit bandwidth prediction in its core form
- usually smoother than pure throughput methods

Limitations
-----------

- requires reasonable buffer and segment-duration assumptions
- parameters such as ``gp`` influence aggressiveness
- more complex than RB or BB

How it relates to Dynamic
-------------------------

The ``dynamic`` policy builds on BOLA by combining it with a throughput rule.
A common pattern is:

- use throughput logic when the buffer is low
- return to BOLA when buffer is healthy

This lets the system ramp up more quickly while preserving BOLA’s steady-state stability.
