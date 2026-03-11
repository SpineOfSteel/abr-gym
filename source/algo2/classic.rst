Classic Heuristic ABR: RB, BB, and Dynamic
==========================================

This page documents lightweight non-RL Adaptive Bitrate (ABR) policies implemented
as SABRE plugins. These algorithms are useful as interpretable baselines before
moving to BOLA, MPC, or RL-based methods such as DQN, PPO, and Pensieve.

Covered here:

- ``rb``: rate-based adaptation using throughput only
- ``bb``: buffer-based adaptation using buffer thresholds
- ``dynamic``: hybrid switching between BOLA and a throughput rule

For a deeper treatment of BOLA itself, including the utility score and equations,
see :doc:`bola`.

What SABRE provides
-------------------

SABRE is the simulator layer that drives all of these policies. It combines:

- a network trace
- a movie manifest
- a player buffer model
- an ABR plugin selected by name
- per-chunk logging for plotting and evaluation

The plugin interface is intentionally simple: the ABR logic selects the next
quality index using the current context and next segment index.

Quickstart
----------

Clone the repository:

.. code-block:: bash

   !git clone https://github.com/SpineOfSteel/abr-gym.git

Run BB on a specific 4G trace:

.. code-block:: bash

   !python sab.py --plugin algo/bb.py -a bb -n /content/abr-gym/DATASET/NETWORK/4Glogs_lum/4g_trace_driving_50015_dr.json -m /content/abr-gym/DATASET/MOVIE/movie_4g.json --chunk-log-start-ts 1608418123 --chunk-log log_BB2_driving_4g -v

Run BB using default paths:

.. code-block:: bash

   !python sab.py --plugin algo/bb.py --chunk-log log_BB_driving_4g.txt -a bb -v

Run RB:

.. code-block:: bash

   !python sab.py --plugin algo/rb.py -a rb --chunk-log log_RB_driving_4g.txt -v

Run Dynamic:

.. code-block:: bash

   !python sab.py --plugin algo/bola-d.py -a dynamic --chunk-log log_BOLA-D_driving_4g.txt -v

RB (Rate-Based)
===============

Plugin name: ``rb``

RB is the simplest throughput-driven baseline.

Decision idea
-------------

RB reads the estimated throughput and chooses the highest representation whose
bitrate is still sustainable according to that estimate.

High-level behavior:

1. Read ``ctx.throughput_est``
2. If it is missing or non-positive, choose quality ``0``
3. Otherwise scan the bitrate ladder in ascending order
4. Return the largest quality index ``q`` such that
   ``bitrates[q] <= throughput_est``

Mathematically, the selected quality is:

.. math::

   q^\* = \max \left\{ q \;:\; R_q \le \hat{T} \right\}

where:

- :math:`R_q` is the bitrate of quality level :math:`q`
- :math:`\hat{T}` is the current throughput estimate

Strengths
---------

- extremely simple
- easy to debug and reason about
- good baseline for testing throughput-estimation quality

Limitations
-----------

- ignores buffer occupancy
- can oscillate if throughput is noisy
- can trigger rebuffering if the estimator is optimistic

When RB is useful
-----------------

RB is useful when you want a clean baseline for the question:

“How far can recent-throughput-only adaptation go before buffer awareness becomes necessary?”

BB (Buffer-Based)
=================

Plugin name: ``bb``

BB is a BBA-like plugin that maps current buffer occupancy to a quality level.

Decision idea
-------------

BB divides the buffer into three zones:

- below ``reservoir_ms``: choose the lowest quality
- above ``reservoir_ms + cushion_ms``: choose the highest quality
- in between: linearly interpolate to an intermediate quality

Using buffer level :math:`B`, reservoir :math:`R`, and cushion :math:`C`:

.. math::

   q(B) =
   \begin{cases}
   q_{\min}, & B \le R \\
   q_{\max}, & B \ge R + C \\
   \text{interp}(B), & R < B < R + C
   \end{cases}

where ``interp(B)`` denotes a linear mapping from the ramp window to a quality index.

Key parameters
--------------

The implementation uses fixed thresholds such as:

- ``reservoir_ms = 5000``
- ``cushion_ms = 6500``

These define the ramp region where BB gradually increases quality as the buffer fills.

Optional throughput guardrail
-----------------------------

BB can apply a throughput-based safety cap:

- ``SAFETY_CAP = 0.85`` by default
- the chosen representation is capped so it does not exceed
  ``0.85 * throughput_est``

Formally, after the buffer-based selection:

.. math::

   q_{\text{final}} = \min \left( q_{\text{buffer}}, q_{\text{safety}} \right)

where :math:`q_{\text{safety}}` is the largest quality index satisfying

.. math::

   R_q \le \alpha \hat{T}

with :math:`\alpha = 0.85`.

Strengths
---------

- naturally buffer-aware
- usually more stable than pure RB
- good at stall avoidance when buffer is low
- easy to interpret and tune

Limitations
-----------

- ramp-up can be slow if thresholds are conservative
- performance depends on sensible threshold choices
- less theoretically grounded than BOLA

Dynamic (BOLA-D Hybrid)
=======================

Plugin name: ``dynamic``

Dynamic is a hybrid ABR policy that combines:

- **BOLA** for stable buffer-aware decisions
- a **Throughput rule** for responsiveness when buffer is low

It switches between the two based on buffer level and which rule appears safer
or more beneficial.

Throughput component
--------------------

The throughput sub-policy uses:

- throughput estimate multiplied by a safety factor
- optional latency-aware insufficient-buffer logic
- abandonment checks when download time looks too long

A simplified throughput-rule selection is:

.. math::

   q_{\text{tput}} = \max \left\{ q : R_q \le \beta \hat{T} \right\}

where :math:`\beta` is a safety multiplier such as ``0.9``.

BOLA component
--------------

The BOLA side is the same utility-driven buffer policy described in :doc:`bola`.

Switching rule
--------------

Dynamic maintains an internal mode.

A representative switching rule is:

- use throughput mode when buffer is low and throughput mode recommends a better
  actionable quality
- use BOLA mode when buffer is healthier and BOLA is at least as good

In your implementation this uses a low-buffer threshold such as:

- ``low_buffer_threshold_ms = 10000``

Conceptually:

.. math::

   \text{mode} =
   \begin{cases}
   \text{throughput}, & B < B_{\text{low}} \text{ and } q_{\text{tput}} > q_{\text{bola}} \\
   \text{bola}, & B \ge B_{\text{low}} \text{ and } q_{\text{bola}} \ge q_{\text{tput}}
   \end{cases}

Strengths
---------

- combines BOLA stability with throughput responsiveness
- avoids some low-buffer traps
- often ramps up faster than pure BOLA

Limitations
-----------

- introduces more parameters
- switching logic can be sensitive to noisy estimates
- harder to analyze than RB or BB

Sample simulator output
-----------------------

A SABRE run prints per-chunk lines followed by an end-of-run summary.

.. code-block:: text

   [130000] Network: bw=23500,lat=20,dur=1000  (q=5: bitrate=20000)
   1608418255.46    20000   24.708865   0.000   796460   291   20
   1608418256.31    20000   24.857546   0.000   596962   142   20
   1608418257.38    20000   24.795098   0.000   947625   205   20

   buffer size: 25000
   total played utility: 349.748002
   time average played utility: 2.143434
   total played bitrate: 2998000.000000
   time average played bitrate: 18373.275138
   total play time: 163.171780
   total rebuffer: 5.146674
   rebuffer ratio: 0.031541
   total rebuffer events: 4.000000
   total bitrate change: 146000.000000
   time average score: 1.985727
   rampup time: 2.002097
   total reaction time: 12.380516

These logs are what you later use for bitrate curves, buffer plots, rebuffer
analysis, and QoE comparisons across algorithms.