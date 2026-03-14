Standard
========

.. contents::
   :local:
   :depth: 2

This chapter combines the repository's **classical heuristic**, **BOLA-style buffer-driven**, **model-predictive control**, and **remote / server-side** Adaptive Bitrate (ABR) methods into one coherent reference page.

These methods are important not only as practical algorithms, but also as interpretable baselines for reinforcement-learning systems. They expose design assumptions clearly, often run with lower operational complexity, and provide strong reference points for evaluating learned policies.

A representative QoE objective used throughout the repository is

.. math::

   R_t = \frac{b_t}{1000}
         - \lambda_r \, \Delta t_{\mathrm{stall}}
         - \lambda_s \frac{|b_t - b_{t-1}|}{1000}

where :math:`b_t` is bitrate in kbps, :math:`\Delta t_{\mathrm{stall}}` is incremental stall time in seconds, and :math:`\lambda_r, \lambda_s` are the rebuffering and smoothness penalties.

The sections below merge the previously separate pages into a single article-like chapter.

Classic Heuristic ABR: RB, BB, and Dynamic
------------------------------------------

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


BOLA
----

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


MPC-Based ABR
-------------

This page covers model-predictive control (MPC) style bitrate adaptation used in
server-side ABR implementations such as FastMPC and RobustMPC.

MPC methods do not simply react to the latest measurement. Instead, they predict
near-future bandwidth over a short horizon and evaluate multiple candidate bitrate
sequences before selecting the next quality.

Core idea
---------

At each chunk decision, MPC:

1. estimates future available bandwidth
2. simulates a short horizon of future chunk downloads
3. evaluates a QoE objective for each candidate action sequence
4. chooses the first action of the best sequence

This is why MPC is often called a **lookahead** ABR method.

QoE objective
-------------

A representative MPC objective over horizon :math:`H` is:

.. math::

   J = \sum_{t=1}^{H}
   \left(
      \frac{b_t}{1000}
      - \lambda_r \, r_t
      - \lambda_s \frac{|b_t - b_{t-1}|}{1000}
   \right)

where:

- :math:`b_t` is bitrate at future step :math:`t`
- :math:`r_t` is rebuffer time at that step
- :math:`\lambda_r` is the rebuffer penalty
- :math:`\lambda_s` is the smoothness penalty

The controller picks the action sequence maximizing :math:`J`.

Bandwidth prediction
--------------------

A simple MPC predictor may use a recent average or harmonic mean of past throughput.
RobustMPC-style methods additionally discount optimistic estimates to be safer under noise.

A generic robust estimate can be written as:

.. math::

   \hat{T}_{\mathrm{robust}} =
   \frac{\hat{T}}{1 + \epsilon}

where :math:`\epsilon` captures prediction uncertainty or recent relative error.

FastMPC
-------

FastMPC uses a lightweight lookahead procedure to choose the next chunk quality quickly.

Typical characteristics:

- short horizon
- simple throughput estimation
- direct QoE scoring of candidate trajectories
- lower computational overhead than more elaborate MPC variants

RobustMPC
---------

RobustMPC adds more conservative bandwidth prediction and is designed to perform
better under uncertain or rapidly changing conditions.

Typical characteristics:

- robustified throughput estimate
- bounded lookahead horizon
- explicit tradeoff between quality, rebuffering, and smoothness

Why MPC matters
---------------

MPC occupies an important middle ground:

- more informed than simple heuristics
- more interpretable than RL
- easier to control than pure learned policies
- still practical for online server-side decision making

Relation to other approaches
----------------------------

- **RB** and **BB** are myopic: they react to the current state only.
- **BOLA** uses a utility score tied to current buffer occupancy.
- **MPC** explicitly simulates future outcomes over a short horizon.
- **RL** learns a policy that may implicitly capture longer-horizon structure.

When to use MPC
---------------

MPC is a strong choice when you want:

- a principled non-RL baseline
- explicit control over the QoE objective
- limited-horizon planning under known segment sizes
- interpretability in server-side decision logic


Remote / Server-Side ABR
------------------------

This page describes the HTTP-style ABR decision flow used by server-side and
shim-compatible implementations in this repository.

Unlike local SABRE plugins, remote ABR components expose a service endpoint that
receives playback statistics from a player and returns the next quality index.

Why remote ABR exists
---------------------

A remote or shim-based ABR service is useful when:

- the player logic should stay thin
- you want to compare multiple algorithms under the same request format
- the ABR decision logic lives in Python while the player runs elsewhere
- training and inference stacks should share a common state representation

Typical request / response flow
-------------------------------

At each chunk decision point:

1. the client POSTs playback and download statistics
2. the server updates its internal ABR state
3. the server optionally computes a reward / QoE log entry
4. the server returns the next quality index as plain text

At end-of-video, the server typically returns ``REFRESH`` and resets its episode state.

Typical request fields
----------------------

The HTTP servers in this repository usually expect fields such as:

- ``lastquality``: previous quality index
- ``lastRequest``: current segment index
- ``buffer``: current buffer level in seconds
- ``RebufferTime``: cumulative rebuffer time in milliseconds
- ``lastChunkStartTime`` and ``lastChunkFinishTime``: timestamps in ms
- ``lastChunkSize``: downloaded chunk size in bytes

Some implementations also accept or ignore summary-style payloads such as
``pastThroughput``.

State update pattern
--------------------

The remote ABR service usually maintains a fixed-shape history tensor. Each request
shifts the state left and appends a new observation column containing:

- previous bitrate
- buffer level
- throughput estimate
- fetch/download time
- next chunk sizes
- remaining chunks

This Pensieve-style representation is shared by several RL servers.

QoE reward logging
------------------

Many remote servers compute a per-chunk reward for analysis even when they are
running only in inference mode.

A typical QoE reward is:

.. math::

   R_t = \frac{b_t}{1000}
         - \lambda_r \Delta t_{\mathrm{stall}}
         - \lambda_s \frac{|b_t - b_{t-1}|}{1000}

where:

- :math:`b_t` is current bitrate in kbps
- :math:`\Delta t_{\mathrm{stall}}` is incremental stall time in seconds
- :math:`\lambda_r` is the rebuffer penalty
- :math:`\lambda_s` is the smoothness penalty

Important implementation detail:

``RebufferTime`` is usually cumulative in the request, so the server converts it to
a per-chunk delta before applying the penalty.

Typical responses
-----------------

The service returns:

- ``"0"`` through ``"5"`` for a valid next quality decision
- ``"REFRESH"`` at end-of-video
- diagnostic strings such as ``BAD_JSON`` or ``MISSING_FIELD:<field>`` on invalid input

Logging
-------

A remote ABR server commonly writes TSV logs per chunk for plotting and later comparison.

A representative line may include:

.. code-block:: text

   time  bitrate_kbps  buffer_s  rebuf_delta_s  chunk_size_bytes  fetch_time_ms  reward

These logs make it easy to compare server-side policies with local SABRE plugin runs.

Relation to other docs
----------------------

- See :doc:`classic` for local rule-based policies.
- See :doc:`bola` for the main BOLA equation and score.
- See :doc:`dqn`, :doc:`pensieve`, and :doc:`ppo` for RL-based remote decision servers.
