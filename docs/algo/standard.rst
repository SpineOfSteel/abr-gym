Standard
========

- Spiteri, K., Urgaonkar, R., and Sitaraman, R. K. *BOLA: Near-optimal bitrate adaptation for online videos.*
- Huang, T.-Y., Johari, R., McKeown, N., Trunnell, M., and Watson, M. *A buffer-based approach to rate adaptation: Evidence from a large video streaming service.*
- Yin, X., Jindal, A., Sekar, V., and Sinopoli, B. *A control-theoretic approach for dynamic adaptive video streaming over HTTP.*
[cite SABRE paper]
[cite MAHIMAHI paper]

.. toctree::
   :maxdepth: 1


.. contents::
   :local:
   :depth: 1

This chapter combines the repository's **classical heuristic**, **BOLA-style buffer-driven**, **model-predictive control**, and **remote / server-side** Adaptive Bitrate (ABR) methods into one coherent reference page.

These methods are important not only as practical algorithms, but also as interpretable baselines for reinforcement-learning systems. They expose design assumptions clearly, often run with lower operational complexity, and provide strong reference points for evaluating learned policies.

A representative QoE objective used throughout the repository is

.. math::

   R_t = \frac{b_t}{1000}
         - \lambda_r \, \Delta t_{\mathrm{stall}}
         - \lambda_s \frac{|b_t - b_{t-1}|}{1000}

where :math:`b_t` is bitrate in kbps, :math:`\Delta t_{\mathrm{stall}}` is incremental stall time in seconds, and :math:`\lambda_r, \lambda_s` are the rebuffering and smoothness penalties.

Quickstart
----------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/SpineOfSteel/abr-gym.git

List available algorithms and utilities:

.. code-block:: bash

   python -m abrGym list

Show simulation command help:

.. code-block:: bash

   python -m abrGym simulate -h

Inspect the BOLA algorithm entry:

.. code-block:: bash

   python -m abrGym info bola

Run RB:

.. code-block:: bash

   python -m abrGym simulate rb

Run BOLA:

.. code-block:: bash

   python -m abrGym simulate bola

Run BBA:

.. code-block:: bash

   python -m abrGym simulate bb

RB
--

Plugin name: ``rb``

RB is the simplest throughput-driven baseline. RB reads the estimated throughput and chooses the highest representation whose bitrate is still sustainable according to that estimate.

High-level behavior:

1. Read ``ctx.throughput_est``
2. If it is missing or non-positive, choose quality ``0``
3. Otherwise scan the bitrate ladder in ascending order
4. Return the largest quality index ``q`` such that ``bitrates[q] <= throughput_est``

Mathematically, the selected quality is:

.. math::

   q^\* = \max \left\{ q \;:\; R_q \le \hat{T} \right\}

where:

- :math:`R_q` is the bitrate of quality level :math:`q`
- :math:`\hat{T}` is the current throughput estimate

Strengths and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~

- extremely simple
- easy to debug and reason about
- good baseline for testing throughput-estimation quality
- ignores buffer occupancy
- can oscillate if throughput is noisy
- can trigger rebuffering if the estimator is optimistic

BBA
---

Plugin name: ``bb``

BBA is a buffer-based adaptation plugin that maps current buffer occupancy to a quality level. It divides the buffer into three zones:

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

Key Parameters
~~~~~~~~~~~~~~

The implementation uses fixed thresholds such as:

- ``reservoir_ms = 5000``
- ``cushion_ms = 6500``

These define the ramp region where BBA gradually increases quality as the buffer fills.

BBA can apply a throughput-based safety cap:

- ``SAFETY_CAP = 0.85`` by default
- the chosen representation is capped so it does not exceed ``0.85 * throughput_est``

Formally, after the buffer-based selection:

.. math::

   q_{\text{final}} = \min \left( q_{\text{buffer}}, q_{\text{safety}} \right)

where :math:`q_{\text{safety}}` is the largest quality index satisfying

.. math::

   R_q \le \alpha \hat{T}

with :math:`\alpha = 0.85`.

Strengths
~~~~~~~~~

- naturally buffer-aware
- usually more stable than pure RB
- good at stall avoidance when buffer is low
- easy to interpret and tune

Limitations
~~~~~~~~~~~

- ramp-up can be slow if thresholds are conservative
- performance depends on sensible threshold choices
- less theoretically grounded than BOLA

BOLA
----

Plugin name: ``bola``

BOLA stands for **Buffer Occupancy based Lyapunov Algorithm**. It is a buffer-aware ABR method that selects the next bitrate by maximizing a utility-per-bit score that balances:

- video quality utility
- current buffer occupancy
- bitrate / chunk cost

Compared with simpler heuristics such as RB or BBA, BOLA is both practically strong and theoretically motivated.

Core intuition
~~~~~~~~~~~~~~

BOLA treats bitrate choice as a utility-optimization problem under finite buffer.

The central idea is:

- high quality should be rewarded
- large buffer gives freedom to be more ambitious
- low buffer should push decisions toward safer choices
- the score should normalize by bitrate cost

Utility model
~~~~~~~~~~~~~

BOLA uses a concave utility function. In this implementation, a typical form is log-bitrate utility:

.. math::

   u(q) = \log(R_q) + u_0

where:

- :math:`R_q` is the bitrate for quality level :math:`q`
- :math:`u_0` is a shift term used to normalize the utility baseline

A concave utility is important because the perceptual gain from increasing bitrate usually has diminishing returns.

Gain parameter and control term
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your implementation uses a gain parameter ``gp`` and a derived control quantity ``Vp``.

A simplified form is:

.. math::

   V_p = \frac{B_{\max} - p}{u_{\max} + \gamma_p}

where:

- :math:`B_{\max}` is the maximum buffer size
- :math:`p` is the segment duration
- :math:`u_{\max}` is the maximum utility value over all qualities
- :math:`\gamma_p` is the gain parameter exposed as ``--gamma-p``

BOLA score
~~~~~~~~~~

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

Delay / oscillation control
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your implementation includes practical oscillation control by inserting delay when the buffer is too full for the chosen quality.

Conceptually, if the current buffer exceeds a quality-dependent threshold :math:`B_{\max}(q)`, then delay is added:

.. math::

   d = \max(0, B - B_{\max}(q))

Abandonment
~~~~~~~~~~~

BOLA can optionally abandon a partially downloaded chunk and restart at a lower quality if the remaining download appears suboptimal.

Strengths
~~~~~~~~~

- strong theoretical motivation
- naturally buffer-aware
- does not require explicit bandwidth prediction in its core form
- usually smoother than pure throughput methods

Limitations
~~~~~~~~~~~

- requires reasonable buffer and segment-duration assumptions
- parameters such as ``gp`` influence aggressiveness
- more complex than RB or BBA

BOLA-TPUT
---------

Plugin name: ``throughput`` (from ``algo/bola-d.py``)

BOLA-TPUT is the throughput-oriented mode exposed by the hybrid BOLA-D implementation. It uses a throughput estimate with a safety margin to choose the highest sustainable quality.

A simplified rule is:

.. math::

   q_{\text{tput}} = \max \left\{ q : R_q \le \beta \hat{T} \right\}

where :math:`\beta` is a safety multiplier such as ``0.9``.

This mode is useful when you want the responsiveness of throughput-based adaptation while still using the shared infrastructure of the hybrid BOLA-D implementation.

BOLA-D
------

Plugin name: ``dynamic``

BOLA-D is a hybrid ABR policy that combines:

- **BOLA** for stable buffer-aware decisions
- a **throughput rule** for responsiveness when buffer is low

It switches between the two based on buffer level and which rule appears safer or more beneficial.

Throughput Component
~~~~~~~~~~~~~~~~~~~~

The throughput sub-policy uses:

- throughput estimate multiplied by a safety factor
- optional latency-aware insufficient-buffer logic
- abandonment checks when download time looks too long

BOLA Component
~~~~~~~~~~~~~~

The BOLA side is the same utility-driven buffer policy described above.

Switching Rule
~~~~~~~~~~~~~~

BOLA-D maintains an internal mode.

A representative switching rule is:

- use throughput mode when buffer is low and throughput mode recommends a better actionable quality
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
~~~~~~~~~

- combines BOLA stability with throughput responsiveness
- avoids some low-buffer traps
- often ramps up faster than pure BOLA

Limitations
~~~~~~~~~~~

- introduces more parameters
- switching logic can be sensitive to noisy estimates
- harder to analyze than RB or BBA

What SABRE provides
~~~~~~~~~~~~~~~~~~~

SABRE is the simulator layer that drives all of these policies. It combines:

- a network trace
- a movie manifest
- a player buffer model
- an ABR plugin selected by name
- per-chunk logging for plotting and evaluation

Sample simulator output
~~~~~~~~~~~~~~~~~~~~~~~

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

FastMPC
-------

FastMPC is a lightweight model-predictive control variant that uses short-horizon lookahead to choose the next chunk quality quickly.

Typical characteristics:

- short horizon
- simple throughput estimation
- direct QoE scoring of candidate trajectories
- lower computational overhead than more elaborate MPC variants

At each chunk decision, FastMPC:

1. estimates future available bandwidth
2. simulates a short horizon of future chunk downloads
3. evaluates a QoE objective for each candidate action sequence
4. chooses the first action of the best sequence

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

RobustMPC
---------

RobustMPC adds more conservative bandwidth prediction and is designed to perform better under uncertain or rapidly changing conditions.

Typical characteristics:

- robustified throughput estimate
- bounded lookahead horizon
- explicit tradeoff between quality, rebuffering, and smoothness

A generic robust estimate can be written as:

.. math::

   \hat{T}_{\mathrm{robust}} =
   \frac{\hat{T}}{1 + \epsilon}

where :math:`\epsilon` captures prediction uncertainty or recent relative error.

Why MPC matters
~~~~~~~~~~~~~~~

MPC occupies an important middle ground:

- more informed than simple heuristics
- more interpretable than RL
- easier to control than pure learned policies
- still practical for online server-side decision making

Remote / Server-Side ABR
------------------------

This section describes the HTTP-style ABR decision flow used by server-side and shim-compatible implementations in this repository.

Unlike local SABRE plugins, remote ABR components expose a service endpoint that receives playback statistics from a player and returns the next quality index.

Why remote ABR exists
~~~~~~~~~~~~~~~~~~~~~

A remote or shim-based ABR service is useful when:

- the player logic should stay thin
- you want to compare multiple algorithms under the same request format
- the ABR decision logic lives in Python while the player runs elsewhere
- training and inference stacks should share a common state representation

Typical request / response flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At each chunk decision point:

1. the client POSTs playback and download statistics
2. the server updates its internal ABR state
3. the server optionally computes a reward / QoE log entry
4. the server returns the next quality index as plain text

At end-of-video, the server typically returns ``REFRESH`` and resets its episode state.

Typical request fields
~~~~~~~~~~~~~~~~~~~~~~

The HTTP servers in this repository usually expect fields such as:

- ``lastquality``: previous quality index
- ``lastRequest``: current segment index
- ``buffer``: current buffer level in seconds
- ``RebufferTime``: cumulative rebuffer time in milliseconds
- ``lastChunkStartTime`` and ``lastChunkFinishTime``: timestamps in ms
- ``lastChunkSize``: downloaded chunk size in bytes

Some implementations also accept or ignore summary-style payloads such as ``pastThroughput``.

State update pattern
~~~~~~~~~~~~~~~~~~~~

The remote ABR service usually maintains a fixed-shape history tensor. Each request shifts the state left and appends a new observation column containing:

- previous bitrate
- buffer level
- throughput estimate
- fetch/download time
- next chunk sizes
- remaining chunks

QoE reward logging
~~~~~~~~~~~~~~~~~~

Many remote servers compute a per-chunk reward for analysis even when they are running only in inference mode.

A typical QoE reward is:

.. math::

   R_t = \frac{b_t}{1000}
         - \lambda_r \Delta t_{\mathrm{stall}}
         - \lambda_s \frac{|b_t - b_{t-1}|}{1000}

Typical responses
~~~~~~~~~~~~~~~~~

The service returns:

- ``"0"`` through ``"5"`` for a valid next quality decision
- ``"REFRESH"`` at end-of-video
- diagnostic strings such as ``BAD_JSON`` or ``MISSING_FIELD:<field>`` on invalid input

Logging
~~~~~~~

A remote ABR server commonly writes TSV logs per chunk for plotting and later comparison.

A representative line may include:

.. code-block:: text

   time  bitrate_kbps  buffer_s  rebuf_delta_s  chunk_size_bytes  fetch_time_ms  reward

References
----------

- Spiteri, K., Urgaonkar, R., and Sitaraman, R. K. *BOLA: Near-optimal bitrate adaptation for online videos.*
- Huang, T.-Y., Johari, R., McKeown, N., Trunnell, M., and Watson, M. *A buffer-based approach to rate adaptation: Evidence from a large video streaming service.*
- Yin, X., Jindal, A., Sekar, V., and Sinopoli, B. *A control-theoretic approach for dynamic adaptive video streaming over HTTP.*
