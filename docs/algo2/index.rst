Algorithms
==========

This documentation collects the Adaptive Bitrate (ABR) algorithms implemented in this
repository across classical heuristics, control-based methods, server-side remote
policies, and reinforcement-learning approaches.

ABR algorithms solve the same core problem: for each video chunk, select the next
representation level using observed bandwidth, current buffer occupancy, latency,
and remaining video horizon so that overall Quality of Experience (QoE) is high.

In this repository, the stack is organized into four broad families:

- **Classic / player-side ABR**: lightweight local decision rules such as
  rate-based, buffer-based, and hybrid heuristics.
- **BOLA family**: utility-driven buffer-based control with a stronger theoretical
  grounding and practical oscillation control.
- **Remote / server-side ABR**: HTTP-based decision services and shim-compatible
  interfaces.
- **RL-based ABR**: policies learned from interaction with the ABR environment,
  including DQN, PPO, and Pensieve/A3C.

Documentation map
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Heuristic and control-based ABR

   classic
   bola
   mpc
   remote

.. toctree::
   :maxdepth: 2
   :caption: RL-based ABR

   dqn
   pensieve
   ppo

Algorithm families
------------------

Classic / heuristic ABR
~~~~~~~~~~~~~~~~~~~~~~~

These algorithms rely on explicit rules rather than learned policies.

- **RB**: rate-based adaptation using recent throughput estimates.
- **BB**: buffer-based quality selection using occupancy thresholds.
- **Dynamic**: hybrid logic that combines throughput and buffer reasoning.
- **BOLA**: utility-driven buffer-based control with score maximization.

MPC / control-based ABR
~~~~~~~~~~~~~~~~~~~~~~~

These methods evaluate future decisions before selecting the next chunk quality.

- **FastMPC**: lightweight server-side lookahead adaptation.
- **RobustMPC**: MPC-style bitrate control with robust throughput prediction.

Remote / shim-based ABR
~~~~~~~~~~~~~~~~~~~~~~~

These components expose ABR logic through an HTTP interface so that a player or
dash.js-like client can POST playback statistics and receive the next quality index.

RL-based ABR
~~~~~~~~~~~~

These methods learn a policy from experience using environment rollouts, QoE rewards,
and either value-based or policy-gradient optimization.

- **DQN**: value-based ABR with replay memory and target network.
- **Pensieve / A3C**: asynchronous actor-critic ABR.
- **PPO**: clipped-policy actor-critic ABR.

Quick SABRE simulator examples
------------------------------

Clone the repository:

.. code-block:: bash

   !git clone https://github.com/SpineOfSteel/abr-gym.git

Run BB on an explicit 4G trace and movie manifest:

.. code-block:: bash

   !python sab.py --plugin algo/bb.py -a bb -n /content/abr-gym/DATASET/NETWORK/4Glogs_lum/4g_trace_driving_50015_dr.json -m /content/abr-gym/DATASET/MOVIE/movie_4g.json --chunk-log-start-ts 1608418123 --chunk-log log_BB2_driving_4g -v

Run BB using a chunk artifact folder:

.. code-block:: bash

   !python sab.py --plugin algo/bb.py -a bb --chunk-folder tmp3 --chunk-log log_BB_driving_4g.txt -v

Run BB using default paths from ``sab.py``:

.. code-block:: bash

   !python sab.py --plugin algo/bb.py --chunk-log log_BB_driving_4g.txt -a bb -v

Run RB:

.. code-block:: bash

   !python sab.py --plugin algo/rb.py -a rb --chunk-log log_RB_driving_4g.txt -v

Run BOLA:

.. code-block:: bash

   !python sab.py --plugin algo/bola.py -a bola --chunk-log log_BOLA_driving_4g.txt -v

Run BOLA-throughput mode:

.. code-block:: bash

   !python sab.py --plugin algo/bola-d.py -a throughput --chunk-log log_BOLA-TPT_driving_4g.txt -v

Run Dynamic / BOLA-D:

.. code-block:: bash

   !python sab.py --plugin algo/bola-d.py -a dynamic --chunk-log log_BOLA-D_driving_4g.txt -v

SABRE command-line arguments
----------------------------

The ``sab.py`` simulator exposes the following important arguments:

- ``--plugin``: load a plugin ``.py`` file that registers ABR and/or averaging logic
- ``-n`` / ``--network``: network trace JSON
- ``-nm`` / ``--network-multiplier``: multiply network bandwidth
- ``-m`` / ``--movie``: movie manifest JSON
- ``-ml`` / ``--movie-length``: optionally truncate the movie duration
- ``-a`` / ``--abr``: registered ABR algorithm name
- ``-ab`` / ``--abr-basic``: simplified ABR mode
- ``-ao`` / ``--abr-osc``: oscillation-control option
- ``-gp`` / ``--gamma-p``: BOLA gain parameter
- ``-noibr`` / ``--no-insufficient-buffer-rule``: disable latency-aware safety rule
- ``-ma`` / ``--moving-average``: throughput estimator registry name
- ``-ws`` / ``--window-size``: averaging window sizes
- ``-hl`` / ``--half-life``: EWMA half-life values
- ``-s`` / ``--seek``: seek control
- ``-r`` / ``--replace``: chunk replacement mode
- ``-b`` / ``--max-buffer``: max buffer size in seconds
- ``-noa`` / ``--no-abandon``: disable abandonment
- ``-rmp`` / ``--rampup-threshold``: ramp-up threshold
- ``-v`` / ``--verbose``: verbose simulator output
- ``--list``: list available ABRs and moving averages
- ``--chunk-log``: per-chunk trace output file
- ``--chunk-folder``: folder for chunk artifact output
- ``--chunk-log-start-ts``: absolute timestamp prefix for chunk logs
- ``--shim``: shim port / identifier
- ``--timeout_s``: timeout setting
- ``--debug_p``: debug plugin behavior
- ``--ping_on_start``: ping on startup

Example simulator output
------------------------

A SABRE run prints chunk-level rows followed by summary metrics:

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

The per-chunk rows typically include:

- timestamp or simulated time
- selected bitrate in kbps
- current buffer level
- incremental stall
- chunk size
- fetch time
- latency or related metric

The summary section reports aggregate QoE-related metrics for direct comparison
across ABR algorithms.