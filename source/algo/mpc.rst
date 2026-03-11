FastMPC
=======

# Reproduction Attempt of [Yin et al.](https://web.stanford.edu/class/cs244/papers/videostreaming-sigcomm2015.pdf)

This module documents the **Model Predictive Control (MPC)** ABR family implemented in this repo:

- **FastMPC**: a practical MPC approximation for real-time use
- **RobustMPC**: a conservative MPC variant that explicitly accounts for prediction errors

The design is inspired by the SIGCOMM’15 paper
*A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP*
(Yin, Jindal, Sekar, Sinopoli). 

Where the code lives
--------------------

Servers (HTTP ABR decision endpoints):

- ``SERVER/fastmpc_server.py`` (FastMPC-like combinational lookahead) 
- ``SERVER/robustmpc_server.py`` (Robust MPC: conservative bandwidth) 

Reference implementation (non-HTTP “component” style):

- ``robust_mpc.py`` (Robust MPC decision logic as a ``Component``) 

If you run MPC via SABRE’s remote shim, the shim-side ABR plugin is registered as ``robustmpc``. :contentReference[oaicite:8]{index=8}


Background: MPC-DASH at a glance
--------------------------------

The SIGCOMM’15 paper frames ABR selection as a **finite-horizon optimal control** problem:
the player picks chunk qualities to maximize a QoE objective while respecting buffer dynamics. 

Key idea:
- predict throughput for a short horizon (next N chunks)
- solve a discrete optimization problem over that horizon (bitrate sequence)
- apply only the first decision (receding horizon / MPC) 

Why MPC:
- it can explicitly trade off bitrate, smoothness, and stalls using a single objective,
  combining throughput prediction + buffer occupancy signals (unlike pure RB or BB). 


QoE objective and mapping to this repo
--------------------------------------

The paper’s general QoE includes:
- video quality (bitrate/utility)
- quality variation penalty (smoothness)
- rebuffer penalty (stall time)
- startup delay penalty :contentReference[oaicite:12]{index=12}

In this repo’s MPC servers, the objective is implemented in a **Pensieve-style** linear form:

.. code-block:: text

   reward = sum(bitrate_kbps)/1000
            - REBUF_PENALTY * rebuffer_seconds
            - sum(|bitrate_kbps - prev_bitrate_kbps|)/1000

FastMPC server computes the same “combo reward” when enumerating candidate sequences. 
RobustMPC server uses the same combo reward but with a conservative bandwidth estimate. 

Note: The servers also log a per-chunk reward for analysis, computed from the incoming request. 


Algorithm 1: FastMPC (this repo’s implementation)
-------------------------------------------------

The SIGCOMM’15 paper proposes FastMPC as a **practical** way to approximate MPC decisions:
solve/enumerate offline and perform lightweight lookup or equivalent fast evaluation online. 

In this repo, ``fastmpc_server.py`` implements a **runtime enumeration** over all quality
sequences of length ``MPC_FUTURE_CHUNK_COUNT=5`` (horizon = 5). 

Key constants
~~~~~~~~~~~~~

- Horizon: ``MPC_FUTURE_CHUNK_COUNT = 5`` :contentReference[oaicite:18]{index=18}
- State shape (for logging / bandwidth prediction): ``S_INFO=5``, ``S_LEN=8`` :contentReference[oaicite:19]{index=19}
- Penalties: ``REBUF_PENALTY=20``, ``SMOOTH_PENALTY=1`` :contentReference[oaicite:20]{index=20}

Throughput prediction (FastMPC server)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FastMPC server uses **harmonic mean** of recent measured throughputs (last up to 5)
as the future bandwidth estimate:

- throughput is derived from ``lastChunkSize / fetch_time`` and stored in the observation
- the server computes a harmonic mean over the last non-zero samples 

This matches the paper’s use of harmonic mean as a robust estimator for short-horizon predictions. 

Lookahead search (FastMPC server)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At each POST:
1) compute predicted bandwidth ``future_bw``
2) compute remaining horizon length (up to 5)
3) enumerate all quality sequences for the horizon
4) simulate buffer evolution under each candidate sequence
5) pick the sequence with maximum objective, return its first element 

Simulation loop details (per candidate sequence):
- download time is computed from segment bytes and predicted bandwidth
- rebuffer increases when ``buffer < download_time``
- buffer decreases by download_time then increases by segment duration
- smoothness penalty is the sum of absolute bitrate changes 


RobustMPC
=========
The SIGCOMM’15 paper defines RobustMPC as optimizing worst-case QoE under bounded throughput,
and shows it is equivalent to running MPC with the **lower-bound throughput**. :contentReference[oaicite:25]{index=25}

The paper also suggests a practical bound:
- baseline prediction: harmonic mean of past 5 chunks
- error bound: max absolute percentage error over past 5 chunks
- conservative throughput: ``C_hat / (1 + err)`` 

RobustMPC in ``robustmpc_server.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This repo implements the same conservative idea:

1) Compute measured throughput from last download:
   ``measured_bw = (chunk_size / fetch_time) / 1000`` (KB/ms scale). 

2) Track prediction errors:
   - compare previous harmonic estimate vs current measured bandwidth
   - store percentage error per step 

3) Compute harmonic mean of recent throughput (past 5), then conservative bandwidth:
   ``future_bw = harmonic_bw / (1 + max_error)`` 

4) Run the exact same combinational lookahead optimizer as FastMPC,
   but using ``future_bw`` above. 

Episode reset behavior:
- if ``end_of_video`` then reply ``REFRESH`` and reset history, including error trackers. 


Reference Robust MPC logic in ``robust_mpc.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file provides a standalone robust MPC implementation (as a ``Component``):

- Horizon: ``HORIZON=5`` :contentReference[oaicite:32]{index=32}
- Bandwidth estimator: harmonic mean of last 5 bandwidth samples :contentReference[oaicite:33]{index=33}
- Error bound: max of last 5 percentage errors :contentReference[oaicite:34]{index=34}
- Conservative bandwidth:
  ``future_bandwidth = harmonic_bandwidth / (1 + max_error)``, clamped by ``MIN_BW_EST_MBPS`` 
- Enumerates all quality sequences of length 5 and picks the first action of the best combo. 


HTTP API contract (FastMPC and RobustMPC servers)
-------------------------------------------------

Both servers implement a simple HTTP endpoint:

- ``OPTIONS``: CORS preflight support
- ``GET``: returns a small JS “ok” message (useful for sanity checks)
- ``POST``: accepts a per-chunk stats payload and returns:
  - a quality index string (e.g., ``"0".."5"``), or
  - ``"REFRESH"`` at end-of-video 

Incoming JSON fields (required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The servers read the following keys from the POST body:

- ``lastquality``: int (previous selected quality)
- ``lastRequest``: int (segment index)
- ``buffer``: float (seconds)
- ``RebufferTime``: float (cumulative rebuffer time, ms)
- ``lastChunkStartTime`` / ``lastChunkFinishTime``: timestamps (ms)
- ``lastChunkSize``: bytes 

These match the “remote shim” payload format described in the repo docs/README. :contentReference[oaicite:39]{index=39}

Special payloads:
- if the JSON contains ``pastThroughput`` the servers treat it as a summary payload and respond ``"0"`` (no decision). 

Response format
~~~~~~~~~~~~~~~

The response body is plain text:

- ``"0".."5"``: the chosen next quality index
- ``"REFRESH"``: end-of-video reset marker 


Quickstart: running the MPC servers
-----------------------------------

FastMPC server
~~~~~~~~~~~~~~

.. code-block:: bash

   python SERVER/fastmpc_server.py \
     --host localhost \
     --port 8395 \
     --movie DATASET/MOVIE/movie_4g.json \
     --debug --verbose

CLI options and defaults are defined in the script. 


RobustMPC server
~~~~~~~~~~~~~~~~

.. code-block:: bash

   python SERVER/robustmpc_server.py \
     --host localhost \
     --port 8390 \
     --movie DATASET/MOVIE/movie_4g.json \
     --debug --verbose

RobustMPC boot and logging behavior (log paths, movie loader, etc.) is implemented in the script. 


Logs produced by MPC servers
----------------------------

Both servers write logs under ``SERVER_LOGS``:

- ``log1...``: per-chunk playback log including bitrate, buffer, rebuffer delta, chunk size, fetch time, reward 
- ``log2...``: bandwidth predictor output over time (future bandwidth estimate) 


Notes and practical considerations
----------------------------------

- Complexity: enumerating all quality sequences of length 5 is ``A_DIM^5`` (for 6 qualities, 7776 combos).
  This is practical for a local server and is how this repo implements “fast enough” MPC online. 
- RobustMPC is intentionally conservative because it shrinks the predicted bandwidth by worst-case recent error,
  matching the paper’s robust lower-bound reasoning. 
- If you later want a closer “table-based FastMPC” to mirror the paper’s offline table idea, you can precompute
  decisions on discretized (buffer, throughput, prev bitrate) bins and replace runtime enumeration with a lookup. 