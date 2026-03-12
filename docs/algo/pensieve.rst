Pensieve (A3C)
==============

Folder: ``SERVER/pensieve``

This module implements the classic **Pensieve** approach: an **A3C actor–critic**
policy that selects the next video chunk quality from a **Pensieve-style state**
(6×8 history tensor).

It includes:

- **Inference server**: ``pensieve_server.py`` (HTTP + CORS, returns next quality index)
- **Training driver**: ``train_a3c.py`` (central learner + worker(s) computing gradients)
- **A3C networks + update utilities**: ``a3c.py`` (ActorNetwork, CriticNetwork, Network, compute_gradients)

Repository files
----------------

.. code-block:: text

   SERVER/pensieve
   ├── a3c.py
   ├── pensieve_server.py
   ├── train_a3c.py
   └── README.md


Quickstart
----------

Install dependencies
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install numpy torch


Run the Pensieve (A3C) server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example:

.. code-block:: bash

   python pensieve_server.py --host localhost --port 8605 \
     --movie movie_4g.json \
     --actor server/models_a3c_min/a3c_actor_ep_600.pth \
     --debug --verbose

Notes:

- If the actor checkpoint is missing, the server continues with randomly initialized weights.
- Logs are written under a server logs folder (see ``SUMMARY_DIR`` / ``LOG_FILE`` in the server).


Train A3C (offline)
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train_a3c.py

Training runs a **central learner** that aggregates worker gradients and applies them
(RMSProp), periodically saving actor and critic checkpoints.


Overview
--------

Runtime flow (server)
~~~~~~~~~~~~~~~~~~~~~

1. **Client (video player)** POSTs per-chunk playback + download stats.
2. **Server** updates the RL state, computes a per-chunk QoE reward for logging,
   and queries the A3C actor to get a probability distribution :math:`\pi(a|s)`.
3. **Action selection** samples a discrete quality index from the distribution.
4. Server returns the **next quality index** as plain text.

At end-of-video, the server returns ``REFRESH`` and resets its internal episode state.


Training flow (A3C)
~~~~~~~~~~~~~~~~~~~

This implementation follows a “parameter-server style” loop:

- **Central learner**
  - Owns master Actor + Critic networks
  - Pushes latest network parameters to workers each epoch
  - Collects worker-computed gradients
  - Averages gradients and applies them (RMSProp)
  - Saves checkpoints periodically

- **Worker(s)**
  - Run rollouts in the ABR environment (``ABREnv``)
  - Compute local gradients (actor + critic) using the A3C update rule
  - Send gradients + summary stats back to the central learner
  - Sync updated parameters and repeat


State, Action, Reward
---------------------

State space (Pensieve-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The state tensor dimensions are:

- ``S_INFO = 6``
- ``S_LEN = 8``

The server maintains a rolling history. Each new chunk shifts the tensor left and
writes the newest “column” at the end.

The 6 rows correspond to:

0. **Last selected bitrate**, normalized by max bitrate
1. **Buffer level**, normalized by ``BUFFER_NORM_FACTOR`` (10 seconds)
2. **Throughput estimate** from last download (KB/ms)
3. **Download time**, normalized (sec/10)
4. **Next chunk sizes** for all qualities (MB)
5. **Remaining chunks**, normalized by a cap

Next-chunk sizes are derived from the movie manifest’s per-segment sizes (converted
from bits to bytes internally).


Action space
~~~~~~~~~~~~

- ``A_DIM = 6`` discrete quality levels, typically indices 0..5.
- The server returns the selected quality index as plain text.


Reward function (QoE)
~~~~~~~~~~~~~~~~~~~~~

The server computes a per-chunk QoE reward:

- Higher bitrate → higher reward
- Rebuffering (stall) → large penalty
- Quality switches → smoothness penalty

Constants used in the server:

- ``REBUF_PENALTY = 20``
- ``SMOOTH_PENALTY = 1``

Reward form:

.. math::

   R = \frac{b}{1000}
       - 20.0 \cdot \Delta t_{stall}
       - 1.0 \cdot \frac{|b - b_{prev}|}{1000}

Where:

- :math:`b` is current selected bitrate (kbps)
- :math:`\Delta t_{stall}` is **incremental** stall time since the last decision (seconds)
- :math:`b_{prev}` is the previous selected bitrate (kbps)

Important: Incoming ``RebufferTime`` from the client is cumulative (ms). The server
converts it into a **delta** from the last request before computing reward.


HTTP API (shim protocol)
------------------------

Request
~~~~~~~

The server expects a JSON POST payload with fields similar to dash.js ABR shims:

- ``lastquality`` (int)
- ``lastRequest`` (int) chunk index
- ``buffer`` (float) seconds
- ``RebufferTime`` (float) cumulative rebuffer time (ms)
- ``lastChunkStartTime`` / ``lastChunkFinishTime`` (ms timestamps)
- ``lastChunkSize`` (bytes)

Error handling:

- Returns ``BAD_JSON`` if payload cannot be parsed as JSON.
- Returns ``MISSING_FIELD:<field>`` if a required field is missing.
- Returns ``BAD_FIELD:<...>`` for invalid field parsing.

Special-case payloads:

- If ``pastThroughput`` is present, the server treats it as a summary payload and replies ``"0"``.


Response
~~~~~~~~

- Returns the next quality index: ``"0"`` … ``"5"``.
- Returns ``"REFRESH"`` at end-of-video and resets internal episode state.


Logging
-------

The server writes one TSV line per chunk for plotting and analysis. A typical line includes:

.. code-block:: text

   time  bitrate_kbps  buffer_s  rebuf_delta_s  chunk_size_bytes  fetch_time_ms  reward

Use these logs to plot bitrate/buffer/stall behavior and QoE distributions.


Movie manifest format (movie_*.json)
------------------------------------

The server expects a movie manifest JSON with keys:

- ``segment_duration_ms``
- ``bitrates_kbps`` (must have length 6)
- ``segment_sizes_bits``: list of segments, each containing 6 sizes in **bits**

Internally, sizes are converted bits → bytes using ceiling division.
If ``total_video_chunks`` is not provided, it is derived from the segment list length.


A3C implementation (a3c.py)
---------------------------

Network structure
~~~~~~~~~~~~~~~~~

The actor/critic share a “Pensieve-like” feature extractor:

- Rows 0, 1, and 5 use FC layers over the most recent scalar.
- Rows 2 and 3 use 1D convolution over history length ``S_LEN``.
- Row 4 uses 1D convolution over the first ``A_DIM`` entries (next chunk sizes).

Actor head outputs a probability distribution :math:`\pi(a|s)` (softmax with clamping).
Critic head outputs a scalar value estimate :math:`V(s)`.

Optimization and stability
~~~~~~~~~~~~~~~~~~~~~~~~~

- Uses RMSProp optimizer (classic A3C style).
- Uses entropy regularization in the actor loss to prevent premature collapse.
- Probability vectors are clamped to avoid log(0) and NaNs.

Training API surface:

- ``ActorNetwork.predict(inputs)`` → action probabilities
- ``CriticNetwork.predict(inputs)`` → value estimate
- ``compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic)`` →
  returns actor gradients, critic gradients, and TD errors


Training driver (train_a3c.py)
------------------------------

Key configuration
~~~~~~~~~~~~~~~~~

The training script sets:

- ``TRACE_JSON_PATH`` (trace JSON, e.g., ``DATASET\\NETWORK\\network.json``)
- ``VIDEO_PATH`` (movie manifest, e.g., ``DATASET\\MOVIE\\movie_4g.json``)
- ``SUMMARY_DIR`` (checkpoint output dir, e.g., ``DATASET\\MODELS``)

Training hyperparameters:

- ``S_DIM = [6, 8]``, ``A_DIM = 6``
- ``TRAIN_SEQ_LEN = 1000``
- ``TRAIN_EPOCH = 2000``
- ``MODEL_SAVE_INTERVAL = 300``
- ``NUM_AGENTS`` controls number of workers
- Multiprocessing start method uses ``spawn`` (safer for PyTorch)

Checkpoints
~~~~~~~~~~~

A3C saves **separate** weights:

- Actor: ``a3c_actor_ep_{epoch}.pth``
- Critic: ``a3c_critic_ep_{epoch}.pth``

For deployment, the HTTP server typically needs only the **actor** checkpoint.


Using a trained actor in the server
-----------------------------------

After training produces actor checkpoints in your model directory, start the server with:

.. code-block:: bash

   python pensieve_server.py --host localhost --port 8605 \
     --movie DATASET/MOVIE/movie_4g.json \
     --actor DATASET/MODELS/a3c_actor_ep_1800.pth \
     --debug --verbose


Troubleshooting
---------------

Import/package issues
~~~~~~~~~~~~~~~~~~~~~

If you use package-style imports (``SERVER.pensieve.a3c``), ensure:

- ``SERVER/__init__.py`` exists
- ``SERVER/pensieve/__init__.py`` exists (recommended)

Windows vs Linux paths
~~~~~~~~~~~~~~~~~~~~~~

Your training script uses Windows-style paths (``DATASET\\...``). For cross-platform
support (Linux/ReadTheDocs), prefer ``os.path.join(...)`` or forward slashes.

Model file not found
~~~~~~~~~~~~~~~~~~~~

If the actor checkpoint path is wrong, the server will warn and run with random weights.
Train first or correct the ``--actor`` path.