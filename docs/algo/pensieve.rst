Pensieve (No Gym)
=================

.. contents::
   :local:
   :depth: 1

Folder: ``SERVER/pensieve``

This module implements the classic **Pensieve** approach: an **A3C actor–critic**
policy that selects the next video chunk quality from a **Pensieve-style state**
(6×8 history tensor).

It includes:

- **Inference server**: ``pensieve_server.py`` (HTTP + CORS, returns next quality index)
- **Training driver**: ``train_a3c.py`` (central learner + worker(s) computing gradients)
- **A3C networks + update utilities**: ``a3c.py`` (ActorNetwork, CriticNetwork, Network, ``compute_gradients``)

Quickstart
----------

Repository files
~~~~~~~~~~~~~~~~

.. code-block:: text

   SERVER/pensieve
   ├── a3c.py
   ├── pensieve_server.py
   ├── train_a3c.py
   └── README.md

Install dependencies
~~~~~~~~~~~~~~~~~~~~

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
with RMSProp, periodically saving actor and critic checkpoints.

Overview
--------

Server runtime flow
~~~~~~~~~~~~~~~~~~~

1. A client video player or ABR shim POSTs per-chunk playback and download statistics.
2. The server updates the RL state, computes a per-chunk QoE reward for logging,
   and queries the A3C actor to get a probability distribution :math:`\pi(a|s)`.
3. An action is sampled from the policy distribution.
4. The server returns the next quality index as plain text.

At end-of-video, the server returns ``REFRESH`` and resets its internal episode state.

Training flow
~~~~~~~~~~~~~

This implementation follows a parameter-server style loop:

- **Central learner** owns the master actor and critic networks.
- It pushes the latest network parameters to workers, receives worker gradients,
  averages them, applies updates, and periodically saves checkpoints.
- **Workers** run rollouts in ``ABREnv``, compute local actor and critic gradients,
  and send those gradients plus summary statistics back to the learner.

State, Action, Reward
---------------------

State space
~~~~~~~~~~~

The state tensor dimensions are:

- ``S_INFO = 6``
- ``S_LEN = 8``

The server maintains a rolling history. Each new chunk shifts the tensor left and
writes the newest observation column at the end.

The 6 rows correspond to:

0. **Last selected bitrate**, normalized by max bitrate
1. **Buffer level**, normalized by ``BUFFER_NORM_FACTOR`` (10 seconds)
2. **Throughput estimate** from last download (KB/ms)
3. **Download time**, normalized (seconds / 10)
4. **Next chunk sizes** for all qualities (MB)
5. **Remaining chunks**, normalized by a cap

Next-chunk sizes are derived from the movie manifest’s per-segment sizes.

Action space
~~~~~~~~~~~~

- ``A_DIM = 6`` discrete quality levels, typically indices ``0`` to ``5``
- The server returns the selected quality index as plain text

Reward function
~~~~~~~~~~~~~~~

The server computes a per-chunk QoE reward that:

- rewards higher bitrate
- penalizes rebuffering heavily
- penalizes quality switches

Constants used in the server:

- ``REBUF_PENALTY = 20``
- ``SMOOTH_PENALTY = 1``

Reward form:

.. math::

   R = \frac{b}{1000}
       - 20.0 \cdot \Delta t_{stall}
       - 1.0 \cdot \frac{|b - b_{prev}|}{1000}

Where:

- :math:`b` is current selected bitrate in kbps
- :math:`\Delta t_{stall}` is incremental stall time since the last decision in seconds
- :math:`b_{prev}` is the previous selected bitrate in kbps

Important: incoming ``RebufferTime`` from the client is cumulative in ms. The server
converts it into a delta before computing reward.

HTTP API
--------

Request
~~~~~~~

The server expects a JSON POST payload with fields similar to dash.js ABR shims:

- ``lastquality`` (int)
- ``lastRequest`` (int) chunk index
- ``buffer`` (float) seconds
- ``RebufferTime`` (float) cumulative rebuffer time in ms
- ``lastChunkStartTime`` / ``lastChunkFinishTime`` (ms timestamps)
- ``lastChunkSize`` (bytes)

Error handling:

- returns ``BAD_JSON`` if payload cannot be parsed as JSON
- returns ``MISSING_FIELD:<field>`` if a required field is missing
- returns ``BAD_FIELD:<...>`` for invalid field parsing

Special-case payloads:

- if ``pastThroughput`` is present, the server treats it as a summary payload and replies ``"0"``

Response
~~~~~~~~

- returns the next quality index: ``"0"`` … ``"5"``
- returns ``"REFRESH"`` at end-of-video and resets internal episode state

Movie manifest format
---------------------

The server expects a movie manifest JSON with keys:

- ``segment_duration_ms``
- ``bitrates_kbps``
- ``segment_sizes_bits``: list of segments, each containing one size per quality in **bits**

Internally, sizes are converted from bits to bytes using ceiling division.
If ``total_video_chunks`` is not provided, it is derived from the segment list length.

Implementation
--------------

A3C implementation (``a3c.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Network structure
^^^^^^^^^^^^^^^^^

The actor and critic share a Pensieve-like feature extractor:

- rows 0, 1, and 5 use fully connected layers over the most recent scalar values
- rows 2 and 3 use 1D convolution over the history dimension
- row 4 uses 1D convolution over the next-chunk-size vector

The actor head outputs a probability distribution :math:`\pi(a|s)`.
The critic head outputs a scalar value estimate :math:`V(s)`.

Optimization and stability
^^^^^^^^^^^^^^^^^^^^^^^^^^

- uses RMSProp optimizer in classic A3C style
- uses entropy regularization in the actor loss to discourage early collapse
- clamps probabilities to avoid ``log(0)`` and numerical instability

Training API surface:

- ``ActorNetwork.predict(inputs)`` → action probabilities
- ``CriticNetwork.predict(inputs)`` → value estimate
- ``compute_gradients(...)`` → actor gradients, critic gradients, and TD errors

Training driver (``train_a3c.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key configuration
^^^^^^^^^^^^^^^^^

Typical constants include:

- ``S_INFO = 6``
- ``S_LEN = 8``
- ``A_DIM = 6``
- ``NUM_AGENTS`` for worker count
- ``TRAIN_SEQ_LEN`` for rollout length
- checkpoint save intervals and model paths

The learner pushes model weights to workers, receives worker-computed gradients,
and applies averaged updates before saving checkpoints.

Using a trained model
---------------------

After training produces checkpoints, start the server with:

.. code-block:: bash

   python pensieve_server.py --host localhost --port 8605 \
     --movie movie_4g.json \
     --actor server/models_a3c_min/a3c_actor_ep_600.pth \
     --debug --verbose

Evaluation note:

- the current server samples from the policy distribution
- for deterministic evaluation, replace sampling with ``argmax(pi)``

Logging and Troubleshooting
---------------------------

Logging
~~~~~~~

The server writes one TSV line per chunk for plotting and analysis, with fields such as:

.. code-block:: text

   time  bitrate_kbps  buffer_s  rebuf_delta_s  chunk_size_bytes  fetch_time_ms  reward

Use these logs to plot bitrate, buffer, stall behavior, and QoE trends.

Troubleshooting
~~~~~~~~~~~~~~~

Model load errors:

- verify the ``--actor`` checkpoint path
- verify the checkpoint was produced by the A3C training code

Import or package issues:

- ensure package ``__init__.py`` files exist where needed
- prefer consistent relative imports between server and training code
