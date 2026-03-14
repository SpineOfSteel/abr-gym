PPO (No Gym)
============

.. contents::
   :local:
   :depth: 1

Folder: ``SERVER/ppo``

This module provides an HTTP ABR decision server backed by a **PPO actor–critic**
policy (PyTorch), using a **Pensieve-style RL state** (history tensor).

It includes:

- **Inference server**: ``ppo_server.py`` (HTTP + CORS, returns next quality index)
- **Trainer**: ``train_ppo.py`` (central learner + worker(s) collecting rollouts)
- **Policy implementation**: ``ppo2.py`` (Actor/Critic networks + PPO2 training)

Quickstart
----------

Repository files
~~~~~~~~~~~~~~~~

.. code-block:: text

   SERVER/ppo
   ├── ppo2.py
   ├── ppo_server.py
   ├── train_ppo.py
   └── README.md

Install dependencies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install numpy torch

Run the PPO ABR server
~~~~~~~~~~~~~~~~~~~~~~

Typical usage:

.. code-block:: bash

   python ppo_server.py ^^host localhost ^^port 8605 \
     ^^movie ../movie_4g.json \
     ^^model models/ppo_model.pth \
     ^^debug ^^verbose

Notes:

- If the checkpoint is missing or fails to load, the server continues with randomly
  initialized weights.
- Logs are written under the server logs folder (configured via ``SUMMARY_DIR`` / ``LOG_FILE``).

Train PPO
~~~~~~~~~

.. code-block:: bash

   python train_ppo.py

The training driver creates a **central learner** process and one or more **worker**
processes. Workers roll out trajectories in ``ABREnv`` and send batches back to the
central learner for PPO updates.

Configuration (paths and hyperparameters) is defined at the top of ``train_ppo.py``.

Overview
--------

Server runtime flow
~~~~~~~~~~~~~~~~~~~

1. A client video player (or shim) POSTs per-chunk playback and download statistics.
2. The server updates the RL state, computes a per-chunk QoE reward for logging,
   and queries the PPO policy to select the next quality.
3. The server responds with the next quality index as **plain text**.

At end-of-video, the server returns ``REFRESH`` and resets its internal episode state.

Training flow
~~~~~~~~~~~~~

- **Central learner** owns the PPO model (actor + critic) and performs PPO updates.
- **Workers** run the ABR environment and collect trajectories:
  state, chosen action (one-hot), old policy probabilities, rewards.
- Workers compute **bootstrapped returns** and send batches back to the learner.
- The learner performs multiple PPO epochs per batch and periodically saves checkpoints.

State, Action, Reward
---------------------

State space
~~~~~~~~~~~

The state uses:

- ``S_INFO = 6``
- ``S_LEN = 8``

State is represented as a ``6 × 8`` tensor (history in columns). Each new chunk
rolls the history and appends the newest observation.

Typical meaning of each row:

0. **Last selected bitrate**, normalized by max bitrate
1. **Buffer level**, normalized by ``BUFFER_NORM_FACTOR`` (10s)
2. **Throughput estimate** from last download (KB/ms)
3. **Download time**, normalized (seconds / 10)
4. **Next chunk sizes** for all qualities (MB)
5. **Remaining chunks**, normalized by cap (``CHUNK_TIL_VIDEO_END_CAP``)

The server computes the “next chunk sizes” feature using the movie manifest
segment sizes for the *next* segment index.

Action space
~~~~~~~~~~~~

- ``A_DIM = 6`` discrete quality levels (for example, 0..5)
- The server response is the next quality index as plain text.

Reward function
~~~~~~~~~~~~~~~

The PPO server uses a QoE reward objective that:

- rewards higher bitrate
- penalizes rebuffering heavily
- penalizes quality switches

In the provided server, the constants are:

- ``REBUF_PENALTY = 4.3``
- ``SMOOTH_PENALTY = 1.0``

The reward is computed using **incremental** stall:

.. math::

   R = \frac{b}{1000}
       - 4.3 \cdot \Delta t_{stall}
       - 1.0 \cdot \frac{|b - b_{prev}|}{1000}

Where:

- :math:`b` is bitrate in kbps
- :math:`\Delta t_{stall}` is incremental stall time in seconds
- :math:`b_{prev}` is previous selected bitrate in kbps

Equivalent form:

.. code-block:: text

   reward = bitrate_mbps - 4.3 * stall_seconds - 1.0 * abs(bitrate_mbps - last_bitrate_mbps)

Important: the server expects ``RebufferTime`` to be **cumulative** time in ms,
and internally converts it to a per-chunk delta before applying the penalty.

HTTP API
--------

Request
~~~~~~~

The server expects JSON fields similar to dash.js ABR shims:

- ``lastquality`` (int)
- ``lastRequest`` (int) chunk index
- ``buffer`` (float) seconds
- ``RebufferTime`` (float) cumulative ms
- ``lastChunkStartTime`` / ``lastChunkFinishTime`` (ms timestamps)
- ``lastChunkSize`` (bytes)

Error responses:

- ``BAD_JSON`` if the payload cannot be parsed as JSON
- ``MISSING_FIELD:<field>`` if a required field is missing
- ``BAD_FIELD:<...>`` if a field cannot be parsed or is invalid

Special-case behavior:

- If the payload contains ``pastThroughput``, it is treated as a summary payload
  and the server replies with ``"0"``.

Response
~~~~~~~~

- Returns the next quality index: ``"0"`` … ``"5"``
- Returns ``"REFRESH"`` at end-of-video and resets state

Implementation
--------------

PPO implementation (``ppo2.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PPO policy is implemented as two PyTorch modules:

- **Actor**: outputs :math:`\pi(a|s)` as a softmax distribution over qualities
- **Critic**: outputs :math:`V(s)`

Both actor and critic use a Pensieve-like feature split:

- scalar FC features for rows such as last bitrate, buffer, remaining chunks
- linear layers over throughput and download-time histories
- linear layer over the next-chunk-size vector

Training objective
~~~~~~~~~~~~~~~~~~

Training uses the PPO probability ratio:

.. math::

   r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}

and applies clipping with :math:`\epsilon = 0.2`.

The implementation also includes:

- multiple PPO passes per batch (``PPO_TRAINING_EPO``)
- critic loss with strong value regression weighting
- entropy bonus with an adaptive entropy weight

Training driver (``train_ppo.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training script uses a learner-worker design:

- **central learner** runs ``central_agent(...)``
- **worker** runs ``agent(...)``

The learner:

- broadcasts latest actor and critic weights
- receives batches and performs PPO updates
- saves checkpoints every ``MODEL_SAVE_INTERVAL`` epochs

Each worker:

- resets ``ABREnv``
- rolls out up to ``TRAIN_SEQ_LEN`` steps
- samples actions from the policy
- computes bootstrapped returns
- sends batches back to the learner
- syncs updated parameters

Key configuration
~~~~~~~~~~~~~~~~~

Paths:

- ``TRACE_JSON_PATH`` → ``DATASET\\NETWORK\\network.json``
- ``VIDEO_PATH`` → ``DATASET\\MOVIE\\movie_4g.json``
- ``SUMMARY_DIR`` → ``DATASET\\MODELS``
- ``LOG_FILE`` → ``SERVER\\SERVER_LOGS``

Hyperparameters:

- ``S_DIM = [6,8]``
- ``A_DIM = 6``
- ``TRAIN_SEQ_LEN = 1000``
- ``TRAIN_EPOCH = 2000``
- ``MODEL_SAVE_INTERVAL = 300``
- ``ACTOR_LR_RATE = 1e-4``
- ``NUM_AGENTS`` controls worker count


Movie manifest
~~~~~~~~~~~~~~

The server loads a movie manifest JSON (``movie_*.json``) that must contain:

- ``segment_duration_ms``
- ``bitrates_kbps`` (length == ``A_DIM``)
- ``segment_sizes_bits``: per-segment sizes in **bits** for each quality

The server converts segment bits to bytes via ceiling division.

Optionally, the JSON may include ``total_video_chunks``; otherwise it derives
the maximum index from the segment list length.


Checkpoints
~~~~~~~~~~~

PPO saves a **single checkpoint file** containing both actor and critic weights.
The server expects this combined checkpoint in ``load_model()``.

Using a trained model
~~~~~~~~~~~~~~~~~~~~~

After training produces checkpoints in ``DATASET\\MODELS``, start the server with:

.. code-block:: bash

   python ppo_server.py ^^host localhost ^^port 8605 \
     ^^movie DATASET/MOVIE/movie_4g.json \
     ^^model DATASET/MODELS/nn_model_ep_300.pth \
     ^^debug ^^verbose

Evaluation note:

- The current server samples from the policy distribution.
- For deterministic evaluation, replace sampling with ``argmax(pi)``.

Logging
--------

The server writes a TSV line per chunk including:

.. code-block:: text

   time  bitrate_kbps  buffer_s  rebuf_delta_s  chunk_size_bytes  fetch_time_ms  entropy  reward

Notes:

- ``entropy`` is computed from the policy distribution :math:`\pi(a|s)`
- logs are flushed on every request for easier tailing and plotting

Troubleshooting
~~~~~~~~~~~~~~~

Import/package issues:

- ensure ``SERVER/__init__.py`` exists
- ensure ``SERVER/ppo/__init__.py`` exists

Model load errors:

- verify the ``^^model`` path
- verify the file is a PPO checkpoint produced by ``ppo2.Network.save_model()``

Windows paths:

- the training script uses Windows-style paths
- on Linux or Read the Docs, prefer forward slashes or ``os.path.join``