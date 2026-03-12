PPO
===

Folder: ``SERVER/ppo``

This module provides an HTTP ABR decision server backed by a **PPO actor–critic**
policy (PyTorch), using a **Pensieve-style RL state** (history tensor).

It includes:

- **Inference server**: ``ppo_server.py`` (HTTP + CORS, returns next quality index)
- **Trainer**: ``train_ppo.py`` (central learner + worker(s) collecting rollouts)
- **Policy implementation**: ``ppo2.py`` (Actor/Critic networks + PPO2 training)

Repository files
----------------

.. code-block:: text

   SERVER/ppo
   ├── ppo2.py
   ├── ppo_server.py
   ├── train_ppo.py
   └── README.md


Quickstart
----------

Install dependencies
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install numpy torch


Run the PPO ABR server
~~~~~~~~~~~~~~~~~~~~~~

Typical usage (from README):

.. code-block:: bash

   python ppo_server.py --host localhost --port 8605 \
     --movie ../movie_4g.json \
     --model models/ppo_model.pth \
     --debug --verbose

Notes:

- If the checkpoint is missing or fails to load, the server continues with randomly
  initialized weights.
- Logs are written under the server logs folder (configured via ``SUMMARY_DIR`` / ``LOG_FILE``).


Train PPO (offline)
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train_ppo.py

The training driver creates a **central learner** process and one or more **worker**
processes. Workers roll out trajectories in ``ABREnv`` and send batches back to the
central learner for PPO updates.

Configuration (paths and hyperparameters) is defined at the top of ``train_ppo.py``.


Overview
--------

Runtime flow (server)
~~~~~~~~~~~~~~~~~~~~~

1. A client video player (or shim) POSTs per-chunk playback & download statistics.
2. The server updates the RL state, computes a per-chunk QoE reward for logging,
   and queries the PPO policy to select the next quality.
3. The server responds with the next quality index as **plain text**.

At end-of-video, the server returns ``REFRESH`` and resets its internal episode state.

Training flow (offline)
~~~~~~~~~~~~~~~~~~~~~~~

- **Central learner** owns the PPO model (actor + critic) and performs PPO updates.
- **Workers** run the ABR environment and collect trajectories:
  state, chosen action (one-hot), old policy probabilities, rewards.
- Workers compute **bootstrapped returns** and send batches back to the learner.
- The learner performs multiple PPO epochs per batch and periodically saves checkpoints.


State, Action, Reward
---------------------

State space (Pensieve-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

- ``A_DIM = 6`` discrete quality levels (e.g., 0..5)
- The server response is the next quality index as plain text.

Reward function (server-side QoE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PPO server uses a QoE reward objective that:

- Rewards higher bitrate
- Penalizes rebuffering heavily
- Penalizes quality switches (smoothness penalty)

In the provided server, the constants are:

- ``REBUF_PENALTY = 4.3``
- ``SMOOTH_PENALTY = 1.0``

The reward is computed using **incremental** stall (delta rebuffer time since the last decision):

.. math::

   R = \frac{b}{1000}
       - 4.3 \cdot \Delta t_{stall}
       - 1.0 \cdot \frac{|b - b_{prev}|}{1000}

Where:

- :math:`b` is bitrate in kbps
- :math:`\Delta t_{stall}` is incremental stall time (seconds)
- :math:`b_{prev}` is previous selected bitrate (kbps)

This corresponds to:

.. code-block:: text

   reward = bitrate_mbps - 4.3 * stall_seconds - 1.0 * abs(bitrate_mbps - last_bitrate_mbps)

Important: The server expects incoming ``RebufferTime`` to be a **cumulative**
time in ms, and it internally converts to a per-chunk delta before applying the penalty.


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

- ``BAD_JSON`` if the payload cannot be parsed as JSON.
- ``MISSING_FIELD:<field>`` if a required field is missing.
- ``BAD_FIELD:<...>`` if a field cannot be parsed or is invalid.

Special-case behavior:

- If the payload contains ``pastThroughput``, it is treated as a “summary”
  payload and the server replies with ``"0"`` (ignored).

Response
~~~~~~~~

- Returns the next quality index: ``"0"`` … ``"5"``
- Returns ``"REFRESH"`` at end-of-video and resets state.


Logging
-------

The server writes a TSV line per chunk including:

.. code-block:: text

   time  bitrate_kbps  buffer_s  rebuf_delta_s  chunk_size_bytes  fetch_time_ms  entropy  reward

Notes:

- ``entropy`` is computed from the policy distribution :math:`\pi(a|s)` and is a
  useful signal for “policy confidence”.
- Logs are flushed on every request for easier tailing and plotting.


Movie manifest format
---------------------

The server loads a movie manifest JSON (``movie_*.json``) that must contain:

- ``segment_duration_ms``
- ``bitrates_kbps`` (length == ``A_DIM``)
- ``segment_sizes_bits``: per-segment sizes in **bits** for each quality

The server converts segment bits → bytes via ceiling division.

Optionally, the JSON may include ``total_video_chunks``; otherwise it derives
the max index from the segment list length.


PPO implementation (ppo2.py)
----------------------------

Networks
~~~~~~~~

The PPO policy is implemented as two PyTorch modules:

- **Actor**: outputs :math:`\pi(a|s)` (softmax distribution over quality indices)
- **Critic**: outputs :math:`V(s)` (state-value)

Both actor and critic share a similar “Pensieve-like” feature split:

- Scalar FC features for specific rows (e.g., last bitrate, buffer, remaining chunks)
- Linear layers over the history rows (throughput and download-time histories)
- Linear layer over the next-chunk-size vector (per quality)

Training objective (PPO2 / clipped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training uses the PPO probability ratio:

.. math::

   r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}

and applies clipping with :math:`\epsilon = 0.2` to limit policy updates.

The implementation also includes:

- Multiple PPO passes per batch (``PPO_TRAINING_EPO``)
- Critic loss (MSE) scaled strongly (value regression)
- Entropy bonus with an adaptive entropy weight that is updated each epoch


Training driver (train_ppo.py)
------------------------------

Processes
~~~~~~~~~

- Central learner process runs ``central_agent(...)``:
  - broadcasts latest actor+critic weights to workers
  - receives batches and performs ``actor.train(...)``
  - saves checkpoints every ``MODEL_SAVE_INTERVAL`` epochs

- Worker process runs ``agent(...)``:
  - resets ``ABREnv``
  - rolls out up to ``TRAIN_SEQ_LEN`` steps
  - uses policy probabilities to sample actions (Gumbel-max sampling over log-probabilities)
  - computes bootstrapped returns (train-side helper)
  - sends (state, one-hot action, old probs, returns) to the learner
  - syncs updated params from the learner

Key configuration
~~~~~~~~~~~~~~~~~

Paths (must exist):

- ``TRACE_JSON_PATH`` → JSON traces: ``DATASET\\NETWORK\\network.json``
- ``VIDEO_PATH`` → movie manifest: ``DATASET\\MOVIE\\movie_4g.json``
- ``SUMMARY_DIR`` → checkpoint directory: ``DATASET\\MODELS``
- ``LOG_FILE`` → server log prefix: ``SERVER\\SERVER_LOGS``

Hyperparameters:

- ``S_DIM = [6,8]``, ``A_DIM = 6``
- ``TRAIN_SEQ_LEN = 1000``
- ``TRAIN_EPOCH = 2000``
- ``MODEL_SAVE_INTERVAL = 300``
- ``ACTOR_LR_RATE = 1e-4``
- ``NUM_AGENTS`` controls worker count (the provided code sets 1)
- Multiprocessing start method is set to ``spawn`` for PyTorch safety.

Checkpoints
~~~~~~~~~~~

PPO saves a **single checkpoint file** that contains both actor and critic weights.
The server expects this combined checkpoint for ``load_model()``.


Using a trained model in the server
-----------------------------------

After training produces checkpoints in ``DATASET\\MODELS`` (e.g. ``nn_model_ep_300.pth``),
start the server with:

.. code-block:: bash

   python ppo_server.py --host localhost --port 8605 \
     --movie DATASET/MOVIE/movie_4g.json \
     --model DATASET/MODELS/nn_model_ep_300.pth \
     --debug --verbose

Evaluation note:

- The server samples from the policy distribution (Gumbel-max).
  If you want deterministic evaluation, modify selection to ``argmax(pi)``.


Troubleshooting
---------------

Import/package issues
~~~~~~~~~~~~~~~~~~~~~

If you document or run with package-style imports (``SERVER.ppo.ppo2``),
ensure:

- ``SERVER/__init__.py`` exists
- ``SERVER/ppo/__init__.py`` exists (recommended)

Model load errors
~~~~~~~~~~~~~~~~~

If the server prints “failed to load model”, verify:

- You passed the correct ``--model`` path
- The file is a PPO checkpoint produced by ``ppo2.Network.save_model()``
  (it must contain `[actor_state_dict, critic_state_dict]`)

Windows paths
~~~~~~~~~~~~~

Your training script uses Windows-style paths (``DATASET\\...``). On Linux/RTD,
prefer forward slashes or use ``os.path.join`` to keep paths portable.