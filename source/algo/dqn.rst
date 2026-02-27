DQN
===

Folder: ``SERVER/dqn``

This module provides an HTTP ABR decision server backed by a DQN (Double-DQN)
value-based policy, using a Pensieve-style state representation.

It includes:

- **Inference server**: ``dqn_server.py`` (HTTP + CORS, returns next quality index)
- **Trainer**: ``train_dqn.py`` (offline training against ``ABREnv``)
- **DQN model**: ``dqn.py`` (PyTorch Double-DQN with replay buffer + target net)

Repository files
----------------

.. code-block:: text

   SERVER/dqn
   ├── dqn.py
   ├── dqn_server.py
   ├── train_dqn.py
   └── README.md

Quickstart
----------

Install dependencies
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install numpy torch

Run the DQN ABR server
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python dqn_server.py --host localhost --port 8606 \
     --movie ../movie_4g.json \
     --model server/models_dqn/dqn_ep_400.pth \
     --debug --verbose

Notes:

- If the model file is missing, the server can still run with randomly initialized weights.
- Logs are written under a server log folder (see ``dqn_server.py`` defaults).
- Optional inference-time exploration may be available via an ``--epsilon`` argument (if enabled).

Training (offline)
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train_dqn.py

Training script behavior (typical flow):

- Initializes ``ABREnv`` (imported from your server env code).
- Iterates episodes, selects actions with epsilon-greedy, collects transitions.
- Trains using replay samples and periodically saves model checkpoints.

Architecture overview
---------------------

Runtime flow:

1. A player (or shim) POSTs per-chunk playback/download statistics to the server.
2. The server updates a history state tensor (``S_INFO × S_LEN``), computes a QoE reward for logging, and chooses the next quality index from Q-values.
3. The response body is the next quality index (integer). At end-of-video, it may return ``REFRESH`` and reset its internal state.

State space
-----------

The server uses a Pensieve-style state tensor with:

- ``S_INFO = 6``
- ``S_LEN = 8``

The state is typically rolled each step, appending a new observation column.

Common meaning for each of the 6 rows:

0. Last selected bitrate (normalized)
1. Buffer level (normalized)
2. Throughput estimate from last download
3. Download time (normalized)
4. Next chunk sizes for all qualities
5. Remaining chunks (normalized)

Action space
------------

The action is the discrete **quality index** in:

- ``[0, A_DIM-1]`` where ``A_DIM`` equals the number of available bitrates.

QoE reward (logged per chunk)
-----------------------------

The server computes a QoE reward (usually for logging/analysis):

- Bitrate utility term (Mbps)
- Stall penalty using incremental rebuffer time
- Smoothness penalty proportional to bitrate change

Typical form:

.. math::

   R = \frac{b}{1000}
       - \alpha \cdot \Delta t_{stall}
       - \beta \cdot \frac{|b - b_{prev}|}{1000}

Where:

- :math:`b` is bitrate in kbps
- :math:`\Delta t_{stall}` is incremental stall time (seconds)
- :math:`\alpha` is rebuffer penalty (e.g., 20)
- :math:`\beta` is smoothness penalty (e.g., 1)

HTTP API
--------

Request
~~~~~~~

The server expects a JSON payload with per-chunk fields such as:

- ``lastquality`` (int)
- ``lastRequest`` (int) segment index
- ``buffer`` (float) seconds
- ``RebufferTime`` (float) cumulative rebuffer time (ms)
- ``lastChunkStartTime`` / ``lastChunkFinishTime`` (ms timestamps)
- ``lastChunkSize`` (bytes)

If malformed or missing fields, the server returns an error response.

Response
~~~~~~~~

- Returns the next quality index as **plain text** (e.g., ``"0"`` … ``"5"``).
- At end-of-video, may return ``"REFRESH"`` and reset episode state.

Logging format
--------------

The server writes one line per chunk with fields like:

.. code-block:: text

   time  bitrate_kbps  buffer_s  rebuf_delta_s  chunk_size_bytes  fetch_time_ms  reward

Use these logs to plot bitrate/buffer/stall behavior and reward distributions.

Movie manifest format
---------------------

The server expects a movie manifest JSON containing:

- segment duration (ms)
- bitrate ladder (kbps)
- per-segment per-quality sizes (often in bits)

Model implementation (dqn.py)
-----------------------------

Typical Double-DQN components:

- Replay buffer (deque)
- Evaluation network and target network
- Double-DQN bootstrapped targets:
  - Next action from eval net
  - Next Q value from target net
- Target network updated periodically or via soft updates (Polyak averaging)

Training driver (train_dqn.py)
------------------------------

The training loop typically performs:

- epsilon-greedy action selection
- step environment
- store transition in replay
- train on minibatches
- periodic checkpointing

Troubleshooting
---------------

Sphinx autodoc import errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use autodoc, ensure your ``docs/conf.py`` includes the repo root in ``sys.path``.
If you want ``SERVER`` imports like ``SERVER.dqn.dqn``, make sure:

- ``SERVER/__init__.py`` exists (so SERVER is a Python package)
- ``SERVER/dqn/__init__.py`` exists (optional but recommended)

Model not found
~~~~~~~~~~~~~~~

If the checkpoint path is wrong, start the server with a valid ``--model`` path
or train a model first and point to the saved ``.pth`` file.