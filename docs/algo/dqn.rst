DQN (No Gym)
============

.. contents::
   :local:
   :depth: 1

Folder: ``SERVER/dqn``

This module provides an HTTP ABR decision server backed by a **DQN value-based**
policy, using a **Pensieve-style state representation**.

It includes:

- **DQN model**: ``dqn.py`` (PyTorch Double-DQN with replay buffer and target network)
- **Inference server**: ``dqn_server.py`` (HTTP + CORS, returns next quality index)
- **Trainer**: ``train_dqn.py`` (offline training against ``ABREnv``)

Quickstart
----------

Repository files
~~~~~~~~~~~~~~~~

.. code-block:: text

   SERVER/dqn
   ├── dqn.py
   ├── dqn_server.py
   ├── train_dqn.py

Run the DQN ABR server
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python dqn_server.py --host localhost --port 8606 \
     --movie ../movie_4g.json \
     --model server/models_dqn/dqn_ep_400.pth \
     --debug --verbose

Notes:

- If the model file is missing, the server can still run with randomly initialized weights.
- Logs are written under a server log folder.
- Optional inference-time exploration may be available via an ``--epsilon`` argument if enabled.

Training
~~~~~~~~

.. code-block:: bash

   python train_dqn.py

Training typically:

- initializes ``ABREnv``
- selects actions with epsilon-greedy exploration
- stores transitions into replay memory
- trains from replay samples
- periodically updates the target network and saves checkpoints

Overview
--------

Server runtime flow
~~~~~~~~~~~~~~~~~~~

1. A player or shim POSTs per-chunk playback and download statistics to the server.
2. The server updates a history state tensor, computes a QoE reward for logging,
   and chooses the next quality index from predicted Q-values.
3. The response body is the next quality index as plain text.

At end-of-video, the server may return ``REFRESH`` and reset its internal episode state.

Training flow
~~~~~~~~~~~~~

The training script follows a standard replay-based value-learning setup:

- step the ABR environment
- collect transitions ``(s, a, r, s', done)``
- sample minibatches from replay memory
- compute Double-DQN targets
- optimize the online network and periodically sync the target network

State, Action, Reward
---------------------

State space
~~~~~~~~~~~

The server uses a Pensieve-style state tensor with:

- ``S_INFO = 6``
- ``S_LEN = 8``

The state is typically rolled each step, appending a new observation column.

Common meaning for each of the 6 rows:

0. last selected bitrate, normalized
1. buffer level, normalized
2. throughput estimate from last download
3. download time, normalized
4. next chunk sizes for all qualities
5. remaining chunks, normalized

Action space
~~~~~~~~~~~~

The action is the discrete quality index in:

- ``[0, A_DIM - 1]`` where ``A_DIM`` equals the number of available bitrates

Reward function
~~~~~~~~~~~~~~~

The server computes a QoE reward, usually for logging and analysis:

- bitrate utility term in Mbps
- stall penalty using incremental rebuffer time
- smoothness penalty proportional to bitrate change

Typical form:

.. math::

   R = \frac{b}{1000}
       - \alpha \cdot \Delta t_{stall}
       - \beta \cdot \frac{|b - b_{prev}|}{1000}

Where:

- :math:`b` is bitrate in kbps
- :math:`\Delta t_{stall}` is incremental stall time in seconds
- :math:`\alpha` is rebuffer penalty
- :math:`\beta` is smoothness penalty

HTTP API
--------

Request
~~~~~~~

The server expects a JSON payload with per-chunk fields such as:

- ``lastquality`` (int)
- ``lastRequest`` (int) segment index
- ``buffer`` (float) seconds
- ``RebufferTime`` (float) cumulative rebuffer time in ms
- ``lastChunkStartTime`` / ``lastChunkFinishTime`` (ms timestamps)
- ``lastChunkSize`` (bytes)

If malformed or missing fields, the server returns an error response.

Response
~~~~~~~~

- returns the next quality index as plain text, for example ``"0"`` to ``"5"``
- at end-of-video, may return ``"REFRESH"`` and reset episode state

Movie manifest format
---------------------

The server expects a movie manifest JSON containing:

- segment duration in ms
- bitrate ladder in kbps
- per-segment, per-quality sizes, often stored in bits

Implementation
--------------

Model implementation (``dqn.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typical Double-DQN components:

- replay buffer
- online evaluation network and target network
- Double-DQN bootstrapped targets
- periodic hard target updates or soft updates, depending on implementation

Core idea
^^^^^^^^^

DQN selects the action with the highest estimated Q-value:

.. math::

   a^* = \arg\max_a Q(s, a)

In Double-DQN, action selection and target evaluation are decoupled to reduce overestimation.

Training driver (``train_dqn.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training loop typically performs:

- epsilon-greedy action selection
- environment stepping
- transition storage in replay memory
- minibatch optimization
- target-network synchronization
- periodic checkpointing

Using a trained model
---------------------

Start the server with a saved checkpoint:

.. code-block:: bash

   python dqn_server.py --host localhost --port 8606 \
     --movie ../movie_4g.json \
     --model server/models_dqn/dqn_ep_400.pth \
     --debug --verbose

Evaluation note:

- for deterministic evaluation, use greedy action selection
- for exploratory testing, keep a small epsilon if supported by the server

Logging and Troubleshooting
---------------------------

Logging
~~~~~~~

The server writes one line per chunk with fields such as:

.. code-block:: text

   time  bitrate_kbps  buffer_s  rebuf_delta_s  chunk_size_bytes  fetch_time_ms  reward

Use these logs to analyze bitrate, buffer, stall behavior, and reward trends.

Troubleshooting
~~~~~~~~~~~~~~~

Model not found:

- verify the ``--model`` path
- train a model first if no checkpoint exists

Runtime issues:

- confirm the movie manifest path is valid
- check that the bitrate ladder matches the action dimension used by the model
