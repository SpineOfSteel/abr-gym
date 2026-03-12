Remote / Server-Side ABR
========================

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