Remote ABR (HTTP SHIM's) 
==============================

SABRE can run ABR logic in an external process (e.g., PPO/A3C/DQN/MPC server) using
a lightweight HTTP “shim” plugin.

The shim keeps the simulator loop unchanged but outsources the ABR decision:

- SABRE simulates download + buffer
- shim builds a JSON payload per completed segment
- shim POSTs to a server
- server returns next quality index (or REFRESH)




HttpShimBase
---------------

``HttpShimBase`` is a generic shim plugin that:

1. Captures per-download stats and prepares a JSON payload
2. POSTs payload to an ABR server endpoint
3. Returns the server's chosen quality index
4. Falls back to a local rate-based choice if the server is unavailable

Server endpoint
---------------

Default URL:

- ``http://127.0.0.1:8333/``

Configurable via:

- ``timeout_s`` (default: 1.0)
- ``debug_p`` (print HTTP/payload logs)
- ``ping_on_start`` (attempt a lightweight ping in ``first_quality()``)

Payload format (Pensieve-style)
-------------------------------

The shim prepares the following fields per completed segment:

- ``lastquality``: just-finished quality index
- ``RebufferTime``: cumulative rebuffer time (ms)
- ``buffer``: current buffer level (seconds)
- ``lastChunkStartTime``: synthetic start time (ms)
- ``lastChunkFinishTime``: synthetic finish time (ms)
- ``lastChunkSize``: segment size (bytes)
- ``lastRequest``: segment index just finished

Important behaviors
-------------------

- Skips **replacement** segments
- Skips **partial / abandoned** downloads
- Uses manifest segment sizes if download progress lacks size_bits
- Uses a synthetic clock derived from simulated download durations

Response handling
-----------------

- If server returns ``REFRESH``: shim resets to quality 0
- Else server returns a string quality index (``"0".."N-1"``)

Failure fallback
----------------

If POST fails or times out:

- shim selects the highest bitrate <= throughput estimate
- if throughput estimate is missing, returns 0

Index consistency check
-----------------------

Shim warns if:

- pending payload lastRequest != (next seg_idx - 1)

This helps detect mismatches caused by skipped downloads or indexing drift.



Wrapper Plugins (PPO / Pensieve / DQN / MPC)
---------------

Wrapper plugins register distinct ABR names and optionally allow selecting a
server port at runtime via ``--shim <port>``.

They all subclass ``HttpShimBase`` and mainly set:

- ``SHIM_NAME`` (for logging)
- ``server_url`` when ``cfg["shim"]`` is provided

Pensieve wrapper
----------------

Registered name: ``pensieve``

Typical run:

.. code-block:: bash

   python sab.py --verbose --debug_p \
     --plugin algo/pensieve.py -a pensieve \
     -n 4Glogs_lum\\4g_trace_driving_50015_dr.json \
     -m movie_4g.json \
     --shim 8605

PPO wrapper
-----------

Registered name: ``ppo``

.. code-block:: bash

   python sab.py --verbose --debug_p \
     --plugin algo/ppo.py -a ppo \
     -n 4Glogs_lum\\4g_trace_driving_50015_dr.json \
     -m movie_4g.json \
     --shim 8606

DQN wrapper
-----------

Registered name: ``dqn``

.. code-block:: bash

   python sab.py --verbose --debug_p \
     --plugin algo/dqn.py -a dqn \
     -n 4Glogs_lum\\4g_trace_driving_50015_dr.json \
     -m movie_4g.json \
     --shim 8605

FastMPC wrapper
---------------

Registered name: ``fastmpc``

.. code-block:: bash

   python sab.py --verbose --debug_p \
     --plugin algo/fastmpc.py -a fastmpc \
     -n 4Glogs_lum\\4g_trace_driving_50015_dr.json \
     -m movie_4g.json \
     --shim 8391

RobustMPC wrapper
-----------------

Registered name: ``robustmpc``

.. code-block:: bash

   python sab.py --verbose --debug_p \
     --plugin algo/robustmpc.py -a robustmpc \
     -n 4Glogs_lum\\4g_trace_driving_50015_dr.json \
     -m movie_4g.json \
     --shim <port>

Server-side expectations
------------------------

Your remote server must accept a JSON POST payload (Pensieve-style fields) and return:

- a quality index as text (e.g., ``"0".."5"``), OR
- ``"REFRESH"`` to signal end-of-video reset

If the server is down, the shim falls back to a local rate-based choice.