ABR Environment (Training Wrapper)
==================================

File: ``SERVER/EnvAbr.py``

This module defines :class:`ABREnv`, a training environment wrapper that:

- loads a trace JSON (your trace format)
- loads a video bitrate ladder
- constructs chunk sizes (CBR approximation)
- runs the network simulator (:class:`SERVER.EnvNetwork.Environment`)
- maintains a Pensieve-style state (6×8)
- computes QoE reward and returns ``(state, reward, done, info)``

Key globals
-----------

State definition:

- ``S_INFO = 6`` and ``S_LEN = 8`` define a rolling state tensor. :contentReference[oaicite:33]{index=33}
- ``BUFFER_NORM_FACTOR = 10.0`` normalizes buffer (seconds → buffer/10s). :contentReference[oaicite:34]{index=34}
- ``DEFAULT_QUALITY = 1`` initial bitrate index. :contentReference[oaicite:35]{index=35}

Class: ABREnv
-------------

Constructor
~~~~~~~~~~~

.. code-block:: text

   ABREnv(
       trace_json_path,
       video_path,
       random_seed=42,
       default_quality=1,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       queue_delay_ms=0.0,
       debug=False
   )

Key behaviors:

- Loads trace JSON using ``load_movie_json()`` (note: despite the name, this function parses *network traces*). :contentReference[oaicite:36]{index=36}
- Infers chunk duration from median of trace slot durations (assumes one chunk per trace slot). :contentReference[oaicite:37]{index=37}
- Loads video bitrate ladder from ``video_path`` (expects ``bitrates_kbps``). :contentReference[oaicite:38]{index=38}
- Sets ``a_dim`` from number of bitrates and validates ``default_quality`` bounds. :contentReference[oaicite:39]{index=39}
- Derives total chunks from the trace length (again: one chunk decision per slot). :contentReference[oaicite:40]{index=40}

Chunk sizes: constant-bitrate approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The environment builds ``video_size_by_bitrate`` as:

.. math::

   size\_bytes(q) \approx bitrate\_kbps(q) \cdot chunk\_len\_ms / 8

Implemented in ``_build_cbr_video_sizes()``. :contentReference[oaicite:41]{index=41}

This is a deliberate simplification when the trace JSON does not include per-segment sizes.

Network simulator integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ABREnv`` creates a :class:`Environment` instance with:

- cooked bandwidth/duration/latency arrays
- computed per-quality chunk sizes
- optional ``queue_delay_ms`` to add extra latency to each download :contentReference[oaicite:42]{index=42}

State representation: 6×8 rolling tensor
----------------------------------------

``ABREnv`` maintains:

- ``self.state``: ``np.zeros((6,8), float32)`` :contentReference[oaicite:43]{index=43}
- ``self.last_bit_rate``: previous quality index
- ``self.buffer_size``: seconds
- ``self.time_stamp``: ms

State update logic lives in ``_update_state(...)``. :contentReference[oaicite:44]{index=44}

Feature rows (by index)
~~~~~~~~~~~~~~~~~~~~~~~

At each step, the environment rolls the history by one column and writes the newest:

0. normalized bitrate (kbps / max_kbps) :contentReference[oaicite:45]{index=45}
1. normalized buffer (buffer_s / 10) :contentReference[oaicite:46]{index=46}
2. throughput estimate KB/ms = chunk_bytes / delay_ms / 1000 :contentReference[oaicite:47]{index=47}
3. download time normalized = delay_ms / 1000 / 10 :contentReference[oaicite:48]{index=48}
4. next chunk sizes for all qualities (MB) :contentReference[oaicite:49]{index=49}
5. remaining chunks fraction :contentReference[oaicite:50]{index=50}

Reset/Step API
--------------

seed()
~~~~~~

Sets NumPy seed and updates network env seed if available. :contentReference[oaicite:51]{index=51}

reset()
~~~~~~~

- clears state
- calls ``net_env.reset_episode()``
- downloads the first chunk at ``default_quality`` to initialize state
- returns the initial state tensor :contentReference[oaicite:52]{index=52}

step(action)
~~~~~~~~~~~~

``action`` is a quality index. ``step``:

1. Downloads the chosen quality chunk using ``net_env.get_video_chunk``.
2. Advances timestamp by ``delay + sleep``.
3. Computes QoE reward:

.. math::

   reward = bitrate\_Mbps
            - rebuf\_penalty \cdot rebuf\_s
            - smooth\_penalty \cdot |bitrate - last\_bitrate|_{Mbps}

Implemented as:

- bitrate term: ``video_bitrates[action] / 1000``
- smoothness term uses absolute kbps difference / 1000 :contentReference[oaicite:53]{index=53}

4. Updates rolling state tensor.
5. Returns ``(state, reward, done, info)`` with a rich info dict. :contentReference[oaicite:54]{index=54}

Info dictionary
~~~~~~~~~~~~~~~

Returned ``info`` includes:

- bitrate_kbps, rebuffer_s, buffer_s
- delay_ms, sleep_ms
- chunk_size_bytes, chunks_remain :contentReference[oaicite:55]{index=55}

Trace JSON format parser: load_movie_json()
-------------------------------------------

Despite the name, this parses your **trace JSON** format:

.. code-block:: json

   [
     {"duration_ms": 1000, "bandwidth_kbps": 45000, "latency_ms": 20.0},
     ...
   ]

It validates:

- duration_ms > 0
- bandwidth_kbps >= 0
- latency_ms >= 0 :contentReference[oaicite:56]{index=56}

It returns a bundle with lists of arrays:

- all_cooked_time: indices only (compatibility)
- all_cooked_bw: kbps
- all_cooked_dur_ms: ms
- all_cooked_latency_ms: ms :contentReference[oaicite:57]{index=57}