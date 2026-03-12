Network Environment
===================

File: ``SERVER/EnvNetwork.py``

This module implements a **trace-driven network + buffer simulator** used by the ABR
training environment. The core class is :class:`Environment`, which replays bandwidth,
duration, and latency slots and simulates downloading video chunks over time.

Key constants
-------------

The simulator uses several global constants:

- ``PACKET_PAYLOAD_PORTION = 0.95``: only this fraction of link capacity is treated as
  usable payload (models protocol overhead). :contentReference[oaicite:0]{index=0}
- ``BUFFER_THRESH_MS = 60000``: buffer cap (60s). Above this, SABRE-style "sleep/drain"
  is applied. :contentReference[oaicite:1]{index=1}
- ``NOISE_LOW/HIGH = 0.9..1.1``: multiplicative noise on download delay. :contentReference[oaicite:2]{index=2}
- ``DRAIN_BUFFER_SLEEP_TIME_MS = 500``: sleep happens in 500ms quanta to advance trace time
  while draining. :contentReference[oaicite:3]{index=3}

Class: Environment
------------------

.. rubric:: Purpose

:class:`Environment` simulates chunk downloads over a time-varying network. Inputs are:

- bandwidth in kbps per slot
- slot duration in ms
- slot latency in ms
- per-quality chunk sizes (bytes)

It maintains internal state:

- current trace index and slot pointer
- progress within the current slot (``slot_offset_ms``)
- current buffer level (ms)
- current video chunk index

See class docstring for the trace bundle format. :contentReference[oaicite:4]{index=4}

Constructor
~~~~~~~~~~~

.. code-block:: text

   Environment(
       all_cooked_time,
       all_cooked_bw,
       all_cooked_dur_ms,
       all_cooked_latency_ms,
       video_size_by_bitrate,
       chunk_len_ms,
       random_seed=42,
       queue_delay_ms=0.0,
   )

Important behavior:

- Chooses a random trace instance and a random starting slot within that trace on reset. :contentReference[oaicite:5]{index=5}
- Stores chunk sizes as ``video_size_by_bitrate[q][chunk_idx]`` in bytes. :contentReference[oaicite:6]{index=6}

Trace selection and time progression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``_reset_trace()`` picks a trace and random start point and converts arrays to NumPy for speed. :contentReference[oaicite:7]{index=7}
- ``_advance_slot()`` moves to the next slot and wraps around at trace end. :contentReference[oaicite:8]{index=8}
- ``_advance_time_without_download(sleep_ms)`` advances the trace pointer without adding download delay, used only during buffer draining when buffer exceeds threshold. :contentReference[oaicite:9]{index=9}

Episode reset
~~~~~~~~~~~~~

``reset_episode()`` resets buffer and chunk counter and picks a new trace starting point. :contentReference[oaicite:10]{index=10}

Chunk download simulation: get_video_chunk()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``get_video_chunk(quality)`` is the main API. It simulates downloading one chunk at a
requested quality index. :contentReference[oaicite:11]{index=11}

Steps:

1. **Validate quality** and look up the chunk size for current ``video_chunk_counter``. :contentReference[oaicite:12]{index=12}
2. **Transmit bytes across slots**:

   - Each slot has ``bw_kbps`` and remaining duration ``slot_remain_ms``.
   - Convert kbps → bytes/sec using ``BITS_IN_BYTE`` and apply ``PACKET_PAYLOAD_PORTION``. :contentReference[oaicite:13]{index=13}
   - If the chunk finishes mid-slot, advance by the fractional time and stop. :contentReference[oaicite:14]{index=14}
   - Otherwise, consume the whole slot and continue. :contentReference[oaicite:15]{index=15}

3. **Add latency + optional queue delay** for the current slot. :contentReference[oaicite:16]{index=16}
4. **Apply delay noise** in ``[0.9, 1.1]``. :contentReference[oaicite:17]{index=17}
5. **Compute rebuffering**:

   ``rebuf_ms = max(delay_ms - buffer_size_ms, 0)`` :contentReference[oaicite:18]{index=18}

6. **Update buffer**:

   - drain by download delay
   - add one chunk duration (``chunk_len_ms``) :contentReference[oaicite:19]{index=19}

7. **Buffer cap drain** if buffer exceeds 60s:

   - compute sleep duration in 500ms quanta
   - subtract from buffer
   - advance trace time without download :contentReference[oaicite:20]{index=20}

8. **Advance chunk index** and compute end-of-video behavior. If end-of-video is reached,
   the environment resets episode automatically. :contentReference[oaicite:21]{index=21}

Return values
~~~~~~~~~~~~~

``get_video_chunk`` returns a tuple:

- ``delay_ms``: download + latency (+queue) + noise delay (ms) :contentReference[oaicite:22]{index=22}
- ``sleep_ms``: buffer-drain sleep time (ms) :contentReference[oaicite:23]{index=23}
- ``buffer_s``: updated buffer level (seconds) :contentReference[oaicite:24]{index=24}
- ``rebuf_s``: rebuffer time (seconds) :contentReference[oaicite:25]{index=25}
- ``chunk_size_bytes``: bytes of downloaded chunk :contentReference[oaicite:26]{index=26}
- ``next_sizes``: list of next chunk sizes for each quality (bytes) :contentReference[oaicite:27]{index=27}
- ``end_of_video``: bool :contentReference[oaicite:28]{index=28}
- ``video_chunk_remain``: number of chunks remaining :contentReference[oaicite:29]{index=29}

Design notes
------------

- The environment models both bandwidth and latency per slot, which is important for
  short chunks and mobile networks. :contentReference[oaicite:30]{index=30}
- Rebuffer is computed using the classic “download delay vs buffer” model. :contentReference[oaicite:31]{index=31}
- The buffer cap drain simulates “player sleeping” when buffer is too high, while still
  advancing trace time. :contentReference[oaicite:32]{index=32}