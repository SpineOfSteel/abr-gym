Dataset
=======

This project reads input files from ``DATASET/`` to drive ABR simulation and training.
The two most important inputs are:

1) **Movie manifest** (bitrates + per-segment sizes)
2) **Network trace** (bandwidth/latency over time)

This page defines the supported formats and provides concrete examples.

.. code-block:: text

   DATASET
   ├── MODELS
   ├── MOVIE
   │   └── manifest
   ├── NETWORK
   │   ├── 3Glogs
   │   ├── 4Glogs
   │   ├── 4Glogs_lum
   │   ├── 5Glogs_lum
   │   ├── hd_fs
   │   ├── mahimahi
   │   └── sd_fs
   └── TRACES
       └── norway_tram


1) MOVIE (Manifest JSON)
------------------------

**Example file:** ``DATASET/MOVIE/movie_4g.json``

Schema
~~~~~~

The movie manifest is a JSON object with required keys:

- ``segment_duration_ms``: segment duration in milliseconds (e.g., 1000)
- ``bitrates_kbps``: list of available qualities (kbps), length = number of qualities
- ``segment_sizes_bits``: 2D list: ``segment_sizes_bits[seg_idx][q]`` = size in **bits**

The decision servers validate that these keys exist. The servers also convert
bits → bytes using ceiling division and derive total chunks from the list length.

Example (from your file)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "segment_duration_ms": 1000,
     "bitrates_kbps": [2000, 4000, 8000, 12000, 16000, 20000],
     "segment_sizes_bits": [
       [229736, 485032, 2672256, 2672176, 2685896, 2675808],
       ...
     ]
   }

Notes
~~~~~

- Your manifest uses **6 qualities** (A_DIM=6), matching your RL servers (PPO/A3C/DQN).
- Segment sizes are variable even within a single “bitrate ladder” (more realistic than pure CBR).

See: ``movie_4g.json`` sample. :contentReference[oaicite:0]{index=0}


2) NETWORK (Slot Trace JSON)
----------------------------
This folder group contains trace formats used by the network simulator and ABR servers. The primary format is the “slot-trace JSON” (list of slot objects), which captures bandwidth, slot duration, and latency over time. This is the format consumed by your EnvNetwork and EnvAbr classes

2A) Slot-trace JSON is the primary format used by ``SERVER/EnvAbr.py`` and ``SERVER/EnvNetwork.py``. A trace is a JSON list of “slots”; each slot is an object:

- ``duration_ms``: slot duration (ms)
- ``bandwidth_kbps``: throughput capacity in kbps
- ``latency_ms``: base latency in ms


**Example files:**
- ``DATASET/NETWORK/4g_trace_driving_50015_dr.json``
- ``DATASET/NETWORK/4g_trace_walking_50007_c.json``


This is the same trace schema referenced by  training scripts (refer to
``network.json`` in the same format).

Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   [
     {"duration_ms": 1000, "bandwidth_kbps": 31500, "latency_ms": 20.0},
     {"duration_ms": 1000, "bandwidth_kbps": 42500, "latency_ms": 20.0},
     {"duration_ms": 1000, "bandwidth_kbps": 39000, "latency_ms": 20.0}
   ]

This trace uses **1-second slots** . Bandwidth values can be very high and can consist occasional drops/outages

How it’s consumed
~~~~~~~~~~~~~~~~~

- **SERVER EnvNetwork / EnvAbr** treat each slot as a time-varying network.
- Downloads are simulated across one or more slots (depending on chunk size and bandwidth).
- Latency is added per slot during the simulated download.

(Those mechanics are implemented in ``EnvNetwork.get_video_chunk()``.)

2B) Mahimahi time series 
--------------------------

Some folders may contain alternative trace representations used by other toolchains.

Example: tab-separated time series (Mahimahi-like)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Example file:** ``DATASET/TRACES/norway_tram/norway_bus_1``

This file is a 2-column, tab-separated time series:

- Column 1: time (seconds, float)
- Column 2: throughput (likely Mbps or similar unit; values are ~4–5)

Example lines:

.. code-block:: text

   0.0    4.03768755221
   0.55   4.79283060109
   0.88   4.49231799163

Usage guidance
~~~~~~~~~~~~~~

- Treat this as a “raw trace” format. If your pipeline expects the slot JSON format,
  write a small converter:
  - resample to fixed ``duration_ms`` (e.g., 1000ms)
  - map throughput to ``bandwidth_kbps``
  - assign a reasonable default or measured ``latency_ms``


3) TRACES/norway_tram (Example simulator output / log)
------------------------------------------------------

Your ``TRACES/norway_tram`` folder can also include:

- raw traces (like ``norway_bus_1``), and/or
- logs from simulator runs for analysis / plotting.

Example: per-chunk simulator log (TSV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Example file:** ``log_sim_bb_norway_bus_1``

This file is TSV with **7 columns per chunk** (no header). A typical line:

.. code-block:: text

   26680.501461844855    750    4.0    0.8872836624630916    450283    887.2836624630917    -3.065319748591294

Interpretation (recommended convention)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 interpret columns as:

1. timestamp (seconds, float)
2. bitrate or quality indicator (often kbps)
3. buffer level (seconds)
4. stall / rebuffer (seconds, delta)
5. chunk size (bytes)
6. download time (ms)
7. reward / QoE (float)

If you want, I can lock this down precisely by matching it against the logger that
writes this file in your SABRE pipeline (and then we’ll document the exact header).

Compatibility matrix
--------------------

.. list-table::
   :header-rows: 1

   * - File type
     - Example
     - Used by
   * - Movie manifest JSON
     - ``movie_4g.json``
     - SABRE, PPO server, A3C server, DQN server
   * - Slot trace JSON
     - ``4g_trace_driving_50015_dr.json``
     - EnvNetwork/EnvAbr, training scripts, SABRE inputs (if converted)
   * - Time series trace (2-col TSV)
     - ``norway_bus_1``
     - Raw trace source; convert to slot JSON for EnvAbr/servers
   * - Per-chunk simulation log (TSV)
     - ``log_sim_bb_norway_bus_1``
     - Analysis/plotting only (not an input)
