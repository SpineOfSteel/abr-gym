Dataset
=======

This page describes the datasets used in the project for network-driven adaptive bitrate (ABR) simulation, training, replay, and evaluation. It focuses on **input data formats**, **conversion utilities**, and the **Norway mobile bandwidth dataset** that appears throughout the repository.

The goal of this page is to help readers answer three practical questions quickly:

#. What files exist under ``DATASET/``?
#. What format does each component expect?
#. How do the Norway traces move from raw measurements to Mahimahi replay or simulator-ready traces?

.. contents:: Table of contents
   :local:
   :depth: 1

Overview
--------

The repository uses several kinds of data inputs:

- **Movie manifests** describing available qualities and per-segment sizes.
- **Network traces** describing time-varying bandwidth and latency.
- **Raw mobility logs** containing timestamped measurements collected in the field.
- **Converted replay traces** for tools such as Mahimahi.
- **Simulation logs** produced after running an ABR algorithm on a trace.

A representative repository layout is shown below.

.. code-block:: text

   DATASET/
   ├── MODELS/
   ├── MOVIE/
   │   ├── manifest
   │   └── *.json
   ├── NETWORK/
   │   ├── 3Glogs/
   │   ├── 4Glogs/
   │   ├── 4Glogs_lum/
   │   ├── 5Glogs_lum/
   │   ├── fcc/
   │   ├── hd_fs/
   │   ├── mahimahi/
   │   ├── norway_raw/
   │   ├── norway_mahimahi/
   │   ├── mahimahi_util.py
   │   └── sd_fs/
   └── TRACES/
       ├── driving_4g/
       └── norway/

Dataset matrix
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 28 26

   * - File type
     - Example
     - Main fields
     - Used by
   * - Movie manifest JSON
     - ``movie_4g.json``
     - segment duration, bitrate ladder, segment sizes
     - SABRE, PPO/A3C/DQN servers, simulators
   * - Slot trace JSON
     - ``4g_trace_driving_50015_dr.json``
     - ``duration_ms``, ``bandwidth_kbps``, ``latency_ms``
     - ``EnvNetwork``, ``EnvAbr``, training scripts
   * - Raw Norway mobility log
     - files in ``DATASET/NETWORK/norway_raw``
     - timestamps, GPS, bytes received, elapsed time
     - source data for conversion and analysis
   * - Mahimahi replay trace
     - files in ``DATASET/NETWORK/norway_mahimahi``
     - packet-delivery opportunity timestamps
     - Mahimahi-based experiments
   * - Simulator / ABR output log
     - ``log_sim_bb_norway_bus_1``
     - bitrate, buffer, stall, reward, download stats
     - evaluation and plotting only

Why dataset organization matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A well-organized dataset layout is essential for ABR research because the same experimental pipeline usually needs to move across **multiple representations of the same underlying scenario**: raw measurements collected in the field, replay-oriented traces for network emulation, simulator-ready traces for training, and post-run logs for analysis. Keeping these representations clearly separated avoids accidental mixing of inputs and outputs, makes experiments easier to reproduce, and helps new contributors understand which files are canonical sources versus which files are derived artifacts.

In practical terms, good dataset organization improves four things:

- **Reproducibility**: readers can trace a result back to the original measurement source.
- **Comparability**: multiple algorithms can be evaluated on the same transport routes and bandwidth conditions.
- **Extensibility**: new converters, trace normalizers, and simulators can be added without changing the raw data.
- **Documentation quality**: route maps, bandwidth profiles, manifests, and trace formats can be explained once and reused consistently across the project.

Movie
-----

**Example file:** ``DATASET/MOVIE/movie_4g.json``

The movie manifest describes the encoded video ladder and the size of each segment at each quality level.

Schema
~~~~~~

The JSON manifest contains the following required keys:

- ``segment_duration_ms``: duration of one segment in milliseconds.
- ``bitrates_kbps``: list of available representation bitrates.
- ``segment_sizes_bits``: 2D array where ``segment_sizes_bits[seg_idx][q]`` is the size of segment ``seg_idx`` at quality ``q`` in **bits**.

Example
~~~~~~~

.. code-block:: json

   {
     "segment_duration_ms": 1000,
     "bitrates_kbps": [2000, 4000, 8000, 12000, 16000, 20000],
     "segment_sizes_bits": [
       [229736, 485032, 2672256, 2672176, 2685896, 2675808]
     ]
   }

Notes
~~~~~

- The bitrate ladder length should match the number of actions used by the ABR controller.
- Segment sizes vary from chunk to chunk, which makes the manifest more realistic than a constant-bitrate approximation.
- Many decision servers internally convert segment sizes from bits to bytes when estimating download time.

Network
-------

The project uses more than one network representation because different tools expect different interfaces. The two most important forms are a **slot-based JSON trace** and the **Norway raw/mobile trace family**.

Slot-trace JSON
~~~~~~~~~~~~~~~

This is the primary format consumed by ``SERVER/EnvNetwork.py`` and ``SERVER/EnvAbr.py``.

Each trace is a JSON list of time slots, with each slot containing:

- ``duration_ms``: slot duration in milliseconds
- ``bandwidth_kbps``: available throughput during that slot
- ``latency_ms``: propagation / base latency during that slot

Example
~~~~~~~

.. code-block:: json

   [
     {"duration_ms": 1000, "bandwidth_kbps": 31500, "latency_ms": 20.0},
     {"duration_ms": 1000, "bandwidth_kbps": 42500, "latency_ms": 20.0},
     {"duration_ms": 1000, "bandwidth_kbps": 39000, "latency_ms": 20.0}
   ]

How it is used
~~~~~~~~~~~~~~

- The simulator walks through the slots in time order.
- A chunk download may span one or more slots depending on chunk size and available bandwidth.
- Latency is added to the transfer model during chunk delivery.
- This is the most convenient format for training environments because it is explicit, compact, and easy to resample.

Other network folders
~~~~~~~~~~~~~~~~~~~~~

Other network-related directories such as ``3Glogs``, ``4Glogs``, ``4Glogs_lum``, ``5Glogs_lum``, ``fcc``, ``hd_fs``, ``sd_fs``, and ``mahimahi`` can stay grouped under this section. They are best understood as alternate network sources, replay inputs, or processed bandwidth datasets that support evaluation, emulation, and comparison.

Traces
------

A central dataset in this repository is the **HSDPA-bandwidth logs for mobile HTTP streaming scenarios** collected in Telenor's 3G/HSDPA mobile network in Norway.

The dataset contains logs from TCP streaming sessions where adaptive video streams were downloaded at maximum speed, with no artificial throttling and no playback buffer cap during collection. The segment duration used during measurement was **2 seconds**. The tests were collected between **2010-09-13** and **2011-04-21**.

This dataset is valuable because it captures **real mobility**, **real route structure**, and **large throughput variation** across different transportation modes. It is therefore useful both for simulator benchmarking and for reproducing realistic stress cases such as tunnel drops, coverage transitions, and fast-changing radio conditions.

The dataset and its analysis are described by Riiser *et al.* in the MMSys 2013 dataset paper, while the earlier TOMCCAP article explains how location-aware bandwidth knowledge can be used for bitrate planning on similar mobile routes. See `dataset paper (PDF) <https://home.simula.no/~paalh/publications/files/mmsys2013-dataset.pdf>`_, `TOMCCAP article <https://dl.acm.org/doi/10.1145/2240136.2240137>`_, and the `dataset website <https://skulddata.cs.umass.edu/traces/mmsys/2013/pathbandwidth/>`_.

Repository locations
~~~~~~~~~~~~~~~~~~~~

The repository includes the Norway dataset in multiple forms:

- Raw data: ``DATASET/NETWORK/norway_raw``
- Mahimahi-formatted traces: ``DATASET/NETWORK/norway_mahimahi``
- Conversion utility: ``DATASET/NETWORK/mahimahi_util.py``
- Converter entry point: ``convert_norway()``
- Simulator-friendly trace groupings: ``DATASET/TRACES/driving_4g`` and ``DATASET/TRACES/norway``

Data pipeline
~~~~~~~~~~~~~

The most common workflow for the Norway dataset is shown below.

.. code-block:: text

   norway_raw/
      original mobile measurements
      (timestamps, GPS, bytes, elapsed ms)
            |
            |  convert_mahimahi_norway()
            v
   norway_mahimahi/
      replay-ready Mahimahi traces
            |
            +--> Mahimahi replay experiments
            |
            +--> optional resampling / conversion
                    to fixed-slot JSON traces
                          |
                          v
                     Python simulators / RL environments
                          |
                          v
                     ABR run logs and plots

Raw log format
~~~~~~~~~~~~~~

A raw Norway log has rows of the form:

.. code-block:: text

   1289406399 549692 59.851754 10.781778 248069 1008
   1289406400 550772 59.851864 10.781833 191698 1080
   1289406401 551773 59.851964 10.781901 280579 1001
   1289406402 552893 59.852060 10.781969 248971 1120

The columns mean:

#. **Unix timestamp** in seconds since 1970-01-01.
#. **Monotonic timestamp** in milliseconds from an unspecified start point.
#. **Latitude** in decimal degrees.
#. **Longitude** in decimal degrees.
#. **Bytes received since the previous sample**.
#. **Milliseconds elapsed since the previous sample**.

The sixth column is effectively the time delta between consecutive samples in column 2.

Throughput calculation
~~~~~~~~~~~~~~~~~~~~~~

A per-sample throughput can be estimated directly from the raw log using columns 5 and 6.

- **Bytes per millisecond** = ``col5 / col6``
- **Kilobytes per second** = ``col5 / col6``
- **Kilobits per second** = ``8 * col5 / col6``

This works because:

.. math::

   \frac{\text{bytes}}{\text{ms}} = \frac{\text{kB}}{\text{s}}

and therefore:

.. math::

   \text{throughput}_{kbps} = 8 \times \frac{\text{bytes received}}{\text{elapsed ms}}

Mahimahi-formatted traces
~~~~~~~~~~~~~~~~~~~~~~~~~

The folder ``DATASET/NETWORK/norway_mahimahi`` stores Norway traces converted for **Mahimahi** replay.

Mahimahi expects traces as packet-delivery opportunity timestamps rather than direct bandwidth-in-kbps slots. That means the raw Norway logs cannot be used directly by Mahimahi without conversion.

Conceptually, ``convert_norway()`` performs the following steps:

#. Read raw samples from ``norway_raw``.
#. Derive throughput from bytes received and elapsed time.
#. Convert the measurement timeline into replay timestamps expected by Mahimahi.
#. Write the resulting traces into ``norway_mahimahi``.

Converting Norway traces for simulator use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your simulator expects fixed-duration slot JSON rather than Mahimahi replay files, you typically need a second conversion step.

A common workflow is:

#. Start from ``norway_raw``.
#. Compute per-sample throughput.
#. Resample or aggregate the samples into fixed windows such as 500 ms or 1000 ms.
#. Store each window as a slot with ``duration_ms``, ``bandwidth_kbps``, and optionally a default ``latency_ms``.

Example target schema
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   [
     {"duration_ms": 1000, "bandwidth_kbps": 2100, "latency_ms": 80.0},
     {"duration_ms": 1000, "bandwidth_kbps": 1750, "latency_ms": 80.0},
     {"duration_ms": 1000, "bandwidth_kbps": 2450, "latency_ms": 80.0}
   ]

Transport groups
~~~~~~~~~~~~~~~~

The Norway dataset is especially useful because it is organized around realistic transport modes. Each transport family captures different mobility structure and therefore stresses ABR logic differently.

Bus
^^^

The bus traces represent urban road mobility with frequent but moderate variation in coverage quality.

.. figure:: ../PLOT/graphs/bus-map.jpg
   :alt: Bus route map from the Norway dataset
   :width: 82%
   :align: center

   Bus route map from the Norway mobile bandwidth dataset.

.. figure:: ../PLOT/graphs/bus-bw.png
   :alt: Measured bandwidth profile for the bus route
   :width: 82%
   :align: center

   Measured bandwidth profile for the bus route.

Car
^^^

The car traces cover longer mobility corridors and often include both strong coverage periods and prolonged weak-signal stretches.

.. figure:: ../PLOT/graphs/car-map.jpg
   :alt: Car route map from the Norway dataset
   :width: 58%
   :align: center

   Car route map from the Norway mobile bandwidth dataset.

.. figure:: ../PLOT/graphs/car-bw.png
   :alt: Measured bandwidth profile for the car route
   :width: 82%
   :align: center

   Measured bandwidth profile for the car route.

Ferry
^^^^^

The ferry traces add waterfront movement and infrastructure transitions that differ from land-only routes.

.. figure:: ../PLOT/graphs/ferry-map.jpg
   :alt: Ferry route map from the Norway dataset
   :width: 68%
   :align: center

   Ferry route map from the Norway mobile bandwidth dataset.

.. figure:: ../PLOT/graphs/ferry-bw.png
   :alt: Measured bandwidth profile for the ferry route
   :width: 82%
   :align: center

   Measured bandwidth profile for the ferry route.

Metro
^^^^^

The metro traces are among the most challenging because part of the route can be underground or tunnel-heavy.

.. figure:: ../PLOT/graphs/metro-map.jpg
   :alt: Metro route map from the Norway dataset
   :width: 78%
   :align: center

   Metro route map from the Norway mobile bandwidth dataset.

.. figure:: ../PLOT/graphs/metro-bw.png
   :alt: Measured bandwidth profile for the metro route
   :width: 82%
   :align: center

   Measured bandwidth profile for the metro route.

Train
^^^^^

The train traces capture longer rail movement with repeated coverage changes over a corridor.

.. figure:: ../PLOT/graphs/train-map.jpg
   :alt: Train route map from the Norway dataset
   :width: 58%
   :align: center

   Train route map from the Norway mobile bandwidth dataset.

.. figure:: ../PLOT/graphs/train-bw.png
   :alt: Measured bandwidth profile for the train route
   :width: 82%
   :align: center

   Measured bandwidth profile for the train route.

Tram
^^^^

The tram traces reflect stop-and-go urban mobility, with recurring changes in observed throughput but usually less route length than car or train settings.

.. figure:: ../PLOT/graphs/tram-map.jpg
   :alt: Tram route map from the Norway dataset
   :width: 82%
   :align: center

   Tram route map from the Norway mobile bandwidth dataset.

.. figure:: ../PLOT/graphs/tram-bw.png
   :alt: Measured bandwidth profile for the tram route
   :width: 82%
   :align: center

   Measured bandwidth profile for the tram route.

Models
------

The ``DATASET/MODELS`` directory should collect learned checkpoints and model artifacts used by the RL and LLM-based controllers.

Recommended usage in this repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For most readers and developers, the following rules of thumb are practical:

- Use ``MOVIE/*.json`` as the canonical source for segment metadata.
- Use slot-trace JSON when training or running Python simulators and decision servers.
- Use ``norway_raw`` when you need the original mobility measurements or want to create new derived formats.
- Use ``norway_mahimahi`` when running replay-based experiments with Mahimahi.
- Store learned weights and reusable checkpoints under ``MODELS/``.

Artifacts
---------

After a trace is used by a simulator or ABR controller, the run usually produces per-chunk logs, derived plots, and other outputs for downstream analysis.

Simulation output logs
~~~~~~~~~~~~~~~~~~~~~~

**Example file:** ``log_sim_bb_norway_bus_1``

A typical row may look like:

.. code-block:: text

   26680.501461844855    750    4.0    0.8872836624630916    450283    887.2836624630917    -3.065319748591294

A common interpretation is:

#. playback or wall-clock timestamp
#. chosen bitrate or representation label
#. player buffer level in seconds
#. stall / rebuffer increment in seconds
#. downloaded chunk size in bytes
#. chunk download time in milliseconds
#. reward / QoE value

These logs are generally **outputs**, not **inputs**. They are most useful for:

- bitrate-over-time plots
- stall analysis
- QoE summaries
- transport-wise comparisons
- CDFs and tradeoff plots

References and source links
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Norway traces documented on this page are associated with the following papers and source page:

- **Commute Path Bandwidth Traces from 3G Networks: Analysis and Applications**. Haakon Riiser, Paul Vigmostad, Carsten Griwodz, and Pål Halvorsen. *Proceedings of the International Conference on Multimedia Systems (MMSys)*, Oslo, Norway, February/March 2013, pp. 114--118. `Dataset paper (PDF) <https://home.simula.no/~paalh/publications/files/mmsys2013-dataset.pdf>`_. `ACM entry <https://dl.acm.org/doi/10.1145/2483977.2483991>`_.

- **Video Streaming Using a Location-based Bandwidth-Lookup Service for Bitrate Planning**. Haakon Riiser, Tore Endestad, Paul Vigmostad, Carsten Griwodz, and Pål Halvorsen. *ACM Transactions on Multimedia Computing, Communications and Applications (TOMCCAP)*, Vol. 8, No. 3, July 2012, Article No. 24. `Publisher page <https://dl.acm.org/doi/10.1145/2240136.2240137>`_. DOI: ``10.1145/2240136.2240137``.

- **Dataset website**: `HSDPA-bandwidth logs for mobile HTTP streaming scenarios <https://skulddata.cs.umass.edu/traces/mmsys/2013/pathbandwidth/>`_. The page describes the collection setting, route grouping, log format, transport categories, and provides transport maps and measured bandwidth plots.

Summary
~~~~~~~

The ``DATASET/`` directory is not a single format but a small ecosystem of related data representations. Among them, the Norway HSDPA dataset is especially important because it provides a real-world mobility benchmark with timestamped throughput measurements, GPS-aware routes, multiple transport categories, replay-ready Mahimahi conversions, and a clear path to simulator-ready JSON traces.
