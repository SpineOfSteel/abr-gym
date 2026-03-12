Dataset
=======

This page describes the datasets used in the project for network-driven adaptive bitrate (ABR) simulation, training, replay, and evaluation. It focuses on **input data formats**, **conversion utilities**, and the **Norway mobile bandwidth dataset** that appears throughout the repository.

The goal of this page is to help readers answer three practical questions quickly:

#. What files exist under ``DATASET/``?
#. What format does each component expect?
#. How do the Norway traces move from raw measurements to Mahimahi replay or simulator-ready traces?

.. contents:: Table of contents
   :local:
   :depth: 2

Overview
--------

The repository uses several kinds of data inputs:

- **Movie manifests** describing available qualities and per-segment sizes.
- **Network traces** describing time-varying bandwidth and latency.
- **Raw mobility logs** containing timestamped measurements collected in the field.
- **Converted replay traces** for tools such as Mahimahi.
- **Simulation logs** produced after running an ABR algorithm on a trace.

Why dataset organization matters
------------------------------

A well-organized dataset layout is essential for ABR research because the same experimental pipeline usually needs to move across **multiple representations of the same underlying scenario**: raw measurements collected in the field, replay-oriented traces for network emulation, simulator-ready traces for training, and post-run logs for analysis. Keeping these representations clearly separated avoids accidental mixing of inputs and outputs, makes experiments easier to reproduce, and helps new contributors understand which files are canonical sources versus which files are derived artifacts.

In practical terms, good dataset organization improves four things:

- **Reproducibility**: readers can trace a result back to the original measurement source.
- **Comparability**: multiple algorithms can be evaluated on the same transport routes and bandwidth conditions.
- **Extensibility**: new converters, trace normalizers, and simulators can be added without changing the raw data.
- **Documentation quality**: route maps, bandwidth profiles, manifests, and trace formats can be explained once and reused consistently across the project.

This is especially important for a repository that combines Mahimahi replay, Python simulators, RL training environments, and plotting utilities. A clean ``DATASET/`` structure makes the data flow understandable before a reader even opens the code.

Data pipeline
~~~~~~~~~~~~~

The most common workflow for the Norway dataset is shown below. This block is intentionally compact so readers can understand the full data path at a glance before diving into format details.

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
   │   ├───fcc
   │   ├── hd_fs/
   │   ├── mahimahi/
   │   ├── norway_raw/
   │   ├── norway_mahimahi/
   │   ├── mahimahi_util.py
   │   └── sd_fs/
   └───TRACES
      ├───driving_4g
      └───norway
Dataset matrix
--------------

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

Movie manifest format
---------------------

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

Network trace formats
---------------------

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

Norway mobile bandwidth dataset
-------------------------------

A central dataset in this repository is the **HSDPA-bandwidth logs for mobile HTTP streaming scenarios** collected in Telenor's 3G/HSDPA mobile network in Norway.

The dataset contains logs from TCP streaming sessions where adaptive video streams were downloaded at maximum speed, with no artificial throttling and no playback buffer cap during collection. The segment duration used during measurement was **2 seconds**. The tests were collected between **2010-09-13** and **2011-04-21**.

This dataset is valuable because it captures **real mobility**, **real route structure**, and **large throughput variation** across different transportation modes. It is therefore useful both for simulator benchmarking and for reproducing realistic stress cases such as tunnel drops, coverage transitions, and fast-changing radio conditions.

The dataset and its analysis are described by Riiser *et al.* in the MMSys 2013 dataset paper, while the earlier TOMCCAP article explains how location-aware bandwidth knowledge can be used for bitrate planning on similar mobile routes. See `dataset paper (PDF) <https://home.simula.no/~paalh/publications/files/mmsys2013-dataset.pdf>`_, `TOMCCAP article <https://dl.acm.org/doi/10.1145/2240136.2240137>`_, and the `dataset website <https://skulddata.cs.umass.edu/traces/mmsys/2013/pathbandwidth/>`_.

Why the Norway dataset is significant
-------------------------------------

The Norway HSDPA dataset is significant because it captures **application-layer throughput under real mobile movement** rather than synthetic or stationary bandwidth traces. The logs were collected on commute routes in and around Oslo across several transport modes, including tram, bus, ferry, metro, train, and a smaller number of car traces. That combination makes the dataset useful for studying how ABR policies behave under realistic coverage transitions, speed changes, and route-specific bandwidth variability. The dataset page hosted by UMass also groups the traces by route and transport type and provides route maps and bandwidth plots, which makes it unusually well suited for both systems work and documentation-driven benchmarking. `Dataset website <https://skulddata.cs.umass.edu/traces/mmsys/2013/pathbandwidth/>`_.

For this repository, the dataset serves several complementary roles:

- as a **grounded source of raw mobility measurements** in ``norway_raw``;
- as a **replay substrate** after conversion to Mahimahi format in ``norway_mahimahi``;
- as a **benchmark family** for comparing controllers across transport types;
- and as a **teaching dataset** because the route maps and bandwidth figures help readers connect network behavior to physical movement.

This also explains why the dataset documentation should stay broad rather than algorithm-specific: the purpose of the page is to document the data, its provenance, its formats, and its experimental value, not to privilege one ABR controller over another.

Repository locations
~~~~~~~~~~~~~~~~~~~~

The repository includes the Norway dataset in multiple forms:

- Raw data: ``DATASET/NETWORK/norway_raw``
- Mahimahi-formatted traces: ``DATASET/NETWORK/norway_mahimahi``
- Conversion utility: ``DATASET/NETWORK/mahimahi_util.py``
- Converter entry point: ``convert_norway()``

These locations correspond to different stages in the workflow:

.. code-block:: text

   norway_raw/
      original field-measurement logs
         -> convert_norway()
            -> norway_mahimahi/
               replay-ready traces for Mahimahi

What the raw dataset contains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The raw archive groups logs by **route** and **transport type**. The transport groups commonly referenced in this project are:

- tram
- car
- bus
- ferry
- metro
- train

The original dataset also includes route maps and measured bandwidth profiles, which are helpful for understanding why some traces are easy and others are failure-heavy.

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

This raw representation is rich because it preserves not only bandwidth observations but also **time alignment** and **GPS positions**, making it suitable for plotting route maps, bandwidth evolution, and mobility-aware analyses.

Mahimahi-formatted traces
-------------------------

The folder ``DATASET/NETWORK/norway_mahimahi`` stores Norway traces converted for **Mahimahi** replay.

Mahimahi expects traces as packet-delivery opportunity timestamps rather than direct bandwidth-in-kbps slots. That means the raw Norway logs cannot be used directly by Mahimahi without conversion.

Why this conversion exists
~~~~~~~~~~~~~~~~~~~~~~~~~~

The raw logs describe measurements such as bytes received over elapsed milliseconds. Mahimahi, by contrast, replays a packet delivery schedule. The conversion step bridges those two worlds by turning measured throughput over time into a sequence of transmission opportunities that approximate the original network behavior.

Converter utility
~~~~~~~~~~~~~~~~~

The repository provides a utility in:

- ``DATASET/NETWORK/mahimahi_util.py``

and the Norway-specific conversion entry point is:

- ``convert_norway()``

Conceptually, ``convert_norway()`` performs the following steps:

#. Read raw samples from ``norway_raw``.
#. Derive throughput from bytes received and elapsed time.
#. Convert the measurement timeline into replay timestamps expected by Mahimahi.
#. Write the resulting traces into ``norway_mahimahi``.

When to use each form
~~~~~~~~~~~~~~~~~~~~~

Use **raw Norway logs** when you want:

- GPS-aware visualizations
- transport-wise route analysis
- custom preprocessing or resampling
- conversion into your own simulator format

Use **Mahimahi traces** when you want:

- packet-level or replay-oriented experiments
- compatibility with Mahimahi networking workflows
- a closer approximation to measured delivery timing

Converting Norway traces for simulator use
------------------------------------------

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

This representation is often easier for RL training environments because:

- it is deterministic and lightweight,
- it aligns cleanly with simulator time steps,
- and it avoids dependence on an external replay tool.

Transport groups
----------------

The Norway dataset is especially useful because it is organized around realistic transport modes. Each transport family captures different mobility structure and therefore stresses ABR logic differently.

Bus
~~~

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

Bus traces are a good benchmark for studying controllers that must react to repeated medium-scale fluctuations without becoming overly unstable.

Car
~~~

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

These traces are useful for evaluating whether an algorithm overcommits during short good periods and then suffers when the route enters a weak-connectivity region.

Ferry
~~~~~

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

Ferry traces are valuable because they often show relatively good connectivity for long portions of the route followed by abrupt degradation, which is challenging for late-session bitrate choices.

Metro
~~~~~

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

These traces are useful for testing whether a player or policy can survive sudden collapses without accumulating severe rebuffering.

Train
~~~~~

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

Train traces are well suited for evaluating long-horizon adaptation, since they mix moderate variability with occasional deeper connectivity drops.

Tram
~~~~

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

Tram traces are helpful when studying stability, since they contain repeated medium-scale changes that can trigger unnecessary oscillation in overly reactive controllers.

Why these transport images matter
---------------------------------

The transport-specific map and bandwidth figures are not just decorative. They provide critical context for interpreting a trace:

- The **map** explains the spatial structure of the route.
- The **bandwidth profile** shows where the route is stable, noisy, or collapse-prone.
- Together, they help explain why a trace family may favor conservative behavior, aggressive opportunism, or stronger stall protection.

For documentation, these figures also make the dataset page easier to navigate because readers can connect each transport name to a concrete mobility pattern.

Simulation output logs
----------------------

After a trace is used by a simulator or ABR controller, the run usually produces per-chunk logs for downstream analysis.

Example
~~~~~~~

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

Recommended usage in this repository
------------------------------------

For most readers and developers, the following rules of thumb are practical:

- Use ``MOVIE/*.json`` as the canonical source for segment metadata.
- Use slot-trace JSON when training or running Python simulators and decision servers.
- Use ``norway_raw`` when you need the original mobility measurements or want to create new derived formats.
- Use ``norway_mahimahi`` when running replay-based experiments with Mahimahi.
- Use simulator output logs only for evaluation and plotting.

References and source links
---------------------------

The Norway traces documented on this page are associated with the following papers and source page:

- **Commute Path Bandwidth Traces from 3G Networks: Analysis and Applications**. Haakon Riiser, Paul Vigmostad, Carsten Griwodz, and Pål Halvorsen. *Proceedings of the International Conference on Multimedia Systems (MMSys)*, Oslo, Norway, February/March 2013, pp. 114--118. `Dataset paper (PDF) <https://home.simula.no/~paalh/publications/files/mmsys2013-dataset.pdf>`_. `ACM entry <https://dl.acm.org/doi/10.1145/2483977.2483991>`_.

- **Video Streaming Using a Location-based Bandwidth-Lookup Service for Bitrate Planning**. Haakon Riiser, Tore Endestad, Paul Vigmostad, Carsten Griwodz, and Pål Halvorsen. *ACM Transactions on Multimedia Computing, Communications and Applications (TOMCCAP)*, Vol. 8, No. 3, July 2012, Article No. 24. `Publisher page <https://dl.acm.org/doi/10.1145/2240136.2240137>`_. DOI: ``10.1145/2240136.2240137``.

- **Dataset website**: `HSDPA-bandwidth logs for mobile HTTP streaming scenarios <https://skulddata.cs.umass.edu/traces/mmsys/2013/pathbandwidth/>`_. The page describes the collection setting, route grouping, log format, transport categories, and provides transport maps and measured bandwidth plots.

Summary
-------

The ``DATASET/`` directory is not a single format but a small ecosystem of related data representations. Among them, the Norway HSDPA dataset is especially important because it provides a real-world mobility benchmark with:

- timestamped throughput measurements,
- GPS-aware routes,
- multiple transport categories,
- replay-ready Mahimahi conversions,
- and a clear path to simulator-ready JSON traces.

By keeping the raw logs, converted Mahimahi traces, and transport visuals together in the documentation, the dataset page can serve both as a **reference** and as a **starting point** for new experiments.
