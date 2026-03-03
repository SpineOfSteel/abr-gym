SABRE Simulator
===============

SABRE is a playback + network simulator for ABR research. It simulates a client
player downloading fixed-duration segments over a network trace, maintaining a
buffer, producing per-chunk logs, and invoking an ABR algorithm plugin to select
the next quality.

This section covers:

- How to run SABRE with a plugin
- Rule-based ABR plugins (BB/RB/BOLA/Dynamic)
- HTTP SHIM plugins for server-based ABR (RL or MPC servers)
- Common plots and evaluation tips



Quickstart
---------------

SABRE is typically run with:

- a **network trace** (JSON)
- a **movie manifest** (segment duration + per-quality segment sizes)
- an **ABR plugin** (Python file)

General pattern:

.. code-block:: bash

   python sab.py \
     --plugin <plugin_file.py> \
     -a <abr_name> \
     -n <trace.json> \
     -m <movie.json> \
     --chunk-log <output_prefix> \
     -v

Recommended flags:

- ``-v``: verbose logs
- ``--debug_p``: plugin debug prints (if the plugin supports it)
- ``--chunk-log-start-ts``: set a start timestamp for aligned plotting
- ``--chunk-log``: writes a per-segment TSV for analysis

Example (Buffer-Based / BB):

.. code-block:: bash

   python sab.py --plugin bb.py -a bb \
     -n 4Glogs_lum\\4g_trace_driving_50015_dr.json \
     -m movie.json \
     --chunk-log-start-ts 1608418123 \
     --chunk-log log_BB_driving_4g \
     -v

Rule-based ABR Plugins
---------------

These are classic non-RL ABR algorithms implemented as SABRE plugins:

- **RB**: rate-based (throughput-only)
- **BB**: buffer-based (BBA-like)
- **BOLA**: Lyapunov-style buffer utility optimization
- **Dynamic (BOLA-D)**: hybrid switching between BOLA and throughput rule

They require no neural networks and are ideal baselines to compare against RL.
