Plots and Evaluation
====================

SABRE produces per-chunk logs suitable for common ABR evaluation plots.

Recommended plots
-----------------

Time series
~~~~~~~~~~~

- Bitrate vs time
- Buffer level vs time
- Stall events / stall time vs time
- Throughput estimate vs time (if logged)

Distributions
~~~~~~~~~~~~~

- CDF of bitrate
- CDF of reward / QoE (per-session or per-chunk)
- Startup delay distribution (if logged)

Stability / smoothness
~~~~~~~~~~~~~~~~~~~~~~

- Number of quality switches per minute
- Magnitude of quality switches (|Δbitrate|)

Comparing algorithms
--------------------

For fair comparison, keep fixed:

- same manifest
- same network trace(s)
- same buffer size / segment duration settings
- identical logging windows

Then compare summary metrics:

- average bitrate
- total rebuffer time
- switch rate
- QoE aggregate score

Tip: Use multiple traces and report mean/median with error bars.