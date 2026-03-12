
# ABR Plot Library

A consolidated plotting library for ABR experiments, 
## What this fixes

### 1) Constants now live in one place
Instead of hard-coding different values in every script, `ABRPlotConfig` centralizes:
- `video_len`
- `num_bins`
- bitrate ladder (`video_bitrates_kbps`)
- unit conversions
- QoE penalties (`rebuffer_penalty`, `smooth_penalty`)
- input/output folders

### 2) Conflicting assumptions are now explicit
The old files used different values depending on dataset or experiment:
- `VIDEO_LEN = 48` in some scripts
- `VIDEO_LEN = 64` in others
- 5-level bitrate ladder in one script
- 6-level bitrate ladder in another
- CDF normalized in some plots, unnormalized in others
- `sim_dp` reward treated as a scalar in one file and reconstructed chunk-wise in another

The library makes those choices configurable instead of silently mixing them.

### 3) One parser supports multiple log styles
Supported:
- standard per-line ABR logs with explicit reward

## Main API

```python
from abr_plot_lib import (
    ABRPlotConfig,
    load_sessions,
    aggregate_metrics,
    plot_reward_by_trace,
    plot_qoe_cdf,
    plot_tradeoff_scatter,
    plot_session_panel,
    save_png,
)
```

## Quick start

```python
from abr_plot_lib import ABRPlotConfig, build_default_report

cfg = ABRPlotConfig(
    results_dir="./results",
    output_dir="./figures",
    video_len=48,
    video_bitrates_kbps=[300, 750, 1200, 1850, 2850, 4300],
)

build_default_report("./results", "./figures", cfg=cfg, schemes=["BB", "BOLA", "RL"])
```

This produces:
- `reward_by_trace.png`
- `qoe_cdf.png`
- `bitrate_vs_stall.png`
- `smoothness_vs_stall.png`
- `bitrate_vs_smoothness.png`
- `session_panel.png` (when aligned sessions exist)

## Example script

Run:

```bash
python example_usage.py
```

It creates a small synthetic dataset and exports multiple PNGs into:

```bash
/mnt/data/example_figures
```

## Notes on session alignment

Many of the old scripts compare only sessions that exist for **all** schemes.  
This behavior is preserved through `common_session_ids(...)` and `aggregate_metrics(...)`.

## Notes on reward handling

For standard logs:
- reward is read directly from column 7

For `sim_dp` logs:
- bitrate is reconstructed from quality index using `video_bitrates_kbps`
- reward is approximated chunk-wise using bitrate, inferred stall, and smoothness penalty
- if a session-level reward exists, the library adjusts the reconstructed vector so the summed reward remains consistent

That keeps the comparison usable while still exposing the assumption in one place.

## Typical custom plots

### Reward by trace

```python
fig, ax = plot_reward_by_trace(metrics)
save_png(fig, "reward_by_trace.png")
```

### QoE CDF

```python
fig, ax = plot_qoe_cdf(metrics)
save_png(fig, "qoe_cdf.png")
```

### Bitrate vs stall ratio

```python
fig, ax = plot_tradeoff_scatter(metrics, x="stall_ratio_pct", y="mean_bitrate_mbps")
save_png(fig, "bitrate_vs_stall.png")
```

### One-session multi-panel view

```python
fig, axes = plot_session_panel(sessions, session_id="trace1.txt", schemes=["BB", "BOLA", "RL"])
save_png(fig, "session_panel.png")
```
