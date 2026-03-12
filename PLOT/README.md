
# Plotting Results with `abrGym`

The `abrGym plot` command generates evaluation plots for **Adaptive Bitrate (ABR) algorithms** using simulation results.

Plots can be generated from:

- **Parquet summary files** (recommended)
- **Directories containing simulation logs (`.txt`)**

These plots help compare algorithms across **network conditions and transport groups**, allowing quick evaluation of streaming quality metrics such as bitrate, stall events, and overall QoE.

---

# Command Overview

Basic usage:

```bash
python -m abrGym plot [OPTIONS]
```

Common options:

| Option | Description |
|------|-------------|
| `--source` | Folder containing log files or a `.parquet` results file |
| `--output-dir` | Directory to store generated plots |
| `--algo` | Algorithms to include |
| `--group` | Transport groups to include |
| `--plot` | Plot types to generate |
| `--include-all` | Generate aggregated plots across all groups |

If `--algo` or `--group` is not specified, **all default algorithms and groups are included**.

---

# Default Algorithms

```
bb
bola
mpc
rl
ppo
netllm
```

These include:

- rule‑based ABR algorithms
- model‑predictive control methods
- reinforcement learning methods
- LLM‑based adaptation approaches

---

# Default Transport Groups

```
tram
car
bus
ferry
metro
train
```

These correspond to network traces collected under different mobility scenarios.

---

# Available Plot Types

```
tradeoff
smoothness
bitrate
stall
qoe
```

Use:

```
--plot all
```

to generate every plot type.

---

# Plot Descriptions

## Tradeoff Plot (Bitrate vs Stall)

The **tradeoff plot** visualizes the relationship between **video bitrate and stall ratio** for each algorithm.

Higher bitrate improves visual quality but can increase the risk of buffering if the network cannot sustain it. ABR algorithms must therefore balance quality and playback stability. This plot shows each algorithm’s mean bitrate and stall percentage along with variability across traces, making it easy to identify algorithms that deliver high quality while minimizing rebuffering.

Output example:

```
baselines-bus-tradeoff.png
```

---

## Bitrate Plot

The **bitrate plot** shows the **average video bitrate achieved across network traces**.

Higher bitrate generally corresponds to better visual quality, but algorithms that aggressively select high bitrates may cause buffering if bandwidth fluctuates. This plot helps evaluate how consistently each algorithm maintains high-quality video across different network environments.

Output example:

```
baselines-bus-br.png
```

---

## Stall Plot

The **stall plot** measures the **percentage of playback time spent stalled (rebuffering)**.

Stall events occur when the playback buffer empties and the player must wait for new segments to download. Rebuffering significantly impacts user experience, often more than small changes in visual quality. This plot helps identify algorithms that maintain smooth playback under varying network conditions.

Output example:

```
baselines-bus-st.png
```

---

## Smoothness Plot

The **smoothness plot** measures **bitrate variation between consecutive segments**.

Frequent quality switches can degrade perceived video quality even if the average bitrate is high. This metric captures the stability of an algorithm's bitrate decisions, rewarding algorithms that maintain consistent quality rather than oscillating between representations.

Output example:

```
baselines-bus-sr.png
```

---

## QoE CDF Plot

The **QoE CDF plot** shows the **distribution of Quality of Experience (QoE) scores** across network traces.

QoE typically combines several metrics such as:

- bitrate
- rebuffering penalties
- quality switching penalties

By plotting the cumulative distribution function (CDF), this graph shows how frequently an algorithm achieves higher QoE compared to others, providing a clear summary of overall performance across traces.

Output example:

```
baselines-bus-qoe.png
```

---

# Basic Usage

Run plotting using default configuration:

```bash
python -m abrGym plot
```

---

# Using a Parquet Results File

Example:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet
```

Specify output directory:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --output-dir graphs/norway
```

Generate all plots including aggregated results:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --output-dir graphs/norway --plot all --include-all
```

---

# Selecting Algorithms

Compare specific algorithms:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --algo bb bola ppo
```

Compare classic and learning-based algorithms:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --algo bb bola mpc rl ppo netllm
```

---

# Selecting Transport Groups

Plot only bus traces:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --group bus
```

Multiple groups:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --group tram car train
```

---

# Combined Filters

Example comparing algorithms on selected groups:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --algo bb bola ppo netllm --group bus tram train --plot tradeoff
```

---

# Using Log Files Instead of Parquet

You can also generate plots directly from simulation logs.

Example:

```bash
python -m abrGym plot --source DATASET/TRACES/norway
```

Recursive search with custom output directory:

```bash
python -m abrGym plot --source DATASET/TRACES/norway --output-dir graphs/from_logs --recursive --include-all
```

---

# Typical Output Files

Each transport group generates plots such as:

```
baselines-bus-tradeoff.png
baselines-bus-br.png
baselines-bus-st.png
baselines-bus-sr.png
baselines-bus-qoe.png
```

If `--include-all` is enabled, additional aggregated plots are generated:

```
baselines-all-tradeoff.png
baselines-all-qoe.png
```

---

# Recommended Workflow

Generate all plots for a full experiment:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --output-dir graphs/paper --algo bb bola mpc rl ppo netllm --group tram car bus ferry metro train --plot all --include-all
```

Quick sanity check:

```bash
python -m abrGym plot --source DATASET/artifacts/norway/results.all.parquet --plot tradeoff qoe
```
