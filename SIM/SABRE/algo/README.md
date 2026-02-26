### ABR Algorithms (BOLA, BOLA‑D/Dynamic, BB, RB) + SABRE Simulator

This README documents the **classic non‑RL ABR algorithms** implemented as **SABRE plugins**:

- **BOLA** (Buffer Occupancy based Lyapunov Algorithm) — `bola.py`
- **BOLA‑D / Dynamic** (switches BOLA ↔ Throughput rule) — `bola-d.py`
- **BB** (Buffer‑Based ABR / BBA‑like) — `bb.py`
- **RB** (Rate‑Based ABR, throughput‑only) — `rb.py`

These are **rule‑based** policies (no neural networks). They are ideal baselines to compare against RL policies like PPO/A3C/DQN.

---

## What is SABRE?

**SABRE** is a playback + network simulator for ABR research. It provides:

- A simulated **player buffer** and **chunk download** pipeline
- Hooks for ABR algorithms via a simple plugin API (`select(ctx, seg_idx) -> AbrDecision`)
- Standard outputs: per‑chunk logs (bitrate, buffer, stall, bandwidth, reward/QoE) for plotting/comparison

You run SABRE with:
- a **network trace** (e.g., 4G/5G bandwidth + latency patterns)
- a **movie manifest** (segment duration + size per quality)
- an **ABR plugin** (BOLA/BB/RB/etc.)

![SABRE flow](docs_nonrl/sabre_flow.png)

---

## Quickstart: run SABRE with a plugin

Example command from the BB plugin header:

```bash
python sab.py --plugin bb.py -a bb -n 4Glogs_lum\4g_trace_driving_50015_dr.json -m movie.json --chunk-log-start-ts 1608418123 --chunk-log log_BB_driving_4g -v
```
(From `bb.py`) fileciteturn5file0

General pattern:

```bash
python sab.py   --plugin <plugin_file.py>   -a <abr_name>   -n <trace.json>   -m <movie.json>   --chunk-log <output_prefix>   -v
```

Where:
- `--plugin` points to the plugin implementation (e.g., `bola.py`)
- `-a` selects the registered ABR name (`bola`, `dynamic`, `bb`, `rb`)
- `-n` is your network trace
- `-m` is your movie manifest
- `--chunk-log` writes a chunk‑level TSV log
- `-v` enables verbose output

---

## SABRE plugin interface (mental model)

Each ABR plugin is a class registered via `@register_abr("<name>")` and implements at minimum:

- `first_quality(ctx) -> int` (initial quality)
- `select(ctx, seg_idx) -> AbrDecision(quality, delay_s)`

SABRE provides a `ctx` (context) that includes:
- `ctx.manifest.bitrates_kbps`
- `ctx.buffer.level_ms()` (buffer occupancy in ms)
- `ctx.max_buffer_ms`
- `ctx.throughput_est` and `ctx.latency_est` (if available)

All of your plugins follow this pattern. fileciteturn5file0turn5file1turn5file2turn5file3

---

# Algorithms

## 1) RB — Rate‑Based (throughput‑only)

**Idea:** choose the highest bitrate that is ≤ the current throughput estimate.

Implementation (`rb.py`):

- If throughput is unknown → choose quality 0
- Else increment quality while `bitrates[q+1] <= throughput` fileciteturn5file3

Pros:
- Simple and responsive to bandwidth changes

Cons:
- Can cause **rebuffering** if throughput is noisy or the estimate is optimistic
- Ignores buffer level entirely (no “safety cushion”)

When to use:
- As a baseline, or when buffer is large and throughput estimator is robust

---

## 2) BB — Buffer‑Based (BBA‑like)

**Idea:** choose bitrate based on buffer occupancy.

Your implementation (`bb.py`) follows a standard BBA mapping:

- If `buffer <= reservoir` → choose **lowest** quality
- If `buffer >= reservoir + cushion` → choose **highest** quality
- Else → **linearly map** buffer to a quality index fileciteturn5file0

In code:
- `reservoir_ms = 5000`
- `cushion_ms = 6500` (hardcoded here) fileciteturn5file0

### Optional safety cap (throughput guardrail)

BB includes an optional “safety cap”:

- `SAFETY_CAP = 0.85` by default
- cap quality so that selected bitrate does not exceed `0.85 * throughput_est` fileciteturn5file0

This prevents “crazy overshoots” when buffer is high but bandwidth is not.

Pros:
- Very stable (buffer‑aware), naturally avoids stalls when buffer is low
- Great for smoothing out noisy throughput estimates

Cons:
- Can be slow to ramp up quality if reservoir/cushion are conservative
- Needs reasonable reservoir/cushion tuning per content/segment duration

---

## 3) BOLA — Buffer Occupancy based Lyapunov Algorithm

**Core idea:** optimize a **utility‑per‑bit** score that balances:
- higher video utility (quality)
- buffer occupancy (avoid stalls)
- bitrate cost (download size)

Your code computes a utility per quality:

- utilities are log‑bitrate based: `u(q) = log(bitrate_q) + u0`
- `u0` shifts the utility baseline (two modes: “zero” and “one”)  

BOLA uses a tunable **gp** (gain parameter) and computes `Vp`:

```text
Vp = (buffer_size_ms - segment_time_ms) / (u_max + gp)
```
(see `BolaBase.__init__`)  

### BOLA scoring rule

For each quality `q`, compute:

```text
score(q) = (Vp * (u(q) + gp) - buffer_level_ms) / bitrate_kbps[q]
```
(see `score_quality`)  

Pick the `q` with the maximum score.  

### Anti‑oscillation & safe upgrade logic

Your `Bola.select()` includes practical safeguards:

- it checks throughput feasibility before increasing quality (via `quality_from_throughput`)  
- it can introduce **delay** (sleep) to keep buffer near a target max level for a quality:
  - `delay = max(0, buffer_level - max_level_for(q))`  
- optional “basic” mode disables some adaptivity (`abr_basic`)
- `abr_osc` controls whether to use the delay‑based oscillation control path  

#### Chunk abandonment (and replace)

BOLA can optionally decide to **abandon** the in‑progress chunk and restart at a lower quality when it becomes suboptimal.  
Your code compares the “score” of staying vs switching and returns a lower `q` when beneficial.  

Pros:
- Strong theoretical grounding + good stability
- Buffer‑aware, avoids stalls, controls oscillation

Cons:
- Needs parameters (`gp`, buffer size assumptions) aligned with playback constraints
- More complex than BB/RB

---

### BOLA‑D / Dynamic

Your `Dynamic` policy (`bola-d.py`) combines:
- **BOLA** (buffer‑optimal)
- **ThroughputRule** (rate‑based with safety factors + latency‑aware IBR) 

#### Throughput Rule

- Selects `q = quality_from_throughput(tput * safety_factor)` with `safety_factor = 0.9` 
- Adds **latency‑aware IBR** (if latency estimate is available):
  - computes a `safe_bits` budget based on `(buffer_level - latency) * throughput` and an adaptive safety multiplier 
- Supports **abandonment** if the remaining download time is likely to exceed an `abandon_multiplier * segment_time` 

### Dynamic switching rule

`Dynamic` flips between BOLA and ThroughputRule based on buffer level:

- threshold: `low_buffer_threshold_ms = 10_000`
- if currently using BOLA and buffer drops below threshold and BOLA’s quality < throughput’s quality → switch to throughput
- else if using throughput and buffer is above threshold and BOLA quality ≥ throughput quality → switch to BOLA 

Pros:
- Combines BOLA’s stability with throughput responsiveness
- Tends to avoid low‑buffer traps while still ramping quality efficiently

Cons:
- More parameters and more moving parts (safety factors, thresholds, abandon rules)
- Switching logic can be sensitive if buffer estimates are noisy

---

## Practical comparison (when to use what)

- **RB**: fastest to react, easiest baseline; can stall if estimator is optimistic.
- **BB**: very robust vs stalls; slower ramp‑up; tuning reservoir/cushion matters.
- **BOLA**: theory + strong stability; handles oscillation & can delay/abandon; needs gp/buffer assumptions.
- **BOLA‑D**: best “hybrid” heuristic; often strong in practice; more complex/tunable.

---

## Plots (from SABRE logs)

 plot:

- bitrate over time
- buffer level over time
- stall time distribution (or stall events)
- QoE / reward CDF (which algorithm “wins” most often)
- startup delay distribution (if logged)
- switch frequency (number of quality changes)

---

## To-Do

- extract the exact equations/definitions for BOLA as written in the paper,
- align the terminology to the paper’s notation,
- mapping algo
- and cite specific paper sections/figures in the README.
