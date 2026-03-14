Simulators
==========

- SABR: *A Stable Adaptive Bitrate Framework Using Behavior Cloning Pretraining
  and Reinforcement Learning Fine-Tuning*.
- ABRBench benchmark datasets introduced alongside SABR.
- Repository simulator components: ``sab.py``, plugin-based ABR modules,
  HTTP shims, and player-side integration code.

The main simulator workflow centers on **SABRE**, with supporting notes
for **HTTP/remote ABR evaluation**, **DASH-based player integration**, and related
benchmarking/evaluation workflows.

.. contents::
   :local:
   :depth: 1

Overview
--------

For Adaptive Bitrate (ABR) research, a simulator usually needs to provide four
things:

- a **video manifest** with per-segment sizes at each quality,
- a **network trace** or synthetic network model,
- a **playback/buffer model** that turns downloads into QoE outcomes,
- and an **ABR policy interface** that chooses the next quality.

In this project, SABRE is the main Python-based simulator for these tasks. It can
run classic rule-based algorithms directly as plugins, and it can also act as a
client-side driver for remote/server-hosted ABR policies over HTTP.

SABRE Simulator
---------------

SABRE is a playback + network simulator for ABR research. It simulates a client
player downloading fixed-duration segments over a network trace, maintaining a
buffer, producing per-chunk logs, and invoking an ABR algorithm plugin to select
the next quality.

SABRE is useful for:

- comparing ABR policies under the same movie and trace,
- generating per-chunk logs for later plotting,
- testing rule-based plugins such as RB, BB, and BOLA,
- evaluating server-side controllers through an HTTP shim.

Quickstart
----------

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

Rule-Based Plugins
------------------

These are classic non-RL ABR algorithms commonly used as SABRE plugins:

- **RB**: rate-based (throughput-only)
- **BB**: buffer-based (BBA-like)
- **BOLA**: Lyapunov-style buffer utility optimization
- **Dynamic / BOLA-D**: hybrid switching between BOLA and a throughput rule
- **FastMPC / RobustMPC**: model-predictive approaches using short-horizon
  bandwidth prediction and QoE optimization

These methods do not require neural network checkpoints, which makes them useful
for reproducible baselines, smoke tests, and quick comparisons before adding RL or
LLM-based controllers.

Server-Side and HTTP-Driven Evaluation
--------------------------------------

SABRE can also evaluate **remote ABR controllers** through an HTTP shim/plugin.
In that setup, the simulator remains responsible for playback, buffering, and log
production, while the policy itself runs in a separate process or server.

This is useful for:

- RL-based controllers such as PPO, A3C, or DQN servers,
- MPC servers that keep planning logic outside the simulator,
- experiments where the decision logic is shared by multiple clients,
- debugging request/response payloads independently of the playback loop.

A typical workflow is:

1. start the remote ABR server,
2. run SABRE with the matching shim/plugin,
3. inspect chunk logs and aggregate results,
4. compare the server-based policy with local baselines.

DASH and Player Integration
---------------------------

Beyond offline simulation, ABR logic is often integrated with a **DASH player**
(such as dash.js) or with browser-side instrumentation that emits playback and
network measurements.

In practice, this repository treats the simulator and the player-facing controller
as complementary layers:

- **SABRE** is the offline evaluation engine,
- **DASH / dash.js logic** is the online playback integration layer,
- **HTTP shims** connect the two when a remote controller is used.

This separation helps keep policy logic portable between pure simulation and
player-driven testing.

QUIC and Transport Experiments
------------------------------

The broader simulator/tooling ecosystem may also include transport-level testbeds,
such as QUIC-based experiments, when the goal is to study how ABR interacts with
lower-layer congestion control and delivery dynamics.

Conceptually, transport experiments answer a different question from core SABRE
runs:

- SABRE focuses on **trace-driven playback simulation**,
- transport testbeds focus on **end-to-end protocol behavior**,
- combining both can help validate whether simulator conclusions remain consistent
  when deployed in more realistic networking stacks.

Logs, Metrics, and Plots
------------------------

A simulator is most useful when it produces logs that can be compared across
algorithms and traces. SABRE chunk logs are typically used to derive metrics such
as:

- average bitrate,
- rebuffer time,
- bitrate switches / instability,
- inefficiency,
- QoE-based reward,
- transport or route-specific comparisons across trace groups.

These logs can then be grouped by transport mode, dataset split, or algorithm to
produce the plotting workflows documented elsewhere in the project.

Relation to SABR and Modern ABR Benchmarks
------------------------------------------

Recent ABR research has emphasized that simulator quality depends not just on the
playback model, but also on **wide-coverage trace benchmarks**, **OOD evaluation**,
and **stable training pipelines** for learning-based methods. The SABR paper
frames this as a two-stage learning setup—behavior-cloning pretraining followed by
PPO fine-tuning—and introduces ABRBench-3G and ABRBench-4G+ to evaluate
robustness across broader network distributions.

For this documentation set, that paper is most useful as a reference point for
how a simulator should support:

- trace-driven training and testing,
- consistent video manifests,
- benchmark splits such as train / test / OOD,
- fair comparison between rule-based, RL-based, and LLM-based ABR methods.

In other words, your local simulator docs can stay implementation-focused, while
SABR provides a research-level framing for why benchmark coverage and stable
training matter.

Practical Guidance
------------------

Use **SABRE** when you want:

- a simple and reproducible offline ABR evaluation loop,
- chunk-level logs for plotting,
- direct comparison of RB / BB / BOLA / MPC / RL controllers,
- a common harness for both local and remote policies.

Use **server-side ABR plugins** when you want:

- policy code isolated from the simulator,
- compatibility with RL inference servers,
- easier debugging of state serialization and decision APIs.

Use **player / DASH integration** when you want:

- browser-facing experimentation,
- playback measurements closer to a real player stack,
- validation beyond offline trace replay.

