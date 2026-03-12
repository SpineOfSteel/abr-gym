Algorithms
==========

This documentation collects the Adaptive Bitrate (ABR) algorithms implemented in this
repository across classical, control-based, reinforcement-learning, and LLM-based
approaches.

ABR algorithms solve the same core problem: for each video chunk, choose the next
bitrate level using recent network measurements, current buffer occupancy, and video
progress so that overall Quality of Experience (QoE) is maximized.

In this repository, the algorithm stack is organized into four broad families:

- **Classic / player-side ABR**: lightweight local decision rules such as rate-based,
  buffer-based, and hybrid heuristics.
- **MPC / control-based ABR**: look-ahead optimization methods that evaluate future
  chunk decisions under predicted bandwidth.
- **RL-based ABR**: policies learned from interaction with the ABR environment using
  actor-critic or value-based reinforcement learning.
- **LLM-based ABR**: large-language-model adaptation for networking decisions, with the
  current documented experiment focus on **Llama2 7B**.

Documentation map
-----------------

Use the sections below depending on what you want to inspect:

- **Classic ABR** for practical heuristic baselines such as RB, BB, BOLA, Dynamic, and
  related player-style algorithms.
- **RL-based** for DQN, PPO, Pensieve/A3C, and the shared RL environment structure.
- **LLM-based** for NetLLM-style adaptation and PLM-backed ABR policies.

.. toctree::
   :maxdepth: 2

   classic
   bola
   dynamic
   mpc
   remote
   rl
   dqn
   pensieve
   ppo
   llm

Algorithm families
------------------

Classic / heuristic ABR
~~~~~~~~~~~~~~~~~~~~~~~

These algorithms rely on explicit rules rather than learned policies.

- **RB**: rate-based adaptation using recent bandwidth estimates.
- **BB / BBA-like**: buffer-based quality selection using occupancy thresholds.
- **BOLA**: utility-driven buffer-based control with a score that balances quality,
  buffer, and bitrate cost.
- **Dynamic**: hybrid logic that combines throughput and buffer reasoning.
- **Festive**: throughput-based adaptation with harmonic mean estimation and switching
  control.

MPC / control-based ABR
~~~~~~~~~~~~~~~~~~~~~~~

These methods evaluate several future decisions before selecting the next chunk quality.

- **FastMPC**: lightweight server-side MPC-style bitrate selection.
- **RobustMPC**: model-predictive control with robust throughput estimation over a
  short horizon.

RL-based ABR
~~~~~~~~~~~~

These methods learn a policy directly from reward.

- **Pensieve / A3C**: asynchronous actor-critic ABR.
- **PPO**: clipped actor-critic optimization for stable policy improvement.
- **DQN**: value-based learning over the discrete bitrate ladder.
- The broader RL overview page also discusses **A2C**, **SAC**, **DDPG**, and **TD3**
  in the shared ABR environment.

LLM-based ABR
~~~~~~~~~~~~~

These methods adapt pretrained language models for networking control.

- **LLM / NetLLM**: PLM-backed ABR using offline trajectories, state encoding, and
  parameter-efficient adaptation.
- The implementation supports several PLM families, but the documented experiment focus
  is **Llama2 7B**.

Repository locations
--------------------

The code is split across simulator, server, and training components.

- **SABRE plugins**: ``SIM/SABRE/algo/*.py``
- **DASH / player-side logic**: ``SIM/DASH/src/algo/*``
- **RL training and inference servers**: ``SERVER/pensieve``, ``SERVER/ppo``,
  ``SERVER/dqn``
- **MPC servers**: ``SERVER/fastmpc_server.py``, ``SERVER/robustmpc_server.py``

Suggested reading order
-----------------------

If you are new to the repository, a good reading path is:

1. Start with ``classic`` for the baseline ABR problem setup.
2. Read ``bola`` and ``mpc`` for strong non-RL baselines.
3. Read ``rl`` for the shared Gym / environment / reward design.
4. Then inspect algorithm-specific pages such as ``dqn``, ``pensieve``, and ``ppo``.
5. Finish with ``llm`` for NetLLM-style PLM adaptation.
