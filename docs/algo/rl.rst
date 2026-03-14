RL Algorithms
=============


This chapter documents the **reinforcement-learning ABR methods** used in the repository. Unlike the rule-based methods in ``standard.rst``, these approaches learn a bitrate-selection policy from interaction data, simulated traces, or offline trajectories.

At a high level, ABR can be modeled as a sequential decision problem in which the agent observes network and playback state, chooses the next representation, and receives a QoE-driven reward.

A representative reward used across several implementations is:

.. math::

   r_t = \frac{b_t}{1000}
         - \alpha \, \mathrm{rebuf}_t
         - \beta \, \frac{|b_t - b_{t-1}|}{1000}

where :math:`b_t` is the chosen bitrate in kbps, :math:`\mathrm{rebuf}_t` is rebuffering in seconds, and :math:`\alpha, \beta` are the rebuffer and smoothness penalties.

References
----------

- Mnih, V. et al. *Playing Atari with Deep Reinforcement Learning*. 2013.
- Mnih, V. et al. *Human-level control through deep reinforcement learning*. Nature, 2015.
- Dabney, W. et al. *Distributional Reinforcement Learning with Quantile Regression*. AAAI, 2018.
- Mnih, V. et al. *Asynchronous Methods for Deep Reinforcement Learning*. ICML, 2016.
- Schulman, J. et al. *Trust Region Policy Optimization*. ICML, 2015.
- Schulman, J. et al. *Proximal Policy Optimization Algorithms*. 2017.
- Mania, H. et al. *Simple random search provides a competitive approach to reinforcement learning*. NeurIPS, 2018.
- Lillicrap, T. et al. *Continuous control with deep reinforcement learning*. 2015.
- Fujimoto, S. et al. *Addressing Function Approximation Error in Actor-Critic Methods*. ICML, 2018.
- Haarnoja, T. et al. *Soft Actor-Critic*. ICML, 2018.

.. contents::
   :local:
   :depth: 1

Overview by algorithm family
----------------------------

The algorithms on this page can be grouped roughly as follows:

- **Discrete value-based methods:** DQN, QR-DQN
- **On-policy actor-critic methods:** A3C, A2C, PPO, RecurrentPPO, TRPO
- **Gradient-free baselines:** ARS
- **Continuous-control actor-critic methods:** DDPG, TD3, SAC

For classic bitrate-ladder ABR, the most natural starting points are usually **DQN** and **PPO**, with **RecurrentPPO** becoming attractive when temporal context is important.


Quickstart
----------

Train a DQN agent:

.. code-block:: bash

   python -m abrGym train gym_dqn \
     --train-trace DATASET/NETWORK/fcc-train \
     --video-meta DATASET/MOVIE/movie_4g.json \
     --model-path DATASET/MODELS/gym_dqn_abr_gym.zip \
     --tensorboard-log DATASET/tb_logs \
     --total-timesteps 50000 \
     --seed 42


Test a PPO agent:

.. code-block:: bash

   python -m abrGym test gym_ppo \
     --split test \
     --test-trace DATASET/NETWORK/fcc-test \
     --video-meta DATASET/MOVIE/movie_4g.json \
     --model-path DATASET/MODELS/gym_ppo_abr_gym.zip \
     --n-eval-episodes 10 \
     --seed 44

Roll out a recurrent PPO policy:

.. code-block:: bash

   python -m abrGym rollout gym_recurrent_ppo \
     --split test \
     --test-trace DATASET/NETWORK/fcc-test \
     --video-meta DATASET/MOVIE/movie_4g.json \
     --model-path DATASET/MODELS/gym_recurrent_ppo_abr_gym.zip \
     --max-steps 200 \
     --seed 44

Environment
-----------

:class:`Environment` simulates chunk downloads over a time-varying network. Inputs are:

- bandwidth in kbps per slot
- slot duration in ms
- slot latency in ms
- per-quality chunk sizes (bytes)

It maintains internal state:

- current trace index and slot pointer
- progress within the current slot (``slot_offset_ms``)
- current buffer level (ms)
- current video chunk index

See ...

 :class:`ABREnv`, a training environment wrapper that:

- loads a trace JSON (your trace format)
- loads a video bitrate ladder
- constructs chunk sizes (CBR approximation)
- runs the network simulator (:class:`SERVER.EnvNetwork.Environment`)
- maintains a Pensieve-style state (6×8)
- computes QoE reward and returns ``(state, reward, done, info)``

Utils like check_env, Monitor and evaluate_policy help check correctness and monitor environment variables for debugging and logging.

DQN
---

Plugin / workflow name: ``gym_dqn``

DQN treats ABR as a **discrete-action value-learning** problem. Each action corresponds to choosing one representation from the bitrate ladder. The model learns an action-value function :math:`Q(s, a)` and selects the action with the highest predicted long-term return.

The Bellman target is:

.. math::

   y_t = r_t + \gamma \max_{a'} Q_{\text{target}}(s_{t+1}, a')

The learned policy is then:

.. math::

   a_t^* = \arg\max_a Q(s_t, a)

Strengths
~~~~~~~~~

- natural fit for discrete bitrate ladders
- simple inference path at deployment time
- strong baseline for value-based ABR learning
- works well when the action set is fixed and small

Limitations
~~~~~~~~~~~

- can be sensitive to reward scaling and exploration design
- may learn unstable values without careful target-network updates
- less natural when extending to continuous or structured actions

QR-DQN
------

QR-DQN extends DQN from predicting a single expected return to predicting a **distribution of returns** through quantile regression. In ABR, this can better capture uncertainty across volatile network traces and diverse QoE outcomes.

Conceptually, instead of learning just :math:`Q(s,a)`, the model learns a set of quantile values for each action. This can make the value estimate richer than standard DQN while keeping the action space discrete.

Typical use in ABR
~~~~~~~~~~~~~~~~~~

- stronger value-based modeling under high network variance
- useful when mean return alone hides tail-risk behaviors
- natural upgrade path from plain DQN-style implementations

PPO
---

Plugin / workflow name: ``gym_ppo``

PPO is a **policy-gradient actor-critic** method that directly optimizes a stochastic policy while constraining policy updates to remain stable. It is often a strong default for ABR because it combines practical robustness with relatively simple training.

The clipped PPO objective is:

.. math::

   L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t \Big[
   \min\big(
   r_t(\theta) \hat{A}_t,
   \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t
   \big)
   \Big]

where :math:`r_t(\theta)` is the policy probability ratio and :math:`\hat{A}_t` is the estimated advantage.

Strengths
~~~~~~~~~

- stable and widely used in practice
- well supported in Stable-Baselines3
- good balance between performance and implementation simplicity
- often easier to tune than older asynchronous methods

Limitations
~~~~~~~~~~~

- still requires careful reward normalization and rollout design
- may need many environment interactions for strong performance
- recurrent variants add engineering complexity

A2C
---

A2C is the synchronous successor to A3C-style actor-critic training. Instead of many asynchronous worker updates, it batches rollouts from parallel environments and performs coordinated gradient steps.

For ABR, A2C keeps the actor-critic structure simple while reducing some of the implementation complexity associated with asynchronous training.

Why it matters for ABR
~~~~~~~~~~~~~~~~~~~~~~

- simpler modern baseline than classic A3C
- works naturally with vectorized environments
- useful when you want an actor-critic method lighter than PPO

RecurrentPPO
------------

Plugin / workflow name: ``gym_recurrent_ppo``

RecurrentPPO augments PPO with recurrent state, typically an LSTM, so the policy can retain temporal context beyond a fixed observation window. This is especially relevant in ABR because throughput, stalls, and buffer dynamics are strongly sequential.

Why it matters for ABR
~~~~~~~~~~~~~~~~~~~~~~

- captures temporal patterns not obvious from one-step observations
- useful for partial observability and bursty networks
- often a strong choice when history matters more than instantaneous state

TRPO
----

TRPO is a trust-region policy-gradient method that constrains each policy update to remain within a safe step size. PPO is often viewed as a more practical approximation of this idea.

In ABR, TRPO is important conceptually because it motivated the stable-update design later popularized by PPO.

Why it matters for ABR
~~~~~~~~~~~~~~~~~~~~~~

- emphasizes conservative, stable policy improvement
- historically important in policy optimization
- often less convenient operationally than PPO

ARS
---

ARS, or Augmented Random Search, is a gradient-free optimization method that updates a policy using random search directions and return comparisons. It is much simpler than many deep RL algorithms.

In ABR, ARS can be attractive as a lightweight experimental baseline when you want to test whether a simpler optimization method is already sufficient.

Why it matters for ABR
~~~~~~~~~~~~~~~~~~~~~~

- simple and fast baseline to prototype
- fewer moving parts than actor-critic methods
- may be useful for low-complexity policies or ablations

SAC
---

SAC, or Soft Actor-Critic, is an off-policy actor-critic method that optimizes both reward and policy entropy. It is best known for continuous-control tasks, but the general design is still relevant when discussing richer ABR action formulations.

In a pure bitrate-ladder ABR setup, SAC is less direct than DQN or PPO because the action space is usually discrete. Still, it becomes more interesting when actions include continuous control variables or hybrid server-side decisions.

Why it matters for ABR
~~~~~~~~~~~~~~~~~~~~~~

- sample-efficient off-policy learning
- strong exploration due to entropy regularization
- more natural for continuous or hybrid action settings

TD3
---

TD3, or Twin Delayed DDPG, improves deterministic actor-critic learning by reducing value overestimation and stabilizing training. Like SAC and DDPG, it is mainly associated with continuous control.

In ABR, TD3 is most relevant when the decision variable is extended beyond selecting from a fixed bitrate ladder.

Why it matters for ABR
~~~~~~~~~~~~~~~~~~~~~~

- stronger continuous-control baseline than plain DDPG
- useful when ABR actions become continuous or parameterized
- less natural for plain discrete bitrate selection

DDPG
----

DDPG is a deterministic off-policy actor-critic algorithm designed for continuous action spaces. It combines a deterministic policy with a learned Q-function and target networks.

For standard ABR with a small discrete bitrate ladder, DDPG is usually not the first choice. It is more relevant in extended formulations such as server-assisted control, bitrate scaling factors, or multi-parameter adaptation decisions.

Why it matters for ABR
~~~~~~~~~~~~~~~~~~~~~~

- foundational continuous-control actor-critic baseline
- useful for non-discrete ABR formulations
- often superseded by TD3 or SAC in modern practice

A3C
---

Legacy / reference family: asynchronous actor-critic / Pensieve-style training

A3C is the classical **asynchronous actor-critic** formulation used by early RL-based ABR systems such as Pensieve-style pipelines. Multiple workers interact with simulated environments in parallel and push gradient updates to shared global parameters.

A typical actor objective is based on the policy gradient, while the critic estimates value to reduce variance. A3C remains historically important even when modern codebases prefer PPO or A2C for easier maintenance and tooling.

Why it matters for ABR
~~~~~~~~~~~~~~~~~~~~~~

- foundational RL approach for Pensieve-style ABR
- historically important for learned bitrate adaptation
- useful reference point when comparing newer actor-critic methods

