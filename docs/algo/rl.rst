
RL Algorithms
=============

.. contents::
   :local:
   :depth: 2

This chapter documents the **Gymnasium / Stable-Baselines3 / SB3-Contrib** reinforcement learning workflows used in the ABR toolkit. It focuses on the modern training-and-evaluation pipeline built around ``AbrStreamingEnv`` and held-out datasets such as ``fcc-train``, ``fcc-valid``, and ``fcc-test`` in **Mahimahi trace format**. The older shim-server workflows (Pensieve server, PPO server, DQN server, FastMPC server, RobustMPC server) should be documented separately in ``rl_shim.rst``. Standard rule-based methods and LLM-based methods should likewise remain in their own chapters.

The goal of this page is twofold:

1. provide a **practical guide** for training, testing, and analyzing ABR RL agents; and
2. present the methods in a more **academic, equation-centered** way so the design choices are easy to justify in papers, reports, and READTHEDOCS-style documentation.

ABR as a RL problem?
--------------------

Adaptive Bitrate (ABR) selection is naturally sequential. At each chunk decision time, the client observes a partial history of bandwidth, delay, buffer occupancy, future chunk sizes, and video progress, then selects a representation level. The current decision influences not only immediate quality, but also future rebuffering, smoothness, and robustness to network variation.

A standard ABR RL formulation is:

- **state**: recent network, playback, and video metadata history
- **action**: representation selection or a continuous control variable mapped to representation selection
- **reward**: a QoE objective balancing quality, rebuffering, and smoothness
- **transition**: determined by the network trace, chunk size, and buffer dynamics

The key reward used throughout this chapter is highlighted repeatedly because it is the central object tying together the algorithmic discussion, evaluation protocol, and trade-off analysis:

.. math::

   r_t = \frac{b_t}{1000} - \alpha \cdot \mathrm{rebuf}_t - \beta \cdot \frac{|b_t - b_{t-1}|}{1000}

where:

- :math:`b_t` is the selected bitrate in kbps,
- :math:`\mathrm{rebuf}_t` is rebuffer time in seconds,
- :math:`\alpha` is the rebuffer penalty,
- :math:`\beta` is the smoothness penalty.

In the examples used throughout this documentation, a common choice is:

- :math:`\alpha = 4.3`
- :math:`\beta = 1.0`

This reward appears again in the algorithm-specific sections because **every model ultimately optimizes some version of this QoE trade-off**.

High-level workflow
-------------------

The Gym/SB3 workflow separates **training**, **validation**, and **testing**. A typical layout is:

::

   DATASET/
   ├── NETWORK/
   │   ├── fcc-train/
   │   ├── fcc-valid/
   │   └── fcc-test/
   └── MOVIE/
       └── movie_4g.json

All three FCC splits are expected to be in **Mahimahi format**. In practice, this means the network traces are replayable time-series traces describing bandwidth over time, and the environment uses them to simulate chunk download delays, buffer drain, and playback progression.

A simple conceptual picture is:

::

   +-------------------+       action a_t        +---------------------+
   |  RL policy/model  |  -------------------->  |   AbrStreamingEnv   |
   +-------------------+                         +---------------------+
             ^                                              |
             |                                              v
             |                state s_{t+1}, reward r_t     |
             +----------------------------------------------+

The environment encapsulates the ABR state update, Mahimahi trace replay, and QoE reward calculation.

Dataset protocol and evaluation splits
--------------------------------------

The ``fcc-train`` / ``fcc-valid`` / ``fcc-test`` split should be treated as a real experimental protocol rather than a convenience folder layout.

- ``fcc-train`` is used for learning policy parameters.
- ``fcc-valid`` is used for hyperparameter tuning, model selection, or early diagnostics.
- ``fcc-test`` is held out for final reporting only.

This matters because ABR agents can overfit to trace characteristics. Even when the reward equation remains unchanged,

.. math::

   r_t = \frac{b_t}{1000} - \alpha \cdot \mathrm{rebuf}_t - \beta \cdot \frac{|b_t - b_{t-1}|}{1000}

the empirical outcome can change substantially depending on the trace distribution seen during training.

A second important protocol choice concerns episode initialization:

- ``fixed_start = False`` is appropriate for **training**, because it increases diversity by sampling different starting positions in traces.
- ``fixed_start = True`` is appropriate for **evaluation**, because it makes results more reproducible and easier to compare across checkpoints.

.. note::

   In the Gym workflow, training should typically use ``fixed_start=False`` and testing should use ``fixed_start=True``. This should be enforced by environment construction, not by ad-hoc manual edits at evaluation time.

State, action, and reward design
--------------------------------

State representation
^^^^^^^^^^^^^^^^^^^^

The ABR environment uses a structured state rather than a raw flattened vector. A common shape is ``(6, 8)``, where the rows encode different semantic quantities over a recent temporal window. Typical components are:

- normalized selected bitrate
- normalized buffer size
- measured throughput
- measured download delay
- next chunk sizes for available qualities
- fraction of chunks remaining

A conceptual view:

::

   state_t =
   [ bitrate history         ]   row 0
   [ buffer history          ]   row 1
   [ throughput history      ]   row 2
   [ delay history           ]   row 3
   [ next chunk sizes        ]   row 4
   [ chunks-remaining signal ]   row 5

This structure motivates custom feature extractors that do **not** simply flatten the state, but instead process semantically different rows differently.

Action design
^^^^^^^^^^^^^

Two action designs are used:

**Discrete action mode**
   The action is an integer bitrate index. This is the natural setting for DQN, QR-DQN, PPO, A2C, RecurrentPPO, and TRPO.

**Continuous action mode**
   The action is a scalar or low-dimensional continuous control variable, typically in :math:`[0,1]`, which is then mapped back to a bitrate level. This is used to make algorithms such as ARS, SAC, TD3, and DDPG runnable in the same ABR environment.

The continuous variant should be understood as a wrapper around the same underlying representation-selection problem.

Reward design
^^^^^^^^^^^^^

The QoE reward is repeated here intentionally because it is the core design contract between the environment and all algorithms:

.. math::

   r_t = \frac{b_t}{1000} - \alpha \cdot \mathrm{rebuf}_t - \beta \cdot \frac{|b_t - b_{t-1}|}{1000}

Interpretation:

- the first term rewards bitrate,
- the second penalizes stalls heavily,
- the third penalizes rapid quality changes.

This is the same trade-off that drives most ABR RL work in practice.

Training, testing, and simulation
---------------------------------

Training
^^^^^^^^

A typical Gym/SB3 training loop looks like this:

1. construct the environment on ``fcc-train``
2. instantiate the algorithm with the chosen policy / feature extractor
3. call ``learn(total_timesteps=...)``
4. save a model checkpoint under ``MODELS/<name>_abr_gym.zip``

Testing
^^^^^^^

Testing should be performed on ``fcc-test`` with ``fixed_start=True``. The most common pattern is:

1. load the trained model
2. construct a test environment
3. call ``evaluate_policy(...)`` for multiple episodes
4. optionally run a manual rollout for qualitative inspection

Simulation
^^^^^^^^^^

In this chapter, “simulation” refers to **environment-driven policy evaluation** inside the Gym/SB3 stack. That is distinct from the older shim-server workflows documented in ``rl_shim.rst``. In the Gym workflow, simulation happens directly through:

- ``env.step(action)``
- ``model.predict(obs, deterministic=True)``

rather than through an external HTTP server loop.

CLI examples
------------

Representative CLI commands are shown below.

Train a DQN model:

.. code-block:: bash

   python -m abrGym train gym_dqn \
     --train-trace DATASET/NETWORK/fcc-train \
     --video-meta DATASET/MOVIE/movie_4g.json \
     --model-path DATASET/MODELS/gym_dqn_abr_gym.zip \
     --tensorboard-log DATASET/tb_logs \
     --total-timesteps 50000 \
     --seed 42

Test a PPO model:

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

Train a continuous-control SAC model:

.. code-block:: bash

   python -m abrGym train gym_sac \
     --train-trace DATASET/NETWORK/fcc-train \
     --video-meta DATASET/MOVIE/movie_4g.json \
     --model-path DATASET/MODELS/gym_sac_abr_gym.zip \
     --tensorboard-log DATASET/tb_logs \
     --continuous-map nearest \
     --seed 42

Figures and placeholders
------------------------

.. comment::
   Replace the placeholders below with generated images once the training plots and architecture figures are finalized.

.. image:: _static/rl/abr_rl_pipeline_placeholder.png
   :alt: Placeholder for ABR RL pipeline figure
   :align: center

.. note::

   Placeholder figure: end-to-end training/evaluation pipeline. A future figure should show
   ``fcc-train -> train model -> save checkpoint -> evaluate on fcc-test -> plot QoE trade-offs``.

.. image:: _static/rl/abr_state_matrix_placeholder.png
   :alt: Placeholder for ABR state matrix figure
   :align: center

.. note::

   Placeholder figure: semantic state layout, ideally labeling the six rows of the ``(6, 8)`` state.

.. image:: _static/rl/fcc_trace_placeholder.png
   :alt: Placeholder for FCC Mahimahi trace example
   :align: center

.. note::

   Placeholder figure: one FCC Mahimahi trace visualization with time on the x-axis and bandwidth on the y-axis.

Algorithm sections
------------------

Each algorithm below appears as a first-level subsection so it can surface cleanly in the sidebar when ``rl_gym.rst`` is included from the top-level documentation tree.

DQN
^^^

Deep Q-Network (DQN) learns an action-value approximation :math:`Q_\theta(s,a)` and uses experience replay plus a target network for stabilization.

Bellman target:

.. math::

   y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')

Loss:

.. math::

   \mathcal{L}(\theta) =
   \mathbb{E}\left[\left(y_t - Q_\theta(s_t, a_t)\right)^2\right]

In ABR, DQN is attractive because the action space is naturally discrete. The bitrate level is a categorical decision, so DQN provides a direct baseline that is easy to interpret and compare.

A key limitation is that DQN estimates only the **expected return**, even though ABR outcomes can be highly variable across traces.

**Primary reference**

- Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015.

.. image:: _static/rl/dqn_architecture_placeholder.png
   :alt: Placeholder for DQN ABR architecture
   :align: center

.. note::

   Placeholder figure: branch-based DQN feature extractor for the ``(6, 8)`` state.

QR-DQN
^^^^^^

QR-DQN extends DQN by learning a **distribution over returns** rather than a single scalar expectation.

Instead of a single Q estimate, it learns quantiles:

.. math::

   Z_\tau(s,a), \qquad \tau \in (0,1)

and approximates the Q-value by averaging across quantiles:

.. math::

   Q(s,a) \approx \frac{1}{N} \sum_{i=1}^N Z_{\tau_i}(s,a)

In ABR, this can be useful when the same state-action pair has highly variable future outcomes due to network uncertainty.

**Primary reference**

- Dabney et al., *Distributional Reinforcement Learning with Quantile Regression*, AAAI, 2018.

PPO
^^^

Proximal Policy Optimization (PPO) is an on-policy actor-critic method that limits destructive policy updates by clipping the likelihood ratio.

Clipped objective:

.. math::

   L^{\mathrm{CLIP}}(\theta) =
   \mathbb{E}\left[
   \min\left(
   r_t(\theta)\hat{A}_t,
   \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
   \right)
   \right]

where:

.. math::

   r_t(\theta) =
   \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}

PPO is often a strong default for ABR because it combines stable updates with relatively simple implementation.

**Primary reference**

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.

A2C
^^^

Advantage Actor-Critic (A2C) is a synchronous actor-critic method from the A3C family.

Policy-gradient form:

.. math::

   \nabla_\theta J(\theta)
   \approx
   \mathbb{E}\left[
   \nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}_t
   \right]

with advantage estimate:

.. math::

   \hat{A}_t = R_t - V_\phi(s_t)

A2C is often simpler than PPO, making it a useful baseline for understanding the value of clipped policy updates.

**Primary reference**

- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, ICML, 2016.

RecurrentPPO
^^^^^^^^^^^^

RecurrentPPO adds an LSTM to PPO so that the policy can carry hidden state across timesteps.

Conceptually:

::

   state window  --->  feature extractor  --->  LSTM  --->  actor / critic heads

This is useful in ABR because short observation windows may not fully capture longer-term temporal dynamics in the network.

The policy objective still inherits PPO’s clipped form, but the hidden state makes rollout bookkeeping more important. During rollout and evaluation, ``lstm_states`` and ``episode_start`` must be passed correctly.

**Primary reference**

- Based on PPO (Schulman et al., 2017), with recurrent implementation support via SB3-Contrib.

TRPO
^^^^

Trust Region Policy Optimization (TRPO) explicitly constrains policy updates to remain in a trust region.

Constrained optimization problem:

.. math::

   \max_\theta \;
   \mathbb{E}\left[
   \frac{\pi_\theta(a|s)}{\pi_{\theta_{\mathrm{old}}}(a|s)} \hat{A}(s,a)
   \right]
   \quad \text{s.t.} \quad
   \mathbb{E}\left[D_{KL}\left(\pi_{\theta_{\mathrm{old}}}\,\|\,\pi_\theta\right)\right] \le \delta

TRPO is more conservative than PPO and can be useful as a strong trust-region baseline.

**Primary reference**

- Schulman et al., *Trust Region Policy Optimization*, ICML, 2015.

ARS
^^^

Augmented Random Search (ARS) is a derivative-free optimization method. Rather than propagating gradients through a value function or policy objective, it explores random parameter directions and updates using the best directions.

A simplified update looks like:

.. math::

   \theta_{k+1}
   =
   \theta_k
   +
   \frac{\alpha}{b \sigma_R}
   \sum_{i=1}^{b} (r_i^+ - r_i^-)\delta_i

ARS becomes applicable here because the environment exposes a **continuous control variable** that is mapped back to bitrate indices. This is a pragmatic wrapper around the same ABR problem.

**Primary reference**

- Mania et al., *Simple random search provides a competitive approach to reinforcement learning*, NeurIPS, 2018.

SAC
^^^

Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm based on maximum-entropy reinforcement learning.

Objective:

.. math::

   J(\pi) =
   \sum_t
   \mathbb{E}_{(s_t,a_t)\sim \rho_\pi}
   \left[
   r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))
   \right]

SAC is attractive for continuous control because it is often sample-efficient and robust. In this ABR setting, however, its continuous action is still mapped to discrete bitrate choices, so its behavior should be interpreted carefully.

**Primary reference**

- Haarnoja et al., *Soft Actor-Critic*, ICML, 2018.

TD3
^^^

Twin Delayed DDPG (TD3) improves deterministic actor-critic learning through:

- two critics,
- delayed actor updates,
- target policy smoothing.

Target:

.. math::

   y = r + \gamma \min_{i=1,2} Q_{\phi_i^-}(s', \tilde{a}')

TD3 is commonly stronger than vanilla DDPG and should usually be preferred when evaluating deterministic continuous-control baselines.

**Primary reference**

- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*, ICML, 2018.

DDPG
^^^^

Deep Deterministic Policy Gradient (DDPG) learns a deterministic actor and critic:

.. math::

   \mu_\theta(s), \qquad Q_\phi(s,a)

with deterministic policy gradient:

.. math::

   \nabla_\theta J
   \approx
   \mathbb{E}_{s \sim \mathcal{D}}
   \left[
   \nabla_a Q_\phi(s,a)\vert_{a=\mu_\theta(s)}
   \nabla_\theta \mu_\theta(s)
   \right]

DDPG is historically important and useful as a reference point, but TD3 generally improves upon it.

**Primary reference**

- Lillicrap et al., *Continuous control with deep reinforcement learning*, 2015.

Trade-offs across algorithm families
------------------------------------

The reward equation makes the central trade-off explicit:

.. math::

   r_t = \frac{b_t}{1000} - \alpha \cdot \mathrm{rebuf}_t - \beta \cdot \frac{|b_t - b_{t-1}|}{1000}

The practical question is not only *which algorithm gets the highest reward*, but *how that reward is achieved*.

Typical trade-offs include:

- **DQN / QR-DQN**
  - natural fit for discrete bitrate actions
  - value-based and relatively easy to interpret
  - QR-DQN can capture uncertainty better than DQN

- **PPO / A2C / RecurrentPPO / TRPO**
  - policy-gradient actor-critic family
  - often smoother training dynamics for structured policies
  - RecurrentPPO can exploit longer temporal patterns

- **ARS / SAC / TD3 / DDPG**
  - enabled through continuous-action wrapper
  - useful for experimentation and cross-family comparison
  - but their outputs are still quantized back into bitrate indices

In practice, **stall reduction**, **smoothness**, and **bitrate utilization** must all be reported alongside reward.

Plots and analysis
------------------

A well-designed RL ABR study should report more than mean test reward.

Recommended plot families include:

- bitrate vs stall trade-off
- QoE distributions across traces
- smoothness penalty / quality-switch counts
- average bitrate by transport group
- per-trace rollout visualizations
- train vs test generalization gap

A useful conceptual summary figure would look like this:

::

   +-----------------+----------------------+------------------------+
   | Metric          | What it captures     | Why it matters         |
   +=================+======================+========================+
   | Avg bitrate     | quality level        | user-perceived quality |
   +-----------------+----------------------+------------------------+
   | Stall duration  | playback interruption| strongest QoE penalty  |
   +-----------------+----------------------+------------------------+
   | Smoothness      | quality variability  | annoyance / stability  |
   +-----------------+----------------------+------------------------+
   | Test reward     | combined objective   | headline metric        |
   +-----------------+----------------------+------------------------+

.. image:: _static/rl/tradeoff_plot_placeholder.png
   :alt: Placeholder for bitrate-stall trade-off plot
   :align: center

.. note::

   Placeholder figure: bitrate versus stall trade-off for multiple algorithms on ``fcc-test``.

.. image:: _static/rl/qoe_boxplot_placeholder.png
   :alt: Placeholder for QoE distribution plot
   :align: center

.. note::

   Placeholder figure: QoE distribution over held-out FCC test traces.

Suggested document tree integration
-----------------------------------

At the top level, ``index.rst`` should link to separate chapters such as:

- ``rl_shim.rst``
- ``rl_gym.rst``
- ``standard_algorithms.rst``
- ``llm.rst``

This chapter should be included as ``rl.rst`` and focus only on the Gym/SB3 workflows.

References
----------

1. Mnih, V. et al. *Playing Atari with Deep Reinforcement Learning*. 2013.
2. Mnih, V. et al. *Human-level control through deep reinforcement learning*. Nature, 2015.
3. Dabney, W. et al. *Distributional Reinforcement Learning with Quantile Regression*. AAAI, 2018.
4. Mnih, V. et al. *Asynchronous Methods for Deep Reinforcement Learning*. ICML, 2016.
5. Schulman, J. et al. *Trust Region Policy Optimization*. ICML, 2015.
6. Schulman, J. et al. *Proximal Policy Optimization Algorithms*. 2017.
7. Mania, H. et al. *Simple random search provides a competitive approach to reinforcement learning*. NeurIPS, 2018.
8. Lillicrap, T. et al. *Continuous control with deep reinforcement learning*. 2015.
9. Fujimoto, S. et al. *Addressing Function Approximation Error in Actor-Critic Methods*. ICML, 2018.
10. Haarnoja, T. et al. *Soft Actor-Critic*. ICML, 2018.

.. comment::
   Future extension ideas:
   - add actual rendered architecture diagrams
   - add a section on hyperparameter sensitivity
   - add ablation studies on reward coefficients
   - add fixed_start discussion with a concrete example trace
