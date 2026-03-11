RL Algorithms for ABR Streaming
===============================

.. contents:: Table of Contents
   :depth: 2
   :local:

Introduction
------------

This page documents the reinforcement learning algorithms used with the uploaded
``rl_plots.ipynb`` notebook for Adaptive Bitrate (ABR) streaming. It is focused on
algorithm design and implementation structure only. It intentionally does **not**
cover dataset analysis, transport-specific trace results, plotting workflows, or
comparative evaluation figures.

The notebook implements a compact ABR stack in three layers:

1. ``Environment`` simulates chunk downloads over a time-varying network trace.
2. ``ABREnv`` adds ABR-specific state, bitrate ladder handling, and QoE reward.
3. ``AbrGymEnv`` exposes a Gymnasium-compatible interface for Stable-Baselines3.

In this setup, the agent chooses a bitrate index for each chunk. The environment
returns a rolling observation with bitrate, buffer occupancy, throughput, delay,
future chunk sizes, and chunks remaining. The reward is QoE-oriented: higher video
quality increases reward, while rebuffering and abrupt quality switches decrease it.

ABR Environment Structure
-------------------------

``Environment``: network and buffer simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The low-level simulator advances through bandwidth, duration, and latency traces and
computes how long a video chunk would take to download at a chosen quality level.
It also updates rebuffering and buffer drain behavior.

.. code-block:: python

   class Environment:
       def __init__(
           self,
           all_cooked_time,
           all_cooked_bw,
           all_cooked_dur_ms,
           all_cooked_latency_ms,
           video_size_by_bitrate,
           chunk_len_ms,
           random_seed=42,
           queue_delay_ms=0.0,
       ):
           ...

       def get_video_chunk(self, quality: int):
           ...

``ABREnv``: ABR state and reward wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ABREnv`` loads the trace JSON and bitrate ladder, infers chunk duration, builds
approximate chunk sizes, and constructs the rolling ABR state used by RL agents.

.. code-block:: python

   class ABREnv:
       def __init__(
           self,
           trace_json_path,
           video_path,
           random_seed=42,
           default_quality=1,
           rebuf_penalty=4.3,
           smooth_penalty=1.0,
           queue_delay_ms=0.0,
           debug=False,
       ):
           ...

Key state constants in the notebook:

.. code-block:: python

   S_INFO = 6
   S_LEN = 8
   BUFFER_NORM_FACTOR = 10.0
   DEFAULT_QUALITY = 1

``AbrGymEnv``: Gymnasium interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AbrGymEnv`` adapts the ABR simulator to the API expected by Stable-Baselines3.
It supports both discrete and continuous action spaces. Discrete actions are used
for algorithms such as A2C, PPO, and DQN. Continuous actions are used for SAC,
DDPG, and TD3, and are mapped back to the nearest bitrate index.

.. code-block:: python

   class AbrGymEnv(gym.Env):
       def __init__(self, trace_json_path, video_path,
                    flatten_obs=True, continuous_actions=False, **abr_kwargs):
           ...
           if self.continuous_actions:
               self.action_space = spaces.Box(
                   low=0,
                   high=self.env.a_dim - 1,
                   shape=(1,),
                   dtype=np.float32,
               )
           else:
               self.action_space = spaces.Discrete(int(self.env.a_dim))

.. code-block:: python

   def step(self, action):
       if self.continuous_actions:
           discrete_action = int(np.round(np.clip(action[0], 0, self.env.a_dim - 1)))
       else:
           discrete_action = int(action)

       obs, reward, done, info = self.env.step(discrete_action)
       terminated = bool(done)
       truncated = False
       return self._format_obs(obs), float(reward), terminated, truncated, info

This design is especially useful for ABR research because it lets one notebook test
both discrete-control and continuous-control RL algorithms against the same state,
reward, and network simulator.

Common Training Pattern in the Notebook
---------------------------------------

All algorithms follow the same high-level pattern:

1. Build ``AbrGymEnv`` with the trace JSON and bitrate ladder.
2. Run ``check_env()`` to validate Gymnasium compatibility.
3. Wrap the environment with ``Monitor``.
4. Instantiate a Stable-Baselines3 model with ``MlpPolicy``.
5. Train with ``learn(total_timesteps=...)``.
6. Save the resulting policy.

Representative setup from the notebook:

.. code-block:: python

   TRACE_JSON_PATH = "network.json"
   VIDEO_PATH = "movie_4g.json"

   env = AbrGymEnv(
       trace_json_path=TRACE_JSON_PATH,
       video_path=VIDEO_PATH,
       flatten_obs=True,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       debug=False,
   )
   check_env(env, warn=True)
   env = Monitor(env, log_dir)

The common observation choice is a flattened 48-dimensional vector derived from the
rolling ``[6, 8]`` ABR state matrix.

.. code-block:: python

   obs_shape = (6 * 8,) if self.flatten_obs else (6, 8)

Algorithms
----------

A2C
~~~

Advantage Actor-Critic (A2C) is the synchronous counterpart to A3C. In Stable-
Baselines3, A2C is an on-policy actor-critic method that uses multiple workers
rather than a replay buffer. For ABR, A2C is a natural fit when bitrate selection
is framed as a discrete action problem.

Why it is relevant to ABR
^^^^^^^^^^^^^^^^^^^^^^^^^

A2C works well when each decision corresponds to selecting one bitrate level from a
fixed ladder. The policy learns to balance immediate bitrate gains against future
stall risk and switching penalties through the shared policy-value structure.

Notebook usage
^^^^^^^^^^^^^^

.. code-block:: python

   from stable_baselines3 import A2C

   env_a2c = AbrGymEnv(
       trace_json_path=TRACE_JSON_PATH,
       video_path=VIDEO_PATH,
       flatten_obs=True,
       continuous_actions=False,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       debug=False,
   )
   check_env(env_a2c, warn=True)
   env_a2c = Monitor(env_a2c, log_dir_a2c)

   model_a2c = A2C("MlpPolicy", env_a2c, verbose=1)
   model_a2c.learn(total_timesteps=100_000)
   model_a2c.save("a2c_sb3_abr_100k.zip")

Implementation note
^^^^^^^^^^^^^^^^^^^

Stable-Baselines3 notes that if A2C training appears unstable, users may prefer
``RMSpropTFLike`` to better match older Stable-Baselines behavior.

References
^^^^^^^^^^

- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*.
- Stable-Baselines3 A2C docs.
- OpenAI Baselines A2C/ACKTR blog post.

PPO
~~~

Proximal Policy Optimization (PPO) is an on-policy actor-critic method that keeps
policy updates from moving too far from the previous policy. Stable-Baselines3’s
implementation documents two practical modifications often used in modern PPO code:
advantage normalization and optional value-function clipping.

Why it is relevant to ABR
^^^^^^^^^^^^^^^^^^^^^^^^^

ABR decisions are sequential and sensitive to unstable updates: a policy that swings
too aggressively can overreact to short-term throughput changes. PPO is attractive
because clipped policy updates often make training easier to reason about when the
state includes buffer, delay, throughput history, and remaining chunks.

Notebook usage
^^^^^^^^^^^^^^

.. code-block:: python

   from stable_baselines3 import PPO

   env = AbrGymEnv(
       trace_json_path=TRACE_JSON_PATH,
       video_path=VIDEO_PATH,
       flatten_obs=True,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       debug=False,
   )
   check_env(env, warn=True)
   env = Monitor(env, log_dir_ppo)

   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000)
   model.save("ppo_sb3_abr_100k.zip")

Practical note
^^^^^^^^^^^^^^

The SB3 documentation also points to recurrent PPO in ``sb3-contrib`` for problems
with longer temporal dependencies, while recommending frame stacking as a simpler
baseline in many cases. In this notebook, the state already carries a short rolling
history, which partially serves that purpose.

References
^^^^^^^^^^

- Schulman et al., *Proximal Policy Optimization Algorithms*.
- Stable-Baselines3 PPO docs.
- OpenAI PPO post and Spinning Up PPO guide.

DQN
~~~

Deep Q-Network (DQN) is a value-based off-policy algorithm for discrete action
spaces. It combines neural-network Q-value approximation with a replay buffer,
target network, and gradient clipping.

Why it is relevant to ABR
^^^^^^^^^^^^^^^^^^^^^^^^^

ABR bitrate selection is naturally discrete when the ladder is fixed. DQN can learn
which bitrate index has the highest long-term value under the current ABR state,
without explicitly learning a stochastic or deterministic actor.

Notebook usage
^^^^^^^^^^^^^^

.. code-block:: python

   from stable_baselines3 import DQN

   env_dqn = AbrGymEnv(
       trace_json_path=TRACE_JSON_PATH,
       video_path=VIDEO_PATH,
       flatten_obs=True,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       debug=False,
   )
   check_env(env_dqn, warn=True)
   env_dqn = Monitor(env_dqn, log_dir_dqn)

   model_dqn = DQN("MlpPolicy", env_dqn, verbose=1)
   model_dqn.learn(total_timesteps=100_000)
   model_dqn.save("dqn_sb3_abr_100k.zip")

Implementation perspective
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this notebook, DQN is the cleanest match for a fixed bitrate ladder because the
agent directly predicts values over discrete quality levels. It is often a useful
baseline when comparing policy-gradient methods against value-based discrete control.

References
^^^^^^^^^^

- Mnih et al., *Playing Atari with Deep Reinforcement Learning*.
- Mnih et al., *Human-level control through deep reinforcement learning*.
- Stable-Baselines3 DQN docs.

SAC
~~~

Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm built around a
maximum-entropy objective. In addition to maximizing return, SAC encourages
stochastic exploration by rewarding policy entropy.

Why it is relevant to ABR
^^^^^^^^^^^^^^^^^^^^^^^^^

Although ABR actions are usually represented as bitrate indices, this notebook also
supports a continuous action interface. That lets SAC output a continuous scalar,
which is then clipped and rounded to the nearest discrete bitrate level. This can be
useful when treating bitrate choice as a smoother control problem before discretizing
at the final interface.

Notebook usage
^^^^^^^^^^^^^^

.. code-block:: python

   from stable_baselines3 import SAC

   env_sac = AbrGymEnv(
       trace_json_path=TRACE_JSON_PATH,
       video_path=VIDEO_PATH,
       flatten_obs=True,
       continuous_actions=True,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       debug=False,
   )
   check_env(env_sac, warn=True)
   env_sac = Monitor(env_sac, log_dir_sac)

   model_sac = SAC("MlpPolicy", env_sac, verbose=1)
   model_sac.learn(total_timesteps=100_000)
   model_sac.save("sac_sb3_abr_100k.zip")

Implementation notes
^^^^^^^^^^^^^^^^^^^^

SB3 documents two details that matter in practice:

- the implementation uses an entropy coefficient rather than the inverse reward
  scale notation used in some SAC descriptions;
- when the entropy coefficient is learned automatically, SB3 optimizes its
  logarithm for improved numerical stability.

It also notes that the default SAC MLP policy uses ReLU activations to match the
original paper.

References
^^^^^^^^^^

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.
- Stable-Baselines3 SAC docs.
- OpenAI Spinning Up SAC guide.

DDPG
~~~~

Deep Deterministic Policy Gradient (DDPG) combines deterministic policy gradients
with replay-buffer-based off-policy learning. It is designed for continuous action
spaces and can be seen as an earlier predecessor to TD3.

Why it is relevant to ABR
^^^^^^^^^^^^^^^^^^^^^^^^^

In this notebook, DDPG is used through the continuous-action wrapper. The actor
produces a scalar action that is mapped to a bitrate index. This makes DDPG a way to
study ABR as continuous control while still deploying a discrete ladder choice at the
simulator boundary.

Notebook usage
^^^^^^^^^^^^^^

.. code-block:: python

   from stable_baselines3 import DDPG

   env_ddpg = AbrGymEnv(
       trace_json_path=TRACE_JSON_PATH,
       video_path=VIDEO_PATH,
       flatten_obs=True,
       continuous_actions=True,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       debug=False,
   )
   check_env(env_ddpg, warn=True)
   env_ddpg = Monitor(env_ddpg, log_dir_ddpg)

   model_ddpg = DDPG("MlpPolicy", env_ddpg, verbose=1)
   model_ddpg.learn(total_timesteps=100_000)
   model_ddpg.save("ddpg_sb3_abr_100k.zip")

Notebook note
^^^^^^^^^^^^^

The notebook also contains a second DDPG training cell using ``50_000`` timesteps
and saving to ``ddpg_sb3_abr_50k.zip``. That appears to be an alternate run rather
than a different algorithmic configuration.

References
^^^^^^^^^^

- Silver et al., *Deterministic Policy Gradient Algorithms*.
- Lillicrap et al., *Continuous control with deep reinforcement learning*.
- Stable-Baselines3 DDPG docs and Spinning Up DDPG guide.

TD3
~~~

Twin Delayed DDPG (TD3) improves on DDPG with three well-known ideas: clipped
Double-Q learning, delayed policy updates, and target policy smoothing. It is an
off-policy actor-critic method for continuous control.

Why it is relevant to ABR
^^^^^^^^^^^^^^^^^^^^^^^^^

For ABR, TD3 fits the same continuous-action wrapper as DDPG and SAC. The policy
outputs a continuous quality signal that is discretized to the nearest ladder level.
Its main appeal is that it keeps the continuous-control formulation while addressing
common overestimation and instability problems associated with DDPG-style learning.

Notebook usage
^^^^^^^^^^^^^^

.. code-block:: python

   from stable_baselines3 import TD3

   env_td3 = AbrGymEnv(
       trace_json_path=TRACE_JSON_PATH,
       video_path=VIDEO_PATH,
       flatten_obs=True,
       continuous_actions=True,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       debug=False,
   )
   check_env(env_td3, warn=True)
   env_td3 = Monitor(env_td3, log_dir_td3)

   model_td3 = TD3("MlpPolicy", env_td3, verbose=1)
   model_td3.learn(total_timesteps=100_000)
   model_td3.save("td3_sb3_abr_100k.zip")

References
^^^^^^^^^^

- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*.
- Stable-Baselines3 TD3 docs.
- OpenAI Spinning Up TD3 guide.

Action-Space Design in This Notebook
------------------------------------

A key implementation idea in the notebook is the split between:

- **discrete-action algorithms**: A2C, PPO, DQN
- **continuous-action algorithms**: SAC, DDPG, TD3

The continuous-action algorithms do not operate on a truly continuous video ladder at
deployment time. Instead, they output a scalar that is converted back to the nearest
valid bitrate index.

.. code-block:: python

   discrete_action = int(np.round(np.clip(action[0], 0, self.env.a_dim - 1)))

This is a practical research compromise. It allows direct comparison between methods
meant for discrete control and methods meant for continuous control, while keeping
all algorithms connected to the same ABR simulator and bitrate ladder.

ABR Reward Design in the Notebook
---------------------------------

The notebook uses a QoE-style reward controlled by:

.. code-block:: python

   rebuf_penalty=4.3
   smooth_penalty=1.0

The reward reflects the classic ABR tradeoff:

- prefer higher bitrate when the network can support it,
- strongly penalize rebuffering,
- discourage abrupt quality oscillations.

That reward structure is what makes RL useful here. Each algorithm is not merely
trying to maximize instantaneous throughput; it is learning a sequential policy over
future playback quality, future stall risk, and switching smoothness.

Minimal End-to-End Example
--------------------------

The following condensed snippet captures the overall notebook workflow for one
algorithm:

.. code-block:: python

   from stable_baselines3 import PPO
   from stable_baselines3.common.env_checker import check_env
   from stable_baselines3.common.monitor import Monitor

   env = AbrGymEnv(
       trace_json_path="network.json",
       video_path="movie_4g.json",
       flatten_obs=True,
       rebuf_penalty=4.3,
       smooth_penalty=1.0,
       debug=False,
   )
   check_env(env, warn=True)
   env = Monitor(env, "logs/ppo/")

   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000)
   model.save("ppo_sb3_abr_100k.zip")

Suggested Organization for a Read the Docs Project
--------------------------------------------------

If you want to keep this material modular in your documentation set, a clean split is:

- ``algo.rst`` for RL algorithm descriptions and code snippets,
- ``env.rst`` for ``Environment``, ``ABREnv``, and ``AbrGymEnv``,
- ``training.rst`` for command examples and saved-model conventions,
- ``plots.rst`` for figures and interpretation,
- ``dataset.rst`` for trace provenance and transport-specific context.

References
----------

Primary algorithm references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Mnih, V. et al. *Asynchronous Methods for Deep Reinforcement Learning*.
  `<https://arxiv.org/abs/1602.01783>`_
- Schulman, J. et al. *Proximal Policy Optimization Algorithms*.
  `<https://arxiv.org/abs/1707.06347>`_
- Mnih, V. et al. *Playing Atari with Deep Reinforcement Learning*.
  `<https://arxiv.org/abs/1312.5602>`_
- Mnih, V. et al. *Human-level control through deep reinforcement learning*.
  `<https://www.nature.com/articles/nature14236>`_
- Haarnoja, T. et al. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.
  `<https://arxiv.org/abs/1801.01290>`_
- Silver, D. et al. *Deterministic Policy Gradient Algorithms*.
  `<https://proceedings.mlr.press/v32/silver14.html>`_
- Lillicrap, T. et al. *Continuous control with deep reinforcement learning*.
  `<https://arxiv.org/abs/1509.02971>`_
- Fujimoto, S. et al. *Addressing Function Approximation Error in Actor-Critic Methods*.
  `<https://arxiv.org/abs/1802.09477>`_

Stable-Baselines3 and tutorial references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Stable-Baselines3 A2C docs:
  `<https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html>`_
- Stable-Baselines3 PPO docs:
  `<https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>`_
- Stable-Baselines3 DQN docs:
  `<https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html>`_
- Stable-Baselines3 SAC docs:
  `<https://stable-baselines3.readthedocs.io/en/master/modules/sac.html>`_
- Stable-Baselines3 TD3 docs:
  `<https://stable-baselines3.readthedocs.io/en/master/modules/td3.html>`_
- Stable-Baselines3 DDPG docs:
  `<https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html>`_
- OpenAI Baselines A2C/ACKTR blog:
  `<https://openai.com/blog/baselines-acktr-a2c/>`_
- OpenAI PPO post:
  `<https://openai.com/research/openai-baselines-ppo>`_
- OpenAI Spinning Up PPO guide:
  `<https://spinningup.openai.com/en/latest/algorithms/ppo.html>`_
- OpenAI Spinning Up SAC guide:
  `<https://spinningup.openai.com/en/latest/algorithms/sac.html>`_
- OpenAI Spinning Up TD3 guide:
  `<https://spinningup.openai.com/en/latest/algorithms/td3.html>`_
- OpenAI Spinning Up DDPG guide:
  `<https://spinningup.openai.com/en/latest/algorithms/ddpg.html>`_
- DQN tutorial by Antonin Raffin:
  `<https://github.com/araffin/rlss23-dqn-tutorial>`_
- PPO implementation details blog:
  `<https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/>`_
