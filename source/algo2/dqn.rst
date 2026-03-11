DQN
===

Folder: ``SERVER/dqn``

This module provides a value-based HTTP ABR decision server using a Deep Q-Network
(DQN) style policy for bitrate selection.

It is designed around a Pensieve-like state representation and a discrete action
space of bitrate levels.

Overview
--------

The DQN setup includes:

- an inference server that returns the next quality index
- a neural network that estimates Q-values for each action
- replay memory for training
- a target network for stabilization

State, action, and reward
-------------------------

State
~~~~~

The state is typically a fixed-shape history tensor using a Pensieve-style layout:

- previous bitrate
- current buffer level
- throughput estimate
- download time
- next chunk sizes
- remaining chunks

Action
~~~~~~

The action space is discrete:

.. math::

   a_t \in \{0, 1, \dots, A-1\}

where each action corresponds to one bitrate level.

Reward
~~~~~~

A standard chunk-level QoE reward is:

.. math::

   R_t = \frac{b_t}{1000}
         - \lambda_r \Delta t_{\mathrm{stall}}
         - \lambda_s \frac{|b_t - b_{t-1}|}{1000}

This rewards higher bitrate and penalizes rebuffering and large bitrate switches.

DQN update rule
---------------

The Q-network is trained toward a Bellman target:

.. math::

   y_t =
   r_t + \gamma \max_{a'} Q_{\mathrm{target}}(s_{t+1}, a')

The online network minimizes:

.. math::

   \mathcal{L}(\theta) =
   \left(
      y_t - Q_{\theta}(s_t, a_t)
   \right)^2

With replay memory, batches of past transitions are sampled to reduce temporal correlation.

Why DQN is useful for ABR
-------------------------

- handles discrete bitrate choices naturally
- learns a state-dependent value function
- simpler than actor-critic methods in some setups
- good baseline against PPO and Pensieve

Limitations
-----------

- can be unstable without replay and target networks
- Q-learning may be sensitive to reward scaling
- policy can be less smooth than actor-critic approaches

Runtime behavior
----------------

At inference time, the server:

1. receives playback statistics via HTTP
2. updates the state tensor
3. evaluates Q-values for all bitrate actions
4. returns the action with highest Q-value

This makes the serving path straightforward even though training is more involved.