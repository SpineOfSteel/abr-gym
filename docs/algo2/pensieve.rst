Pensieve / A3C
==============

Folder: ``SERVER/pensieve``

Pensieve is an actor-critic ABR approach based on asynchronous training.
In this repository it appears as an A3C-style implementation paired with an
HTTP inference server.

Overview
--------

Pensieve-style ABR uses:

- a fixed-shape history state
- a policy network that outputs action probabilities
- a value network that predicts state value
- asynchronous workers collecting rollouts in parallel

State representation
--------------------

The state is typically a ``6 × 8`` history tensor whose rows encode:

0. last selected bitrate
1. current buffer level
2. measured throughput
3. download time
4. next chunk sizes
5. remaining chunks

This is the canonical Pensieve-style ABR state.

Policy and value
----------------

The actor outputs a policy:

.. math::

   \pi_{\theta}(a \mid s)

and the critic predicts a value:

.. math::

   V_{\phi}(s)

The selected bitrate is sampled from or chosen from the policy distribution,
depending on training or inference mode.

A3C objective
-------------

The actor uses a policy-gradient objective with an advantage term:

.. math::

   \nabla_{\theta} J(\theta)
   \approx
   \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \, A_t

where the advantage is commonly:

.. math::

   A_t = R_t - V_{\phi}(s_t)

and :math:`R_t` is the bootstrapped return.

The critic minimizes a value loss such as:

.. math::

   \mathcal{L}_V = \left(R_t - V_{\phi}(s_t)\right)^2

An entropy bonus is often added to encourage exploration:

.. math::

   \mathcal{L}_{\mathrm{entropy}} = - \sum_a \pi_{\theta}(a \mid s) \log \pi_{\theta}(a \mid s)

QoE reward
----------

A typical Pensieve-style chunk reward is:

.. math::

   R_t = \frac{b_t}{1000}
         - \lambda_r \Delta t_{\mathrm{stall}}
         - \lambda_s \frac{|b_t - b_{t-1}|}{1000}

with high penalty on rebuffering and a smaller smoothness penalty.

Why Pensieve matters
--------------------

Pensieve was influential because it showed that ABR can be learned as a sequential
decision problem rather than hand-designed entirely through heuristics.

Strengths
---------

- directly optimizes long-horizon behavior
- naturally handles nonlinear tradeoffs
- foundational RL baseline for ABR

Limitations
-----------

- training can be complex and sensitive
- asynchronous training adds engineering overhead
- inference is easy, but training reproducibility can be harder than PPO