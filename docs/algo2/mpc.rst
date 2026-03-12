MPC-Based ABR
=============

This page covers model-predictive control (MPC) style bitrate adaptation used in
server-side ABR implementations such as FastMPC and RobustMPC.

MPC methods do not simply react to the latest measurement. Instead, they predict
near-future bandwidth over a short horizon and evaluate multiple candidate bitrate
sequences before selecting the next quality.

Core idea
---------

At each chunk decision, MPC:

1. estimates future available bandwidth
2. simulates a short horizon of future chunk downloads
3. evaluates a QoE objective for each candidate action sequence
4. chooses the first action of the best sequence

This is why MPC is often called a **lookahead** ABR method.

QoE objective
-------------

A representative MPC objective over horizon :math:`H` is:

.. math::

   J = \sum_{t=1}^{H}
   \left(
      \frac{b_t}{1000}
      - \lambda_r \, r_t
      - \lambda_s \frac{|b_t - b_{t-1}|}{1000}
   \right)

where:

- :math:`b_t` is bitrate at future step :math:`t`
- :math:`r_t` is rebuffer time at that step
- :math:`\lambda_r` is the rebuffer penalty
- :math:`\lambda_s` is the smoothness penalty

The controller picks the action sequence maximizing :math:`J`.

Bandwidth prediction
--------------------

A simple MPC predictor may use a recent average or harmonic mean of past throughput.
RobustMPC-style methods additionally discount optimistic estimates to be safer under noise.

A generic robust estimate can be written as:

.. math::

   \hat{T}_{\mathrm{robust}} =
   \frac{\hat{T}}{1 + \epsilon}

where :math:`\epsilon` captures prediction uncertainty or recent relative error.

FastMPC
-------

FastMPC uses a lightweight lookahead procedure to choose the next chunk quality quickly.

Typical characteristics:

- short horizon
- simple throughput estimation
- direct QoE scoring of candidate trajectories
- lower computational overhead than more elaborate MPC variants

RobustMPC
---------

RobustMPC adds more conservative bandwidth prediction and is designed to perform
better under uncertain or rapidly changing conditions.

Typical characteristics:

- robustified throughput estimate
- bounded lookahead horizon
- explicit tradeoff between quality, rebuffering, and smoothness

Why MPC matters
---------------

MPC occupies an important middle ground:

- more informed than simple heuristics
- more interpretable than RL
- easier to control than pure learned policies
- still practical for online server-side decision making

Relation to other approaches
----------------------------

- **RB** and **BB** are myopic: they react to the current state only.
- **BOLA** uses a utility score tied to current buffer occupancy.
- **MPC** explicitly simulates future outcomes over a short horizon.
- **RL** learns a policy that may implicitly capture longer-horizon structure.

When to use MPC
---------------

MPC is a strong choice when you want:

- a principled non-RL baseline
- explicit control over the QoE objective
- limited-horizon planning under known segment sizes
- interpretability in server-side decision logic