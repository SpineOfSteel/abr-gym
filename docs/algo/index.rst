Algorithms
==========

This documentation contains Adaptive Bitrate (ABR) algorithms implementaton details.
This repository covers wide range of algorithm from classical heuristics, greedy algorithms, 
control-based methods, reinforcement-learning approaches to state of the art LLM approaches to this problem.

.. toctree::
   :maxdepth: 2
   :caption: Table of contents

   standard
   rl
   llm
   pensieve
   ppo
   dqn
   
Problem: All ABR algorithms address the same decision problem: for each chunk, choose the next representation using data points such as bandwidth, buffer occupancy, latency, Throughut history, video progress, etc., 
in a way that overall Quality of Experience (QoE) remains high.

QoE objective used repeatedly throughout the documentation is:

.. math::

   R_t = \frac{b_t}{1000}
         - \lambda_r \, \Delta t_{\mathrm{stall}}
         - \lambda_s \frac{|b_t - b_{t-1}|}{1000}


where:

- :math:`R_q` is the bitrate of quality level :math:`q`
- :math:`\hat{T}` is the current throughput estimatewhere the first term rewards quality, the second penalizes stalls, and the third penalizes abrupt switching.

We start with :doc:`standard` classical, hueristics and control-based algorithms.
Classical heuristics algorithm like RB, BBA, BOLA, MPC, and remote / server-side ABR methods.

Then discuss both typical torch/TF based :doc:`rl` implementation and then introduce modern Abr Gym RL workflow. 
In traditianl RL approach we discuss complexity and implement :doc:`pensieve`, :doc:`ppo`, and :doc:`dqn` from scratch
Legacy or shim-server RL implementation retained separately because they document server-oriented runtime stacks and training/inference implementations in more detail.
ABR with Gym and Stable-Baselines3 / SB3-Contrib RL workflows over ``AbrStreamingEnv`` using FCC ``fcc-train`` / ``fcc-valid`` / ``fcc-test`` Mahimahi datasets.
when you want server-specific implementation details or historical RL stacks without Gym.
Then we use Gym for LLM ...designed to support multiple pretrained language model (PLM) families and
sizes.

