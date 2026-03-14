Algorithms
==========

This documentation collects the Adaptive Bitrate (ABR) algorithms implemented in this repository across classical heuristics, control-based methods, remote / shim-based services, and reinforcement-learning approaches.

ABR algorithms all address the same sequential decision problem: for each chunk, choose the next representation using bandwidth evidence, buffer occupancy, latency, and video progress so that overall Quality of Experience (QoE) remains high.

A representative chunk-level QoE objective used repeatedly throughout the documentation is

.. math::

   R_t = \frac{b_t}{1000}
         - \lambda_r \, \Delta t_{\mathrm{stall}}
         - \lambda_s \frac{|b_t - b_{t-1}|}{1000}

where the first term rewards quality, the second penalizes stalls, and the third penalizes abrupt switching.

.. toctree::
   :maxdepth: 2
   :caption: ABR Algorithms

   standard
   rl
   pensieve
   ppo
   dqn
   llm




Start with :doc:`standard` for the classical and control-based baselines.
Then read :doc:`rl` for the modern Gym/SB3 workflow and held-out evaluation protocol.
 Use :doc:`pensieve`, :doc:`ppo`, and :doc:`dqn` when you want server-specific implementation details or historical RL stacks without Gym.

Classical heuristics
--------------------
   Classical heuristics algorithm like RB, BBA, BOLA, MPC, and remote / server-side ABR methods.

RL Algorithms with AbrGym
-------------------------
   ABR with Gym and Stable-Baselines3 / SB3-Contrib RL workflows over ``AbrStreamingEnv`` using FCC ``fcc-train`` / ``fcc-valid`` / ``fcc-test`` Mahimahi datasets.

RL implementation without Gym
-------------------------
   Legacy or shim-server RL pages, retained separately because they document server-oriented runtime stacks and training/inference implementations in more detail.

LLM-Based
---------
designed to support multiple pretrained language model (PLM) families and
sizes.

