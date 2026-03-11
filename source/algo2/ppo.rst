PPO
===

Folder: ``SERVER/ppo``

This module provides an HTTP ABR decision server backed by a PPO actor-critic
policy in PyTorch, using a Pensieve-style RL state.

Overview
--------

The PPO stack includes:

- ``ppo_server.py`` for HTTP inference
- ``train_ppo.py`` for offline training
- ``ppo2.py`` for actor-critic and PPO update logic

Quickstart
----------

Install requirements:

.. code-block:: bash

   pip install numpy torch

Run the PPO server:

.. code-block:: bash

   python ppo_server.py --host localhost --port 8605 --movie ../movie_4g.json --model models/ppo_model.pth --debug --verbose

Train PPO:

.. code-block:: bash

   python train_ppo.py

State and action space
----------------------

The PPO policy uses a history tensor with:

- ``S_INFO = 6``
- ``S_LEN = 8``

The action space is discrete bitrate selection:

.. math::

   a_t \in \{0, 1, \dots, A-1\}

Policy and critic
-----------------

The actor outputs a distribution:

.. math::

   \pi_{\theta}(a \mid s)

and the critic predicts:

.. math::

   V_{\phi}(s)

PPO objective
-------------

PPO uses a clipped surrogate objective:

.. math::

   L^{\mathrm{CLIP}}(\theta)
   =
   \mathbb{E}_t
   \left[
      \min
      \left(
         r_t(\theta) A_t,
         \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
      \right)
   \right]

where

.. math::

   r_t(\theta) =
   \frac{\pi_{\theta}(a_t \mid s_t)}
        {\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}

and :math:`A_t` is the estimated advantage.

The clipped objective stabilizes training by limiting excessively large policy updates.

Reward function
---------------

The server-side QoE reward typically follows:

.. math::

   R_t = \frac{b_t}{1000}
         - 4.3 \Delta t_{\mathrm{stall}}
         - 1.0 \frac{|b_t - b_{t-1}|}{1000}

where:

- :math:`b_t` is current bitrate in kbps
- :math:`\Delta t_{\mathrm{stall}}` is incremental stall time in seconds

Why PPO is attractive for ABR
-----------------------------

- more stable than vanilla policy gradients
- easier to tune than asynchronous A3C in many cases
- supports batched trajectory optimization
- strong practical baseline for learned ABR

Inference flow
--------------

At runtime, the server:

1. receives player statistics through HTTP
2. updates the current RL state
3. computes the policy distribution over qualities
4. returns the selected next quality index
5. logs reward and other diagnostics

This makes PPO a practical learned replacement for heuristic server-side ABR.