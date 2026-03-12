Environments
============

The ABR training stack is split into:

- a **network simulator** (trace replay + latency + buffer drain)
- an **ABR training environment** (state, reward, step/reset)
- a **Gymnasium wrapper** (standard RL API)

.. toctree::
   :maxdepth: 2

   network
   abr_env
   abr_gym