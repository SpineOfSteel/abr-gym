Algorithms
==========

This repository contains Adaptive Bitrate (ABR) algorithms implemented in multiple stacks:

- **In-simulator / in-player style (local decisions)**: BB, RB, BOLA, Dynamic, Festive
- **Server-side ABR (remote decisions)**: FastMPC, RobustMPC, DQN, PPO, Pensieve (A3C)


.. toctree::
   :maxdepth: 2

   classic
   bola
   dynamic   
   mpc 
   remote  
   dqn
   pensieve
   ppo


ABR algorithms:
-----------

- **RB**: simple rate-based ABR using recent throughput.
- **BB (BBA-like)**: simplified buffer-based adaptation inspired by
  `BBA <http://yuba.stanford.edu/~nickm/papers/sigcomm2014-video.pdf>`_.
- `BOLA <https://ieeexplore.ieee.org/document/9110784>`_: buffer-based algorithm used widely in practice (also available in DASH.js).
- `Dynamic <https://dl.acm.org/doi/abs/10.1145/3336497>`_: DASH.js default family of adaptive logic (hybrid behavior).
- `Festive <https://dl.acm.org/doi/10.1145/2413176.2413189>`_: rate-based ABR using harmonic mean estimation.
- **fastMPC**
- `RobustMPC <https://users.ece.cmu.edu/~vsekar/papers/sigcomm15_mpcdash.pdf>`_: MPC-based lookahead adaptation.


RL ABR algorithms:
-----------

- `Pensieve <https://github.com/hongzimao/pensieve>`_: RL-based ABR.
- **PPO**: PPO actor-critic ABR (this repo’s training + server implementation).
- **DQN**: Double-DQN value-based ABR (this repo’s training + server implementation).


Algorithm can be found here...
-----------

- SABRE plugins: ``SIM/SABRE/algo/*.py`` (Python)
- RL training + servers: ``SERVER/pensieve``, ``SERVER/ppo``, ``SERVER/dqn``
- MPC servers: ``SERVER/fastmpc_server.py``, ``SERVER/robustmpc_server.py``
- DASH / player-side: ``SIM/DASH/src/algo/*`` (TypeScript)

