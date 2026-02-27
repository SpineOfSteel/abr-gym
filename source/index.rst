abr-gym documentation
=====================

**abr-gym** is an Adaptive Bitrate (ABR) experimentation suite that combines:

- **Datasets** (manifests, network logs, traces)
- **Servers / Gym environments** (ABR env wrappers + algorithm servers)
- **Simulators** (DASH TypeScript simulator + SABRE-based simulator)

Use this documentation to understand the project structure and to run:
training scripts (DQN / A3C-Pensieve / PPO), server-side ABR controllers, and simulators.

Quick navigation
----------------

- :ref:`dataset-section` — where manifests, network logs, and traces live
- :ref:`server-section` — gym env + ABR algorithm servers + training scripts
- :ref:`sim-section` — DASH simulator and SABRE simulator tooling

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   dataset/index
   algo/index
   env/index
   sabre
   plots

Project layout
--------------

This repo is organized into three top-level areas:

- ``DATASET/``: media manifests + network logs + traces for reproducible experiments
- ``SERVER/``: Gym environments + ABR controller servers + training entrypoints
- ``SIM/``: simulation frameworks (DASH + SABRE)

.. _dataset-section:

DATASET
~~~~~~~

The ``DATASET/`` folder contains everything needed to reproduce streaming sessions.

.. code-block:: text

   DATASET
   ├── MODELS
   ├── MOVIE
   │   └── manifest
   ├── NETWORK
   │   ├── 3Glogs
   │   ├── 4Glogs
   │   ├── 4Glogs_lum
   │   ├── 5Glogs_lum
   │   ├── hd_fs
   │   ├── mahimahi
   │   └── sd_fs
   └── TRACES
       └── norway_tram

What each folder is for:

- ``MODELS/``: saved checkpoints / exported policies (if you store them with the dataset).
- ``MOVIE/manifest/``: video manifests and segment metadata used by players/simulators.
- ``NETWORK/*``: bandwidth log collections (3G/4G/5G) and filesystem profiles (e.g., hd/sd).
- ``TRACES/norway_tram``: raw trace sets (example dataset for mobile movement scenarios).

.. _server-section:

SERVER
~~~~~~

The ``SERVER/`` folder contains the ABR gym environments and algorithm implementations.

.. code-block:: text

   SERVER
   │   Env_*.py
   │   fastmpc_server.py
   │   robustmpc_server.py
   │   train_example_abrgym.py
   ├── dqn
   │   ├── dqn.py
   │   ├── dqn_server.py
   │   └── train_dqn.py
   ├── pensieve
   │   ├── a3c.py
   │   ├── pensieve_server.py
   │   └── train_a3c.py
   └── ppo
       ├── ppo2.py
       ├── ppo_server.py
       └── train_ppo.py

Highlights:

- **Environment wrappers**
  - ``EnvAbr.py`` / ``EnvAbrGym.py``: core ABR environment logic and Gym integration.
  - ``EnvNetwork.py``: network model adapter (replay logs / traces / shaping).

- **Classic controllers**
  - ``fastmpc_server.py`` and ``robustmpc_server.py``: MPC baselines served via HTTP.

- **RL algorithms**
  - ``dqn/``: Deep Q Network ABR.
  - ``pensieve/``: A3C-based Pensieve.
  - ``ppo/``: PPO-based ABR policy.

Entry points:

- ``smoke.py``: quick sanity check that your env + traces are wired correctly.
- ``train_example_abrgym.py``: minimal example training loop / usage.

.. _sim-section:

SIM
~~~

The ``SIM/`` folder contains two simulation stacks:

1) A **DASH simulator** (TypeScript/Node + browser frontend)
2) A **SABRE**-based simulator (Python)

.. code-block:: text

   SIM
   ├── DASH
   │   ├── build.js
   │   ├── index.html
   │   ├── package.json
   │   ├── README.md
   │   ├── tsconfig.json
   │   └── src
   │       ├── index.js
   │       ├── types.ts
   │       ├── algo
   │       ├── apps
   │       ├── common
   │       ├── component
   │       ├── controller
   │       └── ts-out
   └── SABRE
       ├── CustomAbr.py
       ├── CustomReplacement.py
       ├── CustomReplacement_new.py
       ├── README.md
       ├── sab.py
       ├── sabre-mmsys18-original.py
       ├── algo
       ├── docs
       └── plot

Notes:

- ``DASH/src/algo`` contains client / server ABR selectors (BB, BOLA, FESTIVE, RB, etc.).
- ``SABRE/sab.py`` is the main simulator entry (plus plotting utilities in ``plot/``).

Next steps
----------

1. Create the pages referenced by the toctree (recommended minimal set):

- ``docs/getting_started.rst``
- ``docs/dataset/index.rst``
- ``docs/server/index.rst``
- ``docs/sim/index.rst``
- ``docs/api/index.rst``

2. Build locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs/ _build/html

3. Push to GitHub and connect ReadTheDocs to the repo.
