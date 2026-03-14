LLM Algorithms
==============


References
----------

The LLM-based ABR pipeline described here builds on the repository implementation and its supported PLM families. When writing broader documentation or a paper section, it is useful to cite both the ABR framework and the underlying pretrained model families used for adaptation and evaluation.

Representative references to include are:

- LLaMA / Llama 2 papers for the LLaMA family
- the GPT-2 paper for GPT-style decoder backbones
- the T5 paper for text-to-text transfer transformers
- the OPT paper for open pretrained transformers
- the Mistral technical report or release paper for the Mistral family

You can later replace this section with full bibliography entries once your documentation build is connected to a shared references file.


.. contents::
   :local:
   :depth: 2

Overview
--------

This chapter documents the **LLM-based adaptive bitrate (ABR) framework** used in this repository. The implementation extends a Genet-style pipeline with experience-pool generation, structured state encoding, parameter-efficient adaptation, and Transformer-based policy evaluation for bitrate selection.

Unlike classical ABR methods such as RB, BB, or BOLA, the LLM pipeline uses a pretrained language model (PLM) backbone as a sequence model over encoded ABR state. The optimization goal is still the same: deliver high visual quality while reducing rebuffering and avoiding unnecessary bitrate oscillation.

The framework supports multiple PLM families. In this repository, the main documented experimental focus is **LLaMA / Llama2 7B**, but the code structure is designed so the same adaptation workflow can be applied to other backbones such as **GPT-2**, **T5-LM**, **OPT**, and **Mistral**.

A typical workflow is:

1. generate or load an ABR experience pool
2. encode ABR observations into a model-ready representation
3. adapt a pretrained backbone using low-rank updates
4. evaluate the adapted model as a bitrate-selection policy


Create the LLM environment:

.. code-block:: bash

   conda create -n abr_netllm python=3.8.10
   conda activate abr_netllm
   pip install torch==2.1.0
   pip install numpy==1.24.4
   pip install munch==4.0.0
   pip install openprompt==1.0.1
   pip install transformers==4.34.1
   pip install peft==0.6.2

The main documented experiment in this repository focuses on a LLaMA-based run:

.. code-block:: bash

   python run_plm.py --adapt \
     --grad-accum-steps 32 \
     --plm-type llama \
     --plm-size small \
     --rank 128 \
     --device cuda:0 \
     --lr 0.0001 \
     --warmup-steps 2000 \
     --num-epochs 80 \
     --eval-per-epoch 2

To test an adapted model:

.. code-block:: bash

   python run_plm.py --test \
     --plm-type llama \
     --plm-size small \
     --rank 128 \
     --device cuda:0

LLaMA
-----

Primary documented configuration: ``--plm-type llama --plm-size small``

**LLaMA** is the primary LLM family emphasized in this repository. The main experiment is described around **Llama2 7B**, adapted for ABR decision-making through parameter-efficient fine-tuning rather than full end-to-end retraining.

In the codebase, a representative configuration is:

.. code-block:: python

   plm_type = "llama"
   plm_size = "small"

The LLaMA workflow typically consists of:

1. collecting or generating an experience pool from ABR trajectories
2. encoding state variables such as throughput history, buffer level, chunk metadata, and prior actions
3. adapting the base LLaMA model using low-rank updates
4. testing the adapted model as a bitrate-selection policy

The default experience pool is stored at:

.. code-block:: text

   artifacts/exp_pools/exp_pool.pkl

To test a specific fine-tuned checkpoint:

.. code-block:: bash

   python run_plm.py --test \
     --plm-type llama \
     --plm-size small \
     --rank 128 \
     --device cuda:0 \
     --model-dir your_finetuned_llm_dir

GPT-2
-----

Supported family in the framework: ``gpt2``

**GPT-2** represents the GPT-style autoregressive Transformer option in this framework. It follows the same ABR adaptation pipeline as LLaMA, but uses the GPT-2 backbone instead.

In this setup, the ABR environment state is not treated as free-form text. Numerical features such as throughput history, buffer occupancy, chunk size information, and past bitrate choices are first transformed into a structured representation that the model can process.

A representative adaptation command is:

.. code-block:: bash

   python run_plm.py --adapt \
     --plm-type gpt2 \
     --plm-size small \
     --rank 128 \
     --device cuda:0

GPT-2 is useful in this repository as a comparison backbone for studying how different Transformer families behave under the same ABR state encoding and evaluation setup.

T5-LM
-----

Supported family in the framework: ``t5-lm``

**T5-LM** provides an alternative Transformer family that can be adapted within the same general pipeline. While architecturally different from GPT-style causal decoders, it is still integrated into the shared PLM runner so that ABR experiments can be compared under a common training and testing interface.

In practice, T5-LM allows the repository to explore whether a different pretrained backbone family changes bitrate-selection quality, robustness, or adaptation efficiency.

A representative invocation pattern is:

.. code-block:: bash

   python run_plm.py --adapt \
     --plm-type t5-lm \
     --plm-size small \
     --rank 128 \
     --device cuda:0

OPT
---

Supported family in the framework: ``opt``

**OPT** is another supported PLM family in the LLM-ABR framework. It can be inserted into the same adaptation loop used for LLaMA and GPT-2, making it useful for controlled experiments where the state encoder, experience pool, and evaluation protocol remain fixed while only the backbone changes.

A representative invocation pattern is:

.. code-block:: bash

   python run_plm.py --adapt \
     --plm-type opt \
     --plm-size small \
     --rank 128 \
     --device cuda:0

Using OPT in this way helps compare whether ABR performance differences are driven by the policy formulation itself or by the characteristics of the underlying Transformer family.

Mistral
-------

Supported family in the framework: ``mistral``

**Mistral** is included as a modern decoder-style model family supported by the framework. Within this repository, it fits into the same offline adaptation structure as the other backbones and can be evaluated under the same ABR traces and QoE objectives.

A representative invocation pattern is:

.. code-block:: bash

   python run_plm.py --adapt \
     --plm-type mistral \
     --plm-size small \
     --rank 128 \
     --device cuda:0

Mistral-based experiments are especially useful when comparing newer open-weight LLMs against older Transformer families under the same ABR state and reward design.

Core Components
---------------

Experience Pool
~~~~~~~~~~~~~~~

The experience pool is the offline dataset used to adapt the PLM. It stores trajectories collected from ABR environments, usually generated by baseline policies.

State Encoder
~~~~~~~~~~~~~

ABR state is numerical and structured rather than textual. The state encoder transforms features such as throughput history, buffer size, remaining chunks, chunk metadata, and previous actions into a representation suitable for the model backbone.

Low-Rank Adaptation
~~~~~~~~~~~~~~~~~~~

The code uses parameter-efficient fine-tuning through low-rank adaptation. Instead of updating all parameters of the underlying model, small trainable low-rank modules are added.

RL Policy
~~~~~~~~~

The ``rl_policy.py`` module implements the Transformer-based policy used for offline RL-style bitrate selection.

Supported PLM Families and Sizes
--------------------------------

Supported model families include:

- ``gpt2``
- ``llama``
- ``llava``
- ``t5-lm``
- ``opt``
- ``mistral``

Supported size labels include:

- ``xxs``
- ``xs``
- ``small``
- ``base``
- ``large``
- ``xl``
- ``xxl``

The actual parameter count depends on the PLM family, so the label ``small`` does not map to the same model size across all backbones.

Running Baselines
-----------------

Baseline methods are typically run from a separate TensorFlow-oriented environment:

.. code-block:: bash

   conda create -n abr_tf python=3.7
   conda activate abr_tf
   pip install tensorflow-gpu==1.15
   pip install tensorboard==1.15.0
   pip install tensorboard-plugin-wit==1.8.0
   pip install tflearn==0.5.0
   pip install numba==0.53.1
   pip install gym==0.18.0
   pip install stable-baselines[mpi]==2.10.1
   pip install pandas==1.1.5
   pip install tqdm==4.62.2

Then run representative baselines:

.. code-block:: bash

   python run_baseline.py --model genet --cuda-id 0
   python run_baseline.py --model mpc
   python run_baseline.py --model bba
