# Robustness_Cascade


This repository studies the robustness of Gatekeeper-style prediction cascades under corrupted and perturbed inputs.

The project investigates how a small-to-large model cascade behaves under distribution shifts, with a particular focus on:
- selective prediction / predict-or-defer behavior,
- deferral reliability,
- confidence separation between accepted and deferred samples,
- robustness under common corruptions and temporal perturbations.

The repository currently includes:
- baseline Gatekeeper cascade training,
- corruption and perturbation evaluation pipelines,
- metric computation for cascade robustness,
- analysis utilities for deferral and error propagation.

## Installation

Install the required dependencies:

```bash
pip install torch torchvision

## Running Code
Follow the sequence:


python train.py

Running `train.py` will:
- train the small model,
- train the large model,
- apply Gatekeeper fine-tuning with an alpha sweep,
- prepare the cascade setup for downstream robustness evaluation.

python evaluate.py

- runs models, collects confidence scores, computes s_o and s_d(see paper) 

python plot.py
- takes those results and produces Figure 3 style plots (see paper)
