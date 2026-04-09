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
