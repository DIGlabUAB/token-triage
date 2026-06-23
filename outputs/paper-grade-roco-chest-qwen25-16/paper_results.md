# Paper-Grade TokenTriage Experiment

## Setup

- Dataset: `Santhosh1705kumar/radiology-reports-chest` / `test`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reports: `16`
- Evaluated tokens: `573`
- Critical tokens: `59`
- Transformer layers: `28`

## Critical Token Mix

- finding: 5
- laterality: 21
- measurement: 27
- measurement_unit: 3
- negation: 2
- severity: 1

## Strongest Baseline Comparison

| Method | Setting | Compute saved | Critical disagreement | HWD | Excess NLL | 95% CI critical |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| entropy | threshold=0.10 | 0.3221 | 0.6102 | 0.5372 | 5.4411 | 0.5075-0.7368 |
| margin | threshold=0.50 | 0.5359 | 0.7458 | 0.7473 | 7.5239 | 0.6809-0.8261 |
| tokentriage-kstable | entropy=0.55,min=0.85,k=4 | 0.8481 | 0.1525 | 0.6462 | 9.5735 | 0.0857-0.2090 |
| tokentriage-kstable | entropy=0.55,min=0.85,k=4 | 0.8481 | 0.1525 | 0.6462 | 9.5735 | 0.0857-0.2090 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8515 | 0.0000 | 0.5847 | 9.5347 | 0.0000-0.0000 |

## Top Deployable Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | fixed-depth | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| 2 | tokentriage-kstable | entropy=0.55,min=0.85,k=4 | 0.8481 | 0.1525 | 0.6462 |
| 3 | tokentriage-kstable | entropy=0.42,min=0.85,k=4 | 0.7528 | 0.1525 | 0.6277 |
| 4 | tokentriage-kstable | entropy=0.30,min=0.85,k=4 | 0.5962 | 0.1525 | 0.5861 |
| 5 | tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.4686 | 0.1525 | 0.5156 |
| 6 | tokentriage-kstable | entropy=0.15,min=0.85,k=4 | 0.3959 | 0.1525 | 0.4567 |
| 7 | tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.2848 | 0.1525 | 0.3515 |
| 8 | tokentriage-kstable | entropy=0.10,min=0.72,k=4 | 0.2880 | 0.2373 | 0.3856 |
| 9 | tokentriage-kstable | entropy=0.10,min=0.85,k=3 | 0.2875 | 0.2373 | 0.3861 |
| 10 | tokentriage-kstable | entropy=0.15,min=0.72,k=4 | 0.3994 | 0.2542 | 0.4977 |
| 11 | tokentriage-kstable | entropy=0.15,min=0.85,k=3 | 0.3987 | 0.2542 | 0.4982 |
| 12 | tokentriage-kstable | entropy=0.20,min=0.72,k=4 | 0.4726 | 0.2712 | 0.5636 |

## Interpretation

This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
