# Paper-Grade TokenTriage Experiment

## Setup

- Dataset: `Santhosh1705kumar/radiology-reports-chest` / `test`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reports: `64`
- Evaluated tokens: `1942`
- Critical tokens: `150`
- Transformer layers: `28`

## Critical Token Mix

- finding: 12
- laterality: 76
- measurement: 38
- measurement_unit: 3
- negation: 9
- severity: 10
- temporality: 2

## Strongest Baseline Comparison

| Method | Setting | Compute saved | Critical disagreement | HWD | Excess NLL | 95% CI critical |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| entropy | threshold=0.10 | 0.3016 | 0.6200 | 0.5101 | 4.9327 | 0.5541-0.6772 |
| margin | threshold=0.50 | 0.5255 | 0.7600 | 0.7262 | 7.2080 | 0.7132-0.8067 |
| tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.4656 | 0.1533 | 0.5449 | 6.5702 | 0.1081-0.1951 |
| tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.4656 | 0.1533 | 0.5449 | 6.5702 | 0.1081-0.1951 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8714 | 0.0000 | 0.6563 | 9.7503 | 0.0000-0.0000 |

## Top Deployable Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | fixed-depth | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| 2 | tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.4656 | 0.1533 | 0.5449 |
| 3 | tokentriage-kstable | entropy=0.15,min=0.85,k=4 | 0.3816 | 0.1533 | 0.4748 |
| 4 | tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.2726 | 0.1533 | 0.3558 |
| 5 | tokentriage-kstable | entropy=0.55,min=0.85,k=4 | 0.8697 | 0.1600 | 0.7087 |
| 6 | tokentriage-kstable | entropy=0.42,min=0.85,k=4 | 0.7642 | 0.1600 | 0.6844 |
| 7 | tokentriage-kstable | entropy=0.30,min=0.85,k=4 | 0.6067 | 0.1600 | 0.6356 |
| 8 | tokentriage-kstable | entropy=0.10,min=0.72,k=4 | 0.2762 | 0.2800 | 0.3976 |
| 9 | tokentriage-kstable | entropy=0.15,min=0.72,k=4 | 0.3856 | 0.3000 | 0.5232 |
| 10 | tokentriage-kstable | entropy=0.20,min=0.72,k=4 | 0.4699 | 0.3133 | 0.5978 |
| 11 | tokentriage-kstable | entropy=0.10,min=0.85,k=3 | 0.2746 | 0.3133 | 0.4086 |
| 12 | tokentriage-kstable | entropy=0.10,min=0.65,k=4 | 0.2783 | 0.3200 | 0.4107 |

## Interpretation

This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
