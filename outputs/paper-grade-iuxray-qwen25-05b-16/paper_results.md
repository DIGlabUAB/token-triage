# Paper-Grade TokenTriage Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Reports: `16`
- Evaluated tokens: `921`
- Critical tokens: `84`
- Transformer layers: `24`

## Critical Token Mix

- finding: 9
- laterality: 12
- negation: 31
- severity: 3
- temporality: 29

## Strongest Baseline Comparison

| Method | Setting | Compute saved | Critical disagreement | HWD | Excess NLL | 95% CI critical |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| entropy | threshold=0.10 | 0.0896 | 0.1310 | 0.1189 | 1.3245 | 0.0800-0.1765 |
| margin | threshold=0.50 | 0.2863 | 0.3929 | 0.3884 | 4.5451 | 0.2787-0.4653 |
| tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.0794 | 0.0000 | 0.0700 | 1.1703 | 0.0000-0.0000 |
| tokentriage-kstable | entropy=0.42,min=0.85,k=4 | 0.4707 | 0.2143 | 0.5092 | 6.7970 | 0.1287-0.3582 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.7955 | 0.0000 | 0.6062 | 9.9724 | 0.0000-0.0000 |

## Top Deployable Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.0794 | 0.0000 | 0.0700 |
| 2 | fixed-depth | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| 3 | tokentriage-kstable | entropy=0.10,min=0.72,k=4 | 0.0798 | 0.0238 | 0.0784 |
| 4 | tokentriage-kstable | entropy=0.10,min=0.85,k=3 | 0.0798 | 0.0238 | 0.0792 |
| 5 | tokentriage-kstable | entropy=0.10,min=0.65,k=4 | 0.0802 | 0.0357 | 0.0826 |
| 6 | tokentriage-kstable | entropy=0.10,min=0.85,k=2 | 0.0800 | 0.0357 | 0.0833 |
| 7 | tokentriage-kstable | entropy=0.10,min=0.50,k=4 | 0.0808 | 0.0476 | 0.0875 |
| 8 | tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.1681 | 0.0595 | 0.1833 |
| 9 | tokentriage-kstable | entropy=0.15,min=0.85,k=4 | 0.1316 | 0.0595 | 0.1497 |
| 10 | tokentriage-kstable | entropy=0.10,min=0.72,k=3 | 0.0808 | 0.0595 | 0.0919 |
| 11 | tokentriage-kstable | entropy=0.30,min=0.85,k=4 | 0.2959 | 0.0833 | 0.3114 |
| 12 | tokentriage-kstable | entropy=0.10,min=0.65,k=3 | 0.0814 | 0.0833 | 0.1009 |

## Interpretation

This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
