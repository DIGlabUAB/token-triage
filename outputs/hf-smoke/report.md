# TokenTriage HF Logit-Lens Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reports: `2`
- Evaluated tokens: `163`
- Critical tokens: `11`

## Main Result

- Entropy-only saved `0.322` simulated layer-compute with critical-token disagreement `1.0`.
- TokenTriage saved `0.291` simulated layer-compute with critical-token disagreement `0.0`.
- Harm-weighted disagreement fell from `0.9544` to `0.6482`.

## Critical Token Mix

- finding: 2
- negation: 5
- temporality: 4

## Interpretation

The result supports the TokenTriage hypothesis when TokenTriage reduces critical-token disagreement while retaining meaningful compute savings. The cost is lower average compute savings because critical tokens are forced to later, stable layers.

## Examples Where Entropy Exited Before Stability

- `CXR578_IM-2176-2001.png` token ` no` (negation) entropy layer 10, stable layer 28, TokenTriage layer 28.
- `CXR578_IM-2176-2001.png` token ` acute` (temporality) entropy layer 13, stable layer 28, TokenTriage layer 28.
- `CXR578_IM-2176-2001.png` token ` without` (negation) entropy layer 16, stable layer 28, TokenTriage layer 28.
- `CXR578_IM-2176-2001.png` token ` consolidation` (finding) entropy layer 19, stable layer 28, TokenTriage layer 28.
- `CXR578_IM-2176-2001.png` token ` prior` (temporality) entropy layer 19, stable layer 28, TokenTriage layer 28.
- `CXR578_IM-2176-2001.png` token ` without` (negation) entropy layer 18, stable layer 28, TokenTriage layer 28.
- `CXR578_IM-2176-2001.png` token ` acute` (temporality) entropy layer 20, stable layer 28, TokenTriage layer 28.
- `CXR776_IM-2319-2001.png` token ` acute` (temporality) entropy layer 2, stable layer 28, TokenTriage layer 28.
- `CXR776_IM-2319-2001.png` token ` negative` (negation) entropy layer 14, stable layer 28, TokenTriage layer 28.
- `CXR776_IM-2319-2001.png` token ` consolidation` (finding) entropy layer 19, stable layer 28, TokenTriage layer 28.
- `CXR776_IM-2319-2001.png` token ` negative` (negation) entropy layer 17, stable layer 28, TokenTriage layer 28.
