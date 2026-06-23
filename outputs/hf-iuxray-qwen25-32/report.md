# TokenTriage HF Logit-Lens Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reports: `32`
- Evaluated tokens: `1799`
- Critical tokens: `167`

## Main Result

- Entropy-only saved `0.332` simulated layer-compute with critical-token disagreement `1.0`.
- TokenTriage saved `0.29` simulated layer-compute with critical-token disagreement `0.0`.
- Harm-weighted disagreement fell from `0.9749` to `0.5946`.

## Critical Token Mix

- finding: 20
- laterality: 33
- negation: 55
- severity: 11
- temporality: 48

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
- `CXR837_IM-2361-1001.png` token ` acute` (temporality) entropy layer 2, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` without` (negation) entropy layer 16, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` consolidation` (finding) entropy layer 20, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` right` (laterality) entropy layer 14, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` prior` (temporality) entropy layer 18, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` no` (negation) entropy layer 14, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` large` (severity) entropy layer 19, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` no` (negation) entropy layer 17, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` acute` (temporality) entropy layer 20, stable layer 28, TokenTriage layer 28.
