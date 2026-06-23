# TokenTriage HF Logit-Lens Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reports: `32`
- Evaluated tokens: `1799`
- Critical tokens: `167`

## Main Result

- Entropy-only saved `0.844` simulated layer-compute with critical-token disagreement `0.988`.
- TokenTriage saved `0.76` simulated layer-compute with critical-token disagreement `0.6647`.
- Harm-weighted disagreement fell from `0.9707` to `0.8474`.

## Critical Token Mix

- finding: 20
- laterality: 33
- negation: 55
- severity: 11
- temporality: 48

## Interpretation

The result supports the TokenTriage hypothesis when TokenTriage reduces critical-token disagreement while retaining meaningful compute savings. The cost is lower average compute savings because critical tokens are forced to later, stable layers.

## Examples Where Entropy Exited Before Stability

- `CXR578_IM-2176-2001.png` token ` no` (negation) entropy layer 2, stable layer 25, TokenTriage layer 25.
- `CXR578_IM-2176-2001.png` token ` acute` (temporality) entropy layer 1, stable layer 27, TokenTriage layer 27.
- `CXR578_IM-2176-2001.png` token ` prior` (temporality) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR578_IM-2176-2001.png` token ` without` (negation) entropy layer 1, stable layer 22, TokenTriage layer 22.
- `CXR776_IM-2319-2001.png` token ` negative` (negation) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR837_IM-2361-1001.png` token ` prior` (temporality) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR1281_IM-0188-2001.png` token ` no` (negation) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR3600_IM-1776-4004.png` token ` no` (negation) entropy layer 2, stable layer 28, TokenTriage layer 28.
- `CXR3600_IM-1776-4004.png` token ` right` (laterality) entropy layer 1, stable layer 27, TokenTriage layer 27.
- `CXR3600_IM-1776-4004.png` token ` chronic` (temporality) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR3600_IM-1776-4004.png` token ` old` (temporality) entropy layer 1, stable layer 27, TokenTriage layer 27.
- `CXR47_IM-2098-2001.png` token ` prior` (temporality) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR864_IM-2384-1002.png` token ` no` (negation) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR864_IM-2384-1002.png` token ` stable` (temporality) entropy layer 1, stable layer 26, TokenTriage layer 26.
- `CXR2390_IM-0944-1002.png` token ` no` (negation) entropy layer 2, stable layer 26, TokenTriage layer 26.
- `CXR2390_IM-0944-1002.png` token ` acute` (temporality) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR2390_IM-0944-1002.png` token ` mild` (severity) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR2390_IM-0944-1002.png` token ` left` (laterality) entropy layer 1, stable layer 28, TokenTriage layer 28.
- `CXR3993_IM-2044-1002.png` token ` without` (negation) entropy layer 22, stable layer 27, TokenTriage layer 27.
- `CXR3993_IM-2044-1002.png` token ` left` (laterality) entropy layer 1, stable layer 28, TokenTriage layer 28.
