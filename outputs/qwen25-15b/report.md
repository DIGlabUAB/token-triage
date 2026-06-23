# Semantic Token Triage Report

## Claim

This local experiment tests whether confidence-only early exit can create disproportionate errors on clinically meaningful tokens such as negation, laterality, severity, temporality, medications, and measurements.

## Run Summary

- **model**: `qwen2.5:1.5b`
- **case_count**: `8`
- **token_count**: `67`
- **critical_token_count**: `15`
- **ollama_logprobs_available**: `False`
- **entropy_only_avg_exit_layer**: `7.34`
- **criticality_aware_avg_exit_layer**: `8.33`
- **entropy_only_simulated_compute_saved**: `0.738`
- **criticality_aware_simulated_compute_saved**: `0.703`
- **entropy_only_dangerous_errors**: `15`
- **criticality_aware_dangerous_errors**: `0`

## Critical Token Mix

- **finding**: 2
- **laterality**: 8
- **negation**: 5

## Example Cases

- `ollama-001` (ollama): Left anterior chest contusion with pleural effusion.
- `ollama-002` (ollama): Right lower extremity deep vein thrombosis without symptoms.
- `ollama-003` (ollama): Abdominal pain with no relevant past medical history.
- `ollama-004` (ollama): Bilateral lower extremity edema, negative for varicose veins.
- `ollama-005` (ollama): Cerebral angiography showing multiple aneurysms.
- `ollama-006` (ollama): Urinary retention due to neurogenic bladder dysfunction.
- `ollama-007` (ollama): Left renal calculi without symptoms or complications.
- `ollama-008` (ollama): Right upper extremity lymphedema with no associated infection.

## Highest-Risk Entropy-Only Exits

- `ollama-001` token `Left` (laterality) exited at layer 8, stabilized at layer 13.
- `ollama-001` token `effusion` (finding) exited at layer 4, stabilized at layer 8.
- `ollama-002` token `Right` (laterality) exited at layer 7, stabilized at layer 11.
- `ollama-002` token `lower` (laterality) exited at layer 9, stabilized at layer 14.
- `ollama-002` token `without` (negation) exited at layer 1, stabilized at layer 6.
- `ollama-003` token `no` (negation) exited at layer 1, stabilized at layer 5.
- `ollama-004` token `Bilateral` (laterality) exited at layer 7, stabilized at layer 11.
- `ollama-004` token `lower` (laterality) exited at layer 10, stabilized at layer 14.
- `ollama-004` token `edema` (finding) exited at layer 5, stabilized at layer 9.
- `ollama-004` token `negative` (negation) exited at layer 3, stabilized at layer 7.
- `ollama-007` token `Left` (laterality) exited at layer 7, stabilized at layer 12.
- `ollama-007` token `without` (negation) exited at layer 3, stabilized at layer 8.
- `ollama-008` token `Right` (laterality) exited at layer 7, stabilized at layer 12.
- `ollama-008` token `upper` (laterality) exited at layer 6, stabilized at layer 10.
- `ollama-008` token `no` (negation) exited at layer 5, stabilized at layer 9.

## Notes

- Ollama was used locally for report generation.
- The installed Ollama endpoint did not expose generated-token logprobs during verification, so `confidence_proxy` is a deterministic stand-in for the first runnable local prototype.
- The package keeps the scoring boundary isolated so a backend that returns true logprobs or hidden states can replace the proxy without changing the triage analysis.
