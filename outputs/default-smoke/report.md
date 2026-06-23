# Semantic Token Triage Report

## Claim

This local experiment tests whether confidence-only early exit can create disproportionate errors on clinically meaningful tokens such as negation, laterality, severity, temporality, medications, and measurements.

## Run Summary

- **model**: `qwen2.5:1.5b`
- **case_count**: `4`
- **token_count**: `35`
- **critical_token_count**: `9`
- **ollama_logprobs_available**: `False`
- **entropy_only_avg_exit_layer**: `7.49`
- **criticality_aware_avg_exit_layer**: `8.63`
- **entropy_only_simulated_compute_saved**: `0.733`
- **criticality_aware_simulated_compute_saved**: `0.692`
- **entropy_only_dangerous_errors**: `9`
- **criticality_aware_dangerous_errors**: `0`

## Critical Token Mix

- **finding**: 1
- **laterality**: 5
- **measurement**: 1
- **negation**: 1
- **temporality**: 1

## Example Cases

- `ollama-001` (ollama): Left upper quadrant pain with elevated liver enzymes.
- `ollama-002` (ollama): Right lower extremity edema without history of trauma.
- `ollama-003` (ollama): Bilateral lung infiltrates consistent with recent viral infection.
- `ollama-004` (ollama): New onset hypertension within 7 days post-surgery.

## Highest-Risk Entropy-Only Exits

- `ollama-001` token `Left` (laterality) exited at layer 8, stabilized at layer 13.
- `ollama-001` token `upper` (laterality) exited at layer 10, stabilized at layer 14.
- `ollama-002` token `Right` (laterality) exited at layer 7, stabilized at layer 11.
- `ollama-002` token `lower` (laterality) exited at layer 9, stabilized at layer 14.
- `ollama-002` token `edema` (finding) exited at layer 7, stabilized at layer 12.
- `ollama-002` token `without` (negation) exited at layer 1, stabilized at layer 6.
- `ollama-003` token `Bilateral` (laterality) exited at layer 8, stabilized at layer 13.
- `ollama-004` token `New` (temporality) exited at layer 9, stabilized at layer 12.
- `ollama-004` token `7` (measurement) exited at layer 4, stabilized at layer 8.

## Notes

- Ollama was used locally for report generation.
- The installed Ollama endpoint did not expose generated-token logprobs during verification, so `confidence_proxy` is a deterministic stand-in for the first runnable local prototype.
- The package keeps the scoring boundary isolated so a backend that returns true logprobs or hidden states can replace the proxy without changing the triage analysis.
