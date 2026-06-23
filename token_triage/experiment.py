from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .clinical_tokens import detect_critical_tokens, token_is_critical, tokenize
from .ollama import OllamaClient
from .synthetic import Case, ollama_cases, template_cases


@dataclass(frozen=True)
class TokenRecord:
    case_id: str
    token_index: int
    token: str
    category: str
    criticality: float
    confidence_proxy: float
    stabilization_layer: int
    entropy_exit_layer: int
    criticality_exit_layer: int
    entropy_dangerous_error: bool
    criticality_dangerous_error: bool


@dataclass(frozen=True)
class ExperimentSummary:
    model: str
    case_count: int
    token_count: int
    critical_token_count: int
    ollama_logprobs_available: bool
    entropy_only_avg_exit_layer: float
    criticality_aware_avg_exit_layer: float
    entropy_only_simulated_compute_saved: float
    criticality_aware_simulated_compute_saved: float
    entropy_only_dangerous_errors: int
    criticality_aware_dangerous_errors: int


def confidence_proxy(token: str, index: int, case_id: str, criticality: float) -> float:
    seed = f"{case_id}:{index}:{token.lower()}"
    jitter = random.Random(seed).uniform(-0.08, 0.08)
    lexical_prior = 0.72
    if token.lower() in {"no", "not", "without", "negative"}:
        lexical_prior = 0.9
    elif token.replace(".", "", 1).isdigit():
        lexical_prior = 0.86
    elif criticality >= 0.9:
        lexical_prior = 0.82
    value = lexical_prior + jitter
    return round(min(0.98, max(0.51, value)), 3)


def simulate_stabilization_layer(confidence: float, criticality: float, max_layers: int) -> int:
    early = 1 + math.floor((1.0 - confidence) * max_layers)
    risk_delay = round(criticality * 4)
    return min(max_layers, max(1, early + risk_delay))


def run_experiment(
    client: OllamaClient,
    *,
    cases: int,
    output_dir: Path,
    use_templates_only: bool = False,
    max_layers: int = 28,
) -> tuple[ExperimentSummary, list[Case], list[TokenRecord]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logprobs_available = client.supports_logprobs()
    generated_cases = template_cases(cases) if use_templates_only else ollama_cases(client, cases)

    records: list[TokenRecord] = []
    for case in generated_cases:
        critical_by_index = {signal.index: signal for signal in detect_critical_tokens(case.report)}
        for index, token in enumerate(tokenize(case.report)):
            signal = critical_by_index.get(index)
            if signal:
                category = signal.category
                criticality = signal.criticality
            else:
                category = "ordinary"
                criticality = 0.15
            confidence = confidence_proxy(token, index, case.case_id, criticality)
            stabilization_layer = simulate_stabilization_layer(confidence, criticality, max_layers)
            entropy_exit_layer = max(1, min(max_layers, round((1.0 - confidence) * max_layers)))
            if token_is_critical(token):
                criticality_exit_layer = max(stabilization_layer, entropy_exit_layer)
            else:
                criticality_exit_layer = entropy_exit_layer
            records.append(
                TokenRecord(
                    case_id=case.case_id,
                    token_index=index,
                    token=token,
                    category=category,
                    criticality=criticality,
                    confidence_proxy=confidence,
                    stabilization_layer=stabilization_layer,
                    entropy_exit_layer=entropy_exit_layer,
                    criticality_exit_layer=criticality_exit_layer,
                    entropy_dangerous_error=criticality >= 0.8 and entropy_exit_layer < stabilization_layer,
                    criticality_dangerous_error=criticality >= 0.8 and criticality_exit_layer < stabilization_layer,
                )
            )

    token_count = len(records)
    entropy_avg = sum(record.entropy_exit_layer for record in records) / token_count
    criticality_avg = sum(record.criticality_exit_layer for record in records) / token_count
    summary = ExperimentSummary(
        model=client.config.model,
        case_count=len(generated_cases),
        token_count=token_count,
        critical_token_count=sum(1 for record in records if record.criticality >= 0.8),
        ollama_logprobs_available=logprobs_available,
        entropy_only_avg_exit_layer=round(entropy_avg, 2),
        criticality_aware_avg_exit_layer=round(criticality_avg, 2),
        entropy_only_simulated_compute_saved=round(1.0 - entropy_avg / max_layers, 3),
        criticality_aware_simulated_compute_saved=round(1.0 - criticality_avg / max_layers, 3),
        entropy_only_dangerous_errors=sum(record.entropy_dangerous_error for record in records),
        criticality_aware_dangerous_errors=sum(record.criticality_dangerous_error for record in records),
    )

    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    (output_dir / "cases.json").write_text(
        json.dumps([asdict(case) for case in generated_cases], indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "token_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return summary, generated_cases, records
