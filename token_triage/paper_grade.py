from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .clinical_tokens import token_is_critical
from .hf_dataset import HfReport, fetch_hf_reports


@dataclass(frozen=True)
class TraceRecord:
    report_id: str
    row_id: int
    position: int
    token_text: str
    category: str
    criticality: float
    final_top_token: str
    final_top_matches_target: bool
    final_target_nll: float
    stable_layer: int
    entropy_by_layer: list[float]
    margin_by_layer: list[float]
    target_nll_by_layer: list[float]
    top_id_by_layer: list[int]
    pred_matches_final_by_layer: list[bool]


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    setting: str
    tokens: int
    critical_tokens: int
    avg_exit_layer: float
    compute_saved: float
    disagreement_all: float
    disagreement_critical: float
    harm_weighted_disagreement: float
    mean_excess_nll: float
    critical_excess_nll: float
    critical_rescue_rate_vs_entropy: float | None = None
    disagreement_critical_ci95: str | None = None
    harm_weighted_ci95: str | None = None


def _load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, torch


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def _criticality(token_text: str) -> tuple[str, float]:
    value = token_is_critical(token_text.strip())
    if value:
        return value
    return "ordinary", 0.15


def _stable_layer(matches_final: list[bool]) -> int:
    for idx in range(len(matches_final)):
        if all(matches_final[idx:]):
            return idx + 1
    return len(matches_final)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_traces(path: Path) -> list[TraceRecord]:
    traces: list[TraceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            traces.append(TraceRecord(**data))
    return traces


def generate_traces(
    *,
    output_dir: Path,
    dataset: str,
    config: str,
    split: str,
    limit: int,
    offset: int,
    text_column: str,
    id_column: str,
    model_name: str,
    device: str,
    max_length: int,
) -> tuple[list[TraceRecord], list[HfReport], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "traces.jsonl"
    reports_path = output_dir / "reports.json"
    meta_path = output_dir / "trace_meta.json"
    if trace_path.exists() and reports_path.exists() and meta_path.exists():
        reports = [HfReport(**row) for row in json.loads(reports_path.read_text(encoding="utf-8"))]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return read_traces(trace_path), reports, int(meta["layers"])

    reports = fetch_hf_reports(
        dataset=dataset,
        config=config,
        split=split,
        limit=limit,
        offset=offset,
        text_column=text_column,
        id_column=id_column,
    )
    tokenizer, model, torch = _load_model(model_name, device)
    actual_device = next(model.parameters()).device
    traces: list[TraceRecord] = []

    with torch.inference_mode():
        for report in reports:
            encoded = tokenizer(
                report.text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(actual_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(actual_device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]
            vocab_norm = math.log(float(model.config.vocab_size))
            final_logits = outputs.logits[0, :-1, :].float()
            final_log_probs = torch.log_softmax(final_logits, dim=-1)
            final_top_ids = torch.argmax(final_logits, dim=-1)
            target_ids = input_ids[0, 1:]

            for pos in range(target_ids.shape[0]):
                target_id = int(target_ids[pos].item())
                token_text = _decode_token(tokenizer, target_id)
                if not token_text.strip():
                    continue
                category, criticality = _criticality(token_text)
                final_pred_id = int(final_top_ids[pos].item())
                entropies: list[float] = []
                margins: list[float] = []
                nlls: list[float] = []
                top_ids: list[int] = []
                matches: list[bool] = []
                for layer_idx, hidden in enumerate(hidden_states):
                    hidden_vec = hidden[0, pos, :]
                    if layer_idx < len(hidden_states) - 1 and hasattr(model, "model") and hasattr(model.model, "norm"):
                        hidden_vec = model.model.norm(hidden_vec)
                    logits = model.lm_head(hidden_vec).float()
                    probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    top2 = torch.topk(probs, k=2)
                    top_id = int(top2.indices[0].item())
                    entropies.append(float((-torch.sum(probs * log_probs).item()) / vocab_norm))
                    margins.append(float(top2.values[0].item() - top2.values[1].item()))
                    nlls.append(float(-log_probs[target_id].item()))
                    top_ids.append(top_id)
                    matches.append(top_id == final_pred_id)

                traces.append(
                    TraceRecord(
                        report_id=report.report_id,
                        row_id=report.row_id,
                        position=pos + 1,
                        token_text=token_text,
                        category=category,
                        criticality=criticality,
                        final_top_token=_decode_token(tokenizer, final_pred_id),
                        final_top_matches_target=final_pred_id == target_id,
                        final_target_nll=float(-final_log_probs[pos, target_id].item()),
                        stable_layer=_stable_layer(matches),
                        entropy_by_layer=entropies,
                        margin_by_layer=margins,
                        target_nll_by_layer=nlls,
                        top_id_by_layer=top_ids,
                        pred_matches_final_by_layer=matches,
                    )
                )

    layers = len(traces[0].entropy_by_layer) if traces else 0
    write_jsonl(trace_path, [asdict(trace) for trace in traces])
    reports_path.write_text(json.dumps([asdict(report) for report in reports], indent=2) + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "model": model_name,
                "limit": limit,
                "offset": offset,
                "max_length": max_length,
                "layers": layers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return traces, reports, layers


def entropy_exit(trace: TraceRecord, threshold: float) -> int:
    for idx, entropy in enumerate(trace.entropy_by_layer, start=1):
        if entropy <= threshold:
            return idx
    return len(trace.entropy_by_layer)


def margin_exit(trace: TraceRecord, threshold: float) -> int:
    for idx, margin in enumerate(trace.margin_by_layer, start=1):
        if margin >= threshold:
            return idx
    return len(trace.margin_by_layer)


def fixed_exit(trace: TraceRecord, layer_fraction: float) -> int:
    return min(len(trace.entropy_by_layer), max(1, math.ceil(len(trace.entropy_by_layer) * layer_fraction)))


def token_triage_exit(trace: TraceRecord, base_exit: int, min_fraction: float, use_stability: bool) -> int:
    if trace.criticality < 0.8:
        return base_exit
    min_layer = fixed_exit(trace, min_fraction)
    if use_stability:
        return max(base_exit, min_layer, trace.stable_layer)
    return max(base_exit, min_layer)


def k_stable_exit(trace: TraceRecord, base_exit: int, min_fraction: float, patience: int) -> int:
    if trace.criticality < 0.8:
        return base_exit
    min_layer = fixed_exit(trace, min_fraction)
    start = max(base_exit, min_layer, patience)
    for layer in range(start, len(trace.top_id_by_layer) + 1):
        window = trace.top_id_by_layer[layer - patience : layer]
        if len(set(window)) == 1:
            return layer
    return len(trace.top_id_by_layer)


def evaluate_policy(
    traces: list[TraceRecord],
    *,
    policy: str,
    setting: str,
    exits: dict[int, int],
    layers: int,
    baseline_entropy_exits: dict[int, int] | None = None,
) -> PolicyResult:
    exit_layers = [exits[idx] for idx in range(len(traces))]
    disagreements = [not trace.pred_matches_final_by_layer[exit_layers[idx] - 1] for idx, trace in enumerate(traces)]
    critical_mask = [trace.criticality >= 0.8 for trace in traces]
    critical_disagreements = [disagree for disagree, is_critical in zip(disagreements, critical_mask) if is_critical]
    weights = [trace.criticality for trace in traces]
    hwd = sum(weight for weight, disagree in zip(weights, disagreements) if disagree) / (sum(weights) or 1.0)
    excess_nll = [
        trace.target_nll_by_layer[exit_layers[idx] - 1] - trace.final_target_nll for idx, trace in enumerate(traces)
    ]
    critical_excess = [value for value, is_critical in zip(excess_nll, critical_mask) if is_critical]
    rescue_rate = None
    if baseline_entropy_exits is not None:
        rescued = 0
        rescue_candidates = 0
        for idx, trace in enumerate(traces):
            if trace.criticality < 0.8:
                continue
            base_exit = baseline_entropy_exits[idx]
            base_disagree = not trace.pred_matches_final_by_layer[base_exit - 1]
            this_disagree = not trace.pred_matches_final_by_layer[exit_layers[idx] - 1]
            if base_disagree:
                rescue_candidates += 1
                if not this_disagree:
                    rescued += 1
        rescue_rate = rescued / rescue_candidates if rescue_candidates else 0.0
    return PolicyResult(
        policy=policy,
        setting=setting,
        tokens=len(traces),
        critical_tokens=sum(critical_mask),
        avg_exit_layer=round(_mean(exit_layers), 3),
        compute_saved=round(1 - _mean(exit_layers) / layers, 4),
        disagreement_all=round(_mean([float(value) for value in disagreements]), 4),
        disagreement_critical=round(_mean([float(value) for value in critical_disagreements]), 4),
        harm_weighted_disagreement=round(hwd, 4),
        mean_excess_nll=round(_mean(excess_nll), 4),
        critical_excess_nll=round(_mean(critical_excess), 4),
        critical_rescue_rate_vs_entropy=round(rescue_rate, 4) if rescue_rate is not None else None,
    )


def bootstrap_policy(
    traces: list[TraceRecord],
    *,
    exits: dict[int, int],
    samples: int,
    seed: int,
) -> tuple[str, str]:
    by_report: dict[str, list[int]] = defaultdict(list)
    for idx, trace in enumerate(traces):
        by_report[trace.report_id].append(idx)
    report_ids = sorted(by_report)
    rng = random.Random(seed)
    crit_rates: list[float] = []
    hwd_rates: list[float] = []
    for _ in range(samples):
        indices: list[int] = []
        for _ in report_ids:
            indices.extend(by_report[rng.choice(report_ids)])
        critical_values: list[float] = []
        weighted_error = 0.0
        weight_total = 0.0
        for idx in indices:
            trace = traces[idx]
            exit_layer = exits[idx]
            disagree = not trace.pred_matches_final_by_layer[exit_layer - 1]
            if trace.criticality >= 0.8:
                critical_values.append(float(disagree))
            weight_total += trace.criticality
            if disagree:
                weighted_error += trace.criticality
        crit_rates.append(_mean(critical_values))
        hwd_rates.append(weighted_error / (weight_total or 1.0))
    crit_ci = f"{_quantile(crit_rates, 0.025):.4f}-{_quantile(crit_rates, 0.975):.4f}"
    hwd_ci = f"{_quantile(hwd_rates, 0.025):.4f}-{_quantile(hwd_rates, 0.975):.4f}"
    return crit_ci, hwd_ci


def build_policy_suite(
    traces: list[TraceRecord],
    *,
    layers: int,
    entropy_thresholds: list[float],
    margin_thresholds: list[float],
    triage_min_fractions: list[float],
    bootstrap_samples: int,
) -> list[PolicyResult]:
    results: list[PolicyResult] = []
    exit_maps: list[tuple[str, str, dict[int, int], dict[int, int] | None]] = []

    for fraction in [0.25, 0.5, 0.75, 1.0]:
        exits = {idx: fixed_exit(trace, fraction) for idx, trace in enumerate(traces)}
        exit_maps.append(("fixed-depth", f"{fraction:.2f}", exits, None))

    stability_exits = {idx: trace.stable_layer for idx, trace in enumerate(traces)}
    exit_maps.append(("stability-oracle", "first-final-stable", stability_exits, None))

    for threshold in entropy_thresholds:
        base = {idx: entropy_exit(trace, threshold) for idx, trace in enumerate(traces)}
        exit_maps.append(("entropy", f"threshold={threshold:.2f}", base, None))
        for fraction in triage_min_fractions:
            exits = {
                idx: token_triage_exit(trace, base[idx], min_fraction=fraction, use_stability=False)
                for idx, trace in enumerate(traces)
            }
            exit_maps.append(("tokentriage-minlayer", f"entropy={threshold:.2f},min={fraction:.2f}", exits, base))
            exits_stable = {
                idx: token_triage_exit(trace, base[idx], min_fraction=fraction, use_stability=True)
                for idx, trace in enumerate(traces)
            }
            exit_maps.append(("tokentriage-stable", f"entropy={threshold:.2f},min={fraction:.2f}", exits_stable, base))
            for patience in [2, 3, 4]:
                exits_kstable = {
                    idx: k_stable_exit(trace, base[idx], min_fraction=fraction, patience=patience)
                    for idx, trace in enumerate(traces)
                }
                exit_maps.append(
                    (
                        "tokentriage-kstable",
                        f"entropy={threshold:.2f},min={fraction:.2f},k={patience}",
                        exits_kstable,
                        base,
                    )
                )

    for threshold in margin_thresholds:
        exits = {idx: margin_exit(trace, threshold) for idx, trace in enumerate(traces)}
        exit_maps.append(("margin", f"threshold={threshold:.2f}", exits, None))

    for name, setting, exits, baseline in exit_maps:
        result = evaluate_policy(
            traces,
            policy=name,
            setting=setting,
            exits=exits,
            layers=layers,
            baseline_entropy_exits=baseline,
        )
        if bootstrap_samples > 0 and name in {
            "entropy",
            "margin",
            "tokentriage-minlayer",
            "tokentriage-kstable",
            "tokentriage-stable",
        }:
            crit_ci, hwd_ci = bootstrap_policy(traces, exits=exits, samples=bootstrap_samples, seed=17)
            result = PolicyResult(
                **{
                    **asdict(result),
                    "disagreement_critical_ci95": crit_ci,
                    "harm_weighted_ci95": hwd_ci,
                }
            )
        results.append(result)
    return results


def write_paper_outputs(
    output_dir: Path,
    *,
    results: list[PolicyResult],
    traces: list[TraceRecord],
    reports: list[HfReport],
    layers: int,
    model_name: str,
    dataset: str,
    split: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "policy_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    (output_dir / "policy_results.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )

    critical = [trace for trace in traces if trace.criticality >= 0.8]
    category_counts = Counter(trace.category for trace in critical)
    deployable_policies = {"entropy", "margin", "fixed-depth", "tokentriage-minlayer", "tokentriage-kstable"}
    deployable_pareto = sorted(
        [result for result in results if result.policy in deployable_policies],
        key=lambda r: (r.disagreement_critical, -r.compute_saved, r.avg_exit_layer),
    )
    best_entropy = min((r for r in results if r.policy == "entropy"), key=lambda r: r.disagreement_critical)
    best_margin = min((r for r in results if r.policy == "margin"), key=lambda r: r.disagreement_critical)
    deployable_triage = [r for r in results if r.policy in {"tokentriage-minlayer", "tokentriage-kstable"}]
    best_triage = min(
        deployable_triage,
        key=lambda r: (r.disagreement_critical, -r.compute_saved),
    )
    high_saving_triage = max(
        (r for r in deployable_triage if r.compute_saved >= 0.45),
        key=lambda r: (-r.disagreement_critical, r.compute_saved),
        default=best_triage,
    )
    oracle_triage = min(
        (r for r in results if r.policy == "tokentriage-stable"),
        key=lambda r: (r.disagreement_critical, -r.compute_saved),
    )
    lines = [
        "# Paper-Grade TokenTriage Experiment",
        "",
        "## Setup",
        "",
        f"- Dataset: `{dataset}` / `{split}`",
        f"- Model: `{model_name}`",
        f"- Reports: `{len(reports)}`",
        f"- Evaluated tokens: `{len(traces)}`",
        f"- Critical tokens: `{len(critical)}`",
        f"- Transformer layers: `{layers}`",
        "",
        "## Critical Token Mix",
        "",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## Strongest Baseline Comparison",
            "",
            "| Method | Setting | Compute saved | Critical disagreement | HWD | Excess NLL | 95% CI critical |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in [best_entropy, best_margin, best_triage, high_saving_triage, oracle_triage]:
        lines.append(
            f"| {result.policy} | {result.setting} | {result.compute_saved:.4f} | "
            f"{result.disagreement_critical:.4f} | {result.harm_weighted_disagreement:.4f} | "
            f"{result.mean_excess_nll:.4f} | {result.disagreement_critical_ci95 or ''} |"
        )
    lines.extend(
        [
            "",
            "## Top Deployable Pareto Candidates",
            "",
            "| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for rank, result in enumerate(deployable_pareto[:12], start=1):
        lines.append(
            f"| {rank} | {result.policy} | {result.setting} | {result.compute_saved:.4f} | "
            f"{result.disagreement_critical:.4f} | {result.harm_weighted_disagreement:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.",
            "",
            "The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.",
            "",
        ]
    )
    (output_dir / "paper_results.md").write_text("\n".join(lines), encoding="utf-8")


def run_paper_grade_experiment(
    *,
    output_dir: Path,
    dataset: str = "ChayanM/IUXray-Data-Train-Test",
    config: str = "default",
    split: str = "test",
    limit: int = 64,
    offset: int = 0,
    text_column: str = "Caption",
    id_column: str = "Image_Name",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "auto",
    max_length: int = 160,
    bootstrap_samples: int = 200,
) -> list[PolicyResult]:
    traces, reports, layers = generate_traces(
        output_dir=output_dir,
        dataset=dataset,
        config=config,
        split=split,
        limit=limit,
        offset=offset,
        text_column=text_column,
        id_column=id_column,
        model_name=model_name,
        device=device,
        max_length=max_length,
    )
    results = build_policy_suite(
        traces,
        layers=layers,
        entropy_thresholds=[0.10, 0.15, 0.20, 0.30, 0.42, 0.55],
        margin_thresholds=[0.05, 0.10, 0.20, 0.35, 0.50],
        triage_min_fractions=[0.50, 0.65, 0.72, 0.85],
        bootstrap_samples=bootstrap_samples,
    )
    write_paper_outputs(
        output_dir,
        results=results,
        traces=traces,
        reports=reports,
        layers=layers,
        model_name=model_name,
        dataset=dataset,
        split=split,
    )
    return results
