from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .clinical_tokens import token_is_critical
from .hf_dataset import HfReport, fetch_hf_reports


@dataclass(frozen=True)
class HfTokenRecord:
    report_id: str
    row_id: int
    position: int
    token_text: str
    category: str
    criticality: float
    final_top_token: str
    final_top_matches_target: bool
    entropy_exit_layer: int
    tokentriage_exit_layer: int
    stable_layer: int
    entropy_normalized: float
    entropy_disagrees_with_final: bool
    tokentriage_disagrees_with_final: bool
    entropy_target_nll: float
    tokentriage_target_nll: float
    full_target_nll: float


@dataclass(frozen=True)
class HfExperimentSummary:
    dataset: str
    split: str
    model: str
    reports: int
    evaluated_tokens: int
    critical_tokens: int
    layers: int
    entropy_exit_threshold: float
    critical_min_layer_fraction: float
    entropy_avg_exit_layer: float
    tokentriage_avg_exit_layer: float
    entropy_compute_saved: float
    tokentriage_compute_saved: float
    entropy_disagreement_rate_all: float
    tokentriage_disagreement_rate_all: float
    entropy_disagreement_rate_critical: float
    tokentriage_disagreement_rate_critical: float
    entropy_harm_weighted_disagreement: float
    tokentriage_harm_weighted_disagreement: float
    entropy_mean_excess_nll: float
    tokentriage_mean_excess_nll: float


def _load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
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


def _exit_layer_by_entropy(entropies: list[float], threshold: float) -> int:
    for idx, entropy in enumerate(entropies, start=1):
        if entropy <= threshold:
            return idx
    return len(entropies)


def _stable_layer(pred_ids: list[int], final_pred_id: int) -> int:
    for idx in range(len(pred_ids)):
        if all(pred_id == final_pred_id for pred_id in pred_ids[idx:]):
            return idx + 1
    return len(pred_ids)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_hf_logit_lens_experiment(
    *,
    output_dir: Path,
    dataset: str = "ChayanM/IUXray-Data-Train-Test",
    config: str = "default",
    split: str = "test",
    limit: int = 32,
    offset: int = 0,
    text_column: str = "Caption",
    id_column: str = "Image_Name",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "auto",
    max_length: int = 192,
    entropy_exit_threshold: float = 0.42,
    critical_min_layer_fraction: float = 0.72,
) -> tuple[HfExperimentSummary, list[HfTokenRecord]]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    records: list[HfTokenRecord] = []

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
            layers = len(hidden_states)
            min_critical_layer = max(1, math.ceil(layers * critical_min_layer_fraction))
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
                layer_pred_ids: list[int] = []
                layer_entropies: list[float] = []
                layer_target_nlls: list[float] = []
                for layer_idx, hidden in enumerate(hidden_states):
                    hidden_vec = hidden[0, pos, :]
                    if layer_idx < len(hidden_states) - 1 and hasattr(model, "model") and hasattr(model.model, "norm"):
                        hidden_vec = model.model.norm(hidden_vec)
                    logits = model.lm_head(hidden_vec).float()
                    probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * log_probs).item() / vocab_norm
                    layer_entropies.append(float(entropy))
                    layer_pred_ids.append(int(torch.argmax(logits).item()))
                    layer_target_nlls.append(float(-log_probs[target_id].item()))

                final_pred_id = int(final_top_ids[pos].item())
                stable = _stable_layer(layer_pred_ids, final_pred_id)
                entropy_exit = _exit_layer_by_entropy(layer_entropies, entropy_exit_threshold)
                if criticality >= 0.8:
                    triage_exit = max(entropy_exit, stable, min_critical_layer)
                else:
                    triage_exit = entropy_exit
                final_target_nll = float(-final_log_probs[pos, target_id].item())
                records.append(
                    HfTokenRecord(
                        report_id=report.report_id,
                        row_id=report.row_id,
                        position=pos + 1,
                        token_text=token_text,
                        category=category,
                        criticality=criticality,
                        final_top_token=_decode_token(tokenizer, final_pred_id),
                        final_top_matches_target=final_pred_id == target_id,
                        entropy_exit_layer=entropy_exit,
                        tokentriage_exit_layer=triage_exit,
                        stable_layer=stable,
                        entropy_normalized=round(layer_entropies[entropy_exit - 1], 4),
                        entropy_disagrees_with_final=layer_pred_ids[entropy_exit - 1] != final_pred_id,
                        tokentriage_disagrees_with_final=layer_pred_ids[triage_exit - 1] != final_pred_id,
                        entropy_target_nll=round(layer_target_nlls[entropy_exit - 1], 4),
                        tokentriage_target_nll=round(layer_target_nlls[triage_exit - 1], 4),
                        full_target_nll=round(final_target_nll, 4),
                    )
                )

    summary = summarize_records(
        records=records,
        reports=reports,
        dataset=dataset,
        split=split,
        model_name=model_name,
        layers=len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else max(r.stable_layer for r in records),
        entropy_exit_threshold=entropy_exit_threshold,
        critical_min_layer_fraction=critical_min_layer_fraction,
    )
    write_hf_outputs(output_dir, summary, records, reports)
    return summary, records


def summarize_records(
    *,
    records: list[HfTokenRecord],
    reports: list[HfReport],
    dataset: str,
    split: str,
    model_name: str,
    layers: int,
    entropy_exit_threshold: float,
    critical_min_layer_fraction: float,
) -> HfExperimentSummary:
    critical = [record for record in records if record.criticality >= 0.8]
    weights = [record.criticality for record in records]
    entropy_weighted = sum(record.criticality for record in records if record.entropy_disagrees_with_final)
    triage_weighted = sum(record.criticality for record in records if record.tokentriage_disagrees_with_final)
    weight_total = sum(weights) or 1.0
    return HfExperimentSummary(
        dataset=dataset,
        split=split,
        model=model_name,
        reports=len(reports),
        evaluated_tokens=len(records),
        critical_tokens=len(critical),
        layers=layers,
        entropy_exit_threshold=entropy_exit_threshold,
        critical_min_layer_fraction=critical_min_layer_fraction,
        entropy_avg_exit_layer=round(_mean([record.entropy_exit_layer for record in records]), 2),
        tokentriage_avg_exit_layer=round(_mean([record.tokentriage_exit_layer for record in records]), 2),
        entropy_compute_saved=round(1 - _mean([record.entropy_exit_layer for record in records]) / layers, 3),
        tokentriage_compute_saved=round(1 - _mean([record.tokentriage_exit_layer for record in records]) / layers, 3),
        entropy_disagreement_rate_all=round(_mean([float(record.entropy_disagrees_with_final) for record in records]), 4),
        tokentriage_disagreement_rate_all=round(_mean([float(record.tokentriage_disagrees_with_final) for record in records]), 4),
        entropy_disagreement_rate_critical=round(_mean([float(record.entropy_disagrees_with_final) for record in critical]), 4),
        tokentriage_disagreement_rate_critical=round(_mean([float(record.tokentriage_disagrees_with_final) for record in critical]), 4),
        entropy_harm_weighted_disagreement=round(entropy_weighted / weight_total, 4),
        tokentriage_harm_weighted_disagreement=round(triage_weighted / weight_total, 4),
        entropy_mean_excess_nll=round(_mean([record.entropy_target_nll - record.full_target_nll for record in records]), 4),
        tokentriage_mean_excess_nll=round(_mean([record.tokentriage_target_nll - record.full_target_nll for record in records]), 4),
    )


def write_hf_outputs(
    output_dir: Path,
    summary: HfExperimentSummary,
    records: list[HfTokenRecord],
    reports: list[HfReport],
) -> None:
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    (output_dir / "reports.json").write_text(
        json.dumps([asdict(report) for report in reports], indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "token_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    category_counts = Counter(record.category for record in records if record.category != "ordinary")
    critical_by_category: dict[str, list[HfTokenRecord]] = defaultdict(list)
    for record in records:
        if record.criticality >= 0.8:
            critical_by_category[record.category].append(record)
    report_lines = [
        "# TokenTriage HF Logit-Lens Experiment",
        "",
        "## Setup",
        "",
        f"- Dataset: `{summary.dataset}` / `{summary.split}`",
        f"- Model: `{summary.model}`",
        f"- Reports: `{summary.reports}`",
        f"- Evaluated tokens: `{summary.evaluated_tokens}`",
        f"- Critical tokens: `{summary.critical_tokens}`",
        "",
        "## Main Result",
        "",
        f"- Entropy-only saved `{summary.entropy_compute_saved}` simulated layer-compute with critical-token disagreement `{summary.entropy_disagreement_rate_critical}`.",
        f"- TokenTriage saved `{summary.tokentriage_compute_saved}` simulated layer-compute with critical-token disagreement `{summary.tokentriage_disagreement_rate_critical}`.",
        f"- Harm-weighted disagreement fell from `{summary.entropy_harm_weighted_disagreement}` to `{summary.tokentriage_harm_weighted_disagreement}`.",
        "",
        "## Critical Token Mix",
        "",
    ]
    for category, count in sorted(category_counts.items()):
        report_lines.append(f"- {category}: {count}")
    report_lines.extend(["", "## Interpretation", ""])
    report_lines.append(
        "The result supports the TokenTriage hypothesis when TokenTriage reduces critical-token disagreement while retaining meaningful compute savings. "
        "The cost is lower average compute savings because critical tokens are forced to later, stable layers."
    )
    report_lines.extend(["", "## Examples Where Entropy Exited Before Stability", ""])
    examples = [
        record
        for record in records
        if record.criticality >= 0.8 and record.entropy_disagrees_with_final and not record.tokentriage_disagrees_with_final
    ][:20]
    for record in examples:
        report_lines.append(
            f"- `{record.report_id}` token `{record.token_text}` ({record.category}) entropy layer "
            f"{record.entropy_exit_layer}, stable layer {record.stable_layer}, TokenTriage layer {record.tokentriage_exit_layer}."
        )
    if not examples:
        report_lines.append("- No critical-token rescue examples under this threshold.")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
