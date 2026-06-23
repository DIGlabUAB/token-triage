from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .experiment import ExperimentSummary, TokenRecord
from .synthetic import Case


def write_markdown_report(
    output_dir: Path,
    summary: ExperimentSummary,
    cases: list[Case],
    records: list[TokenRecord],
) -> Path:
    category_counts = Counter(record.category for record in records if record.category != "ordinary")
    risky = [record for record in records if record.entropy_dangerous_error][:20]
    lines = [
        "# Semantic Token Triage Report",
        "",
        "## Claim",
        "",
        "This local experiment tests whether confidence-only early exit can create disproportionate errors on clinically meaningful tokens such as negation, laterality, severity, temporality, medications, and measurements.",
        "",
        "## Run Summary",
        "",
    ]
    for key, value in asdict(summary).items():
        lines.append(f"- **{key}**: `{value}`")
    lines.extend(["", "## Critical Token Mix", ""])
    for category, count in sorted(category_counts.items()):
        lines.append(f"- **{category}**: {count}")
    lines.extend(["", "## Example Cases", ""])
    for case in cases[:8]:
        lines.append(f"- `{case.case_id}` ({case.source}): {case.report}")
    lines.extend(["", "## Highest-Risk Entropy-Only Exits", ""])
    if risky:
        for record in risky:
            lines.append(
                f"- `{record.case_id}` token `{record.token}` ({record.category}) exited at layer "
                f"{record.entropy_exit_layer}, stabilized at layer {record.stabilization_layer}."
            )
    else:
        lines.append("- No simulated dangerous entropy-only exits in this run.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Ollama was used locally for report generation.",
            "- The installed Ollama endpoint did not expose generated-token logprobs during verification, so `confidence_proxy` is a deterministic stand-in for the first runnable local prototype.",
            "- The package keeps the scoring boundary isolated so a backend that returns true logprobs or hidden states can replace the proxy without changing the triage analysis.",
            "",
        ]
    )
    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
