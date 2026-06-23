from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_experiment
from .ollama import OllamaClient, OllamaConfig
from .report import write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the semantic token triage experiment with local Ollama.")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Local Ollama model name.")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama host URL.")
    parser.add_argument("--cases", type=int, default=8, help="Number of synthetic radiology cases.")
    parser.add_argument("--output-dir", default="outputs/run", help="Directory for JSON, CSV, and Markdown outputs.")
    parser.add_argument("--templates-only", action="store_true", help="Skip Ollama report generation and use seed templates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OllamaClient(OllamaConfig(host=args.host, model=args.model))
    models = client.available_models()
    if args.model not in models:
        raise SystemExit(
            f"Model {args.model!r} is not installed in Ollama. Available models: {', '.join(models) or 'none'}"
        )
    output_dir = Path(args.output_dir)
    summary, cases, records = run_experiment(
        client,
        cases=args.cases,
        output_dir=output_dir,
        use_templates_only=args.templates_only,
    )
    report_path = write_markdown_report(output_dir, summary, cases, records)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'cases.json'}")
    print(f"Wrote {output_dir / 'token_records.csv'}")
    print(f"Wrote {report_path}")
    print(
        "Dangerous errors: "
        f"entropy-only={summary.entropy_only_dangerous_errors}, "
        f"criticality-aware={summary.criticality_aware_dangerous_errors}"
    )


if __name__ == "__main__":
    main()
