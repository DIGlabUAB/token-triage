from __future__ import annotations

import argparse
from pathlib import Path

from token_triage.paper_grade import run_paper_grade_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-grade TokenTriage baseline suite.")
    parser.add_argument("--output-dir", default="outputs/paper-grade-iuxray-qwen25")
    parser.add_argument("--dataset", default="ChayanM/IUXray-Data-Train-Test")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--text-column", default="Caption")
    parser.add_argument("--id-column", default="Image_Name")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    args = parser.parse_args()
    results = run_paper_grade_experiment(
        output_dir=Path(args.output_dir),
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.limit,
        offset=args.offset,
        text_column=args.text_column,
        id_column=args.id_column,
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        bootstrap_samples=args.bootstrap_samples,
    )
    best = min(
        (result for result in results if result.policy in {"tokentriage-minlayer", "tokentriage-kstable"}),
        key=lambda result: (result.disagreement_critical, -result.compute_saved),
    )
    print(f"Wrote {args.output_dir}/traces.jsonl")
    print(f"Wrote {args.output_dir}/policy_results.csv")
    print(f"Wrote {args.output_dir}/paper_results.md")
    print(
        "Best TokenTriage: "
        f"{best.setting}, compute_saved={best.compute_saved}, "
        f"critical_disagreement={best.disagreement_critical}"
    )


if __name__ == "__main__":
    main()
