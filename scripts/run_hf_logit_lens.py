from __future__ import annotations

import argparse
from pathlib import Path

from token_triage.hf_logit_lens import run_hf_logit_lens_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF Dataset + Qwen logit-lens TokenTriage experiment.")
    parser.add_argument("--output-dir", default="outputs/hf-logit-lens")
    parser.add_argument("--dataset", default="ChayanM/IUXray-Data-Train-Test")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--text-column", default="Caption")
    parser.add_argument("--id-column", default="Image_Name")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--entropy-threshold", type=float, default=0.42)
    parser.add_argument("--critical-min-layer-fraction", type=float, default=0.72)
    args = parser.parse_args()
    summary, _ = run_hf_logit_lens_experiment(
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
        entropy_exit_threshold=args.entropy_threshold,
        critical_min_layer_fraction=args.critical_min_layer_fraction,
    )
    print(f"Wrote {args.output_dir}/summary.json")
    print(f"Wrote {args.output_dir}/token_records.csv")
    print(f"Wrote {args.output_dir}/report.md")
    print(
        "Critical disagreement: "
        f"entropy={summary.entropy_disagreement_rate_critical}, "
        f"tokentriage={summary.tokentriage_disagreement_rate_critical}"
    )


if __name__ == "__main__":
    main()
