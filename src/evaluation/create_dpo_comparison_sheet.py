import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-path')
    parser.add_argument('--dpo-path')
    parser.add_argument('--output-path')
    return parser

def main():
    parser = parse_args()
    args = parser.parse_args()

    baseline_path = Path(args.baseline_path)
    dpo_path = Path(args.dpo_path)
    output_path = Path(args.output_path)

    baseline_eval_df = pd.read_csv(baseline_path)
    baseline_eval_df = baseline_eval_df.rename(
        columns={
            "clarity_score": "baseline_clarity_score",
            "correctness_score": "baseline_correctness_score",
            "exam_usefulness_score": "baseline_exam_usefulness_score",
            "conciseness_score": "baseline_conciseness_score",
            "overall_score": "baseline_overall_score",
            "evaluation_notes": "baseline_evaluation_notes",
        }
    )
    dpo_df = pd.read_csv(dpo_path)

    merge_keys = ["id", "topic", "subtopic", "prompt", "difficulty"]
    comparison_df = baseline_eval_df.merge(
                    dpo_df,
                    on=merge_keys,
                    how="inner",
                )
    print("Baseline rows:", len(baseline_eval_df))
    print("DPO rows:", len(dpo_df))
    print("Merged rows:", len(comparison_df))

    score_columns = [
        "dpo_clarity_score",
        "dpo_correctness_score",
        "dpo_exam_usefulness_score",
        "dpo_conciseness_score",
        "dpo_overall_score",
        "comparison_winner",
        "evaluation_notes",
    ]

    for col in score_columns:
        comparison_df[col] = ""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    print(f"Successfully saved baseline vs DPO comparison sheet at {output_path}")

if __name__ == "__main__":
    main()