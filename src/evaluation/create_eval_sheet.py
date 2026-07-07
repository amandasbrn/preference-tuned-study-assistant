import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    parser.add_argument('--output-path')
    return parser

def generate_eval_file(input_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    score_columns = [
        "clarity_score",
        "correctness_score",
        "exam_usefulness_score",
        "conciseness_score",
        "overall_score",
        "evaluation_notes",
    ]
    for col in score_columns:
        data[col] = ""
    return data

def main():
    parser = parse_args()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    eval_data = generate_eval_file(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_data.to_csv(output_path, index=False)

    print(f"Successfully saved baseline evaluation file at {output_path}")

if __name__ == "__main__":
    main()