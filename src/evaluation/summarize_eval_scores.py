import argparse
import pandas as pd
from pathlib import Path

SCORE_COLUMNS = [
    "clarity_score",
    "correctness_score",
    "exam_usefulness_score",
    "conciseness_score",
    "overall_score",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    parser.add_argument('--output-path')
    return parser.parse_args()

def read_csv(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    for col in SCORE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def generate_metrics(data: pd.DataFrame):
    avg_scores = data[SCORE_COLUMNS].mean()
    topic_scores = data.groupby("topic")[SCORE_COLUMNS].mean()
    lowest = data.sort_values("overall_score").head(3)
    highest = data.sort_values("overall_score", ascending=False).head(3)

    return avg_scores, topic_scores, lowest, highest

def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    data = read_csv(input_path)
    avg_scores, topic_scores, lowest, highest = generate_metrics(data)
    row_num = data.shape[0]

    markdown_content = f"""# Baseline Evaluation Report

    ## Overview
    This report summarizes the baseline performance of the base model before DPO fine-tuning.

    ## Dataset
    - Evaluation prompts: {row_num}
    - Topics: Deep Learning and NLP

    ## Average Scores
    {avg_scores}

    ## Topic-level Scores
    {topic_scores}

    ## Lowest-scoring Examples
    {lowest}

    ## Highest-scoring Examples
    {highest}

    ## Baseline Takeaway
    The base model can produce readable explanations, but quality varies across concepts. Common weaknesses include vague examples, occasional technical imprecision, and answers that are clear but not always exam-focused.

    """

    # Create and write to the .md file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(markdown_content)

    print("Markdown file created successfully and stored in {output_path}!")

if __name__ == "__main__":
    main()
