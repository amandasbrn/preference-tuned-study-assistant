import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/study_prompts.csv")

REQUIRED_COLUMNS = ["id", "topic", "subtopic", "prompt", "difficulty"]
VALID_TOPICS = {"deep_learning", "nlp"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Dataset preview:")
    print(df.head())
    print()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df["id"].duplicated().any():
        duplicated_ids = df.loc[df["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicated IDs found: {duplicated_ids}")

    invalid_topics = set(df["topic"]) - VALID_TOPICS
    if invalid_topics:
        raise ValueError(f"Invalid topics found: {invalid_topics}")

    invalid_difficulties = set(df["difficulty"]) - VALID_DIFFICULTIES
    if invalid_difficulties:
        raise ValueError(f"Invalid difficulties found: {invalid_difficulties}")

    if df["prompt"].isna().any():
        raise ValueError("Some prompts are missing.")

    if (df["prompt"].str.len() < 10).any():
        raise ValueError("Some prompts are too short.")

    print("Validation passed.")
    print()
    print("Number of prompts:", len(df))
    print()
    print("Topic distribution:")
    print(df["topic"].value_counts())
    print()
    print("Difficulty distribution:")
    print(df["difficulty"].value_counts())
    print()
    print("Subtopic distribution:")
    print(df["subtopic"].value_counts())


if __name__ == "__main__":
    main()