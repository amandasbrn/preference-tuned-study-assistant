import json
import pandas as pd
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    parser.add_argument('--output-path')
    return parser.parse_args()

def df_preference_label(input_path: Path) -> pd.DataFrame:
    records = []

    with open(input_path, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)

    df = pd.DataFrame(records)
    df['label'] = ""
    df['notes'] = ""

    return df

def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    data = df_preference_label(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
