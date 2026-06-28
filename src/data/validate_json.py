import json
import argparse
from pathlib import Path

REQUIRED_FIELDS = {
    "id",
    "topic",
    "subtopic",
    "prompt",
    "difficulty",
    "answer_a",
    "answer_b",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    return parser.parse_args()

def validate_file(jsonl: Path):
    with open(jsonl, 'r') as f:
        records = []
        for line in f:
            record = json.loads(line)
            records.append(record)
    
    missing_fields = REQUIRED_FIELDS - set(record.keys())
    if missing_fields:
        raise ValueError(f"Missing columns: {missing_fields}")
    
    if not record["answer_a"].strip():
        raise ValueError("Some empty answer exists.")
    
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicated IDs found: {ids}")
    
    topic_counts = {}

    for record in records:
        topic = record["topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print('Validation passed.')
    print('')
    print('File:', jsonl)
    print('Number of record:', len(records))
    print('Unique ids:', len(set(ids)))
    print('')
    print("Topic distribution:")
    for topic, count in topic_counts.items():
        print(f"{topic}: {count}")

def main():
    args = parse_args()
    input_path = Path(args.input_path)

    validate_file(input_path)

if __name__ == "__main__":
    main()
