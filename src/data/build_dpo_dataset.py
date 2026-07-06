import json
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/preferences/preference_labels_100.csv")

def input_pref_data(input_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    data["label"] = data["label"].astype(str).str.strip().str.upper()
    return data

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data_clean = data.copy()
    data_clean = data[data['label']!='skip']
    data_clean = data.drop('notes',axis=1)

    data_clean['chosen'] = ""
    data_clean['rejected'] = ""

    for i, row in data_clean.iterrows():
        if row['label'] == 'A':
            data_clean.loc[i, 'chosen'] = data_clean.loc[i, 'answer_a']
            data_clean.loc[i, 'rejected'] = data_clean.loc[i, 'answer_b']
        elif row['label'] == 'B':
            data_clean.loc[i, 'chosen'] = data_clean.loc[i, 'answer_b']
            data_clean.loc[i, 'rejected'] = data_clean.loc[i, 'answer_a']

    return data_clean

def save_to_json(data: pd.DataFrame, file_name: str):
    data.to_json(f'data/preferences/{file_name}.jsonl', orient='records', lines=True)

def main():
    pref_data = input_pref_data(DATA_PATH)
    clean_pref_data = preprocessing(pref_data)

    #used_cols = ['prompt', 'chosen', 'rejected']

    train_data, eval_data = train_test_split(clean_pref_data, test_size = 0.1, random_state=42)

    save_to_json(train_data, 'dpo_train')
    save_to_json(eval_data, 'dpo_eval')

if __name__ == "__main__":
    main()


