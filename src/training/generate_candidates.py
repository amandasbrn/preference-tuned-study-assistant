import pandas as pd
from pathlib import Path
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.prompt_templates import build_study_friendly_prompt, build_formal_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_PATH = Path("data/raw/study_prompts.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=3)
    return parser

def load_prompt(data: Path) -> pd.DataFrame:
    if not data.exists():
        raise FileNotFoundError(f"File not found: {data}")
    df = pd.read_csv(data)
    return df

def format_prompt(question: str) -> str:
    prompt_a = build_study_friendly_prompt(question)
    prompt_b = build_formal_prompt(question)
    return prompt_a, prompt_b

def generate_output(formatted_prompt: str):
    messages = [
    {
        "role": "system",
        "content": "You are a precise study assistant. Answer clearly for exam revision. Do not use emojis, hashtags, or motivational comments."
    },
    {
        "role": "user",
        "content": formatted_prompt
    }
    ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt")
    answer = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
        )
    
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = answer[0][input_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded

def main():
    prompt = load_prompt(PROMPT_PATH)

    parser = parse_args()
    args = parser.parse_args()

    limit = args.limit
    
    df_subset = prompt.head(limit).copy()
    answer_a = []
    answer_b = []
    for p in df_subset['prompt']:
        prompt_a, prompt_b = format_prompt(p)
        output_a = generate_output(prompt_a)
        answer_a.append(output_a)
        output_b = generate_output(prompt_b)
        answer_b.append(output_b)
    df_subset['answer_a'] = answer_a
    df_subset['answer_b'] = answer_b

    row_dicts = df_subset.to_dict(orient="records")

    output_path = Path(f"data/processed/candidate_answers_{limit}_sample.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(row_dicts)):
            f.write(json.dumps(row_dicts[i], ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(df_subset)}] {row_dicts[i]['id']} - {row_dicts[i]['subtopic']}")
            print(f"Saved candidate pair for {row_dicts[i]['id']}")
            print('')

if __name__ == "__main__":
    main()
