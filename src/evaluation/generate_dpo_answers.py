import pandas as pd
from pathlib import Path
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.data.prompt_templates import build_study_friendly_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--input-path')
    parser.add_argument('--output-path')
    parser.add_argument('--adapter-path')
    return parser

def load_prompt(data: Path) -> pd.DataFrame:
    if not data.exists():
        raise FileNotFoundError(f"File not found: {data}")
    df = pd.read_csv(data)
    return df

def format_prompt(question: str) -> str:
    baseline = build_study_friendly_prompt(question)
    return baseline

def generate_output(formatted_prompt: str, adapter_path: Path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
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
    parser = parse_args()
    args = parser.parse_args()

    limit = args.limit
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    adapter_path = Path(args.adapter_path)

    prompt = load_prompt(input_path)
    
    df_subset = prompt.head(limit).copy()
    dpo_answer = []
    for p in df_subset['prompt']:
        baseline_prompt = format_prompt(p)
        answer = generate_output(baseline_prompt, adapter_path)
        dpo_answer.append(answer)

    df_subset['dpo_answer'] = dpo_answer

    #row_dicts = df_subset.to_dict(orient="records")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_csv(output_path, index=False)
    print(f"Successfully saved DPO answers at {output_path}")

if __name__ == "__main__":
    main()