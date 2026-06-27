import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.prompt_templates import build_study_friendly_prompt, build_formal_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_PATH = Path("data/raw/study_prompts.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def load_prompt(data: Path) -> pd.DataFrame:
    if not data.exists():
        raise FileNotFoundError(f"File not found: {data}")
    df = pd.read_csv(data)
    return df

def format_prompt(data: pd.DataFrame) -> str:
    question = data['prompt'].iloc[0]
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
    answer_a = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
        )
    
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = answer_a[0][input_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded

def main():
    prompt = load_prompt(PROMPT_PATH)
    prompt_a, prompt_b = format_prompt(prompt)
    output_a = generate_output(prompt_a)
    output_b = generate_output(prompt_b)
    print("STUDY FRIENDLY")
    print(output_a)
    print('-------------------------------------')
    print("FORMAL")
    print(output_b)

if __name__ == "__main__":
    main()
