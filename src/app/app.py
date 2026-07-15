import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import torch
from src.data.prompt_templates import build_study_friendly_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "crispytempura/dpo-study-assistant-lora"

@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base_model():
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base_model.eval()
    return base_model

@st.cache_resource
def load_dpo_model():
    '''
        separate base model + LoRA adapter
    '''
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    dpo_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    dpo_model = dpo_model.to(torch.float32)
    dpo_model.eval()
    return dpo_model

def format_prompt(question: str) -> str:
    baseline = build_study_friendly_prompt(question)
    return baseline

def generate_output(tokenizer, model, formatted_prompt: str):
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
    st.title("Preference-Tuned Study Assistant")
    st.markdown("Compare a base model with a DPO-tuned model for simple, exam-ready study explanations.")

    with st.sidebar:
        st.header("Model Info")
        st.write("Base model: Qwen/Qwen2.5-0.5B-Instruct")
        st.write("Fine-tuning: DPO + LoRA")
        st.write("Best DPO run:")
        st.code("lr=1e-6, beta=0.1, steps=20, r=16")
        st.write("Purpose: simple, exam-ready study explanations")

    st.divider()

    tokenizer = load_tokenizer()
    base_model = load_base_model()
    dpo_model = load_dpo_model()

    question = st.text_area(
        "Enter a study question",
        value="Type your study question here",
        height=100,
    )

    st.info(
    "This demo compares model behavior before and after DPO tuning. "
    "Outputs may still contain mistakes, so technical answers should be verified."
    
    )
    generate_button = st.button("Generate Answers")

    if generate_button:
        # with st.spinner("Generating base model answer..."):
        #     base_output = generate_output(tokenizer, base_model, question)

        with st.spinner("Generating DPO-tuned answer..."):
            dpo_output = generate_output(tokenizer, dpo_model, question)

        #col1, col2 = st.columns(2)

        # with col1:
        #     st.subheader("Base Model")
        #     st.write(base_output)

        # with col2:
        #     st.subheader("DPO-Tuned Model")
        #     st.write(dpo_output)
        
        st.subheader("DPO-Tuned Model")
        st.write(dpo_output)


if __name__ == "__main__":
    main()