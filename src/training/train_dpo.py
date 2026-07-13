import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name')
    parser.add_argument('--train-path')
    parser.add_argument('--val-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-steps', type=int, default=2, help="maximum number of training steps")
    return parser.parse_args()

def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }

def main():
    args = parse_args()
    model_name = args.model_name
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    output_dir = Path(args.output_dir)
    max_steps = args.max_steps

    dataset = load_dataset("json",
                           data_files={
                               'train':str(train_path),
                               'validation':str(val_path)})

    dataset = dataset.map(preprocess_function, remove_columns=["id", "topic", "subtopic",'difficulty','answer_a','answer_b','label'])

    train_data = dataset['train']
    val_data = dataset['validation']

    # print("Dataset:", dataset)
    # print("Train rows:", len(train_data))
    # print("Eval rows:", len(val_data))
    # print("Train columns:", train_data.column_names)
    # print("First train example:", train_data[0])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print("Loaded tokenizer:", tokenizer.__class__.__name__)
    print("Loaded model:", model.__class__.__name__)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = DPOConfig(
                    output_dir=str(output_dir),
                    max_steps=max_steps,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=1,
                    learning_rate=5e-6,
                    logging_steps=1,
                    save_steps=10,
                    eval_steps=10,
                    report_to="none",
                )
    
    trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            processing_class=tokenizer
        )
    
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

if __name__ == "__main__":
    main()


