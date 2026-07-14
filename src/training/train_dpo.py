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
    parser.add_argument('--eval-path')
    parser.add_argument('--output-dir')
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=30, help="maximum number of training steps")
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
    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)

    dataset = load_dataset("json",
                           data_files={
                               'train':str(train_path),
                               'validation':str(eval_path)})

    dataset = dataset.map(preprocess_function, remove_columns=["id", "topic", "subtopic",'difficulty','answer_a','answer_b'])

    train_data = dataset['train']
    eval_data = dataset['validation']

    # print("Dataset:", dataset)
    # print("Train rows:", len(train_data))
    # print("Eval rows:", len(val_data))
    # print("Train columns:", train_data.column_names)
    # print("First train example:", train_data[0])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Loaded tokenizer:", tokenizer.__class__.__name__)
    print("Loaded model:", model.__class__.__name__)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = DPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        report_to="none",
    )
    
    trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            processing_class=tokenizer
        )
    
    print("Training configuration")
    print(f"Model: {args.model_name}")
    print(f"Train path: {train_path}")
    print(f"Eval path: {eval_path}")
    print(f"Output dir: {output_dir}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Beta: {args.beta}")
    print(f"LoRA r: {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"LoRA dropout: {args.lora_dropout}")
    
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

if __name__ == "__main__":
    main()


