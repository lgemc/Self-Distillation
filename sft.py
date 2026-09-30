"""SFT baseline: train on the same demonstrations SDFT uses, off-policy.

Same CLI and optimizer settings as main.py, so the only difference between the
two runs is the method. The model sees the plain prompt (the SDFT student
prompt) and is trained to reproduce the golden demonstration (the text SDFT
puts in the teacher prompt); the loss covers the completion only.
"""
import argparse

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Longest completion SDFT trains on; SFT gets the same budget.
MAX_COMPLETION_LENGTH = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="SFT baseline")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science", "medical"])
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Max prompt tokens (sequence cap is this + completion length)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps (-1 = run all epochs)")
    return parser.parse_args()


def load_dataset(name, seed=42):
    """Prompt/completion pairs matching main.py's student prompts and demonstrations."""
    if name == "tooluse":
        ds = load_from_disk("data/tooluse_data/train_data")
        fmt = lambda ex: {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "completion": [{"role": "assistant", "content": "\n".join(ex["golden_response"])}],
        }
    elif name == "science":
        ds = load_from_disk("data/science_data/train_data")
        fmt = lambda ex: {
            "prompt": ex["messages"],
            "completion": [{"role": "assistant", "content": ex["output_text"]}],
        }
    elif name == "medical":
        ds = load_from_disk("data/medical_data/train_data")
        fmt = lambda ex: {
            "prompt": [{"role": "user", "content": ex["question"]}],
            "completion": [{"role": "assistant", "content": ex["output_text"]}],
        }
    else:
        raise ValueError(f"Invalid dataset name: {name}")
    ds = ds.map(fmt, remove_columns=ds.column_names)
    print(f"Loaded {len(ds)} training examples")
    return ds.shuffle(seed=seed)


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(args.dataset_name, args.seed)

    per_device = 4
    config = SFTConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=args.num_prompts_per_batch // per_device,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_length=args.max_prompt_length + MAX_COMPLETION_LENGTH,
        completion_only_loss=True,
        gradient_checkpointing=True,
        save_steps=100,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=args.output_dir,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
