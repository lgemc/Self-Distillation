from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science", "medical"])
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Max prompt tokens; longer prompts are cut from the left")
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="Micro-batch size; gradient accumulation fills up to num_prompts_per_batch")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Keep activations instead of recomputing them (faster, more memory)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3, help="Fraction of GPU memory vLLM reserves")
    parser.add_argument("--no_vllm_sleep", action="store_true", help="Keep vLLM resident between generations instead of sleeping it")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint every N steps; larger than the run saves only the final model")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps (-1 = run all epochs)")
    args = parser.parse_args()
    if args.num_prompts_per_batch % args.per_device_batch_size:
        parser.error("--num_prompts_per_batch must be a multiple of --per_device_batch_size")
    return args

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(train_dir) 

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


def load_medical_dataset(seed=42) -> Dataset:
    """Load and prepare medical dataset (built by prepare_medical.py) with formatted prompts."""
    path = 'data/medical_data/train_data'
    print(f"Loading medical dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['question']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                orig_content=example['question'],
                output_text=example['output_text'],
            )}],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    elif args.dataset_name == "medical":
        dataset, _ = load_medical_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=not args.no_vllm_sleep, 
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = args.per_device_batch_size,
        gradient_accumulation_steps = args.num_prompts_per_batch // args.per_device_batch_size,
        gradient_checkpointing = not args.no_gradient_checkpointing,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = 1024,
        num_train_epochs = args.num_train_epochs,
        max_steps = args.max_steps,
        num_iterations = 1,
        num_generations = 1,
        save_steps = args.save_steps,
        max_grad_norm = 1,
        report_to = "wandb",
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = 3,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    # The EMA teacher trails the student (mixup alpha per step), so "the model after SDFT" is two
    # weight sets. Save the teacher beside the student so a diff can read both.
    teacher = trainer.accelerator.unwrap_model(trainer.ref_model)
    teacher.save_pretrained(os.path.join(args.output_dir, "teacher"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "teacher"))
