"""Build data/medical_data from HuatuoGPT-o1, following the paper's recipe.

The Medical dataset was never released with this repo, so it is rebuilt here:
  train: medical-o1-reasoning-SFT (English, ~20k). The demonstration is the
         final `Response` only -- the paper drops the chain of thought.
  eval:  1000 questions sampled at random from medical-o1-verifiable-problem,
         scored against `Ground-True Answer` by an LLM judge (eval_medical.py).

The paper's exact 1000-question sample is not published, so eval numbers are
comparable in kind, not directly to its table.
"""
import argparse
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Build the medical dataset")
    parser.add_argument("--num_eval", type=int, default=1000, help="Number of eval questions to sample")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the eval sample")
    parser.add_argument("--output_dir", type=str, default="data/medical_data", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    train = train.map(
        lambda ex: {"question": ex["Question"], "output_text": ex["Response"]},
        remove_columns=train.column_names,
    )

    # Drop eval questions that also appear in train, so eval stays held out.
    train_questions = set(q.strip() for q in train["question"])
    problems = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem", split="train")
    problems = problems.filter(lambda ex: ex["Open-ended Verifiable Question"].strip() not in train_questions)
    eval_data = problems.shuffle(seed=args.seed).select(range(args.num_eval))
    eval_data = eval_data.map(
        lambda ex: {
            "prompt": [{"role": "user", "content": ex["Open-ended Verifiable Question"]}],
            "answer": ex["Ground-True Answer"],
        },
        remove_columns=eval_data.column_names,
    )

    train.save_to_disk(f"{args.output_dir}/train_data")
    eval_data.save_to_disk(f"{args.output_dir}/eval_data")
    print(f"train: {len(train)}  eval: {len(eval_data)}  ->  {args.output_dir}")
