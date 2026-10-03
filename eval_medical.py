import argparse
import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on medical test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to model_path)")
    # Qwen3's recommended thinking-mode sampler (model card, and the checkpoint's
    # generation_config.json). Greedy decoding in thinking mode is what the card
    # warns against: it degrades answers and loops until the length cap.
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling cutoff (-1 to disable)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed; run several to report a spread rather than one draw")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini",
                        help="OpenAI model used as the correctness judge (the paper uses gpt-5-mini)")
    parser.add_argument("--judge_workers", type=int, default=16,
                        help="Concurrent judge requests")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, gpu_memory_utilization=0.8):
    """Load model using vLLM and tokenizer from the given path."""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        max_model_len=4096,
        trust_remote_code=True,
    )
    return llm, tokenizer


def load_test_data():
    """Load medical test dataset (built by prepare_medical.py)."""
    path = 'data/medical_data/eval_data'
    print(f"Loading medical test dataset from {path}")
    data = Dataset.load_from_disk(path)
    return data


def generate_responses(llm, tokenizer, prompts, max_new_tokens=2048, temperature=0.6,
                       top_p=0.95, top_k=20, seed=0):
    """Generate responses from the model using vLLM."""
    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )

    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


JUDGE_PROMPT = """You are evaluating a response to a medical question.

Question:
{question}

Reference answer:
{answer}

Response:
{response}

Judge only on medical accuracy and completeness, not on writing style or verbosity.
The response is CORRECT if it contains the key medical information from the reference
answer, even if phrased differently or includes additional correct medical details.
Otherwise it is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


def strip_thinking(text: str) -> str:
    """Drop a Qwen3-style <think> block so the judge sees only the answer."""
    return text.split("</think>")[-1].strip()


def evaluate_correctness(questions, responses, answers, judge_model, workers):
    """
    Ask an LLM judge whether each response matches the reference answer.
    Returns list of scores (1 for correct, 0 for incorrect).
    """
    client = OpenAI()

    def judge(args):
        question, response, answer = args
        reply = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, answer=answer, response=strip_thinking(response))}],
        )
        verdict = reply.choices[0].message.content.strip().upper()
        return 1 if verdict.startswith("CORRECT") else 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(judge, zip(questions, responses, answers)))


def main():
    args = parse_args()

    # Load model and data
    llm, tokenizer = load_model_and_tokenizer(args.model_path)
    test_data = load_test_data()

    prompts = [example['prompt'] for example in test_data]
    answers = [example['answer'] for example in test_data]
    questions = [prompt[-1]['content'] for prompt in prompts]

    # Generate responses
    responses = generate_responses(
        llm, tokenizer, prompts,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.seed,
    )

    # Evaluate correctness
    print("\nEvaluating responses...")
    scores = evaluate_correctness(questions, responses, answers, args.judge_model, args.judge_workers)
    accuracy = np.mean(scores)

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    # Save results
    output_dir = args.output_dir if args.output_dir else args.model_path
    os.makedirs(output_dir, exist_ok=True)

    results_to_save = {
        "accuracy": float(accuracy),
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "per_sample_scores": scores,
        "config": {
            "model_path": args.model_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "judge_model": args.judge_model,
        }
    }

    output_path = os.path.join(output_dir, f"eval_results-seed{args.seed}.json")
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Save responses for inspection
    responses_path = os.path.join(output_dir, f"eval_responses-seed{args.seed}.json")
    with open(responses_path, "w") as f:
        json.dump([
            {
                "prompt": prompts[i],
                "response": responses[i],
                "answer": answers[i],
                "correct": bool(scores[i])
            }
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved responses to {responses_path}")


if __name__ == "__main__":
    main()
