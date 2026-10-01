# SDFT vs SFT on Qwen3-0.6B: tool use → science

Run on 2026-09-30, on a single NVIDIA GB10 (DGX Spark class, 128 GB unified memory).

## Summary

Both methods learned the science task about equally well, but **SFT lost its tool-use skill while doing so and SDFT did not**:

- **Science:** both reach about 60%, up from 32% for the base model.
- **Tool use after the science stage:** SFT dropped by 10 questions, which is statistically significant (p = 0.02) and leaves it 5 points below the untrained base model. SDFT moved by −2 questions, which is indistinguishable from no change (p = 0.73).

This is the paper's central claim, reproduced at small scale. It is one seed, though, and the tool-use eval has only 97 questions; see [Caveats](#caveats).

## Results

Accuracy on each task's held-out eval split, greedy decoding, 2048 new tokens.

| model | tool use (n=97) | science (n=507) |
|---|---|---|
| Qwen3-0.6B base | 51.5% | 31.8% |
| SDFT, after tool use | 54.6% | 28.4% |
| **SDFT, after tool use → science** | **52.6%** | **59.2%** |
| SFT, after tool use | 56.7% | 6.9% |
| **SFT, after tool use → science** | **46.4%** | **60.0%** |

### Forgetting: tool use before vs after the science stage

Paired per-question comparison of the same model before and after the science stage, using an exact McNemar test.

| method | newly wrong | newly right | net | p |
|---|---|---|---|---|
| SDFT | 5 | 3 | −2 | 0.73 |
| SFT | 13 | 3 | **−10** | **0.021** |

Comparing the two final models directly on tool use, 10 questions are right only for SDFT and 4 only for SFT (p = 0.18). With 97 questions the direct gap is not significant on its own; the within-method forgetting is the clearer signal.

### Science gains are knowledge, not formatting

The science eval scores the letter inside `<answer>…</answer>`. Counting only responses that contain the tag separates knowing the answer from producing the format:

| model | responses with `<answer>` | accuracy on those |
|---|---|---|
| base | 505 / 507 | 31.9% |
| SDFT, after tool use | 426 / 507 | 33.8% |
| SDFT, after tool use → science | 471 / 507 | 63.7% |
| SFT, after tool use | **99 / 507** | 35.4% |
| SFT, after tool use → science | 495 / 507 | 61.4% |

Both methods roughly double accuracy among tagged answers, so the science improvement reflects real knowledge.

The SFT tool-use model's 6.9% on science is a **format collapse, not lost knowledge**. It answers science questions in the tool-use demonstration style (`I need to …`), repeats that until it hits the length cap, and tags only 99 of 507 answers. Where it does tag, it is at base-level accuracy. The SDFT tool-use model keeps answering in the requested format.

### Tool-use gains are small

The tool-use stage itself moved tool-use accuracy by only +3 points (SDFT) and +5 points (SFT) over the base model, each about one standard error at n = 97. At this model size the tool-use task gives a weak learning signal, so the forgetting result is the informative part of this experiment.

## Setup

Both methods start from `Qwen/Qwen3-0.6B`, train on tool use, then continue from that model on science. They share every setting except the method.

| setting | value |
|---|---|
| learning rate | 5e-5, cosine, 10% warmup |
| epochs per stage | 2 |
| prompts per optimizer step | 32 |
| gradient clipping | 1.0 |
| weight decay | 0 |
| precision | bf16 |
| max prompt length | 2048 tokens (no prompt is truncated) |
| max completion length | 1024 tokens |
| seed | 42 |

- **SDFT** (`main.py`): on-policy self-distillation, using the repo defaults:
  - **Generation:** the student generates with vLLM from the plain prompt.
  - **Teacher:** the teacher sees the prompt plus the golden demonstration. It is an EMA of the student, with α = 0.01, synced every step.
  - **Loss:** per-token forward KL(teacher ‖ student) over the full vocabulary.
  - **Other settings:** vLLM importance-sampling correction is on, and the first 3 completion tokens are skipped in the loss.
- **SFT** (`sft.py`): next-token cross-entropy on the same demonstrations, with the plain prompt as input and the loss on the completion only. Qwen3's chat template renders each demonstration after an empty `<think></think>` block.
- **Evaluation:** `eval_tooluse.py` and `eval_science.py` via `make eval`, using vLLM greedy decoding with `max_new_tokens=2048`.

### Reproduce

From commit `3d889e4` or later:

```bash
make train-seq SEQ="tooluse science"     # SDFT -> runs/seq-Qwen3-0.6B/{1-tooluse,2-science}
make sft-seq   SEQ="tooluse science"     # SFT  -> runs/sft-seq-Qwen3-0.6B/{1-tooluse,2-science}
make eval DATASET=tooluse CKPT=runs/seq-Qwen3-0.6B/2-science    # etc., per model and task
```

SFT's tool-use stage in this run used micro-batch 4. The Makefile now defaults SFT to micro-batch 1 for memory. TRL normalizes the loss by the token count of the whole accumulated batch, so the update is the same.

### Training runs (wandb project `sdft`)

| stage | run id | wall-clock |
|---|---|---|
| SDFT tool use | `uw83i3pk` | 4h22m (2h of it sharing the GPU with SFT) |
| SDFT science | `sndpe1uf` | 1h35m |
| SFT tool use | `byy653w0` | 1h31m (sharing the GPU with SDFT) |
| SFT science | `zax3bhqu` | 36m |

Training loss fell smoothly in every stage, with no NaNs and stable entropy:

| stage | loss, start → end |
|---|---|
| SDFT tool use | 0.140 → 0.066 |
| SDFT science | 0.189 → 0.099 |
| SFT tool use | 0.50 → 0.16 |
| SFT science | 1.56 → 1.10 |

## Caveats

- **One seed, fixed hyperparameters, final checkpoints.** The paper sweeps learning rate, batch size and epochs per method and reports the checkpoint with the best validation score. This is a same-settings comparison, not best vs best.
- **Small tool-use eval.** With n = 97, differences of a few points are noise.
- **Different model.** The paper's sequential experiment uses Qwen2.5-7B-Instruct, and its text does not state the task order.
- **Format sensitivity.** SFT trains the model to answer after an empty `<think></think>` block, while SDFT keeps the model's own reasoning. Part of what is measured here is how much each method disturbs the model's output format, which is itself a form of forgetting.

## Changes made along the way

- **Truncated teacher prompts.** At the old 1024-token prompt limit, about 8% of tool-use and 1.5% of science teacher prompts were cut from the left, losing the question. This run uses 2048. (`MAX_PROMPT`)
- **`kl_approx` logging.** The metric logged `nan` when a completion was no longer than the skipped tokens. Logging only; training was unaffected.
- **SDFT speed.** About 54 → 44 s/step on the GB10:
  - Full-vocabulary log-softmax and KL took about half of each training step's GPU time. The token log-probs and entropy are now derived from the one full log-softmax, with bit-identical outputs.
  - Micro-batch 1 is fastest, because padding at larger sizes costs more than it saves.
  - Gradient checkpointing and vLLM sleep mode are off.
- **SFT memory.** At micro-batch 4 its full-vocabulary logits reached about 79 GB and pushed the machine into swap. It now defaults to micro-batch 1.
