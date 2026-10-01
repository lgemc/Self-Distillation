# SDFT runs. Override any variable on the command line, e.g.
#   make train DATASET=science
#   make train MODEL=Qwen/Qwen2.5-7B-Instruct   # paper default, needs an H200-class GPU

PY       ?= .venv/bin/python
DATASET  ?= tooluse
MODEL    ?= Qwen/Qwen3-0.6B
LR       ?= 5e-5
EPOCHS   ?= 2
SEED     ?= 42
# Teacher prompts carry the golden answer; at 1024 about 8% of tooluse and 1.5%
# of science teacher prompts get left-truncated (losing the question). 2048 fits all.
MAX_PROMPT ?= 2048
RUN_NAME ?= $(DATASET)-$(notdir $(MODEL))

# Speed knobs for the GB10 (unified memory, small model). None changes the
# update: MICRO_BS x (32 / MICRO_BS accumulation) is still 32 prompts per step.
#   MICRO_BS    micro-batch size. For SDFT 1 is fastest: larger batches add padding,
#               which costs more in the full-vocab KL and attention than it saves
#               (measured 47 / 49 / 51 s per step at 1 / 2 / 4). SFT uses SFT_MICRO_BS;
#               at 4 its full-vocab logits grew to ~79 GB and pushed the GB10 into swap.
#   GRAD_CKPT   1 = recompute activations in backward (saves memory, slower)
#   VLLM_SLEEP  1 = sleep vLLM between generations (frees its memory, ~2 s/step)
#   VLLM_MEM    fraction of memory vLLM reserves; 32 x 3k tokens of 0.6B KV is ~11 GB
MICRO_BS   ?= 1
SFT_MICRO_BS ?= 1
GRAD_CKPT  ?= 0
VLLM_SLEEP ?= 0
VLLM_MEM   ?= 0.15
SPEED_ARGS  = --per_device_batch_size $(MICRO_BS) \
	$(if $(filter 0,$(GRAD_CKPT)),--no_gradient_checkpointing) \
	$(if $(filter main.py,$(SCRIPT)),--vllm_gpu_memory_utilization $(VLLM_MEM) $(if $(filter 0,$(VLLM_SLEEP)),--no_vllm_sleep))
OUT      ?= runs/$(RUN_NAME)

# main.py is SDFT; sft.py is the SFT baseline (same CLI, same demonstrations).
SCRIPT   ?= main.py

export WANDB_PROJECT ?= sdft

# The GB10 is sm_121a. Triton's bundled ptxas (CUDA 12.x) doesn't know that
# arch, so torch.compile inside vLLM fails; the system CUDA 13 ptxas does.
export TRITON_PTXAS_PATH ?= /usr/local/cuda/bin/ptxas

.PHONY: train smoke train-seq sft sft-seq data-medical eval

# Sequential continual learning over SEQ, in the order given. The paper (Sec. 4.3,
# Fig. 3) trains the three skills sequentially but its text doesn't state the order.
# Each stage starts from the previous stage's final model.
SEQ     ?= science tooluse medical
SEQ_TAG ?= seq-$(notdir $(MODEL))
SEQ_DIR ?= runs/$(SEQ_TAG)

# The real run: full epochs over the train split.
train:
	WANDB_NAME=$(RUN_NAME) $(PY) $(SCRIPT) \
	  --dataset_name $(DATASET) \
	  --model_name $(MODEL) \
	  --output_dir $(OUT) \
	  --learning_rate $(LR) \
	  --num_train_epochs $(EPOCHS) \
	  --max_prompt_length $(MAX_PROMPT) $(SPEED_ARGS) \
	  --seed $(SEED)

# Smoke test: a few optimizer steps with small batches, just to confirm the
# pipeline (model load, vLLM, loss, wandb) works end to end.
smoke:
	WANDB_NAME=smoke-$(RUN_NAME) $(PY) main.py \
	  --dataset_name $(DATASET) \
	  --model_name $(MODEL) \
	  --output_dir runs/smoke-$(RUN_NAME) \
	  --learning_rate $(LR) \
	  --num_prompts_per_batch $(MICRO_BS) \
	  --max_steps 3 \
	  --max_prompt_length $(MAX_PROMPT) $(SPEED_ARGS) \
	  --seed $(SEED)

# SFT baseline: the same runs with sft.py, under sft-* names.
sft:
	$(MAKE) train SCRIPT=sft.py RUN_NAME=sft-$(RUN_NAME) MICRO_BS=$(SFT_MICRO_BS)

sft-seq:
	$(MAKE) train-seq SCRIPT=sft.py SEQ_TAG=sft-$(SEQ_TAG) MICRO_BS=$(SFT_MICRO_BS)

# Medical is not shipped with the repo; rebuild it from HuatuoGPT-o1.
data-medical: data/medical_data/train_data
data/medical_data/train_data:
	$(PY) prepare_medical.py

# Stage i trains on the i-th dataset in SEQ, starting from stage i-1's output.
train-seq: data/medical_data/train_data
	@prev=$(MODEL); i=0; \
	for ds in $(SEQ); do \
	  i=$$((i+1)); out=$(SEQ_DIR)/$$i-$$ds; \
	  echo ">>> stage $$i: $$ds  from $$prev  ->  $$out"; \
	  WANDB_NAME=$(SEQ_TAG)-$$i-$$ds $(PY) $(SCRIPT) \
	    --dataset_name $$ds \
	    --model_name $$prev \
	    --output_dir $$out \
	    --learning_rate $(LR) \
	    --num_train_epochs $(EPOCHS) \
	    --max_prompt_length $(MAX_PROMPT) $(SPEED_ARGS) \
	    --seed $(SEED) || exit 1; \
	  prev=$$out; \
	done

# Score a model on DATASET's held-out eval split. CKPT is a run directory, a
# checkpoint, or an HF id (for the baseline), e.g.
#   make eval CKPT=Qwen/Qwen3-0.6B
#   make eval CKPT=runs/tooluse-Qwen3-0.6B/checkpoint-252
# MAX_NEW_TOKENS is the eval generation budget; Qwen3 spends a lot of it inside
# <think>, and at 1024 some answers are cut off before the answer.
CKPT           ?= $(OUT)
MAX_NEW_TOKENS ?= 2048
EVAL_OUT       ?= results/$(subst /,_,$(CKPT:/=))/$(DATASET)-$(MAX_NEW_TOKENS)

eval:
	$(PY) eval_$(DATASET).py --model_path $(CKPT) --output_dir $(EVAL_OUT) --max_new_tokens $(MAX_NEW_TOKENS)
