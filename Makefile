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
OUT      ?= runs/$(RUN_NAME)

export WANDB_PROJECT ?= sdft

# The GB10 is sm_121a. Triton's bundled ptxas (CUDA 12.x) doesn't know that
# arch, so torch.compile inside vLLM fails; the system CUDA 13 ptxas does.
export TRITON_PTXAS_PATH ?= /usr/local/cuda/bin/ptxas

.PHONY: train smoke train-seq data-medical eval

# Sequential continual learning, the paper's order: Science -> Tool Use -> Medical.
# Each stage starts from the previous stage's final model.
SEQ     ?= science tooluse medical
SEQ_DIR ?= runs/seq-$(notdir $(MODEL))

# The real run: full epochs over the train split.
train:
	WANDB_NAME=$(RUN_NAME) $(PY) main.py \
	  --dataset_name $(DATASET) \
	  --model_name $(MODEL) \
	  --output_dir $(OUT) \
	  --learning_rate $(LR) \
	  --num_train_epochs $(EPOCHS) \
	  --max_prompt_length $(MAX_PROMPT) \
	  --seed $(SEED)

# Smoke test: a few optimizer steps with small batches, just to confirm the
# pipeline (model load, vLLM, loss, wandb) works end to end.
smoke:
	WANDB_NAME=smoke-$(RUN_NAME) $(PY) main.py \
	  --dataset_name $(DATASET) \
	  --model_name $(MODEL) \
	  --output_dir runs/smoke-$(RUN_NAME) \
	  --learning_rate $(LR) \
	  --num_prompts_per_batch 2 \
	  --max_steps 3 \
	  --max_prompt_length $(MAX_PROMPT) \
	  --seed $(SEED)

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
	  WANDB_NAME=seq-$(notdir $(MODEL))-$$i-$$ds $(PY) main.py \
	    --dataset_name $$ds \
	    --model_name $$prev \
	    --output_dir $$out \
	    --learning_rate $(LR) \
	    --num_train_epochs $(EPOCHS) \
	    --max_prompt_length $(MAX_PROMPT) \
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
