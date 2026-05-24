#!/usr/bin/env bash
# Phase 2 v2: SDPO on Qwen2.5-Math-1.5B with rich feedback + aligned hyperparams
# Fixes from failed v1 run:
#   1. Rich format+correctness feedback (like code/tooluse tasks have)
#   2. correctness_feedback=True → tells model "you said X, correct is Y"
#   3. teacher_update_rate=0.05 (was 0.0001, 500× too slow)
#   4. distillation_topk=20 (was 100, matching original SDPO)
#   5. success_reward_threshold=0.5 (was 1.0, now partial credit counts)
#   6. remove_thinking_from_demonstration=True (cleaner demos)
#   7. environment_feedback_only_without_solution=True (feedback only when no demo available)
#
# Architecture:
#   - EMA teacher provides logit targets on reprompted sequences
#   - Correct solutions from batch used as demonstrations (thinking stripped)
#   - When no solution available: rich feedback tells model what went wrong
#   - JSD (alpha=0.5) between student and teacher on reprompted sequences

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
ulimit -c 0

CONFIG_NAME="sdpo"
TASK=math
export TASK

# Data
train_path=data/math/train.parquet
val_path="['data/math/math500.parquet','data/math/minerva.parquet','data/math/amc2024.parquet','data/math/olympiadbench.parquet']"

# Hyperparameters (same base as Phase 1 GRPO)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
# Qwen2.5-Math-1.5B has max_position_embeddings=4096.
# vLLM max_model_len = prompt + response = 4096 (same as GRPO).
# Reprompt (for teacher logits) must also fit in 4096: reprompt_prompt + response ≤ 4096.
# We set max_reprompt_len=1536 so truncated reprompt + response (≤2560) fits.
MAX_REPROMPT_LENGTH=${MAX_REPROMPT_LENGTH:-1536}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# Self-distillation hyperparameters (aligned with original SDPO defaults)
ALPHA=${ALPHA:-0.5}                        # 0.5 = JSD (symmetric KL)
EMA_WEIGHT=${EMA_WEIGHT:-0.05}             # fast EMA teacher update (original default)
DISTILLATION_TOPK=${DISTILLATION_TOPK:-20}   # top-20 logits (original default)
IS_CLIP=${IS_CLIP:-2.0}                    # importance sampling clip

# Smoke / full toggle
SMOKE_STEPS=${SMOKE_STEPS:-0}

# Logger
LOGGER=${LOGGER:-'["console"]'}

project_name=${PROJECT_NAME:-'codistill_repro_math'}
exp_name=${EXP_NAME:-'qwen25math_1.5b_sdpo_phase2_v2'}

DATA=(
  data.train_files=${train_path}
  data.val_files=${val_path}
  data.train_batch_size=${TRAIN_BATCH_SIZE}
  data.max_prompt_length=${MAX_PROMPT_LENGTH}
  data.max_response_length=${MAX_RESPONSE_LENGTH}
  data.truncation='error'
  data.filter_overlong_prompts=True
  data.shuffle=True
  "data.apply_chat_template_kwargs={enable_thinking: False}"
  custom_reward_function.path=selfevolve/resd/feedback/math.py
  custom_reward_function.name=compute_score_r1zero
  +custom_reward_function.reward_kwargs.format_feedback=True
  +custom_reward_function.reward_kwargs.correctness_feedback=True
)

MODEL=(
  actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.optim.lr=$LR
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.optim.lr_warmup_steps=10
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LEN}
)

DISTILLATION=(
  actor_rollout_ref.actor.self_distillation.alpha=${ALPHA}
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=${EMA_WEIGHT}
  actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK}
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True
  actor_rollout_ref.actor.self_distillation.is_clip=${IS_CLIP}
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LENGTH}
  actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5
  actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=True
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=True
  actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=True
)

ROLLOUT=(
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE
  actor_rollout_ref.rollout.calculate_log_probs=True
  # Validation: greedy decoding (DeepSeek-Math / CoDistill protocol).
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.temperature=0
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=0.95
)

REF=(
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ALGORITHM=(
  algorithm.rollout_correction.rollout_is=token
)

TRAINER=(
  trainer.logger="${LOGGER}"
  trainer.total_epochs=${TOTAL_EPOCHS}
  trainer.project_name=${project_name}
  trainer.experiment_name=${exp_name}
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.max_actor_ckpt_to_keep=1
  trainer.save_freq=${SAVE_FREQ:-50}
  trainer.test_freq=${TEST_FREQ:-100}
  trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
  trainer.rollout_data_dir="checkpoints/${project_name}/${exp_name}/rollouts"
  trainer.validation_data_dir="checkpoints/${project_name}/${exp_name}/val_generations"
  trainer.reprompt_data_dir="checkpoints/${project_name}/${exp_name}/reprompts"
)

# Override total_training_steps when SMOKE_STEPS or TOTAL_STEPS is set.
if [[ "${SMOKE_STEPS}" -gt 0 ]]; then
  TRAINER+=(
    trainer.total_training_steps=${SMOKE_STEPS}
    trainer.test_freq=999
    trainer.save_freq=0
    trainer.val_before_train=False
  )
elif [[ -n "${TOTAL_STEPS:-}" ]]; then
  TRAINER+=(
    trainer.total_training_steps=${TOTAL_STEPS}
  )
fi

python -m selfevolve.resd.trainer.main_ppo \
  --config-name=${CONFIG_NAME} \
  "${DATA[@]}" \
  "${ALGORITHM[@]}" \
  "${MODEL[@]}" \
  "${ROLLOUT[@]}" \
  "${ACTOR[@]}" \
  "${DISTILLATION[@]}" \
  "${REF[@]}" \
  "${TRAINER[@]}"
