#!/usr/bin/env bash
# CoDistill-GRPO baseline reproduction: vanilla GRPO on Qwen2.5-Math-1.5B
# trained on Hendrycks MATH, evaluated on Minerva/MATH500/AMC2024/OlympiadBench.
#
# Follows CoDistill-GRPO recipe (Sun et al. 2026, arXiv:2605.08873) Appendix A:
#   - DeepSeek-R1-Zero chat template (applied in the preprocessor)
#   - Format reward: 0.25 each for </think>, </answer> (Total possible: 1.0
#     accuracy + 0.5 format = 1.5). Implemented via compute_score_r1zero in
#     selfevolve/resd/feedback/math.py.
#   - GRPO with no KL-to-ref (uses RESD's grpo.yaml defaults: adv_estimator=grpo,
#     norm_adv_by_std_in_grpo=True, vanilla policy loss, rollout.n=8)
#   - 8 epochs (CoDistill setting for Qwen models)
#
# Hardware: 4x A6000 (48 GB each). Qwen2.5-Math-1.5B is ~3.5 GB bf16, so we
# have plenty of headroom for KV cache and FSDP-4 sharding.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
ulimit -c 0

CONFIG_NAME="grpo"
TASK=math
export TASK

# Data
train_path=data/math/train.parquet
# Multiple eval datasets in one config; verl handles them as a list.
val_path="['data/math/math500.parquet','data/math/minerva.parquet','data/math/amc2024.parquet','data/math/olympiadbench.parquet']"

# Hyperparameters (CoDistill Table 4, Qwen recipe)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}     # CoDistill: 32 prompts per RL step
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}  # G = 8 rollouts per prompt
LR=${LR:-1e-6}                                # CoDistill: 1e-6 for Qwen-Math
TOTAL_EPOCHS=${TOTAL_EPOCHS:-8}              # CoDistill: 8 epochs for Qwen
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# Smoke / full toggle (override via env: SMOKE_STEPS=3 for smoke test)
SMOKE_STEPS=${SMOKE_STEPS:-0}

# Logger: console only by default; export LOGGER='["console","wandb"]' to add wandb
LOGGER=${LOGGER:-'["console"]'}

project_name=${PROJECT_NAME:-'codistill_repro_math'}
exp_name=${EXP_NAME:-'qwen25math_1.5b_grpo'}

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
  +custom_reward_function.reward_kwargs.correctness_feedback=False
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

ROLLOUT=(
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE
  actor_rollout_ref.rollout.val_kwargs.n=4
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
  "${REF[@]}" \
  "${TRAINER[@]}"
