#!/usr/bin/env bash
# Smoke-test SDPO on FiNER with Qwen3-4B-Thinking, adapted for 4x A6000 (48GB each).
#
# Differences vs selfevolve/resd/run_finer_sdpo_stream_qwen3_4b_fsdp.sh:
#   - n_gpus_per_node: 8 -> 4
#   - rollout.tensor_model_parallel_size: 4 -> 2 (one TP group per pair of GPUs)
#   - max_prompt_length: 49152 -> 4096 (FiNER prompts are ~few KB)
#   - max_response_length: 20480 -> 4096
#   - max_reprompt_length: 49152 -> 8192
#   - rollout.max_model_len: 83968 -> 12288
#   - ppo_max_token_len_per_gpu: 83968 -> 12288
#   - trainer.total_training_steps: cap at 3 for the smoke test
#   - trainer.val_before_train: True -> False
#   - trainer.test_freq: 2 -> 999
#   - trainer.save_freq: 2 -> 0
#   - trainer.logger: console only (no wandb)
#
# These lower a single-rollout step's memory peak well within 48 GB / GPU.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export WANDB_MODE=offline
ulimit -c 0

CONFIG_NAME="sdpo"

train_path=data/finer/train_-1.parquet
val_path=data/finer/val.parquet

TASK=finer
export TASK

TRAIN_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=4
LR=1e-5
LAMBDA=0.0
EMA_WEIGHT=0.05
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096
MAX_REPROMPT_LENGTH=8192
ALPHA=0.5
DISTILLATION_TOPK=100
DONTS_REPROMPT_ON_SELF_SUCCESS=True

project_name='smoke_sdpo_finer'
exp_name='smoke_qwen3_4b_a6000'

DATA=(
  data.train_files=${train_path}
  data.val_files=${val_path}
  data.train_batch_size=${TRAIN_BATCH_SIZE}
  data.max_prompt_length=${MAX_PROMPT_LENGTH}
  data.max_response_length=${MAX_RESPONSE_LENGTH}
  data.truncation='error'
  data.filter_overlong_prompts=True
  data.shuffle=False
  "data.apply_chat_template_kwargs={enable_thinking: True}"
  custom_reward_function.path=selfevolve/resd/feedback/finer.py
  custom_reward_function.name=compute_score_count
  +custom_reward_function.reward_kwargs.correctness_feedback=True
)

MODEL=(
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Thinking-2507
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.optim.lr=$LR
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.optim.lr_warmup_steps=2
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288
)

DISTILLATION=(
  actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS}
  actor_rollout_ref.actor.self_distillation.alpha=$ALPHA
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=$EMA_WEIGHT
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LENGTH}
  actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0
  actor_rollout_ref.actor.self_distillation.success_rate_weighting=False
)

CONTEXT_UPDATER=(
  actor_rollout_ref.actor.self_distillation.context_updater.enabled=False
)

TEACHER=(
  actor_rollout_ref.actor.self_distillation.teacher.server_ip="127.0.0.1"
)

ROLLOUT=(
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE
  actor_rollout_ref.rollout.val_kwargs.n=2
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
  actor_rollout_ref.rollout.max_model_len=12288
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=0.95
)

REF=(
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ALGORITHM=(
  algorithm.lam=${LAMBDA}
  algorithm.rollout_correction.rollout_is=token
)

TRAINER=(
  trainer.use_stream_trainer=True
  trainer.max_updates_per_batch=1
  trainer.min_updates_per_batch=1
  trainer.early_stop_improvement_threshold=0.0
  trainer.logger='["console"]'
  trainer.total_epochs=1
  trainer.total_training_steps=3
  trainer.project_name=${project_name}
  trainer.experiment_name=${exp_name}
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.max_actor_ckpt_to_keep=1
  trainer.save_freq=0
  trainer.test_freq=999
  trainer.val_before_train=False
  trainer.forget_eval.eval_freq=0
  trainer.rollout_data_dir="checkpoints/${project_name}/${exp_name}/rollouts"
  trainer.validation_data_dir="checkpoints/${project_name}/${exp_name}/val_generations"
  trainer.reprompt_data_dir="checkpoints/${project_name}/${exp_name}/reprompts"
)

python -m selfevolve.resd.trainer.main_ppo \
  --config-name=${CONFIG_NAME} \
  "${DATA[@]}" \
  "${ALGORITHM[@]}" \
  "${MODEL[@]}" \
  "${ROLLOUT[@]}" \
  "${ACTOR[@]}" \
  "${DISTILLATION[@]}" \
  "${CONTEXT_UPDATER[@]}" \
  "${TEACHER[@]}" \
  "${REF[@]}" \
  "${TRAINER[@]}"
