#!/bin/bash
# Docker wrapper for the Qwen2.5-Math-1.5B GRPO baseline run.
# Usage:
#   bash run_math_grpo_qwen25math_1.5b_docker.sh           # full 8-epoch run
#   SMOKE_STEPS=3 bash run_math_grpo_qwen25math_1.5b_docker.sh   # smoke test
set -euo pipefail

IMAGE=brandonzyw/resd:v2
HOST_REPO=/data/ariy/RESD
HOST_HF=/data/ariy/hf_cache
HOST_LOGS=/data/ariy/logs
mkdir -p "$HOST_HF" "$HOST_LOGS"

CONTAINER_NAME=${CONTAINER_NAME:-math_grpo_qwen25math_1p5b}

docker run --rm \
  --gpus all \
  --shm-size=64g \
  --ipc=host \
  --net=host \
  --name "$CONTAINER_NAME" \
  -e SMOKE_STEPS="${SMOKE_STEPS:-0}" \
  -e TOTAL_STEPS="${TOTAL_STEPS:-}" \
  -e SAVE_FREQ="${SAVE_FREQ:-50}" \
  -e TEST_FREQ="${TEST_FREQ:-100}" \
  -e VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}" \
  -e PROJECT_NAME="${PROJECT_NAME:-codistill_repro_math}" \
  -e EXP_NAME="${EXP_NAME:-qwen25math_1.5b_grpo}" \
  -e LOGGER="${LOGGER:-[\"console\"]}" \
  -v "$HOST_REPO":/workspace/RESD \
  -v "$HOST_HF":/root/.cache/huggingface \
  -w /workspace/RESD \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -c "source /opt/conda/etc/profile.d/conda.sh && conda activate opd && WANDB_MODE=offline bash /workspace/RESD/run_math_grpo_qwen25math_1.5b.sh"
