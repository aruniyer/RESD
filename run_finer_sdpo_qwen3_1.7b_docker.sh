#!/bin/bash
# Docker wrapper for the Qwen3-1.7B FiNER SDPO run.
set -euo pipefail

IMAGE=brandonzyw/resd:v2
HOST_REPO=/data/ariy/RESD
HOST_HF=/data/ariy/hf_cache
HOST_LOGS=/data/ariy/logs
mkdir -p "$HOST_HF" "$HOST_LOGS"

docker run --rm \
  --gpus all \
  --shm-size=64g \
  --ipc=host \
  --net=host \
  --name finer_sdpo_qwen3_1p7b \
  -v "$HOST_REPO":/workspace/RESD \
  -v "$HOST_HF":/root/.cache/huggingface \
  -w /workspace/RESD \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -c "source /opt/conda/etc/profile.d/conda.sh && conda activate opd && WANDB_MODE=offline bash /workspace/RESD/run_finer_sdpo_qwen3_1.7b.sh"
