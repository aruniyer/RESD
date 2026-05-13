#!/bin/bash
# Docker wrapper: launches the FiNER SDPO smoke test inside the brandonzyw/resd:v2 container.
# Usage: bash run_smoke_finer_docker.sh [tag]
set -euo pipefail

TAG=${1:-smoke}
IMAGE=brandonzyw/resd:v2
HOST_REPO=/data/ariy/RESD
HOST_HF=/data/ariy/hf_cache
HOST_LOGS=/data/ariy/logs
mkdir -p "$HOST_HF" "$HOST_LOGS"

# Run the smoke training script inside the opd conda env
docker run --rm \
  --gpus all \
  --shm-size=64g \
  --ipc=host \
  --net=host \
  -v "$HOST_REPO":/workspace/RESD \
  -v "$HOST_HF":/root/.cache/huggingface \
  -w /workspace/RESD \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -c "source /opt/conda/etc/profile.d/conda.sh && conda activate opd && WANDB_MODE=offline bash /workspace/RESD/run_smoke_finer.sh"
