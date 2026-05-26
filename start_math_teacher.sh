#!/usr/bin/env bash
# Start the teacher feedback server (Qwen2.5-Math-7B-Instruct) on GPU 0.
# This runs inside the training container and provides structured feedback
# for incorrect student math attempts.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash start_math_teacher.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/selfevolve/resd/teacher"

export PROXY_FRONTEND_PORT=${TEACHER_PORT:-15555}
export PROXY_BACKEND_PORT=${TEACHER_BACKEND_PORT:-15556}
export VLLM_USE_V1=1
# Qwen2.5-Math-7B-Instruct has max_position_embeddings=4096
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-4096}

TEACHER_MODEL=${TEACHER_MODEL:-"Qwen/Qwen2.5-Math-7B-Instruct"}

# Kill any existing teacher processes
ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9 2>/dev/null || true
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9 2>/dev/null || true

echo "[Teacher] Starting proxy on port ${PROXY_FRONTEND_PORT} (backend: ${PROXY_BACKEND_PORT})..."
nohup python proxy.py &> /tmp/teacher_proxy.log &
PROXY_PID=$!

# Wait for proxy to be ready
sleep 3
echo "[Teacher] Proxy started (PID: ${PROXY_PID})"

echo "[Teacher] Starting vLLM worker with model: ${TEACHER_MODEL}..."
# For 7B model on 1 GPU (48GB A6000), TP=1 is sufficient
nohup python worker.py \
  --backend vllm \
  --tp-size 1 \
  --n-logprobs 0 \
  --ckpt-path "${TEACHER_MODEL}" \
  --seq-len 8192 \
  &> /tmp/teacher_worker.log &
WORKER_PID=$!

echo "[Teacher] Worker starting (PID: ${WORKER_PID}). Waiting for model to load..."

# Wait for worker to connect (model loading takes ~30-60s for 7B)
MAX_WAIT=180
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  if grep -q "worker started" /tmp/teacher_worker.log 2>/dev/null; then
    echo "[Teacher] Worker ready! Server is up on port ${PROXY_FRONTEND_PORT}"
    echo "[Teacher] Proxy PID: ${PROXY_PID}, Worker PID: ${WORKER_PID}"
    exit 0
  fi
  if ! kill -0 $WORKER_PID 2>/dev/null; then
    echo "[Teacher] ERROR: Worker died during startup. Check /tmp/teacher_worker.log"
    cat /tmp/teacher_worker.log | tail -20
    exit 1
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  echo "[Teacher] Still loading model... (${WAITED}s/${MAX_WAIT}s)"
done

echo "[Teacher] WARNING: Worker did not report ready within ${MAX_WAIT}s. Check /tmp/teacher_worker.log"
echo "[Teacher] Last worker log lines:"
tail -10 /tmp/teacher_worker.log
