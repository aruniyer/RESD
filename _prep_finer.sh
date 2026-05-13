#!/bin/bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD
mkdir -p data/finer

python selfevolve/resd/data/format/finer.py \
  --task_name finer \
  --input selfevolve/ace/data/finer_train_batched_1000_samples.jsonl \
  --num_data -1 \
  --output data/finer/train_-1.parquet

python selfevolve/resd/data/format/finer.py \
  --task_name finer \
  --input selfevolve/ace/data/finer_val_batched_500_samples.jsonl \
  --output data/finer/val.parquet

ls -lh data/finer/
