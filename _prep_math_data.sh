#!/bin/bash
# Build all MATH-family parquet datasets inside the brandonzyw/resd:v2 container.
# Run once to populate data/math/{train,math500,minerva,amc2024,olympiadbench}.parquet.
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD

mkdir -p data/math

echo "=== Train: Hendrycks MATH ==="
python selfevolve/resd/data/format/math_dataset.py \
  --source math --split train \
  --output data/math/train.parquet

echo "=== Eval: MATH500 ==="
python selfevolve/resd/data/format/math_dataset.py \
  --source math500 \
  --output data/math/math500.parquet

echo "=== Eval: Minerva ==="
python selfevolve/resd/data/format/math_dataset.py \
  --source minerva \
  --output data/math/minerva.parquet

echo "=== Eval: AMC 2024 ==="
python selfevolve/resd/data/format/math_dataset.py \
  --source amc2024 \
  --output data/math/amc2024.parquet || echo "AMC2024 failed (may need different HF dataset id)"

echo "=== Eval: OlympiadBench ==="
python selfevolve/resd/data/format/math_dataset.py \
  --source olympiadbench \
  --output data/math/olympiadbench.parquet || echo "OlympiadBench failed (may need different HF dataset id)"

echo "=== Done ==="
ls -lh data/math/
