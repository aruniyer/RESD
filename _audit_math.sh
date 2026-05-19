#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD
python - << 'PY'
import pandas as pd
for name in ["train", "math500", "minerva", "amc2024", "olympiadbench"]:
    df = pd.read_parquet(f"data/math/{name}.parquet")
    print(f"\n=== {name}: {len(df)} rows ===")
    r = df.iloc[0]
    print("data_source:", r["data_source"])
    print("ground_truth:", str(r["reward_model"]["ground_truth"])[:80])
    print("prompt[:200]:", r["prompt"][0]["content"][:200])
    print("...")
PY
