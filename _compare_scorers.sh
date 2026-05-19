#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD
python - << 'PY'
"""Rescore the v3 Minerva failures with verl's math_metric (proper math_verify)
to see if it handles the unit/precision/symbolic issues."""
import json, os, sys, importlib.util
import pandas as pd

# Use verl's wrapper
sys.path.insert(0, "/workspace/RESD")
from verl.utils.reward_score.math_verify import compute_score as mv_compute_score

# For comparison, our existing scorer
spec = importlib.util.spec_from_file_location("_math_feedback",
    "/workspace/RESD/selfevolve/resd/feedback/math.py")
math_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(math_mod)
ours = math_mod.compute_score_r1zero

path = "/workspace/RESD/checkpoints/codistill_repro_math/qwen25math_1.5b_grpo_phase1_v3/val_generations/0.jsonl"
recs = [json.loads(l) for l in open(path) if l.strip()]
print(f"Loaded {len(recs)} records")

# Build per-benchmark ground-truth lookup
gt_by_q_per_ds = {}
import glob
for p in glob.glob("/workspace/RESD/data/math/*.parquet"):
    if "train" in p: continue
    df = pd.read_parquet(p)
    ds_name = df.iloc[0]["data_source"]
    gt_by_q_per_ds[ds_name] = {r["question"].strip(): r["reward_model"]["ground_truth"] for _, r in df.iterrows()}

# Score every record with BOTH scorers
results = {ds: {"ours": 0, "mv": 0, "total": 0} for ds in gt_by_q_per_ds}
for r in recs:
    inp = r["input"]
    if "user\n" not in inp or "\nassistant" not in inp:
        continue
    q = inp.split("user\n", 1)[1].rsplit("\nassistant", 1)[0].strip()
    matched_ds = None
    gt = None
    for ds, q2gt in gt_by_q_per_ds.items():
        if q in q2gt:
            gt = q2gt[q]; matched_ds = ds; break
    if gt is None:
        continue
    results[matched_ds]["total"] += 1
    if ours(r["output"], gt, extra_info={"split": "test"})["acc"] >= 1.0:
        results[matched_ds]["ours"] += 1
    try:
        if mv_compute_score(r["output"], gt) >= 1.0:
            results[matched_ds]["mv"] += 1
    except Exception:
        pass

print("\n=== Comparison: our scorer vs verl's math_verify wrapper ===")
print(f"{'Benchmark':<15} {'Total':>6} {'Ours':>8} {'Ours%':>8} {'MV':>8} {'MV%':>8}")
for ds in sorted(results):
    tot = results[ds]["total"]; ours_c = results[ds]["ours"]; mv_c = results[ds]["mv"]
    print(f"{ds:<15} {tot:>6} {ours_c:>8} {100*ours_c/max(tot,1):>7.2f}% {mv_c:>8} {100*mv_c/max(tot,1):>7.2f}%")
PY
