#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD
python - << 'PY'
"""Replay the existing val_generations dump with the fixed verifier."""
import json, os, sys, importlib.util
import pandas as pd

# Import the math.py module directly (skip the package __init__ which has
# broken relative imports we don't need).
spec = importlib.util.spec_from_file_location(
    "_math_feedback",
    "/workspace/RESD/selfevolve/resd/feedback/math.py",
)
math_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_mod)
compute_score_r1zero = math_mod.compute_score_r1zero

paths_to_try = [
    "/workspace/RESD/checkpoints/codistill_repro_math/qwen25math_1.5b_grpo_phase1_v2/val_generations/0.jsonl",
    "/workspace/RESD/checkpoints/codistill_repro_math/qwen25math_1.5b_grpo_phase1/val_generations/0.jsonl",
]
path = next((p for p in paths_to_try if os.path.exists(p)), None)
print(f"Reading {path}")
recs = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            recs.append(json.loads(line))
print(f"Loaded {len(recs)} val generations")

# Build a ground-truth lookup keyed on question text
import glob
gt_by_question = {}
for p in glob.glob("/workspace/RESD/data/math/*.parquet"):
    if "train" in p:
        continue
    df = pd.read_parquet(p)
    for _, r in df.iterrows():
        q = r["question"].strip()
        gt_by_question[q] = (r["reward_model"]["ground_truth"], r["data_source"])
print(f"Ground truth lookup: {len(gt_by_question)} unique questions")

def find_gt(input_text):
    # The input field has 'system\n...\nuser\n{question}\nassistant\n'
    if "user\n" in input_text and "\nassistant" in input_text:
        q = input_text.split("user\n", 1)[1].rsplit("\nassistant", 1)[0].strip()
        return gt_by_question.get(q, (None, None))
    # fallback: linear search
    for q, (gt, ds) in gt_by_question.items():
        if q in input_text:
            return gt, ds
    return None, None

new_correct = 0
matched = 0
new_correct_by_ds = {}
total_by_ds = {}
for r in recs:
    gt, ds = find_gt(r["input"])
    if gt is None:
        continue
    matched += 1
    total_by_ds[ds] = total_by_ds.get(ds, 0) + 1
    new_out = compute_score_r1zero(r["output"], gt, extra_info={"split": "test"})
    if new_out["acc"] >= 1.0:
        new_correct += 1
        new_correct_by_ds[ds] = new_correct_by_ds.get(ds, 0) + 1

print(f"\nMatched {matched}/{len(recs)} records to ground truth")
print(f"\n=== NEW verifier results ===")
print(f"Overall: {new_correct}/{matched} = {new_correct/max(matched,1):.3%}")
for ds, total in sorted(total_by_ds.items()):
    correct = new_correct_by_ds.get(ds, 0)
    print(f"  {ds:20s}: {correct:4d}/{total:4d} = {correct/total:.3%}")
PY
