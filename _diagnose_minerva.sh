#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD
python - << 'PY'
"""Dump 5 Minerva failures and 2 Minerva successes from v3 val_generations,
so we can diagnose what's going wrong vs the paper baseline."""
import json, os, sys, importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "_math_feedback",
    "/workspace/RESD/selfevolve/resd/feedback/math.py",
)
math_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_mod)
compute_score_r1zero = math_mod.compute_score_r1zero

# Use the v3 val dump (greedy@1, our recent run)
paths = [
    "/workspace/RESD/checkpoints/codistill_repro_math/qwen25math_1.5b_grpo_phase1_v3/val_generations/0.jsonl",
    "/workspace/RESD/checkpoints/codistill_repro_math/qwen25math_1.5b_grpo_phase1_v2/val_generations/0.jsonl",
]
path = next((p for p in paths if os.path.exists(p)), None)
print(f"Reading {path}\n")

recs = [json.loads(l) for l in open(path) if l.strip()]
print(f"{len(recs)} records loaded")

# Build ground truth lookup
minerva_df = pd.read_parquet("/workspace/RESD/data/math/minerva.parquet")
gt_by_q = {r["question"].strip(): r["reward_model"]["ground_truth"] for _, r in minerva_df.iterrows()}

minerva_recs = []
for r in recs:
    inp = r["input"]
    if "user\n" in inp and "\nassistant" in inp:
        q = inp.split("user\n", 1)[1].rsplit("\nassistant", 1)[0].strip()
        if q in gt_by_q:
            gt = gt_by_q[q]
            score = compute_score_r1zero(r["output"], gt, extra_info={"split": "test"})
            minerva_recs.append((q, r["output"], gt, score["acc"], score["pred"]))

print(f"Found {len(minerva_recs)} Minerva records")
n_correct = sum(1 for _, _, _, acc, _ in minerva_recs if acc >= 1.0)
print(f"Accuracy: {n_correct}/{len(minerva_recs)} = {n_correct/len(minerva_recs):.3%}\n")

# Show 5 failures
print("===== FAILURES =====")
failures = [r for r in minerva_recs if r[3] < 1.0][:5]
for i, (q, out, gt, acc, pred) in enumerate(failures, 1):
    print(f"\n--- Failure {i} ---")
    print(f"GROUND TRUTH: {repr(gt)}")
    print(f"EXTRACTED PRED: {repr(pred)}")
    print(f"QUESTION (first 200): {q[:200]}")
    print(f"MODEL OUTPUT (full, since usually short for Minerva):")
    print(out[:2000])
    print("..." if len(out) > 2000 else "")

# Show 2 successes
print("\n===== SUCCESSES =====")
successes = [r for r in minerva_recs if r[3] >= 1.0][:2]
for i, (q, out, gt, acc, pred) in enumerate(successes, 1):
    print(f"\n--- Success {i} ---")
    print(f"GROUND TRUTH: {repr(gt)}")
    print(f"EXTRACTED PRED: {repr(pred)}")
    print(f"MODEL OUTPUT (first 800 chars):")
    print(out[:800])
PY
