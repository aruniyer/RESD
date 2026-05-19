#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD
python - << 'PY'
"""Standalone baseline eval: Qwen2.5-Math-1.5B + simple math prompt + greedy
+ fixed verifier, on all 4 benchmarks. Target: match CoDistill paper Base
column (Minerva 20.4, MATH500 54.9, AMC 32.0, Olympiad 23.1)."""
import json, os, sys, importlib.util, time
import pandas as pd
from vllm import LLM, SamplingParams

# Load our fixed verifier
spec = importlib.util.spec_from_file_location(
    "_math_feedback",
    "/workspace/RESD/selfevolve/resd/feedback/math.py",
)
math_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_mod)
compute_score_r1zero = math_mod.compute_score_r1zero

MODEL = "Qwen/Qwen2.5-Math-1.5B"

# DeepSeek-Math zero-shot prompt for instruct/RL models (paper Table 6, Llama):
SIMPLE_PROMPT_TEMPLATE = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

print(f"Loading {MODEL}...")
llm = LLM(model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          max_model_len=4096, enforce_eager=True, dtype="bfloat16")
print("Model loaded.")

sp = SamplingParams(temperature=0, max_tokens=3072, top_p=1.0, top_k=-1)

for benchmark in ["math500", "minerva", "amc2024", "olympiadbench"]:
    df = pd.read_parquet(f"data/math/{benchmark}.parquet")
    # Build prompts using SIMPLE template (NOT R1-Zero) - just the math problem
    # passed through the model's standard chat template as a user message.
    tokenizer = llm.get_tokenizer()
    prompts = []
    for _, r in df.iterrows():
        q = r["question"]
        user_msg = SIMPLE_PROMPT_TEMPLATE.format(question=q)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(text)

    print(f"\n=== {benchmark}: {len(prompts)} prompts ===")
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"Generated in {time.time()-t0:.1f}s")

    n_correct = 0
    for out, (_, r) in zip(outs, df.iterrows()):
        completion = out.outputs[0].text
        gt = r["reward_model"]["ground_truth"]
        score = compute_score_r1zero(completion, gt, extra_info={"split": "test"})
        if score["acc"] >= 1.0:
            n_correct += 1
    acc = n_correct / len(df)
    print(f"  greedy@1: {n_correct}/{len(df)} = {acc:.4f}")
PY
