#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD
python - << 'PY'
"""Quick sanity check: does Qwen2.5-Math-1.5B emit <think>/<answer> tags
under the R1-Zero system prompt? Run 3 example completions and inspect."""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading Qwen/Qwen2.5-Math-1.5B...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()

df = pd.read_parquet("data/math/train.parquet")
print(f"Loaded {len(df)} training records")
print()

for i in range(3):
    record = df.iloc[i]
    messages = list(record["prompt"])
    print(f"=== EXAMPLE {i+1}: target = {record['reward_model']['ground_truth'][:60]} ===")
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True, temperature=1.0, top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    completion = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("--- completion (first 500 chars) ---")
    print(completion[:500])
    print("...")
    print("Has <think>?", "<think>" in completion)
    print("Has </think>?", "</think>" in completion)
    print("Has <answer>?", "<answer>" in completion)
    print("Has </answer>?", "</answer>" in completion)
    print("Has \\boxed?", "\\boxed" in completion)
    print()
PY
