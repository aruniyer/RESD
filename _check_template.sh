#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
cd /workspace/RESD
python - << 'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
print("chat template present:", tok.chat_template is not None)
# Render an example
msgs = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("--- rendered template ---")
print(text)
print("--- end ---")
PY
