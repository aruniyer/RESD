"""Prepare MATH-family datasets for SDPO/GRPO training.

Generates parquet files compatible with the RESD/verl rl_dataset pipeline,
mirroring the conventions used by selfevolve/resd/data/format/finer.py.

Supported sources (loaded via HF `datasets`):
  - "math"          : Hendrycks et al. 2021 (12,500 train / 5,000 test).
                      HF dataset id: EleutherAI/hendrycks_math (config="all")
  - "math500"       : HuggingFaceH4/MATH-500 (500 evaluation problems)
  - "minerva"       : math-ai/minerva-math (272 problems, formal eval set)
  - "amc2024"       : AI-MO/aimo-validation-amc (83 problems)
  - "olympiadbench" : Hothan/OlympiadBench (math-text subset)

For training, follows the CoDistill-GRPO recipe (Sun et al. 2026, arXiv:2605.08873):
  - chat template = DeepSeek-R1-Zero ("Please reason step by step, and put your
    final answer within \\boxed{}.")
  - ground_truth = the extracted boxed answer from the dataset's solution field
  - reward dispatch via data_source field (see verl reward_score)

Usage:
    python selfevolve/resd/data/format/math_dataset.py \
        --source math \
        --split train \
        --output data/math/train.parquet

    python selfevolve/resd/data/format/math_dataset.py \
        --source math500 \
        --output data/math/math500.parquet
"""

import argparse
import os
import re
from typing import Optional

import pandas as pd

# DeepSeek-R1-Zero chat template from CoDistill-GRPO Appendix A, Table 5
MATH_PROMPT_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {prompt}. Answer:"


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} expression. Mirror of math.py's helper."""
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        # also handle "\fbox{...}"
        idx = string.rfind(r"\fbox{")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    if s is None:
        return ""
    for left in (r"\boxed{", r"\fbox{"):
        if s.startswith(left) and s.endswith("}"):
            return s[len(left):-1]
    return ""


def extract_answer_from_solution(solution: str) -> str:
    """Pull the final boxed answer from a Hendrycks-style solution text."""
    boxed = last_boxed_only_string(solution)
    if boxed:
        return remove_boxed(boxed).strip()
    # fallback: look for "answer is X" patterns
    m = re.search(r"final answer is[:\s]*\$?([^.$\n]+)\$?", solution, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return solution.strip().split("\n")[-1]


def make_record(idx: int, problem: str, answer: str, data_source: str,
                split: str, extra: Optional[dict] = None) -> dict:
    """One processed sample matching the schema expected by rl_dataset."""
    prompt_text = MATH_PROMPT_TEMPLATE.format(prompt=problem.strip())
    extra_info = {"index": str(idx), "split": split}
    if extra:
        extra_info.update(extra)
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "question": problem.strip(),
        "target": answer,
        "others": {"task": "math", "data_source": data_source},
        "data_source": data_source,
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": extra_info,
    }


def load_math(split: str = "train"):
    """Hendrycks MATH via EleutherAI/hendrycks_math."""
    from datasets import load_dataset, concatenate_datasets
    # The 'all' config loads all 7 subjects.
    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus"]
    parts = []
    for sub in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", sub,
                          split=split, trust_remote_code=True)
        parts.append(ds)
    return concatenate_datasets(parts)


def load_math500():
    from datasets import load_dataset
    return load_dataset("HuggingFaceH4/MATH-500", split="test")


def load_minerva():
    from datasets import load_dataset
    return load_dataset("math-ai/minervamath", split="test")


def load_amc2024():
    from datasets import load_dataset
    return load_dataset("AI-MO/aimo-validation-amc", split="train")


def load_olympiadbench():
    from datasets import load_dataset
    # Use the text-only English math subset
    return load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP",
                        split="train")


def convert_source(source: str, split: str = "train",
                   num_data: int = -1) -> list[dict]:
    if source == "math":
        ds = load_math(split=split)
        data_source = "math"
        records = []
        for i, item in enumerate(ds):
            if num_data > 0 and i >= num_data:
                break
            problem = item["problem"]
            answer = extract_answer_from_solution(item["solution"])
            if not answer:
                continue
            records.append(make_record(i, problem, answer, data_source, split,
                                        extra={"level": item.get("level", "")}))
        return records

    if source == "math500":
        ds = load_math500()
        records = []
        for i, item in enumerate(ds):
            if num_data > 0 and i >= num_data:
                break
            problem = item["problem"]
            answer = item.get("answer") or extract_answer_from_solution(
                item.get("solution", ""))
            if not answer:
                continue
            records.append(make_record(i, problem, answer, "math500", "test"))
        return records

    if source == "minerva":
        ds = load_minerva()
        records = []
        for i, item in enumerate(ds):
            if num_data > 0 and i >= num_data:
                break
            problem = item["question"]
            answer = item["answer"]
            records.append(make_record(i, problem, answer, "minerva", "test"))
        return records

    if source == "amc2024":
        ds = load_amc2024()
        records = []
        for i, item in enumerate(ds):
            if num_data > 0 and i >= num_data:
                break
            problem = item["problem"]
            answer = str(item["answer"])
            records.append(make_record(i, problem, answer, "amc2024", "test"))
        return records

    if source == "olympiadbench":
        ds = load_olympiadbench()
        records = []
        for i, item in enumerate(ds):
            if num_data > 0 and i >= num_data:
                break
            problem = item["question"]
            answer = item["final_answer"][0] if isinstance(item.get("final_answer"), list) else item.get("final_answer", "")
            if not answer:
                continue
            records.append(make_record(i, problem, str(answer),
                                        "olympiadbench", "test"))
        return records

    raise ValueError(f"Unknown source: {source}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a MATH-family dataset for SDPO/GRPO training."
    )
    parser.add_argument("--source", required=True,
                        choices=["math", "math500", "minerva",
                                 "amc2024", "olympiadbench"],
                        help="Which dataset to convert.")
    parser.add_argument("--split", default="train",
                        help="HF split (only relevant for source=math).")
    parser.add_argument("--num_data", type=int, default=-1,
                        help="Cap the number of samples (-1 = all).")
    parser.add_argument("--output", "-o", required=True,
                        help="Output parquet path.")
    args = parser.parse_args()

    records = convert_source(args.source, split=args.split,
                              num_data=args.num_data)
    print(f"Built {len(records)} records from source={args.source}, "
          f"split={args.split}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} samples to {args.output}")


if __name__ == "__main__":
    main()
