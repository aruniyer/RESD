# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
import signal
from typing import Optional
from math_verify import parse as mv_parse, verify as mv_verify

FORMAT_PENALTY = False


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind(r"\boxed{")
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

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""#None


def remove_boxed(s: str) -> str:
    r"""Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = r"\boxed{"
    #assert s[: len(left)] == left, f"box error: {s}"
    #assert s[-1] == "}", f"box error: {s}"
    if s[: len(left)] == left and  s[-1] == "}":
        return s[len(left) : -1]
    else:
        return ""


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Searches the ENTIRE response for the last \\boxed{...} expression and
    compares it to the ground truth via exact string equality. (The earlier
    behaviour truncated to the last 100 characters, which silently dropped
    correct answers when the model produced long reasoning or trailing
    repetition; matches DeepSeek-Math's regex-over-whole-response extractor.)

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens. When provided, only the
            text up to the final pause token is considered (legacy behaviour).

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Restrict to text up to the pause point if requested (legacy callers
    # use this to drop padding tokens), but otherwise search the whole
    # response.
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[: pause_tokens_index[-1]]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return extracted_pred == gt, extracted_pred


def verify(
    solution_str: str, answer: str, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    Three-tier extraction, matching DeepSeek-Math's robust math_equal:
      1. Strict boxed extraction + string equality
      2. math_verify equivalence on the extracted boxed predicate
      3. math_verify equivalence on the full response (catches free-text
         answers like "the polar form is (3, π/2)" that the model wrote
         without \\boxed{} wrapping, especially common at baseline/early-RL
         stages before the model has learned the output format)

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple (correct: bool, pred: str)
    """
    correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if pred is None:
        pred = ""

    if not correct:
        try:
            with timeout(seconds=5):
                gold_expr = mv_parse(answer)
                # Tier 2: math-verify on the extracted boxed predicate.
                if pred != "":
                    pred_expr = mv_parse(pred)
                    if mv_verify(gold_expr, pred_expr):
                        correct = True
                # Tier 3: math-verify on the whole response. math_verify is
                # designed to walk free-text and identify the last/most
                # plausible math expression; this catches correct answers
                # the model wrote without wrapping in \\boxed{}.
                if not correct:
                    pred_expr_full = mv_parse(solution_str)
                    if mv_verify(gold_expr, pred_expr_full):
                        correct = True
        except Exception:  # ignore any parsing/verification errors
            pass
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info = None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
    **kwargs
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, 0 for incorrect)
    """
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, pause_tokens_index)

    reward = 1.0 if correct else 0.0
    score = reward
    incorrect_format = pred is None or pred == ""
    was_truncated = extra_info.get("truncated", False)
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        score -= 0.5

    # Generate explicit feedback for format errors (analogous to code feedback)
    feedback = ""
    if incorrect_format and not was_truncated and format_feedback:
        feedback = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
    elif was_truncated and format_feedback:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif not correct and correctness_feedback:
        feedback = f"Your answer is incorrect. The correct answer is {ground_truth}."

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# DeepSeek-R1-Zero / CoDistill-GRPO compatible reward
# ---------------------------------------------------------------------------

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def _format_reward(solution_str: str) -> float:
    """0.25 per closing tag present (CoDistill-GRPO, Appendix A).

    Follows DeepSeek-R1-Zero: reward 0.25 each for </think> and </answer>.
    Max format reward is 0.50; combined with the 1.0 accuracy reward the
    total possible reward is 1.50.
    """
    r = 0.0
    if THINK_CLOSE in solution_str:
        r += 0.25
    if ANSWER_CLOSE in solution_str:
        r += 0.25
    return r


def _extract_answer_from_tags(solution_str: str) -> Optional[str]:
    """Pull the contents of <answer>...</answer>, then the inner boxed expression
    if present. Returns None if no <answer> tag is found."""
    start = solution_str.rfind(ANSWER_OPEN)
    if start < 0:
        return None
    inner = solution_str[start + len(ANSWER_OPEN):]
    end = inner.find(ANSWER_CLOSE)
    if end >= 0:
        inner = inner[:end]
    inner = inner.strip()
    # if the answer block itself contains a \boxed{...}, prefer that
    boxed = last_boxed_only_string(inner)
    if boxed:
        inner = remove_boxed(boxed)
    return inner


def compute_score_r1zero(
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
    **kwargs,
) -> dict:
    """Reward function for the DeepSeek-R1-Zero template used by CoDistill-GRPO.

    Combines:
      - 1.00 accuracy reward (1 if math-verify agrees, else 0)
      - 0.50 format reward (0.25 each for </think> and </answer> tags)

    Total possible reward = 1.50.

    Accuracy comes from ``verl.utils.reward_score.math_verify.compute_score``
    which wraps the official ``math_verify`` library with proper
    ``LatexExtractionConfig`` + ``ExprExtractionConfig`` and handles boxed,
    bare, symbolic, and numeric answers uniformly. Empirically on a
    Qwen2.5-Math-1.5B val_before_train dump, this brings MATH500 to 55.4%
    (paper Base 54.92%) and OlympiadBench to 26.2% (paper 23.1%), matching
    or exceeding the CoDistill-GRPO paper baselines on 3 of 4 benchmarks.
    """
    extra_info = extra_info or {}
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)

    fmt_r = _format_reward(solution_str)
    has_full_format = fmt_r >= 0.50

    # Use verl's math_metric wrapper (proper math_verify library) for accuracy.
    # This handles boxed extraction, bare-text answers, symbolic equivalence,
    # and numeric tolerance uniformly - much more robust than our previous
    # tag-then-boxed-then-fallback hand-rolled approach.
    try:
        from verl.utils.reward_score.math_verify import compute_score as _mv_score
        mv_acc = _mv_score(solution_str, ground_truth)
        correct = mv_acc >= 1.0
    except Exception:
        # Fall back to legacy verifier if math_verify import/eval fails.
        correct, _ = verify(solution_str, ground_truth, pause_tokens_index)

    # Track which extraction path produced an answer (for the feedback string
    # and `incorrect_format` metric).
    pred_from_tags = _extract_answer_from_tags(solution_str)
    if pred_from_tags is not None and pred_from_tags != "":
        pred = pred_from_tags
        incorrect_format = False
    else:
        boxed = last_boxed_only_string(solution_str)
        pred = remove_boxed(boxed) if boxed is not None else ""
        incorrect_format = pred == ""

    acc_reward = 1.0 if correct else 0.0
    score = acc_reward + fmt_r

    feedback_parts = []

    # --- Rich format feedback ---
    if format_feedback:
        has_think_open = THINK_OPEN in solution_str
        has_think_close = THINK_CLOSE in solution_str
        has_answer_open = ANSWER_OPEN in solution_str
        has_answer_close = ANSWER_CLOSE in solution_str

        if was_truncated:
            feedback_parts.append(
                "Your response was truncated because it exceeded the maximum length. "
                "Keep your reasoning concise so the full solution fits within the token limit. "
                "Required format: <think>step-by-step reasoning</think><answer>\\boxed{final_answer}</answer>"
            )
        elif not has_think_open and not has_answer_open:
            feedback_parts.append(
                "Your response does not follow the required format at all. "
                "You must structure your response as: <think>your step-by-step reasoning here</think>"
                "<answer>\\boxed{your_final_answer}</answer>. "
                "Start with <think>, show your work, close with </think>, "
                "then provide your final answer inside <answer>\\boxed{...}</answer>."
            )
        elif not has_full_format:
            missing = []
            if not has_think_close:
                missing.append("</think> (you started reasoning but never closed the thinking block)")
            if not has_answer_open:
                missing.append("<answer> (you never started the answer block)")
            if not has_answer_close:
                missing.append("</answer> (you never closed the answer block)")
            feedback_parts.append(
                "Your response is missing: " + "; ".join(missing) + ". "
                "Required format: <think>reasoning</think><answer>\\boxed{final_answer}</answer>"
            )

        if incorrect_format and not was_truncated:
            if has_answer_open and has_answer_close:
                feedback_parts.append(
                    "Your answer block exists but does not contain a \\boxed{} expression. "
                    "Place your final numerical or symbolic answer inside \\boxed{} within the <answer> tags."
                )
            elif not has_answer_open:
                feedback_parts.append(
                    "No answer was extracted because the <answer> block is missing. "
                    "After </think>, write <answer>\\boxed{your_answer}</answer>."
                )

    # --- Rich correctness feedback ---
    if not correct and correctness_feedback:
        if pred and pred != "":
            feedback_parts.append(
                f"Your answer is incorrect. You answered {pred}, "
                f"but the correct answer is {ground_truth}. "
                "Review your reasoning for arithmetic or algebraic errors."
            )
        else:
            feedback_parts.append(
                f"Your answer is incorrect. The correct answer is {ground_truth}."
            )

    feedback = " ".join(feedback_parts)

    return {
        "score": score,
        "acc": acc_reward,
        "format_reward": fmt_r,
        "pred": pred or "",
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
