"""Math examples from the MATH train split and the GSM8K train split.

Targets use each dataset's own human-written solution, rewritten so the final line is in the
answer format the prompt asked for (see src/formats.py).
"""

import re

from datasets import load_dataset
from inspect_evals.math.utils import last_boxed_only_string, remove_boxed

from src.config import MathConfig
from src.formats import math_prompt, math_target
from src.schema import Example, Message, bucket


def pick_style(example_id: str, harness_share: float, alternatives: list[str]) -> str:
    """Deterministically choose 'harness' for most examples, otherwise one alternative style."""
    if bucket(example_id, "math-style") < harness_share:
        return "harness"
    return alternatives[int(bucket(example_id, "math-alt") * len(alternatives))]


def unbox_last(solution: str) -> tuple[str, str] | None:
    """Replace the last \\boxed{X} in a MATH solution with X. Returns (text, X).

    Returns None when the boxed answer is empty (2 of the 7,500 train solutions).
    """
    boxed = last_boxed_only_string(solution)
    answer = remove_boxed(boxed).strip()
    if not answer:
        return None
    start = solution.rfind(boxed)
    return solution[:start] + answer + solution[start + len(boxed) :], answer


def make_example(example_id: str, source: str, style: str, problem: str, target: str, meta: dict) -> Example:
    messages = [Message(role="user", content=math_prompt(style, problem)), Message(role="assistant", content=target)]
    return Example(id=example_id, skill="math", source=source, style=style, messages=messages, meta=meta)


def math_example(example_id: str, row: dict, config: MathConfig) -> Example | None:
    unboxed = unbox_last(row["solution"])
    if unboxed is None:
        return None
    text, answer = unboxed
    style = pick_style(example_id, config.harness_share, ["boxed", "answer_is"])
    # The "boxed" style keeps the dataset's solution unchanged: it already ends in \boxed{answer}.
    target = row["solution"].strip() if style == "boxed" else math_target(style, text, answer)
    meta = {"answer": answer, "problem": row["problem"], "level": row["level"], "type": row["type"]}
    return make_example(example_id, "math_train", style, row["problem"], target, meta)


def gsm8k_example(example_id: str, row: dict, config: MathConfig) -> Example:
    """Every GSM8K train answer ends with '#### <number>'; 79 of them use thousands commas."""
    steps, final = row["answer"].rsplit("####", 1)
    steps = re.sub(r"<<[^>]*>>", "", steps)  # remove calculator annotations
    answer = final.strip().replace(",", "")
    style = pick_style(example_id, config.harness_share, ["hashes", "answer_is"])
    meta = {"answer": answer, "problem": row["question"]}
    return make_example(example_id, "gsm8k_train", style, row["question"], math_target(style, steps, answer), meta)


def build_math(config: MathConfig) -> tuple[list[Example], dict]:
    """Return (examples, counts). Only training splits are loaded."""
    math_rows = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    math_examples = [math_example(f"math_train_{i}", row, config) for i, row in enumerate(math_rows)]
    math_examples = [example for example in math_examples if example is not None]

    gsm_rows = load_dataset("openai/gsm8k", "main", split="train")
    gsm_examples = [gsm8k_example(f"gsm8k_train_{i}", row, config) for i, row in enumerate(gsm_rows)]

    counts = {
        "math_train": {"dataset": "DigitalLearningGmbH/MATH-lighteval", "split": "train",
                       "loaded": len(math_rows), "formatted": len(math_examples)},
        "gsm8k_train": {"dataset": "openai/gsm8k (main)", "split": "train",
                        "loaded": len(gsm_rows), "formatted": len(gsm_examples)},
    }  # fmt: skip
    return math_examples + gsm_examples, counts
