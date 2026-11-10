"""Code examples from nvidia/OpenCodeInstruct (train split).

An example is kept only if its solution is 'definitions only', passes the static safety check,
does not overlap any benchmark, and passes all of its unit tests when we execute it ourselves.

Three prompt styles (shares set in CodeConfig):
  tests     - same layout as the MBPP grader prompt: task + assert tests, with three worked
              examples taken from OUR OWN training data (never from MBPP)
  complete  - a function signature + docstring to complete
  plain     - the dataset's problem statement as-is
The target is always exactly one ```python block (see src/formats.py).
"""

import ast
import json
import random
import re

from datasets import load_dataset

from src.config import CodeConfig
from src.data.code_exec import BANNED_NAMES, passes_tests_many, static_problem, top_level_functions
from src.data.decontam import BenchmarkIndex
from src.formats import code_target, extract_code, mbpp_layout_prompt
from src.schema import Example, Message, bucket

SHARD = "hf://datasets/nvidia/OpenCodeInstruct/data/train-{:05d}-of-00050.parquet"


def unsafe_test(test: str) -> bool:
    """True if a test line is not a single, safe assert statement. Tests are executed too."""
    try:
        body = ast.parse(test).body
    except SyntaxError:
        return True
    if len(body) != 1 or not isinstance(body[0], ast.Assert):
        return True
    for node in ast.walk(body[0]):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return True
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return True
    return False


def parse_candidate(row: dict, config: CodeConfig) -> dict | None:
    """Turn one dataset row into a candidate, or None if it fails the static filters."""
    if float(row["average_test_score"]) < 1.0:
        return None
    tests = [test.strip() for test in json.loads(row["unit_tests"])]
    code = extract_code(row["output"])
    if code is None or len(tests) < config.tests_in_prompt:
        return None
    code = code.strip()
    if static_problem(code) is not None or any(unsafe_test(test) for test in tests):
        return None
    called = re.match(r"assert\s+(\w+)\(", tests[0])
    if not called or called.group(1) not in top_level_functions(code):
        return None
    return {"id": f"oci_{row['id']}", "problem": row["input"].strip(), "code": code, "tests": tests}


def complete_prompt(code: str) -> str | None:
    """Signature + docstring of a single documented function, or None if the code is not that."""
    definitions = [node for node in ast.parse(code).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        return None
    function = definitions[0]
    if ast.get_docstring(function) is None or len(function.body) < 2:
        return None
    stub = "\n".join(code.split("\n")[: function.body[0].end_lineno])  # imports + def line + docstring
    return f"Complete the following Python function.\n\n```python\n{stub}\n```"


def code_example(candidate: dict, worked_pool: list[dict], rng: random.Random, config: CodeConfig) -> Example:
    shown_tests = candidate["tests"][: config.tests_in_prompt]
    roll = bucket(candidate["id"], "code-style")
    style, prompt = "plain", candidate["problem"]
    if roll < config.tests_share:
        others = [c for c in rng.sample(worked_pool, 4) if c["id"] != candidate["id"]][:3]
        worked = [{**c, "tests": c["tests"][: config.tests_in_prompt]} for c in others]
        style, prompt = "tests", mbpp_layout_prompt(candidate["problem"], shown_tests, worked)
    elif roll < config.tests_share + config.complete_share:
        stub_prompt = complete_prompt(candidate["code"])
        if stub_prompt:
            style, prompt = "complete", stub_prompt
    messages = [Message(role="user", content=prompt), Message(role="assistant", content=code_target(candidate["code"]))]
    meta = {"problem": candidate["problem"], "tests": candidate["tests"]}
    return Example(id=candidate["id"], skill="code", source="opencodeinstruct_train", style=style,
                   messages=messages, meta=meta)  # fmt: skip


def build_code(config: CodeConfig, seed: int, index: BenchmarkIndex) -> tuple[list[Example], dict]:
    """Return (examples, counts). Reads only the configured parquet shards of the train split.

    Candidates are checked against the benchmark index BEFORE prompts are built, because a
    worked example that overlaps a benchmark would otherwise contaminate every prompt that uses it.
    """
    files = [SHARD.format(i) for i in config.shards]
    # Read the shard files directly: loading a subset through the dataset name fails its row-count check.
    rows = load_dataset("parquet", data_files=files, split="train").shuffle(seed=seed)

    candidates, scanned, overlapping = [], 0, 0
    for row in rows:
        scanned += 1
        candidate = parse_candidate(row, config)
        if candidate is None:
            continue
        if index.find(candidate["problem"]) or index.find(candidate["code"]):
            overlapping += 1
            continue
        candidates.append(candidate)
        if len(candidates) >= config.max_candidates:
            break

    passed = passes_tests_many([(c["code"], c["tests"]) for c in candidates])
    kept = [candidate for candidate, ok in zip(candidates, passed) if ok]
    worked_pool = [c for c in kept if len(c["problem"]) + len(c["code"]) <= config.fewshot_max_chars]
    rng = random.Random(seed)
    examples = [code_example(candidate, worked_pool, rng, config) for candidate in kept]

    counts = {"opencodeinstruct_train": {
        "dataset": "nvidia/OpenCodeInstruct", "split": "train", "shards": config.shards,
        "rows_scanned": scanned, "removed_benchmark_overlap": overlapping,
        "passed_static_filters": len(candidates), "passed_tests": len(kept),
        "worked_example_pool": len(worked_pool), "formatted": len(examples),
    }}  # fmt: skip
    return examples, counts
