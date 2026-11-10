import json
import random

import pytest
from inspect_evals.math.utils import extract_answer

from src.config import CodeConfig, MathConfig, MixConfig
from src.data.chat_data import clean_messages
from src.data.code_data import code_example, complete_prompt, parse_candidate
from src.data.code_exec import passes_tests, static_problem
from src.data.decontam import BenchmarkIndex
from src.data.math_data import gsm8k_example, math_example, unbox_last
from src.data.mix import build_training_set
from src.data.pools import split_dev
from src.formats import target_problem
from src.schema import Example, Message, write_jsonl

LONG = (
    "Pat is to select six cookies from a tray containing only chocolate chip, oatmeal, "
    "and peanut butter cookies. How many different assortments of six cookies can be selected?"
)
SHORTER = "Write a python function to check whether a number is even."  # 11 words
SOLUTION = "We add: $2+2=4$. So the answer is $\\boxed{\\frac{8}{2}}$."
MATH_ROW = {"problem": "What is 2+2?", "solution": SOLUTION, "level": "Level 1", "type": "Algebra"}
CODE = 'def add(a, b):\n    """Return the sum of a and b."""\n    return a + b'


def make_index() -> BenchmarkIndex:
    index = BenchmarkIndex()
    index.add("math_test", LONG)
    index.add("mbpp", SHORTER)
    index.add("mbpp", "Add two numbers.")  # under 8 words: exact match only
    return index


def chat_example(example_id: str, source: str, text: str = "hello") -> Example:
    messages = [Message(role="user", content=text), Message(role="assistant", content="ok")]
    return Example(id=example_id, skill="general", source=source, style="chat", messages=messages)


def with_style(style: str, make):
    """Find an id whose deterministic style choice equals `style`."""
    return next(e for e in (make(f"id_{i}") for i in range(500)) if e.style == style)


def oci_row(**overrides) -> dict:
    row = {
        "id": "abc",
        "input": "Write a function `add` that returns the sum of two numbers.",
        "output": f"```python\n{CODE}\n```\nExample usage below.",
        "unit_tests": json.dumps(["\nassert add(1, 2) == 3\n", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]),
        "average_test_score": "1",
    }
    return {**row, **overrides}


# ---------------------------------------------------------------- benchmark overlap


def test_overlap_exact_ngram_and_short_text():
    index = make_index()
    assert index.find(LONG.upper().replace(",", " ;")) == "math_test"
    assert index.find("Here is a problem. " + LONG + " Answer carefully.") == "math_test"
    assert index.find("Task: " + SHORTER + " Use Python 3.") == "mbpp"
    assert index.find("add two numbers") == "mbpp"
    assert index.find("Write code to add two numbers together") is None
    assert index.find("A farmer has 12 cows and buys 7 more. How many cows now?") is None


def test_overlap_checks_every_message():
    assert make_index().find_in_example(chat_example("a", "s", "hi")) is None
    contaminated = chat_example("a", "s", "hi").model_copy(
        update={"messages": [Message(role="user", content="hi"), Message(role="assistant", content=LONG)]}
    )
    assert make_index().find_in_example(contaminated) == "math_test"


# ---------------------------------------------------------------- math


def test_unbox_last():
    text, answer = unbox_last(SOLUTION)
    assert answer == "\\frac{8}{2}" and "\\boxed" not in text
    assert unbox_last("So we get $\\boxed{ }$.") is None


def test_math_examples_follow_the_requested_format():
    harness = with_style("harness", lambda i: math_example(i, MATH_ROW, MathConfig()))
    assert harness.target.endswith("\nANSWER: \\frac{8}{2}")
    assert extract_answer(harness.target) == harness.meta["answer"] and target_problem(harness) is None
    boxed = with_style("boxed", lambda i: math_example(i, MATH_ROW, MathConfig()))
    assert "\\boxed{" in boxed.messages[0].content and boxed.target == SOLUTION
    assert with_style("answer_is", lambda i: math_example(i, MATH_ROW, MathConfig())).target.endswith(
        "The answer is \\frac{8}{2}"
    )
    assert all(math_example(f"id_{i}", MATH_ROW, MathConfig(harness_share=1.0)).style == "harness" for i in range(20))


def test_gsm8k_example_strips_annotations_and_commas():
    row = {"question": "How many clips?", "answer": "48/2 = <<48/2=24>>24 clips.\n#### 1,024"}
    harness = with_style("harness", lambda i: gsm8k_example(i, row, MathConfig()))
    assert harness.target == "48/2 = 24 clips.\nANSWER: 1024" and harness.source == "gsm8k_train"
    assert with_style("hashes", lambda i: gsm8k_example(i, row, MathConfig())).target.endswith("\n#### 1024")


# ---------------------------------------------------------------- code


def test_static_problem_and_execution():
    assert static_problem("import math\n" + CODE) is None
    assert static_problem(CODE + "\nprint(add(1, 2))") == "top-level statement"
    assert static_problem("import os\n" + CODE) == "import outside allowlist"
    assert static_problem("def f():\n    return open('x').read()") == "banned builtin"
    assert passes_tests(CODE, ["assert add(1, 2) == 3"]) is True
    assert passes_tests(CODE, ["assert add(1, 2) == 4"]) is False
    assert passes_tests("def loop():\n    while True:\n        pass", ["assert loop() is None"], timeout=1) is False


def test_parse_candidate_filters():
    config = CodeConfig()
    candidate = parse_candidate(oci_row(), config)
    assert candidate["code"] == CODE and len(candidate["tests"]) == 3
    assert parse_candidate(oci_row(average_test_score="0.9"), config) is None
    assert parse_candidate(oci_row(output="no code block"), config) is None
    assert parse_candidate(oci_row(unit_tests=json.dumps(["assert other(1) == 1"] * 3)), config) is None
    assert parse_candidate(oci_row(unit_tests=json.dumps(["assert add(1, 2) == 3"])), config) is None
    unsafe = json.dumps(["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert __import__('os')"])
    assert parse_candidate(oci_row(unit_tests=unsafe), config) is None


def test_code_example_styles_and_target():
    candidate = parse_candidate(oci_row(), CodeConfig())
    pool = [{**candidate, "id": f"oci_{i}"} for i in range(6)]
    tests_style = code_example(candidate, pool, random.Random(0), CodeConfig(tests_share=1.0))
    assert tests_style.style == "tests" and tests_style.messages[0].content.endswith("## Completion:\n")
    plain = code_example(candidate, pool, random.Random(0), CodeConfig(tests_share=0.0, complete_share=0.0))
    assert plain.style == "plain" and plain.messages[0].content == candidate["problem"]
    assert target_problem(tests_style) is None and tests_style.target == f"```python\n{CODE}\n```"


def test_complete_prompt_gives_signature_and_docstring_only():
    prompt = complete_prompt("import math\n" + CODE)
    assert prompt.endswith('```python\nimport math\ndef add(a, b):\n    """Return the sum of a and b."""\n```')
    assert complete_prompt("def f(x):\n    return x") is None


# ---------------------------------------------------------------- chat, dev split, mixing


def test_clean_messages():
    ok = [{"role": "user", "content": " hi "}, {"role": "assistant", "content": "hello"}]
    assert [m.content for m in clean_messages(ok)] == ["hi", "hello"]
    assert clean_messages([{"role": "system", "content": "x"}] + ok) is None
    assert clean_messages([ok[0], {"role": "assistant", "content": " "}]) is None


def test_split_dev_is_disjoint_and_deterministic():
    examples = [chat_example(f"e{i}", "s") for i in range(50)]
    pool, dev = split_dev(examples, 10)
    assert len(dev) == 10 and len(pool) == 40 and not {e.id for e in dev} & {e.id for e in pool}
    assert [e.id for e in split_dev(examples, 10)[1]] == [e.id for e in dev]


def test_build_training_set_samples_counts_deterministically(tmp_path):
    write_jsonl(str(tmp_path / "a.jsonl"), [chat_example(f"a{i}", "a") for i in range(20)])
    write_jsonl(str(tmp_path / "b.jsonl"), [chat_example(f"b{i}", "b") for i in range(20)])
    config = MixConfig(pools_dir=str(tmp_path), counts={"a": 5, "b": 3})
    examples, record = build_training_set(config, make_index())
    assert len(examples) == 8 and record["sources"]["a"] == {"split": "train", "sampled": 5, "pool_size": 20}
    assert record["overlap_check"]["overlaps_found"] == 0
    assert [e.id for e in build_training_set(config, make_index())[0]] == [e.id for e in examples]
    with pytest.raises(ValueError):
        build_training_set(MixConfig(pools_dir=str(tmp_path), counts={"a": 21}), make_index())


def test_build_training_set_refuses_benchmark_overlap(tmp_path):
    write_jsonl(str(tmp_path / "a.jsonl"), [chat_example("a0", "a", LONG)])
    with pytest.raises(RuntimeError):
        build_training_set(MixConfig(pools_dir=str(tmp_path), counts={"a": 1}), make_index())


# ---------------------------------------------------------------- pool filters


class WordTokenizer:
    """Stand-in tokenizer: one token per word."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


def test_filter_examples_applies_each_filter_and_counts_it():
    from src.config import PoolsConfig
    from src.data.pools import filter_examples

    def math(example_id: str, problem: str, target: str) -> Example:
        messages = [Message(role="user", content=problem), Message(role="assistant", content=target)]
        return Example(id=example_id, skill="math", source="math_train", style="harness", messages=messages,
                       meta={"answer": "4", "problem": problem})  # fmt: skip

    examples = [
        math("ok", "What is 2+2?", "Add.\nANSWER: 4"),
        math("overlap", LONG, "Count.\nANSWER: 4"),
        math("bad_format", "What is 1+3?", "ANSWER: 4\nHope this helps!"),
        math("too_long", "What is 3+1?", "word " * 20 + "\nANSWER: 4"),
        math("duplicate", "What is 2+2?", "Sum.\nANSWER: 4"),
    ]
    config = PoolsConfig(max_target_tokens={"math": 10, "code": 10, "ifollow": 10, "general": 10})
    removed_log: list[dict] = []
    kept, counts = filter_examples(examples, make_index(), WordTokenizer(), config, removed_log)
    assert [e.id for e in kept] == ["ok"]
    assert counts["math_train"] == {"removed_overlap_math_test": 1, "removed_bad_format": 1,
                                    "removed_too_long": 1, "removed_duplicate": 1, "kept": 1}  # fmt: skip
    assert removed_log[0]["id"] == "overlap" and removed_log[0]["benchmark"] == "math_test"
