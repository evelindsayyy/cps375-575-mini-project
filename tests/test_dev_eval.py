import json

import src.dev_eval as dev_eval
from src.config import DevEvalConfig
from src.formats import math_prompt
from src.schema import Example, Message, write_jsonl


def math_dev(example_id: str, style: str, answer: str) -> Example:
    messages = [Message(role="user", content=math_prompt(style, "Q")), Message(role="assistant", content="x")]
    return Example(id=example_id, skill="math", source="math_train", style=style, messages=messages,
                   meta={"answer": answer, "problem": "Q"})  # fmt: skip


def code_dev(example_id: str) -> Example:
    messages = [Message(role="user", content="Write add."), Message(role="assistant", content="x")]
    meta = {"problem": "Write add.", "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]}
    return Example(id=example_id, skill="code", source="opencodeinstruct_train", style="plain", messages=messages, meta=meta)


def ifollow_dev(example_id: str) -> Example:
    messages = [Message(role="user", content="Answer in lowercase."), Message(role="assistant", content="x")]
    return Example(id=example_id, skill="ifollow", source="tulu3_persona_if", style="chat", messages=messages)


def test_ifollow_is_scored_by_the_judge_and_unreadable_verdicts_count_as_failures(tmp_path, monkeypatch):
    write_jsonl(str(tmp_path / "dev" / "ifollow.jsonl"), [ifollow_dev(f"i{n}") for n in range(4)])
    monkeypatch.setattr(dev_eval, "sample_student", lambda path, examples, max_tokens: ["ok"] * len(examples))
    verdicts = [("pass", "VERDICT: PASS"), ("fail", "VERDICT: FAIL"), (None, "?"), ("pass", "VERDICT: PASS")]
    monkeypatch.setattr(dev_eval, "judge_responses", lambda config, prompts, responses: verdicts)
    config = DevEvalConfig(dev_dir=str(tmp_path / "dev"), skills=["ifollow"])
    scores = dev_eval.evaluate_dev(config, "tinker://x", str(tmp_path / "dev_eval.json"))
    assert scores == {"ifollow": 50.0, "ifollow_samples": 4, "ifollow_unreadable_verdicts": 1}
    saved = json.loads((tmp_path / "dev_eval.json").read_text())["samples"]
    assert saved[1]["judge"] == "VERDICT: FAIL" and saved[2]["correct"] is False


def test_math_completion_is_scored_by_its_final_line_and_equivalence():
    assert dev_eval.math_correct(math_dev("m", "harness", "\\frac{1}{2}"), "Work.\nANSWER: 1/2")
    assert not dev_eval.math_correct(math_dev("m", "harness", "\\frac{1}{2}"), "ANSWER: 1/2\nMore text.")
    assert dev_eval.math_correct(math_dev("m", "boxed", "4"), "So it is $\\boxed{4}$.")
    assert not dev_eval.math_correct(math_dev("m", "answer_is", "4"), "No final line.")


def test_code_to_run_follows_the_grader():
    assert dev_eval.code_to_run("Text\n```python\na = 1\n```\n```python\nb = 2\n```") == "a = 1\n"
    assert dev_eval.code_to_run("def f():\n    return 1") == "def f():\n    return 1"  # no block: run everything


def test_evaluate_dev_scores_each_skill_and_saves_every_sample(tmp_path, monkeypatch):
    write_jsonl(str(tmp_path / "dev" / "math.jsonl"), [math_dev("m1", "harness", "2"), math_dev("m2", "harness", "3")])
    write_jsonl(str(tmp_path / "dev" / "code.jsonl"), [code_dev("c1"), code_dev("c2")])
    completions = {
        "m1": "1+1.\nANSWER: 2", "m2": "Wrong.\nANSWER: 4",
        "c1": "```python\ndef add(a, b):\n    return a + b\n```", "c2": "```python\ndef add(a, b):\n    return a - b\n```",
    }  # fmt: skip
    seen_limits = {}

    def fake_sample(checkpoint_path, examples, max_tokens):
        seen_limits[examples[0].skill] = max_tokens
        return [completions[example.id] for example in examples]

    monkeypatch.setattr(dev_eval, "sample_student", fake_sample)
    out_path = tmp_path / "dev_eval.json"
    config = DevEvalConfig(dev_dir=str(tmp_path / "dev"), skills=["math", "code"])
    scores = dev_eval.evaluate_dev(config, "tinker://x", str(out_path))

    assert scores == {"math": 50.0, "math_samples": 2, "code": 50.0, "code_samples": 2}
    assert seen_limits == {"math": 4096, "code": 4096}  # the grader's limits for these tasks
    saved = json.loads(out_path.read_text())
    assert saved["scores"] == scores and saved["checkpoint_path"] == "tinker://x"
    assert {(s["id"], s["correct"]) for s in saved["samples"]} == {("m1", True), ("m2", False), ("c1", True), ("c2", False)}
