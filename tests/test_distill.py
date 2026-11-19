from types import SimpleNamespace

from src.config import DistillConfig, TeacherConfig
from src.distill.math_distill import (
    anthropic_sample,
    select_solutions,
    teacher_messages,
)
from src.distill.verify import answers_match
from src.formats import final_answer, math_prompt, target_problem
from src.schema import Example, Message

TEACHER = TeacherConfig(model="Qwen/Qwen3.8-27B", renderer="qwen3_8_disable_thinking")


def config(**overrides) -> DistillConfig:
    fields = {"name": "math_train_distilled_test", "source": "math_train", "problems": 10, "teacher": TEACHER}
    return DistillConfig(**{**fields, **overrides})


def question(style: str = "harness", answer: str = "\\frac{3}{2}") -> Example:
    messages = [Message(role="user", content=math_prompt(style, "Find r.")), Message(role="assistant", content="x")]
    meta = {"answer": answer, "problem": "Find r."}
    return Example(id="math_train_7", skill="math", source="math_train", style=style, messages=messages, meta=meta)


def sample(text: str, stopped: bool = True) -> dict:
    return {"text": text, "stopped": stopped}


# ---------------------------------------------------------------- answer extraction and matching


def test_final_answer_for_each_style():
    assert final_answer("harness", "Steps.\nANSWER: 3/2") == "3/2"
    assert final_answer("boxed", "So $r = \\boxed{\\frac{3}{2}}$.") == "\\frac{3}{2}"
    assert final_answer("answer_is", "Steps.\nThe answer is 12.") == "12"
    assert final_answer("hashes", "Steps.\n#### 12") == "12"


def test_final_answer_requires_the_final_line_last():
    assert final_answer("harness", "ANSWER: 3/2\nLet me double-check.") is None
    assert final_answer("answer_is", "The answer is 12\nDone.") is None
    assert final_answer("hashes", "#### 12\nDone.") is None
    assert final_answer("boxed", "No box here.") is None


def test_answers_match_uses_mathematical_equivalence():
    assert answers_match("3/2", "\\frac{3}{2}")
    assert answers_match("(0, 5)", "(0,5)")
    assert not answers_match("5", "6")


# ---------------------------------------------------------------- which solutions are kept


def test_select_keeps_only_correct_finished_answers():
    samples = [
        sample("Work.\nANSWER: 3/2"),
        sample("Work.\nANSWER: 2"),  # wrong value
        sample("Work.\nANSWER: 3/2", stopped=False),  # hit the token limit
        sample("Work, no final line."),  # no answer the grader could read
    ]
    kept, counts = select_solutions(question(), samples, config())
    assert [e.target for e in kept] == ["Work.\nANSWER: 3/2"]
    assert counts == {"sampled": 4, "truncated": 1, "no_answer": 1, "wrong": 1, "correct": 1, "duplicate": 0}


def test_select_drops_duplicates_and_caps_per_problem():
    texts = ["A.\nANSWER: 3/2", "a.  \nANSWER: 3/2", "B.\nANSWER: 3/2", "C.\nANSWER: 3/2", "D.\nANSWER: 3/2"]
    kept, counts = select_solutions(question(), [sample(t) for t in texts], config(keep_per_problem=3))
    assert [e.target for e in kept] == ["A.\nANSWER: 3/2", "B.\nANSWER: 3/2", "C.\nANSWER: 3/2"]
    assert counts["duplicate"] == 1 and counts["correct"] == 5


def test_kept_examples_keep_the_question_prompt_and_pass_the_format_check():
    kept, _ = select_solutions(question(), [sample("Work.\nANSWER: \\frac{3}{2}")], config())
    example = kept[0]
    assert example.source == "math_train_distilled_test" and example.split == "train"
    assert example.messages[0].content == question().messages[0].content
    assert example.meta["teacher"] == "Qwen/Qwen3.8-27B" and example.meta["question_id"] == "math_train_7"
    assert target_problem(example) is None


def test_teacher_sees_the_format_spec_then_the_question():
    messages = teacher_messages(question())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == question().messages[0].content


# ---------------------------------------------------------------- Anthropic teacher responses


def anthropic_response(stop_reason: str, *texts: str) -> SimpleNamespace:
    blocks = [SimpleNamespace(type="text", text=text) for text in texts]
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=SimpleNamespace(output_tokens=42))


def test_anthropic_answer_that_ended_on_its_own_counts_as_finished():
    row = anthropic_sample(anthropic_response("end_turn", "Step 1.\n", "ANSWER: 3"))
    assert row == {"text": "Step 1.\nANSWER: 3", "stopped": True, "tokens": 42}


def test_anthropic_answer_cut_at_the_token_limit_counts_as_truncated():
    assert anthropic_sample(anthropic_response("max_tokens", "Step 1. Step 2."))["stopped"] is False

