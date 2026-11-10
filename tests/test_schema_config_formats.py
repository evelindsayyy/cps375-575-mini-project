import pytest
from pydantic import ValidationError

from src.config import ExperimentConfig, MixConfig, PoolsConfig, TrainConfig, load_config
from src.formats import code_target, extract_code, math_prompt, math_target, mbpp_layout_prompt, target_problem
from src.schema import Example, Message, bucket, normalize, read_jsonl, write_jsonl


def example(skill="math", style="harness", target="x\nANSWER: 4", meta=None, **extra):
    messages = [Message(role="user", content="q"), Message(role="assistant", content=target)]
    return Example(id="e1", skill=skill, source="s", style=style, messages=messages, meta=meta or {"answer": "4"}, **extra)


def test_example_only_accepts_the_train_split():
    assert example().split == "train"
    with pytest.raises(ValidationError):
        example(split="test")


def test_example_rejects_system_messages_and_unknown_fields():
    with pytest.raises(ValidationError):
        Message(role="system", content="x")
    with pytest.raises(ValidationError):
        example(extra_field=1)


def test_jsonl_round_trip(tmp_path):
    path = str(tmp_path / "sub" / "a.jsonl")
    write_jsonl(path, [example()])
    assert read_jsonl(path) == [example()]


def test_bucket_and_normalize():
    assert bucket("abc", "s") == bucket("abc", "s") != bucket("abc", "t")
    assert 0.0 <= bucket("abc") < 1.0
    assert normalize("Hello,  WORLD!\n$x^2$") == "hello world x 2"


def test_configs_reject_unknown_fields_and_wrong_base_model():
    with pytest.raises(ValidationError):
        MixConfig(counts={"math_train": 1}, typo=1)
    with pytest.raises(ValidationError):
        TrainConfig(base_model="Qwen/Qwen3.5-9B")
    with pytest.raises(ValidationError):
        PoolsConfig(math={"harness_share": 1.5})


def test_repo_config_files_load():
    assert load_config("configs/pools.yaml", PoolsConfig) == PoolsConfig()
    smoke = load_config("configs/s0_smoke.yaml", ExperimentConfig)
    assert smoke.name == "s0_smoke" and smoke.eval.limit == 50 and sum(smoke.mix.counts.values()) == 4500


def test_math_formats_follow_the_spec():
    prompt = math_prompt("harness", " What is 2+2? ")
    assert prompt.startswith("Solve the following math problem step by step.")
    assert "\n\nWhat is 2+2?\n\n" in prompt and "{prompt}" not in prompt
    assert math_target("harness", "Add.\n", "\\frac{8}{2}") == "Add.\nANSWER: \\frac{8}{2}"
    assert math_target("hashes", "Add.", "4").endswith("\n#### 4")
    assert "\\boxed{}" in math_prompt("boxed", "q")


def test_target_problem_math():
    assert target_problem(example()) is None
    assert target_problem(example(target="ANSWER: 4\nHope this helps!")) is not None
    money = example(target="x\nANSWER: \\$42409", meta={"answer": "\\$42409"})
    assert target_problem(money) is None


def test_target_problem_code():
    good = code_target("def add(a, b):\n    return a + b\n")
    assert good == "```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(good) == "def add(a, b):\n    return a + b\n"
    assert target_problem(example(skill="code", style="plain", target=good)) is None
    for bad in ("Here:\n" + good, good + "\n" + good, "```python\ndef f(:\n```", "```\nx = 1\n```"):
        assert target_problem(example(skill="code", style="plain", target=bad)) is not None


def test_mbpp_layout_prompt_matches_grader_layout():
    shot = {"problem": "Double a number.", "tests": ["assert d(1) == 2"], "code": "def d(x):\n    return 2 * x"}
    prompt = mbpp_layout_prompt("Add two numbers.", ["assert add(1, 2) == 3", "assert add(0, 0) == 0"], [shot] * 3)
    assert prompt.startswith("You are an expert Python programmer.")
    assert "## Completion 3\n```python\ndef d(x):\n    return 2 * x\n```\n\n\n# Now, do it" in prompt
    assert prompt.endswith(
        "## Prompt:\n```python\nAdd two numbers.\n```\n\n## Test Case:\n```python\n"
        "assert add(1, 2) == 3\nassert add(0, 0) == 0\n```\n\n## Completion:\n"
    )
