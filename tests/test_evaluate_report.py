import json

from src.evaluate import read_scores
from src.report import runs_table

RESULTS = {
    "ifeval": {"metrics": {"google/IFEval/final_acc": 0.6012}, "samples": [{}] * 541},
    "math500": {"metrics": {"math500/accuracy": 0.47}, "samples": [{}] * 500},
    "mbpp": {"metrics": {"mbpp/accuracy": 0.7198}, "samples": [{}] * 257},
}


def summary(name: str, finished: str, full_eval: bool) -> dict:
    return {
        "name": name, "finished": finished, "full_eval": full_eval, "scores": read_scores(RESULTS),
        "config": {"notes": "n", "train": {"lora_rank": 64, "epochs": 2}, "eval": {"limit": None if full_eval else 50}},
        "data": {"sources": {"math_train": {"sampled": 10}}},
        "training": {"start_learning_rate": 0.0002, "checkpoint_path": f"tinker://{name}"},
    }  # fmt: skip


def test_read_scores():
    assert read_scores(RESULTS) == {"ifeval": 60.1, "ifeval_samples": 541, "math500": 47.0,
                                    "math500_samples": 500, "mbpp": 72.0, "mbpp_samples": 257}  # fmt: skip


def test_runs_table_is_ordered_and_marks_threshold(tmp_path):
    for name, finished, full in [("b", "2026-11-12", True), ("a", "2026-11-11", False)]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(summary(name, finished, full)))
    rows = runs_table(str(tmp_path)).strip().split("\n")
    assert len(rows) == 4 and rows[2].startswith("| a |") and rows[3].startswith("| b |")
    assert "| limit 50 |" in rows[2] and "| no |" in rows[2]  # a quick eval never counts as meeting
    assert "| full |" in rows[3] and "| 60.1 | 47.0 | 72.0 | yes |" in rows[3]


def test_evaluate_rejects_results_with_a_failed_task(tmp_path, monkeypatch):
    import pytest

    import src.evaluate as evaluate_module
    from src.config import EvalConfig

    out_path = tmp_path / "eval.json"
    broken = {**RESULTS, "mbpp": {"error": "tokenizer download failed"}}
    monkeypatch.setattr(evaluate_module.subprocess, "run", lambda command, check: out_path.write_text(json.dumps(broken)))
    with pytest.raises(RuntimeError, match="mbpp"):
        evaluate_module.evaluate(EvalConfig(), "tinker://x", "Qwen/Qwen3.5-9B-Base", str(out_path))
    assert not out_path.exists() and (tmp_path / "eval.json.failed").exists()

    monkeypatch.setattr(evaluate_module.subprocess, "run", lambda command, check: out_path.write_text(json.dumps(RESULTS)))
    assert evaluate_module.evaluate(EvalConfig(), "tinker://x", "Qwen/Qwen3.5-9B-Base", str(out_path))["mbpp"] == 72.0
