import json
import os

import pytest

import src.experiment as experiment
from src.config import ExperimentConfig, MixConfig
from src.schema import Example, Message

RESULTS = {
    "ifeval": {"metrics": {"google/IFEval/final_acc": 0.60}, "samples": [{}] * 541},
    "math500": {"metrics": {"math500/accuracy": 0.47}, "samples": [{}] * 500},
    "mbpp": {"metrics": {"mbpp/accuracy": 0.72}, "samples": [{}] * 257},
}


def make_config(**overrides) -> ExperimentConfig:
    return ExperimentConfig(name="run1", mix=MixConfig(counts={"a": 1}), **overrides)


@pytest.fixture
def stages(monkeypatch):
    """Replace the three expensive stages with fakes that count how often they run."""
    calls = {"data": 0, "train": 0, "eval": 0, "dev": 0}

    def fake_data(mix_config, index):
        calls["data"] += 1
        messages = [Message(role="user", content="q"), Message(role="assistant", content="a")]
        example = Example(id="e1", skill="general", source="a", style="chat", messages=messages)
        return [example], {"sources": {"a": {"sampled": 1}}, "total": 1}

    def fake_train(train_config, examples, name, run_dir):
        calls["train"] += 1
        return {"checkpoint_path": "tinker://fake", "examples": len(examples)}

    def fake_evaluate(eval_config, checkpoint_path, base_model, out_path):
        calls["eval"] += 1
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(RESULTS, f)
        return experiment.read_scores(RESULTS)

    def fake_dev(dev_config, checkpoint_path, out_path):
        calls["dev"] += 1
        scores = {"math": 40.0, "math_samples": 300, "code": 70.0, "code_samples": 300}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"scores": scores}, f)
        return scores

    monkeypatch.setattr(experiment, "build_benchmark_index", lambda: None)
    monkeypatch.setattr(experiment, "evaluate_dev", fake_dev)
    monkeypatch.setattr(experiment, "build_training_set", fake_data)
    monkeypatch.setattr(experiment, "train", fake_train)
    monkeypatch.setattr(experiment, "evaluate", fake_evaluate)
    return calls


def test_fresh_run_executes_every_stage_and_saves_each_result(tmp_path, stages):
    summary = experiment.run_experiment(make_config(), str(tmp_path))
    assert stages == {"data": 1, "train": 1, "eval": 1, "dev": 1}
    assert summary["training"]["checkpoint_path"] == "tinker://fake" and summary["scores"]["mbpp"] == 72.0
    assert summary["dev_scores"]["math"] == 40.0
    saved = sorted(os.listdir(tmp_path / "run1"))
    assert saved == ["config.json", "data.json", "dev_eval.json", "eval.json", "summary.json", "train.jsonl", "training.json"]


def test_crash_during_evaluation_keeps_the_checkpoint_and_does_not_retrain(tmp_path, stages, monkeypatch):
    def failing_evaluate(*args):
        raise RuntimeError("evaluation crashed")

    working_evaluate = experiment.evaluate
    monkeypatch.setattr(experiment, "evaluate", failing_evaluate)
    with pytest.raises(RuntimeError):
        experiment.run_experiment(make_config(), str(tmp_path))
    with open(tmp_path / "run1" / "training.json", encoding="utf-8") as f:
        assert json.load(f)["checkpoint_path"] == "tinker://fake"

    monkeypatch.setattr(experiment, "evaluate", working_evaluate)
    summary = experiment.run_experiment(make_config(), str(tmp_path))
    assert stages == {"data": 1, "train": 1, "eval": 1, "dev": 1}  # data and training were not repeated
    assert summary["scores"]["ifeval"] == 60.0


def test_finished_run_is_returned_without_running_anything(tmp_path, stages):
    first = experiment.run_experiment(make_config(), str(tmp_path))
    assert experiment.run_experiment(make_config(), str(tmp_path)) == first
    assert stages == {"data": 1, "train": 1, "eval": 1, "dev": 1}


def test_run_folder_with_a_different_config_is_refused(tmp_path, stages):
    experiment.run_experiment(make_config(), str(tmp_path))
    with pytest.raises(ValueError):
        experiment.run_experiment(make_config(notes="changed"), str(tmp_path))
    assert stages == {"data": 1, "train": 1, "eval": 1, "dev": 1}
