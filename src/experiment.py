"""Run one experiment from one config file:  data mix -> train -> evaluate -> saved record.

    python -m src.experiment configs/<name>.yaml

Everything about the run is stored in runs/<name>/, each file written as soon as its stage ends:
    config.json     the exact ExperimentConfig
    train.jsonl     the exact training examples
    data.json       what was sampled from each source + the benchmark-overlap check
    losses.jsonl    loss at every step
    training.json   checkpoint path and training record (written the moment training ends)
    eval.json       full output of the provided evaluation script
    dev_eval.json   scores and every sample on our own held-out dev sets (src/dev_eval.py)
    summary.json    config + data counts + checkpoint path + scores (read by src/report.py)

Runs are resumable: running the same config again skips every stage whose file already exists,
so an interruption during evaluation never costs a second training run. A run folder that was
created with a different config is refused; a changed experiment needs a new name.
"""

import json
import os
import sys
from datetime import datetime, timezone

from src.config import DevEvalConfig, ExperimentConfig, load_config
from src.data.decontam import build_benchmark_index
from src.data.mix import build_training_set
from src.dev_eval import evaluate_dev
from src.evaluate import evaluate, read_scores
from src.schema import read_jsonl, write_jsonl
from src.train.sft import train

RUNS_DIR = "runs"


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_experiment(config: ExperimentConfig, runs_dir: str = RUNS_DIR) -> dict:
    run_dir = os.path.join(runs_dir, config.name)
    paths = {name: os.path.join(run_dir, name) for name in
             ("config.json", "train.jsonl", "data.json", "training.json", "eval.json", "dev_eval.json", "summary.json")}  # fmt: skip
    config_record = config.model_dump(mode="json")

    if os.path.exists(paths["config.json"]):
        if load_json(paths["config.json"]) != config_record:
            raise ValueError(f"{run_dir} was created with a different config; use a new experiment name")
    else:
        os.makedirs(run_dir, exist_ok=True)
        save_json(paths["config.json"], config_record)
    if os.path.exists(paths["summary.json"]):
        return load_json(paths["summary.json"])

    # Stage 1: sample the training set. data.json is written last, so its presence means done.
    if os.path.exists(paths["data.json"]):
        examples, data_record = read_jsonl(paths["train.jsonl"]), load_json(paths["data.json"])
    else:
        examples, data_record = build_training_set(config.mix, build_benchmark_index())
        write_jsonl(paths["train.jsonl"], examples)
        save_json(paths["data.json"], data_record)

    # Stage 2: train and publish. The checkpoint path is saved before anything else happens.
    if os.path.exists(paths["training.json"]):
        training = load_json(paths["training.json"])
    else:
        training = train(config.train, examples, config.name, run_dir)
        save_json(paths["training.json"], training)

    # Stage 3: evaluate. The provided script writes eval.json only when it finishes.
    if os.path.exists(paths["eval.json"]):
        scores = read_scores(load_json(paths["eval.json"]))
    else:
        scores = evaluate(config.eval, training["checkpoint_path"], config.train.base_model, paths["eval.json"])

    # Stage 4: our own dev sets, scored the same way for every run (DevEvalConfig defaults).
    if os.path.exists(paths["dev_eval.json"]):
        dev_scores = load_json(paths["dev_eval.json"])["scores"]
    else:
        dev_scores = evaluate_dev(DevEvalConfig(), training["checkpoint_path"], paths["dev_eval.json"])

    summary = {
        "name": config.name,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config_record,
        "data": data_record,
        "training": training,
        "full_eval": config.eval.limit is None,
        "scores": scores,
        "dev_scores": dev_scores,
    }
    save_json(paths["summary.json"], summary)
    return summary


if __name__ == "__main__":
    result = run_experiment(load_config(sys.argv[1], ExperimentConfig))
    print(json.dumps(result["scores"], indent=2))
