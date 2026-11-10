"""Step 4 of the pipeline: evaluate a checkpoint with the PROVIDED evaluation script.

The script (evaluation/eval_all.py) is run unchanged as a subprocess, exactly as the README
describes, so our numbers are produced the same way as the graded ones.
"""

import json
import os
import subprocess
import sys

from src.config import EvalConfig

SCORE_KEYS = {
    "ifeval": "google/IFEval/final_acc",
    "math500": "math500/accuracy",
    "mbpp": "mbpp/accuracy",
}


def read_scores(results: dict) -> dict:
    """Pull the three headline scores (in %) and sample counts out of a results JSON."""
    scores = {}
    for task, key in SCORE_KEYS.items():
        scores[task] = round(100 * results[task]["metrics"][key], 1)
        scores[f"{task}_samples"] = len(results[task]["samples"])
    return scores


def evaluate(config: EvalConfig, checkpoint_path: str, base_model: str, out_path: str) -> dict:
    command = [
        sys.executable, "-m", "evaluation.eval_all",
        "--checkpoint_path", checkpoint_path,
        "--base_model", base_model,
        "--temperature", str(config.temperature),
        "--top_p", str(config.top_p),
        "--output_path", out_path,
    ]  # fmt: skip
    if config.limit is not None:
        command += ["--limit", str(config.limit)]
    subprocess.run(command, check=True)
    with open(out_path, encoding="utf-8") as f:
        results = json.load(f)
    # The provided script exits normally even when a task fails; it records {"error": ...} instead
    # (seen on 2026-11-10 when the tokenizer download failed). Such a file must not count as done.
    failed = {task: results[task]["error"] for task in SCORE_KEYS if "error" in results[task]}
    if failed:
        os.replace(out_path, out_path + ".failed")
        raise RuntimeError(f"Evaluation failed for {sorted(failed)}: {failed}")
    return read_scores(results)
