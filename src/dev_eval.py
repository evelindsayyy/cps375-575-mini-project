"""Score a checkpoint on our own held-out dev sets (data/dev/), to compare runs without the public benchmarks.

    python -m src.dev_eval runs/<name>      # score a finished run; writes runs/<name>/dev_eval.json

The dev sets are training-split examples held out when the pools were built and never trained on.
Generation follows the provided evaluation: the grader's renderer for the base model, greedy
decoding (temperature 0, top-p 1), and the grader's token limit per task. Scoring follows the
grader's rules:
  math  the final line the prompt asked for (ANSWER:, \\boxed{}, "The answer is", ####) must be
        the last line, and its answer must equal the gold answer by the grader's equivalence checks
  code  the first ```python block (the whole completion if there is none, as the MBPP grader
        does) is run with all of the example's tests, in a separate process with a timeout
"""

import json
import os
import sys

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.config import BASE_MODEL, DevEvalConfig
from src.data.code_exec import passes_tests_many
from src.distill.verify import answers_match
from src.formats import extract_code, final_answer
from src.schema import Example, read_jsonl


def math_correct(example: Example, completion: str) -> bool:
    answer = final_answer(example.style, completion)
    return answer is not None and answers_match(answer, example.meta["answer"])


def code_to_run(completion: str) -> str:
    code = extract_code(completion)
    return completion if code is None else code


def sample_student(checkpoint_path: str, examples: list[Example], max_tokens: int) -> list[str]:
    """One greedy completion per example, in input order."""
    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(BASE_MODEL), get_tokenizer(BASE_MODEL))
    client = tinker.ServiceClient().create_sampling_client(model_path=checkpoint_path).result()
    params = tinker.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=renderer.get_stop_sequences())
    futures = [
        client.sample(renderer.build_generation_prompt([{"role": "user", "content": example.messages[0].content}]),
                      num_samples=1, sampling_params=params)
        for example in examples
    ]  # fmt: skip
    return [renderers.get_text_content(renderer.parse_response(f.result().sequences[0].tokens)[0]) for f in futures]


def evaluate_dev(config: DevEvalConfig, checkpoint_path: str, out_path: str) -> dict:
    """Score every configured skill; write all samples and the scores to out_path."""
    scores, samples = {}, []
    for skill in config.skills:
        examples = read_jsonl(os.path.join(config.dev_dir, f"{skill}.jsonl"))
        completions = sample_student(checkpoint_path, examples, config.max_tokens[skill])
        if skill == "math":
            correct = [math_correct(example, text) for example, text in zip(examples, completions)]
        else:
            correct = passes_tests_many([(code_to_run(text), ex.meta["tests"]) for ex, text in zip(examples, completions)])
        scores[skill] = round(100 * sum(correct) / len(examples), 1)
        scores[f"{skill}_samples"] = len(examples)
        samples += [{"id": example.id, "skill": skill, "style": example.style, "correct": ok, "completion": text}
                    for example, text, ok in zip(examples, completions, correct)]  # fmt: skip
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": config.model_dump(), "checkpoint_path": checkpoint_path, "scores": scores,
                   "samples": samples}, f, indent=2)  # fmt: skip
    return scores


if __name__ == "__main__":
    run_dir = sys.argv[1]
    with open(os.path.join(run_dir, "training.json"), encoding="utf-8") as f:
        checkpoint = json.load(f)["checkpoint_path"]
    print(json.dumps(evaluate_dev(DevEvalConfig(), checkpoint, os.path.join(run_dir, "dev_eval.json")), indent=2))
