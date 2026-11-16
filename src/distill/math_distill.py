"""Teacher-written math solutions, kept only when the final answer equals the gold answer.

    python -m src.distill.math_distill configs/distill/<name>.yaml

Questions come from an existing math pool (MATH train or GSM8K train, already de-duplicated
against every benchmark). The teacher gets the output-format spec (formats.MATH_TEACHER_SPEC)
as its system message and the unchanged training prompt as the question. A solution is kept
only if it finished before the token limit, its last line is the final-answer line the prompt
asked for, and that answer equals the gold answer by the grader's own equivalence checks.
Up to `keep_per_problem` distinct correct solutions are kept per question: training on several
distinct correct solutions improved over plain SFT in Yuan et al. (arXiv 2308.01825).

Outputs:
    <out_dir>/distill/<name>/samples.jsonl   every teacher sample, written batch by batch (resumable)
    <pools_dir>/<name>.jsonl                 the kept examples (a new pool)
    <out_dir>/manifests/<name>.json          config, token counts, and counts at every step
"""

import json
import logging
import os
import sys
from collections import Counter

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.config import BASE_MODEL, DistillConfig, load_config
from src.data.decontam import build_benchmark_index, split_clean
from src.data.pools import within_token_limit
from src.distill.verify import answers_match
from src.formats import MATH_TEACHER_SPEC, final_answer, target_problem
from src.schema import Example, Message, bucket, normalize, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

def teacher_messages(question: Example) -> list[dict]:
    return [{"role": "system", "content": MATH_TEACHER_SPEC}, {"role": "user", "content": question.messages[0].content}]


def select_solutions(question: Example, samples: list[dict], config: DistillConfig) -> tuple[list[Example], dict]:
    """Keep the distinct correct solutions among one question's samples, at most keep_per_problem."""
    counts = Counter({"sampled": len(samples), "truncated": 0, "no_answer": 0, "wrong": 0, "correct": 0, "duplicate": 0})
    kept, seen = [], set()
    for sample in samples:
        text = sample["text"].strip()
        answer = final_answer(question.style, text)
        if not sample["stopped"]:
            counts["truncated"] += 1
        elif answer is None:
            counts["no_answer"] += 1
        elif not answers_match(answer, question.meta["answer"]):
            counts["wrong"] += 1
        else:
            counts["correct"] += 1
            if normalize(text) in seen:
                counts["duplicate"] += 1
            elif len(kept) < config.keep_per_problem:
                seen.add(normalize(text))
                kept.append(solution_example(question, text, answer, len(kept), config))
    return kept, dict(counts)


def solution_example(question: Example, text: str, answer: str, k: int, config: DistillConfig) -> Example:
    # meta["answer"] is the teacher's own final answer (verified equal to the gold one), so the
    # format check confirms the grader reads exactly this answer from the last line.
    meta = {**question.meta, "answer": answer, "gold_answer": question.meta["answer"],
            "question_id": question.id, "teacher": config.teacher.model}  # fmt: skip
    messages = [question.messages[0], Message(role="assistant", content=text)]
    return Example(id=f"{config.name}_{question.id}_{k}", skill="math", source=config.name,
                   style=question.style, messages=messages, meta=meta)  # fmt: skip


def pick_questions(config: DistillConfig) -> list[Example]:
    """The first `problems` questions of the source pool in a seeded hash order (reproducible)."""
    pool = read_jsonl(os.path.join(config.pools_dir, f"{config.source}.jsonl"))
    if config.problems > len(pool):
        raise ValueError(f"{config.source}: asked for {config.problems} questions but the pool has {len(pool)}")
    return sorted(pool, key=lambda example: bucket(example.id, f"distill-{config.seed}"))[: config.problems]


def sample_teacher(questions: list[Example], config: DistillConfig, samples_path: str) -> dict[str, dict]:
    """Ask the teacher for samples_per_problem solutions per question, one batch at a time.

    Each finished batch is appended to samples_path, so an interrupted job resumes where it stopped.
    """
    done = {}
    if os.path.exists(samples_path):
        with open(samples_path, encoding="utf-8") as f:
            done = {row["question_id"]: row for row in map(json.loads, f)}
    todo = [question for question in questions if question.id not in done]
    if not todo:
        return done

    teacher = config.teacher
    renderer = renderers.get_renderer(teacher.renderer, get_tokenizer(teacher.model))
    client = tinker.ServiceClient().create_sampling_client(base_model=teacher.model).result()
    params = tinker.SamplingParams(temperature=teacher.temperature, max_tokens=teacher.max_tokens,
                                   stop=renderer.get_stop_sequences())  # fmt: skip
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)
    for start in range(0, len(todo), config.chunk_size):
        batch = todo[start : start + config.chunk_size]
        prompts = [renderer.build_generation_prompt(teacher_messages(question)) for question in batch]
        futures = [client.sample(p, num_samples=config.samples_per_problem, sampling_params=params) for p in prompts]
        with open(samples_path, "a", encoding="utf-8") as f:
            for question, prompt, future in zip(batch, prompts, futures):
                row = {"question_id": question.id, "prompt_tokens": prompt.length, "samples": []}
                for sequence in future.result().sequences:
                    message, termination = renderer.parse_response(sequence.tokens)
                    row["samples"].append({"text": renderers.get_text_content(message),
                                           "stopped": termination == "stop_sequence", "tokens": len(sequence.tokens)})  # fmt: skip
                done[question.id] = row
                f.write(json.dumps(row) + "\n")
        logger.info("teacher samples: %d of %d questions", len(done), len(questions))
    return done


def build_distilled_pool(config: DistillConfig) -> dict:
    questions = pick_questions(config)
    samples = sample_teacher(questions, config, os.path.join(config.out_dir, "distill", config.name, "samples.jsonl"))

    candidates, counts = [], Counter()
    for question in questions:
        kept, question_counts = select_solutions(question, samples[question.id]["samples"], config)
        candidates.extend(kept)
        counts.update(question_counts)
        counts["questions_with_a_kept_solution"] += bool(kept)

    # Teacher text is new text, so it is checked against the benchmarks again, like every pool.
    clean, removed = split_clean(candidates, build_benchmark_index())
    counts.update(f"removed_overlap_{benchmark}" for _, benchmark in removed)
    tokenizer = get_tokenizer(BASE_MODEL)
    pool = []
    for example in clean:
        if target_problem(example) is not None:
            counts["removed_bad_format"] += 1
        elif not within_token_limit(example, tokenizer, config.max_target_tokens):
            counts["removed_too_long"] += 1
        else:
            pool.append(example)
    write_jsonl(os.path.join(config.pools_dir, f"{config.name}.jsonl"), pool)

    rows = samples.values()
    manifest = {
        "config": config.model_dump(),
        "split": "train",
        "questions": len(questions),
        "counts": {**counts, "pool": len(pool)},
        "tokens": {"prompt_per_request": sum(r["prompt_tokens"] for r in rows),
                   "sampled": sum(s["tokens"] for r in rows for s in r["samples"])},  # fmt: skip
        "styles": dict(Counter(example.style for example in pool)),
    }
    manifest_path = os.path.join(config.out_dir, "manifests", f"{config.name}.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = build_distilled_pool(load_config(sys.argv[1], DistillConfig))
    print(json.dumps({key: result[key] for key in ("counts", "tokens")}, indent=2))
