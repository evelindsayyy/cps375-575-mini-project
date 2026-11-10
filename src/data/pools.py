"""Step 1 of the pipeline: build the verified example pools that experiments sample from.

    python -m src.data.pools configs/pools.yaml

For every source: load the train split -> format -> remove benchmark overlap -> check the target
format -> apply the token limit -> drop duplicate problems. Then hold out a dev set per skill.

Outputs (under PoolsConfig.out_dir):
    pools/<source>.jsonl      one file per data source
    dev/<skill>.jsonl         held-out examples, never used for training
    manifests/pools.json      the config used + counts at every step (the report's data table)
    manifests/removed.jsonl   every example removed for benchmark overlap, and which benchmark
"""

import json
import logging
import os
import sys
from collections import Counter

from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.config import BASE_MODEL, PoolsConfig, load_config
from src.data.chat_data import build_chat
from src.data.code_data import build_code
from src.data.decontam import BenchmarkIndex, build_benchmark_index, split_clean
from src.data.math_data import build_math
from src.formats import target_problem
from src.schema import Example, bucket, write_jsonl

logger = logging.getLogger(__name__)

# The dev set of each skill is held out from that skill's main source.
DEV_SOURCES = {"math_train", "opencodeinstruct_train", "tulu3_persona_if"}


def within_token_limit(example: Example, tokenizer, limit: int) -> bool:
    assistant_texts = [m.content for m in example.messages if m.role == "assistant"]
    return all(len(tokenizer.encode(text, add_special_tokens=False)) <= limit for text in assistant_texts)


def filter_examples(
    examples: list[Example], index: BenchmarkIndex, tokenizer, config: PoolsConfig, removed_log: list[dict]
) -> tuple[list[Example], dict]:
    """Apply the four filters in order and count what each one removes, per source."""
    counts: dict[str, Counter] = {}
    clean, removed = split_clean(examples, index)
    for example, benchmark in removed:
        counts.setdefault(example.source, Counter())[f"removed_overlap_{benchmark}"] += 1
        removed_log.append({"id": example.id, "source": example.source, "benchmark": benchmark,
                            "prompt_start": example.messages[0].content[:300]})  # fmt: skip
    kept, seen = [], set()
    for example in clean:
        source_counts = counts.setdefault(example.source, Counter())
        problem = example.meta.get("problem") or example.messages[0].content
        if target_problem(example) is not None:
            source_counts["removed_bad_format"] += 1
        elif not within_token_limit(example, tokenizer, config.max_target_tokens[example.skill]):
            source_counts["removed_too_long"] += 1
        elif problem in seen:
            source_counts["removed_duplicate"] += 1
        else:
            seen.add(problem)
            source_counts["kept"] += 1
            kept.append(example)
    return kept, {source: dict(c) for source, c in counts.items()}


def split_dev(examples: list[Example], dev_size: int) -> tuple[list[Example], list[Example]]:
    """Hold out the `dev_size` examples with the lowest hash values."""
    ranked = sorted(examples, key=lambda example: bucket(example.id, "dev"))
    return ranked[dev_size:], ranked[:dev_size]


def build_pools(config: PoolsConfig) -> dict:
    index = build_benchmark_index()
    tokenizer = get_tokenizer(BASE_MODEL)
    manifest = {"config": config.model_dump(), "benchmark_texts_indexed": index.counts, "sources": {}}
    removed_log: list[dict] = []

    builders = [
        lambda: build_math(config.math),
        lambda: build_code(config.code, config.seed, index),
        lambda: build_chat(config.chat),
    ]
    for build in builders:
        examples, load_counts = build()
        kept, filter_counts = filter_examples(examples, index, tokenizer, config, removed_log)
        for source, loaded in load_counts.items():
            pool = [example for example in kept if example.source == source]
            dev: list[Example] = []
            if source in DEV_SOURCES:
                pool, dev = split_dev(pool, config.dev_size)
                write_jsonl(os.path.join(config.out_dir, "dev", f"{pool[0].skill}.jsonl"), dev)
            write_jsonl(os.path.join(config.out_dir, "pools", f"{source}.jsonl"), pool)
            manifest["sources"][source] = {
                **loaded, **filter_counts.get(source, {}),
                "pool": len(pool), "dev": len(dev), "styles": dict(Counter(e.style for e in pool)),
            }  # fmt: skip
            logger.info("%s: %s", source, manifest["sources"][source])

    manifest_dir = os.path.join(config.out_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "pools.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(manifest_dir, "removed.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in removed_log)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_pools(load_config(sys.argv[1], PoolsConfig))
