"""Step 2 of the pipeline: sample a training set from the pools according to a MixConfig.

The sample is deterministic (same config -> same examples). Before the set is returned, every
example is checked once more against all benchmarks: this is the final check on exactly the
data that will be trained on, and its result is stored with the experiment.
"""

import os
import random

from src.config import MixConfig
from src.data.decontam import BenchmarkIndex
from src.schema import Example, bucket, read_jsonl


def build_training_set(config: MixConfig, index: BenchmarkIndex) -> tuple[list[Example], dict]:
    """Return (shuffled examples, record of what was sampled and checked)."""
    examples: list[Example] = []
    record: dict = {"sources": {}}
    for source, count in config.counts.items():
        pool = read_jsonl(os.path.join(config.pools_dir, f"{source}.jsonl"))
        if count > len(pool):
            raise ValueError(f"{source}: asked for {count} examples but the pool has {len(pool)}")
        ranked = sorted(pool, key=lambda example: bucket(example.id, f"mix-{config.seed}"))
        examples.extend(ranked[:count])
        record["sources"][source] = {"split": "train", "sampled": count, "pool_size": len(pool)}

    for example in examples:
        benchmark = index.find_in_example(example)
        if benchmark:
            raise RuntimeError(f"Training example {example.id} overlaps benchmark {benchmark}")
    record["overlap_check"] = {"examples_checked": len(examples), "overlaps_found": 0,
                               "benchmark_texts_indexed": index.counts}  # fmt: skip

    random.Random(config.seed).shuffle(examples)
    record["total"] = len(examples)
    return examples, record
