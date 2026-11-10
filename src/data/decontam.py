"""Decontamination: remove training examples that overlap with any evaluation benchmark.

A training text is contaminated if, after normalization, it
  1. is identical to a benchmark text, or
  2. shares any 13-word sequence with a benchmark text, or
  3. contains a whole benchmark text that is 8-12 words long (too short to have a 13-gram).
Benchmark texts shorter than 8 words are matched by rule 1 only.
"""

from datasets import load_dataset

from src.schema import Example, normalize

NGRAM = 13
MIN_SHORT = 8


class BenchmarkIndex:
    def __init__(self) -> None:
        self.exact: dict[str, str] = {}  # normalized full text -> benchmark name
        self.ngrams: dict[str, str] = {}  # 13-word window -> benchmark name
        self.short: dict[str, str] = {}  # whole short text (8-12 words) -> benchmark name
        self.counts: dict[str, int] = {}

    def add(self, name: str, text: str) -> None:
        norm = normalize(text)
        if not norm:
            return
        self.counts[name] = self.counts.get(name, 0) + 1
        self.exact[norm] = name
        words = norm.split()
        if len(words) >= NGRAM:
            for i in range(len(words) - NGRAM + 1):
                self.ngrams[" ".join(words[i : i + NGRAM])] = name
        elif len(words) >= MIN_SHORT:
            self.short[norm] = name

    def find(self, text: str) -> str | None:
        """Return the benchmark name this text overlaps with, or None."""
        norm = normalize(text)
        if norm in self.exact:
            return self.exact[norm]
        words = norm.split()
        for i in range(len(words) - NGRAM + 1):
            hit = self.ngrams.get(" ".join(words[i : i + NGRAM]))
            if hit:
                return hit
        if self.short:
            for size in range(MIN_SHORT, NGRAM):
                for i in range(len(words) - size + 1):
                    hit = self.short.get(" ".join(words[i : i + size]))
                    if hit:
                        return hit
        return None

    def find_in_example(self, example: Example) -> str | None:
        """Check every message of a training example."""
        for message in example.messages:
            hit = self.find(message.content)
            if hit:
                return hit
        return None


def build_benchmark_index() -> BenchmarkIndex:
    """Index every benchmark we are graded on or use as a held-out proxy.

    These datasets are loaded ONLY to detect overlap. Nothing here is ever written to training data.
    """
    index = BenchmarkIndex()

    for row in load_dataset("google/IFEval", split="train"):
        index.add("ifeval", row["prompt"])

    # Full MATH test split (5,000 problems); MATH-500 is a subset of it.
    for row in load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test"):
        index.add("math_test", row["problem"])

    for split in ("train", "test", "validation", "prompt"):
        for row in load_dataset("google-research-datasets/mbpp", "full", split=split):
            index.add("mbpp", row["text"])
            index.add("mbpp", row["code"])
        for row in load_dataset("google-research-datasets/mbpp", "sanitized", split=split):
            index.add("mbpp", row["prompt"])
            index.add("mbpp", row["code"])

    # Proxy held-out suite.
    for row in load_dataset("openai/gsm8k", "main", split="test"):
        index.add("gsm8k_test", row["question"])
    for row in load_dataset("openai/openai_humaneval", split="test"):
        index.add("humaneval", row["prompt"])
    for row in load_dataset("Maxwell-Jia/AIME_2024", split="train"):  # its only split; an evaluation set
        index.add("aime2024", row["Problem"])
    for row in load_dataset("math-ai/aime25", split="test"):
        index.add("aime2025", row["problem"])
    for row in load_dataset("allenai/IFBench_test", split="train"):  # its only split; an evaluation set
        index.add("ifbench", row["prompt"])
    return index


def split_clean(examples: list[Example], index: BenchmarkIndex) -> tuple[list[Example], list[tuple[Example, str]]]:
    """Return (clean, removed) where removed is a list of (example, benchmark_name)."""
    clean, removed = [], []
    for example in examples:
        hit = index.find_in_example(example)
        if hit:
            removed.append((example, hit))
        else:
            clean.append(example)
    return clean, removed
