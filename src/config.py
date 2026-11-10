"""Every module's input, defined as a Pydantic model.

An experiment is one ExperimentConfig: which examples to sample from which source, how to train,
how to evaluate. Changing one field gives a new experiment; the config is saved with its results.
Unknown fields are rejected, so a typo in a config file fails immediately.
"""

from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

BASE_MODEL = "Qwen/Qwen3.5-9B-Base"  # the only base model allowed for the graded checkpoint


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


ConfigT = TypeVar("ConfigT", bound=Config)


# ---------------------------------------------------------------- data pools


class MathConfig(Config):
    """MATH train split + GSM8K train split, using each dataset's own solutions."""

    harness_share: float = Field(0.75, ge=0, le=1)  # share of prompts using the grader's wording


class CodeConfig(Config):
    """nvidia/OpenCodeInstruct train split, kept only if the solution passes its tests."""

    shards: list[int] = [0, 17, 34]  # which of the 50 parquet files to read (~100k rows each)
    max_candidates: int = 20000  # stop scanning once this many pass the static filters
    tests_share: float = Field(0.50, ge=0, le=1)  # prompts laid out like the MBPP grader prompt
    complete_share: float = Field(0.15, ge=0, le=1)  # "complete this function" prompts
    tests_in_prompt: int = 3  # MBPP prompts show three asserts
    fewshot_max_chars: int = 1000  # worked examples come from the shortest ~12% of kept examples


class ChatConfig(Config):
    """Subsets of allenai/tulu-3-sft-mixture (train split), selected by its `source` field."""

    ifollow_source: str = "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980"
    general_sources: list[str] = ["ai2-adapt-dev/no_robots_converted", "ai2-adapt-dev/oasst1_converted"]


class PoolsConfig(Config):
    """Builds the verified example pools that experiments sample from."""

    out_dir: str = "data"
    seed: int = 0
    dev_size: int = 300  # examples per skill held out for our own evaluation
    # Limits on target length in tokens. The grader stops generation at 1024 tokens (IFEval)
    # and 4096 tokens (MATH, MBPP).
    max_target_tokens: dict[str, int] = {"math": 1500, "code": 1500, "ifollow": 900, "general": 900}
    math: MathConfig = MathConfig()
    code: CodeConfig = CodeConfig()
    chat: ChatConfig = ChatConfig()


# ---------------------------------------------------------------- one experiment


class MixConfig(Config):
    """How many examples to sample from each source. This is the data mixture."""

    pools_dir: str = "data/pools"
    counts: dict[str, int]  # source name -> number of examples, e.g. {"math_train": 6000}
    seed: int = 0


class TrainConfig(Config):
    base_model: Literal["Qwen/Qwen3.5-9B-Base"] = BASE_MODEL
    lora_rank: int = 64
    learning_rate: float | None = None  # None = the Tinker Cookbook's recommended value for the model
    epochs: int = 2
    batch_size: int = 64
    max_length: int = 4096  # tokens per example; longer examples are truncated
    seed: int = 0


class EvalConfig(Config):
    """Passed to the provided evaluation script. Defaults are the official settings."""

    limit: int | None = None  # None = full evaluation; a number = quick check only
    temperature: float = 0.0
    top_p: float = 1.0


class ExperimentConfig(Config):
    name: str  # also the folder name under runs/
    notes: str = ""  # what this experiment is testing, in one sentence
    mix: MixConfig
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()


def load_config(path: str, model: type[ConfigT]) -> ConfigT:
    with open(path, encoding="utf-8") as f:
        return model.model_validate(yaml.safe_load(f))
