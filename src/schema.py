"""The one data structure every module exchanges: a training example.

Rule 1 of the project (train only on training splits) is built into the schema: an Example
cannot be created with any split other than "train".
"""

import hashlib
import json
import os
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

Skill = Literal["math", "code", "ifollow", "general"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    skill: Skill
    source: str  # which dataset the example came from, e.g. "math_train"
    split: Literal["train"] = "train"
    style: str  # which prompt format was used, e.g. "harness"
    messages: list[Message]
    meta: dict = {}

    @property
    def target(self) -> str:
        """The final assistant message: the text the model is trained to produce."""
        return self.messages[-1].content


def bucket(key: str, salt: str = "") -> float:
    """Deterministic number in [0, 1) derived from a string. Used for reproducible choices."""
    digest = hashlib.sha1(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def normalize(text: str) -> str:
    """Lowercase, replace every non-alphanumeric character with a space, collapse spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def write_jsonl(path: str, examples: list[Example]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(example.model_dump_json() + "\n")


def read_jsonl(path: str) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [Example.model_validate(json.loads(line)) for line in f]
