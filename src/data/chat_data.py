"""Instruction-following and general chat examples from allenai/tulu-3-sft-mixture (train split).

The subsets are selected by the dataset's `source` field (names set in ChatConfig).
"""

from datasets import load_dataset

from src.config import ChatConfig
from src.schema import Example, Message


def clean_messages(messages: list[dict]) -> list[Message] | None:
    """Strip whitespace; drop conversations with a system message or an empty message.

    Measured on these subsets: 795 no_robots conversations start with a system message (the
    grader never sends one) and 1 conversation has an empty message.
    """
    if messages[0]["role"] == "system" or any(not m["content"].strip() for m in messages):
        return None
    return [Message(role=m["role"], content=m["content"].strip()) for m in messages]


def is_generation_request(prompt: str) -> bool:
    """True when the prompt is the wording used to generate this dataset, not a task for the model.

    Measured on the instruction-following subset: 71 prompts contain "Create a verifiable
    instruction that the following persona might ask you to do", and their responses disagree
    on what to do with it (some answer in JSON, some repeat it, some answer an unrelated task).
    """
    return "verifiable instruction" in prompt.lower()


def build_chat(config: ChatConfig) -> tuple[list[Example], dict]:
    """Return (examples, counts) for the instruction-following and general subsets."""
    wanted = {config.ifollow_source: ("ifollow", "tulu3_persona_if")}
    wanted.update({source: ("general", "tulu3_general") for source in config.general_sources})
    counts = {name: {"dataset": "allenai/tulu-3-sft-mixture", "split": "train", "loaded": 0, "removed_generation_request": 0, "formatted": 0}
              for _, name in wanted.values()}  # fmt: skip

    tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    examples = []
    for tulu_source, row_id, messages in zip(tulu["source"], tulu["id"], tulu["messages"]):
        if tulu_source not in wanted:
            continue
        skill, name = wanted[tulu_source]
        counts[name]["loaded"] += 1
        cleaned = clean_messages(messages)
        if cleaned is None:
            continue
        if is_generation_request(cleaned[0].content):
            counts[name]["removed_generation_request"] += 1
            continue
        counts[name]["formatted"] += 1
        examples.append(Example(id=f"tulu3_{row_id}", skill=skill, source=name, style="chat",
                                messages=cleaned, meta={"tulu_source": tulu_source}))  # fmt: skip
    return examples, counts
