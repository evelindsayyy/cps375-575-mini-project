"""Step 3 of the pipeline: supervised fine-tuning (LoRA) on Tinker.

The training loop follows the provided starter script (evaluation/train_and_publish.py):
conversation_to_datum -> forward_backward -> optim_step -> save_weights_for_sampler -> publish.
Added here: real data, shuffled epochs, a linearly decaying learning rate, and a loss log.
"""

import json
import math
import os
import random

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.config import TrainConfig
from src.schema import Example


def to_datums(examples: list[Example], config: TrainConfig) -> list[types.Datum]:
    """Render each conversation the way the grader will, with loss on assistant text only."""
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(config.base_model), tokenizer)
    return [
        conversation_to_datum(
            [message.model_dump() for message in example.messages],
            renderer,
            max_length=config.max_length,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        for example in examples
    ]


def batch_loss(result, batch: list[types.Datum]) -> float:
    """Mean negative log-likelihood per trained token (same formula as the starter script)."""
    logprobs = np.concatenate([output["logprobs"].tolist() for output in result.loss_fn_outputs])
    weights = np.concatenate([datum.loss_fn_inputs["weights"].tolist() for datum in batch])
    return float(-np.dot(logprobs, weights) / max(weights.sum(), 1))


def train(config: TrainConfig, examples: list[Example], name: str, run_dir: str) -> dict:
    """Train, save, and publish one checkpoint. Returns a record of the run."""
    datums = to_datums(examples, config)
    start_lr = config.learning_rate or get_lr(config.base_model)
    steps_per_epoch = math.ceil(len(datums) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    client = tinker.ServiceClient().create_lora_training_client(
        base_model=config.base_model, rank=config.lora_rank
    ).result()

    rng = random.Random(config.seed)
    step, loss = 0, float("nan")
    with open(os.path.join(run_dir, "losses.jsonl"), "w", encoding="utf-8") as loss_log:
        for epoch in range(config.epochs):
            order = list(range(len(datums)))
            rng.shuffle(order)
            for start in range(0, len(order), config.batch_size):
                batch = [datums[i] for i in order[start : start + config.batch_size]]
                learning_rate = start_lr * (1 - step / total_steps)  # linear decay to zero
                forward = client.forward_backward(batch, loss_fn="cross_entropy")
                update = client.optim_step(
                    types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
                )
                loss = batch_loss(forward.result(), batch)
                update.result()
                step += 1
                loss_log.write(json.dumps({"step": step, "epoch": epoch, "lr": learning_rate, "loss": loss}) + "\n")

    checkpoint_path = client.save_weights_for_sampler(name=name).result().path
    # Published explicitly so staff can load it; unpublished checkpoints cannot be graded.
    tinker.RestClient().publish_checkpoint_from_tinker_path(checkpoint_path).result()
    return {"checkpoint_path": checkpoint_path, "published": True, "examples": len(datums),
            "steps": total_steps, "start_learning_rate": start_lr, "final_loss": loss}  # fmt: skip
