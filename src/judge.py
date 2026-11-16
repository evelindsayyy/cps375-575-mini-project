"""Second-model review: does a response follow every instruction in its prompt?

Used where there is no gold answer to compare with: scoring the instruction-following dev set,
and (later) reviewing teacher answers to instruction-following prompts. The judge lists each
explicit instruction, marks it followed or not, and ends with one verdict line. A response
passes only if every instruction is followed, like IFEval's strict prompt-level score.
"""

import re

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.config import JudgeConfig

JUDGE_SPEC = (
    "You check whether a response follows the instructions in a request. Judge only the explicit "
    "instructions (format, length, number of sentences, paragraphs, sections or bullet points, words "
    "or letters to include or avoid and how often, case, punctuation, language, how to start or end, "
    "repeating the request, and similar). Do not judge quality, style, or accuracy. Count carefully "
    "when an instruction sets a number."
)
JUDGE_TASK = """REQUEST:
<<<
{prompt}
>>>

RESPONSE:
<<<
{response}
>>>

MEASURED BY A PROGRAM (use this number instead of counting words yourself):
{measured}

List every explicit instruction in the request, one per line, as
- <instruction>: FOLLOWED
or
- <instruction>: NOT FOLLOWED (<short reason>)
Then end with exactly one final line: VERDICT: PASS if every instruction is followed, otherwise VERDICT: FAIL."""

VERDICT = re.compile(r"VERDICT:\s*(PASS|FAIL)\s*\Z")


def measured_counts(response: str) -> str:
    """The response's word count. Words are runs of word characters, as IFEval counts them.

    Given to the judge because it miscounted: on the s1_a dev run it got 5 of about 10 total
    word-count limits wrong (e.g. 99 words called "about 108"; 62 words passed a 50-word limit);
    with the count given, 13 of 14 agreed. A blank-line paragraph count was tried too and removed:
    the judge then failed answers whose headers it counted as paragraphs (37 of 43 new failures).
    """
    words = len(re.findall(r"\w+", response))
    return f"- words in the response: {words}"


def judge_messages(prompt: str, response: str) -> list[dict]:
    # str.format parses only the template, so braces inside the prompt or response are safe.
    task = JUDGE_TASK.format(prompt=prompt, response=response, measured=measured_counts(response))
    return [{"role": "system", "content": JUDGE_SPEC}, {"role": "user", "content": task}]


def parse_verdict(text: str) -> str | None:
    """'pass' or 'fail' from the final line, or None if the judge did not end with a verdict."""
    match = VERDICT.search(text.strip())
    return match.group(1).lower() if match else None


def judge_responses(config: JudgeConfig, prompts: list[str], responses: list[str]) -> list[tuple[str | None, str]]:
    """(verdict, full judge text) for each prompt/response pair, in input order. Greedy decoding."""
    renderer = renderers.get_renderer(config.renderer, get_tokenizer(config.model))
    client = tinker.ServiceClient().create_sampling_client(base_model=config.model).result()
    params = tinker.SamplingParams(temperature=0.0, max_tokens=config.max_tokens, stop=renderer.get_stop_sequences())
    futures = [
        client.sample(renderer.build_generation_prompt(judge_messages(prompt, response)), num_samples=1, sampling_params=params)
        for prompt, response in zip(prompts, responses)
    ]  # fmt: skip
    texts = [renderers.get_text_content(renderer.parse_response(f.result().sequences[0].tokens)[0]) for f in futures]
    return [(parse_verdict(text), text) for text in texts]
