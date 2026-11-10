"""OUTPUT FORMAT SPEC — what a training target must look like, and why.

Every format below is aligned with how the provided evaluation scripts extract answers
(read from the pinned inspect_evals / tinker_cookbook source).

MATH
  - The grader takes the answer from the LAST line, which must be `ANSWER: <answer>`
    (regex `ANSWER\\s*:\\s*([^\\n]+)\\s*\\Z`). Any text after that line means "no answer".
  - So a math target is: step-by-step solution, then one final line `ANSWER: <answer>`.
  - When a prompt asks for a different final format (\\boxed{}, "The answer is X", "#### X"),
    the target ends in that format instead. The target always obeys the prompt.

CODE
  - The grader runs the FIRST block that starts with a line ```python. Nothing is added to it:
    the code must contain its own imports.
  - So a code target is exactly one ```python block and no other text.

CHAT (instruction following, general)
  - No extraction; the response is checked as-is. Generation stops at 1024 tokens,
    so targets must be shorter than that.

All formats: the conversation is rendered with the `role_colon` renderer the grader uses for
this base model, which also teaches the model to stop after its answer.
"""

import ast
import re
import textwrap

from inspect_evals.math.math import USER_PROMPT_TEMPLATE  # the math grader's prompt template
from inspect_evals.math.utils import extract_answer  # the math grader's answer extraction
from inspect_evals.mbpp.mbpp import PROMPT_TEMPLATE  # the MBPP grader's instruction header

from src.schema import Example

# ---------------------------------------------------------------- math

MATH_PROMPTS = {
    "harness": USER_PROMPT_TEMPLATE.strip(),
    "boxed": "{prompt}\n\nSolve the problem step by step. Put your final answer in \\boxed{}.",
    "answer_is": (
        "{prompt}\n\nExplain your reasoning step by step, then end with a final line of the form "
        '"The answer is X".'
    ),
    "hashes": (
        "{prompt}\n\nWork through the problem step by step. On the last line, write #### followed "
        "by the final numeric answer."
    ),
}


def math_prompt(style: str, problem: str) -> str:
    """Insert the problem with str.replace, because LaTeX problems contain braces."""
    return MATH_PROMPTS[style].replace("{prompt}", problem.strip())


def math_target(style: str, solution: str, answer: str) -> str:
    """Solution text followed by the final line in the format `style` asks for."""
    final_line = {
        "harness": f"ANSWER: {answer}",
        "answer_is": f"The answer is {answer}",
        "hashes": f"#### {answer}",
    }[style]
    return f"{solution.strip()}\n{final_line}"


# ---------------------------------------------------------------- code

CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)  # the MBPP grader's pattern

MBPP_TASK_SECTION = textwrap.dedent(
    """
    # Now, do it for the following task.

    ## Prompt:
    ```python
    {prompt}
    ```

    ## Test Case:
    ```python
    {test_list_str}
    ```

    ## Completion:
    """
)


def code_target(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def extract_code(text: str) -> str | None:
    """The code the MBPP grader would run: the first ```python block."""
    blocks = CODE_BLOCK.findall(text)
    return blocks[0] if blocks else None


def mbpp_layout_prompt(problem: str, tests: list[str], worked_examples: list[dict]) -> str:
    """Same layout as the MBPP grader prompt.

    `worked_examples` are dicts with problem / tests / code taken from OUR training data.
    The grader's own worked examples are MBPP items and must never appear in training.
    """
    prompt = PROMPT_TEMPLATE + "\n\nFor example:\n\n"
    for i, shot in enumerate(worked_examples, start=1):
        shot_tests = "\n".join(shot["tests"])
        prompt += (
            f"## Prompt {i}\n```python\n{shot['problem']}\n```\n\n"
            f"## Test Case {i}\n```python\n{shot_tests}\n```\n\n"
            f"## Completion {i}\n```python\n{shot['code']}\n```\n\n"
        )
    task = MBPP_TASK_SECTION.replace("{prompt}", problem)
    return prompt + task.replace("{test_list_str}", "\n".join(tests))


# ---------------------------------------------------------------- validation


def target_problem(example: Example) -> str | None:
    """Check a target against the spec. Returns None if valid, else a short reason."""
    if example.skill == "code":
        blocks = CODE_BLOCK.findall(example.target)
        if len(blocks) != 1 or example.target.strip() != f"```python\n{blocks[0]}```":
            return "not exactly one python block"
        try:
            ast.parse(blocks[0])
        except SyntaxError:
            return "syntax error"
    if example.skill == "math" and example.style == "harness":
        if extract_answer(example.target) != example.meta["answer"]:
            return "grader would not extract the answer"
    return None
