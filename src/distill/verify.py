"""Is a teacher's final answer equal to the gold answer? Decided the same way the grader decides.

Mirrors the three checks in evaluation/math_scorer.py (which we must not import from or modify):
an answer counts as correct if math_verify, the string-normalising check, or the sympy check
says the two are equivalent.
"""

import asyncio

from inspect_evals.math.utils import is_equiv, is_equiv_sympy, normalize_final_answer
from math_verify import parse, verify


def math_verify_match(answer: str, gold: str) -> bool:
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${answer}$")))
    except Exception:  # the grader treats any math_verify failure as "not equal"
        return False


async def grader_equivalent(answer: str, gold: str) -> bool:
    if await is_equiv(answer, gold):
        return True
    return await is_equiv_sympy(await normalize_final_answer(answer), await normalize_final_answer(gold))


def answers_match(answer: str, gold: str) -> bool:
    return math_verify_match(answer, gold) or asyncio.run(grader_equivalent(answer, gold))
