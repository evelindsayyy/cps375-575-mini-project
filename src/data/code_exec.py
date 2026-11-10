"""Static safety checks and sandboxed execution for dataset code.

Dataset code is untrusted. Before running anything we require that it only defines functions and
classes and only imports from a small allowlist. It is then run in a separate Python process with
a timeout, in an empty temporary directory.
"""

import ast
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ALLOWED_MODULES = {
    "math", "cmath", "itertools", "collections", "functools", "re", "heapq", "bisect", "string",
    "typing", "datetime", "decimal", "fractions", "operator", "statistics", "copy", "json",
    "dataclasses", "enum", "abc", "numbers", "textwrap", "calendar", "random",
}  # fmt: skip
BANNED_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "breakpoint"}


def static_problem(code: str) -> str | None:
    """Return None if the code is safe to run and is 'definitions only', else a reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax error"
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) and not is_docstring:
            return "top-level statement"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [(node.module or "").split(".")[0]]
        else:
            modules = []
        if any(module not in ALLOWED_MODULES for module in modules):
            return "import outside allowlist"
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return "banned builtin"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute"
    return None


def top_level_functions(code: str) -> list[str]:
    return [node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)]


def passes_tests(code: str, tests: list[str], timeout: float = 10.0) -> bool:
    """Run `code` followed by the assert statements. True only if the process exits cleanly."""
    script = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as workdir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=workdir,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def passes_tests_many(jobs: list[tuple[str, list[str]]], workers: int = 8) -> list[bool]:
    """Run many (code, tests) jobs in parallel. Results keep the input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: passes_tests(job[0], job[1]), jobs))
