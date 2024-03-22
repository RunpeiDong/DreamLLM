#!/usr/bin/env python3
import re
import ast
from typing import List

from .eval_utils import sft_answer

__all__ = [
    "minimum_code", "extract_markdown", "sft_extract", "extract_function",
]


def minimum_code(code: str) -> str:
    code_lines = code.split("\ndef ")
    for i in range(2, len(code_lines) + 1):
        new_code = "def ".join(code_lines[:i])
        try:
            ast.parse(new_code)
            return new_code
        except Exception:
            pass
    return code  # fallback: return original code


def extract_markdown(text: str, python_only: bool = True) -> List[str]:
    """
    extract text quoted by ```python ... ``` or ``` ... ```

    Args:
        text (str): text to extract
        python_only (bool): whether to extract python code only. Default: True.
    """
    # Regular expression to match Python code blocks in Markdown
    pattern = r"```python\n([\s\S]*?)\n```" if python_only else r"```(?:python\n)?([\s\S]*?)```"

    # Find all matches of the pattern in the given text
    matches = re.findall(pattern, text)

    # Return the list of Python code blocks
    return [m.strip() for m in matches]


def sft_extract(text) -> str:
    text = sft_answer(text)
    code = extract_markdown(text)
    if not code:
        code = extract_markdown(text, python_only=False)
    return "\n\n".join(code)


def extract_function(code: str, markdown: bool = False) -> str:
    """Extracts the function definition from a code snippet.

    Args:
        code (str): code snippet.
        markdown (bool): whether the code is in markdown format or not. Default: False.
    """
    # post processing for code-gen task
    if markdown:
        return sft_extract(code)
    try:
        ast.parse(code)
    except Exception:
        code = minimum_code(code)
    return code
