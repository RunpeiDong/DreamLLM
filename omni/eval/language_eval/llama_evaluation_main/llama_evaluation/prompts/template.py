#!/usr/bin/env python3

from typing import List, Tuple

__all__ = [
    "BASE_TEMPLATE",
    "BASE_FEWSHOT_TEMPLATE",
    "SFT_TEMPLATE",
    "SFT_FEWSHOT_TEMPLATE",
    "fewshot_prompt",
]


BASE_TEMPLATE = "Problem:\n{}\n\nSolution:\n"
BASE_FEWSHOT_TEMPLATE = "Problem:\n{}\n\nSolution:\n{}\n\n"
SFT_TEMPLATE = "Human: {}\n\nAssistant:"
# SFT_TEMPLATE = "USER: {}\n\nASSISTANT:"
SFT_FEWSHOT_TEMPLATE = "Human: {}\n\nAssistant: {}</t>\n\n"
# SFT_FEWSHOT_TEMPLATE = "USER: {}\n\nASSISTANT: {}</t>\n\n"


def fewshot_prompt(template, few_shot_examples: List[Tuple[str, str]]) -> str:
    return "".join(
        [template.format(question, answer) for question, answer in few_shot_examples]
    )
