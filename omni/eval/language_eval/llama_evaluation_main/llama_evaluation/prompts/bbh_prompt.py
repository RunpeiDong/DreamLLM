#!/usr/bin/env python3

import os

from .prompt import Prompt
from .template import BASE_FEWSHOT_TEMPLATE, fewshot_prompt

__all__ = [
    "file2prompt",
    "load_bbh_prompt",
]


def file2prompt(sub_task: str, template = BASE_FEWSHOT_TEMPLATE) -> str:
    filename = os.path.join(os.path.dirname(__file__), "bbh_prompts_text", f"{sub_task}.txt")
    with open(filename, "r") as f:
        content = f.read()

    instruction, *samples = content.split("\n\nQ: ")
    q_and_a = [i.split("\nA: ") for i in samples]
    return instruction + "\n\n" + fewshot_prompt(template, q_and_a)


def load_bbh_prompt(sub_task: str, template: str, few_shot_template: str = BASE_FEWSHOT_TEMPLATE) -> Prompt:
    prompts = file2prompt(sub_task, few_shot_template)
    return Prompt(prefix=prompts, template=template)
