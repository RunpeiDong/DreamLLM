#!/usr/bin/env python3

import os
import json

from llama_evaluation_main.llama_evaluation.utils.task_utils import GAOKAO_CATS
from llama_evaluation_main.llama_evaluation.prompts.prompt import Prompt
from llama_evaluation_main.llama_evaluation.prompts.template import BASE_FEWSHOT_TEMPLATE, fewshot_prompt


__all__ = ["load_gaokao_prompt"]


def load_prompt(sub_task):
    filename = os.path.join(os.path.dirname(__file__), "GAOKAO_MCQ_prompt.json")
    all_prompts = json.load(open(filename, 'r'))
    return all_prompts[sub_task] + "{}"


def sample2prompt(sub_task, template):
    return ""


def load_gaokao_prompt(sub_task: str, template: str, few_shot_template: str = BASE_FEWSHOT_TEMPLATE) -> Prompt:
    prefix = sample2prompt(sub_task, few_shot_template)
    return Prompt(prefix, template.format(load_prompt(sub_task)))
