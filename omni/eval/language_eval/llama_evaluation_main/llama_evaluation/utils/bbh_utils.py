#!/usr/bin/env python3
import re

from .task_utils import TASK_MAPPING

__all__ = ["get_max_gen", "get_post_process", "get_metrics"]


BBH_PATTERNS = [
    "True|False|true|false",
    "Yes|No|yes|no",
    "[A-F]",
    "[ABC]",
    "[\<\>\[\]\(\)\{\}]",
    "invalid|valid|Invalid|Valid",
    "[A-K]",
    "[AB]",
    "[A-E]",
    "[A-G]",
    "[ABC]",
    "[A-E]",
    "Yes|No|yes|no",
    "[A-E]",
    "[A-R]",
    "[A-D]",
    "[A-F]",
    "[AB]",
    "Yes|No|yes|no",
    "[A-D]",
    "[A-E]",
    "[A-G]",
    "[ABC]",
    "Yes|No|yes|no",
]


BBH_PATTERNS = [r'(?<![a-zA-Z0-9_])(%s)(?![a-zA-Z0-9_])'%i for i in BBH_PATTERNS]
BBH_PATTERNS.insert(12, "-?\d+\.?\d*")
BBH_PATTERNS.insert(14, "-?\d+\.?\d*")
BBH_PATTERNS.append(None)
BBH_PATTERNS_DICT = dict(zip(TASK_MAPPING["bbh"], BBH_PATTERNS))


def get_max_gen(dataset):
    return 512


def get_post_process(result, dataset=None):
    result = result.split("\n\n")[0]
    pattern = BBH_PATTERNS_DICT[dataset]
    if pattern is None:
        return result
    result = re.findall(pattern, result)
    if len(result) == 0:
        return ""
    elif dataset in ["boolean_expressions", "multistep_arithmetic_two"]:
        return result[-1].capitalize()
    elif dataset == "dyck_languages":
        return  "".join(result)
    else:
        return result[0].capitalize()


def get_metrics(result, label, dataset=None):
    if dataset == "word_sorting":
        return label.replace(" ", "").lower() in result.replace(" ", "").lower()
    else:
        return (get_post_process(result, dataset) == get_post_process(label, dataset))
