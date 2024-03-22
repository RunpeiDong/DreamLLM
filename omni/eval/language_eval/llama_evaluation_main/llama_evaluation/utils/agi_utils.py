import os
import re
import json
import pandas as pd

from .math_utils import is_equiv, is_latex_equal, get_answer_str
from .task_utils import AGI_DATAPATH


__all__ = ["load_dataset", "get_post_process", "get_max_gen", "get_metrics"]


def load_dataset(subset: str) -> list:
    filename = os.path.join(AGI_DATAPATH, subset.replace("_", "-")+".jsonl")
    with open(filename, 'r') as f:
        content = [json.loads(i) for i in f.readlines()]
    return content


def get_max_gen(subset: str, cot: str = False) -> int:
    if cot or subset in ["math", "gaokao_mathcloze"]:
        return 512
    else:
        return 10
    

def get_post_process(result: str, subset: str = None) -> str:
    prefix_list = ["The answer is therefore", "The answer is", "答案是"]
    result = result.split("\n\n")[0]
    for prefix in prefix_list:
        if prefix in result:
            result = result.split(prefix)[1]

    if subset in ["math", "gaokao_mathcloze"]:
        return result.strip()
    pattern = r'(?<![a-zA-Z0-9_])([A-G])(?![a-zA-Z0-9_])'
    result = re.findall(pattern, result)
    if len(result) == 0:
        return ""
    elif subset == "gaokao_physics":
        return ",".join(result)
    else:
        return result[0]


def get_metrics(result: str, label: str, subset: str = None) -> bool:
    result = get_post_process(result, subset)
    if subset in ["math", "gaokao_mathcloze"]:
        return (is_equiv(result, label) or is_latex_equal(result, label))
    else:
        return (result == label)
