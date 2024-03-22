#!/usr/bin/env python3

import os
import ast
import pandas as pd

from llama_evaluation_main.llama_evaluation.prompts.prompt import Prompt
from llama_evaluation_main.llama_evaluation.prompts.template import BASE_FEWSHOT_TEMPLATE, fewshot_prompt

__all__ = [
    "agi_n_shot",
    "csv2prompt",
    "load_agi_prompt",
    "agi_fewshot_template",
]

# define the datasets
english_qa_datasets = ["lsat_ar", "lsat_lr", "lsat_rc", "logiqa_en", "sat_math", "sat_en", "aqua_rat",
                       "sat_en_without_passage", "gaokao_english"]
chinese_qa_datasets = ["logiqa_zh", "jec_qa_kd", "jec_qa_ca", 
                       "gaokao_chinese", "gaokao_geography", "gaokao_history",
                       "gaokao_biology", "gaokao_chemistry", "gaokao_physics", "gaokao_mathqa"]
english_cloze_datasets = ["math"]
chinese_cloze_datasets = ["gaokao_mathcloze"]

agi_n_shot = {"gaokao_chinese": 3,
"gaokao_geography": 5,
"gaokao_history": 5,
"gaokao_biology": 5,
"gaokao_chemistry": 5,
"gaokao_physics": 5,
"gaokao_mathqa": 5,
"gaokao_english": 3,
"sat_math": 5,
"sat_en": 3,
"aqua_rat": 5,
"lsat_ar": 3,
"lsat_lr": 3,
"lsat_rc": 3,
"logiqa_en": 3,
"logiqa_zh": 3,
"gaokao_mathcloze": 5,
"math": 4,
"sat_en": 3,
"sat_en_without_passage": 3}


def agi_fewshot_template(sub_task):
    if sub_task in english_qa_datasets:
        prompt = 'Problem {}.   {} {}\nChoose from the following options:   {}'
    if sub_task in chinese_qa_datasets:
        prompt = '问题 {}.   {} {}\n从以下选项中选择:   {}'
    if sub_task in english_cloze_datasets:
        prompt = 'Problem {}.   {}{}{}'
    if sub_task in chinese_cloze_datasets:
        prompt = '问题 {}.   {}{}{}'
    return prompt


def agi_explanation_template(sub_task, cot=False):
    if sub_task in english_qa_datasets + english_cloze_datasets:
        prompt = ('Explanation for Problem {}:   {}\n' if cot else '{}{}') + 'The answer is therefore {}'
    if sub_task in chinese_qa_datasets + chinese_cloze_datasets:
        prompt = ('问题 {}的解析:   {}\n' if cot else '{}{}') + '答案是 {}'
    return prompt


# process few_shot raw_prompts
def combine_prompt(prompt_path, sub_task, cot=False):
    skip_passage = False
    if sub_task == 'sat_en_without_passage':
        skip_passage = True
        sub_task = "sat_en"
    demostrations = []
    # read the prompts by context and explanation
    context_row = [0, 1, 3, 5, 7, 9]
    explanation_row = [0, 2, 4, 6, 8, 10]
    filename = os.path.join(os.path.dirname(__file__), "agi_few_shot_prompts.csv")
    raw_prompts_context = pd.read_csv(filename, header=0, 
                                      skiprows=lambda x: x not in context_row,
                                      keep_default_na=False)
    raw_prompts_explanation = pd.read_csv(filename, header=0,
                                          skiprows=lambda x: x not in explanation_row,
                                          keep_default_na=False).replace(r'\n\n', '\n', regex=True)
    contexts = []
    for line in list(raw_prompts_context[sub_task.replace("_", "-")]):
        if line:
            contexts.append(ast.literal_eval(line))
    explanations = [exp for exp in raw_prompts_explanation[sub_task.replace("_", "-")] if exp]

    question_prompt = agi_fewshot_template(sub_task)
    answer_prompt = agi_explanation_template(sub_task, cot=cot)

    for idx, (con, exp) in enumerate(zip(contexts, explanations)):
        passage = con["passage"].strip() if con["passage"] is not None and not skip_passage else ""
        question = con["question"].strip()
        options = " ".join(con["options"]) if con["options"] is not None else ""
        label = con["label"] if con["label"] is not None else con["answer"].strip()
        if isinstance(label, list):
            label = ",".join(label)
        idx = (idx + 1) if cot else ""
        exp = exp if cot else ""
        demostrations.append((question_prompt.format(idx, passage, question, options),
                              answer_prompt.format(idx, exp, label)))

    return demostrations


def csv2prompt(sub_task: str, template = '{}\n{}\n\n') -> str:
    content = combine_prompt(os.path.dirname(__file__), sub_task)
    return fewshot_prompt(template, content)


def load_agi_prompt(sub_task: str, template: str, few_shot_template: str = BASE_FEWSHOT_TEMPLATE) -> Prompt:
    prompts = csv2prompt(sub_task, few_shot_template)
    return Prompt(prefix=prompts, template=template)
