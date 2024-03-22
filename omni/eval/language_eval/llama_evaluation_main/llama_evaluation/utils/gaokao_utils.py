import re
import os
import json
import numpy as np

from .task_utils import GAOKAO_DATAPATH, GAOKAO_CATS


def load_dataset(subset):
    return json.load(open(os.path.join(GAOKAO_DATAPATH, GAOKAO_CATS[subset][0]+'.json')))['example']


def get_post_process(result: str, answer_length: str = None, subset: str = None) -> str:
    question_type = GAOKAO_CATS[subset][1]
    """
    Extract choice answer from model output

    Format of result that is expected:
    'single_choice': choice answer should be the last Capital Letter of the result, e.g.: "...【答案】 A <eoa>"
    'multi_question_choice': "...【答案】A ... 【答案】C ..." or write the choice answers at the beginning of the result, e.g. "A C D E F...."
    'multi_choice': "...【答案】 ABD " or write the choice answers at the end of the result, e.g. "... ACD"
    'five_out_of_seven': choice answers should be the first five Capital Letters of the result, e.g. "A C D F B ...."
    """
    if question_type == 'single_choice':
        model_answer = []
        temp = re.findall(r'[A-D]', result[::-1])
        if len(temp) != 0:
            model_answer.append(temp[0])

    elif question_type == 'multi_question_choice':
        model_answer = []
        temp = re.findall(r"【答案】\s*[:：]*\s*[A-Z]", result)
            
        if len(temp) == answer_length:
            for t in temp:
                model_answer.append(re.findall(r'[A-Z]', t)[0])
        else:
            temp = re.findall(r"[A-Z]", result)
            if len(temp) > 0:
                for k in range(min(len(temp), answer_length)):
                    model_answer.append(temp[k])

    elif question_type == 'multi_choice':
        model_answer = []
        answer = ''
        content = re.sub(r'\s+', '', result)
        answer_index = content.find('【答案】')
        if answer_index > 0:
            temp = content[answer_index:]
            if len(re.findall(r'[A-D]', temp)) > 0:
                for t in re.findall(r'[A-D]', temp):
                    answer += t
        else:
            temp = content[-10:]
            if len(re.findall(r'[A-D]', temp)) > 0:
                for t in re.findall(r'[A-D]', temp):
                    answer += t
        if len(answer) != 0:
            model_answer.append(answer)
    
    elif question_type == 'five_out_of_seven':
        model_answer = []
        temp = re.findall(r'[A-G]', result)
        if len(temp) > 0:
            for k in range(min(5, len(temp))):
                model_answer.append(temp[k])

    return model_answer


def get_metrics(result: str, label: str, subset: str = None) -> bool:
    scores = [i == j for i, j in zip(result, label)]
    return np.mean(scores)
