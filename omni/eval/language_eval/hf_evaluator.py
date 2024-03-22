import os
import re
import torch
import string
import numpy as np

import sqlite3
from db_utils import DATABASE_PATH, get_dataset_info, ensure_model_info_exist, get_max_eval_count, insert_data, update_data


def llama_forward(tokens, generator):
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    logits = generator.model.forward_all(tokens, 0)
    return logits


def open_db(no_db):
    if torch.distributed.get_rank() == 0 and not no_db:
        conn = sqlite3.connect(os.path.join(DATABASE_PATH, "evaluate_info.db"))
        return conn
    else:
        return None


def close_db(conn):
    if conn is not None:
        conn.commit()
        conn.close()


def post_process(result):
    result = re.findall(r"(?<![a-zA-Z0-9_])([ABCD])(?![a-zA-Z0-9_])", result)
    if len(result) == 0:
        result = ""
    else:
        result = result[0]
    return result


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def exact_match(res, labels):
    if not isinstance(labels, list):
        labels = [labels]
    else:
        labels = labels[: len(labels) - labels.count("NAN")]
    res = normalize_answer(res)
    for label in labels:
        if res == normalize_answer(label):
            return True
    return False


def include_answer(res, labels):
    if not isinstance(labels, list):
        labels = [labels]
    else:
        labels = labels[: len(labels) - labels.count("NAN")]
    res = normalize_answer(res)
    for label in labels:
        if normalize_answer(label) in res:
            return True
    return False


def f1_score(res, labels):
    if not isinstance(labels, list):
        labels = [labels]
    else:
        labels = labels[: len(labels) - labels.count("NAN")]
    res = normalize_answer(res)
    if len(res) == 0:
        return 0

    res_split = res.split()
    precision = recall = count = 0
    for label in labels:
        label = normalize_answer(label)
        label_split = label.split()
        if len(label_split) == 0:
            continue
        common = [i for i in res_split if i in label_split]
        precision += len(common) / len(res_split)
        recall += len(common) / len(label_split)
        count += 1

    if precision == 0 or recall == 0:
        return 0

    precision = precision / count
    recall = recall / count

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def f1_score_cn(res, labels):
    if not isinstance(labels, list):
        labels = [labels]
    else:
        labels = labels[: len(labels) - labels.count("NAN")]
    if len(res) == 0:
        return 0

    precision = recall = count = 0
    for label in labels:
        if len(label) == 0:
            continue
        common = [i for i in res if i in label]
        precision += len(common) / len(res)
        recall += len(common) / len(label)
        count += 1

    if precision == 0 or recall == 0:
        return 0

    precision = precision / count
    recall = recall / count

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def short_generation_evaluator(dataloader, generator, data2db, logger):
    task_type, check_repeat, no_db = data2db[-3:]
    data2db = data2db[:-3]
    keys2check = ["model_id", "dataset_id", "eval_count", "sample_id"]

    cors = []
    incs = []
    f1s = []
    count = 1
    for i, (questions, labels) in enumerate(dataloader):
        labels = [[labels[i][j] for i in range(len(labels))] for j in range(len(labels[0]))]
        results = generator.generate(questions, 100, temperature=0)
        results = [i.split("Assistant: ")[1] for i in results]
        cor = [exact_match(" ".join(result.split()[:10]), label) for result, label in zip(results, labels)]
        inc = [include_answer(result, label) for result, label in zip(results, labels)]
        f1 = [f1_score(result, label) for result, label in zip(results, labels)]
        cors.extend(cor)
        incs.extend(inc)
        f1s.extend(f1)

        for question, label, result in zip(questions, labels, results):
            logger.info("question:%s" % question)
            logger.info("answer:%s" % result)
            logger.info("label:%s" % (", ".join(label[: len(label) - label.count("NAN")])))

    return {"accuracy": np.mean(cors), "include": np.mean(incs), "F1 score": np.mean(f1s)}


def short_generation_cn_evaluator(dataloader, generator, data2db, logger):
    task_type, check_repeat, no_db = data2db[-3:]
    data2db = data2db[:-3]
    keys2check = ["model_id", "dataset_id", "eval_count", "sample_id"]

    cors = []
    incs = []
    f1s = []
    count = 1
    for i, (questions, labels) in enumerate(dataloader):
        labels = [[labels[i][j] for i in range(len(labels))] for j in range(len(labels[0]))]
        results = generator.generate(questions, 100, temperature=0)
        results = [i.split("Assistant: ")[1] for i in results]
        cor = [exact_match(result, label) for result, label in zip(results, labels)]
        inc = [include_answer(result, label) for result, label in zip(results, labels)]
        f1 = [f1_score_cn(result, label) for result, label in zip(results, labels)]
        cors.extend(cor)
        incs.extend(inc)
        f1s.extend(f1)

        for question, label, result in zip(questions, labels, results):
            logger.info("question:%s" % question)
            logger.info("answer:%s" % result)
            logger.info("label:%s" % (", ".join(label[: len(label) - label.count("NAN")])))

    return {"accuracy": np.mean(cors), "include": np.mean(incs), "F1 score": np.mean(f1s)}
