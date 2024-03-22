import torch
import numpy as np

from tqdm import tqdm
from .evaluator_utils import (
    local_forward,
    get_logprob_by_tokens,
    exact_match,
    include_answer,
    f1_score,
    post_process,
    write_jsonl,
)


__all__ = [
    "multiple_choice_evaluator",
    "concat_logprob_evaluator",
    "wsc_concat_logprob_evaluator",
    "predict_last_evaluator",
    "concat_acc_norm_evaluator",
    "generation_for_choice_evaluator",
    "short_generation_evaluator",
    "short_generation_cn_evaluator",
]


def multiple_choice_evaluator(dataset, generator, batch_size, fs):
    cors = torch.zeros(len(dataset))
    target_ids = [generator.tokenizer.encode(i, bos=False, eos=False) for i in dataset.labels]
    target_ids = torch.tensor(target_ids)

    for i, (prompt, label) in tqdm(enumerate(dataset)):
        _, logits = local_forward(prompt, generator)
        logits = logits[:, -1].flatten()
        logits = torch.nn.functional.log_softmax(logits, dim=0).cpu()
        logits = logits[target_ids].sum(dim=-1)

        pred = torch.argmax(logits, dim=0).cpu()
        cors[i] = pred == label
        torch.distributed.barrier()

        write_jsonl(fs, [i + 1, pred.item(), str(logits.tolist()), str((pred == label).item())])

    return {"accuracy": cors.mean().item()}


def concat_logprob_evaluator(dataset, generator, batch_size, fs):
    cors = torch.zeros(len(dataset))

    for i, (question, choices, label) in tqdm(enumerate(dataset)):
        len_ch = len(choices)
        pred = torch.ones(len_ch) * -1e6
        for j in range(len_ch):
            sample = " ".join([question, choices[j]])
            logits = get_logprob_by_tokens(sample, generator, choices[j])
            torch.distributed.barrier()
            pred[j] = logits.sum() / len(choices[j])
        cors[i] = pred.argmax() == label

        write_jsonl(fs, [i + 1, pred.argmax().item(), str(pred.tolist()), str((pred.argmax() == label).item())])

    return {"accuracy": cors.mean().item()}


def wsc_concat_logprob_evaluator(dataset, generator, batch_size, fs):
    cors = torch.zeros(len(dataset))

    for i, (question, choices, label) in tqdm(enumerate(dataset)):
        len_ch = len(choices)
        pred = torch.ones(len_ch) * -1e6
        for j in range(len_ch):
            sample = " ".join([question, choices[j]])
            logits = get_logprob_by_tokens(sample, generator, choices[j])
            torch.distributed.barrier()
            pred[j] = logits.sum() / len(choices[j])
        cors[i] = (pred.argmax() % 2) == label

        write_jsonl(fs, [i + 1, pred.argmax().item() % 2, str(pred.tolist()), str((pred.argmax() == label).item())])

    return {"accuracy": cors.mean().item()}


def predict_last_evaluator(dataset, generator, batch_size, fs):
    cors = torch.zeros(len(dataset))

    for i, (question, choices, label) in tqdm(enumerate(dataset)):
        post_ctx = choices.pop()
        len_ch = len(choices)
        pred = torch.ones(len_ch) * -1e6
        for j in range(len_ch):
            sample = " ".join([question, choices[j], post_ctx])
            logits = get_logprob_by_tokens(sample, generator, post_ctx)
            torch.distributed.barrier()
            pred[j] = logits.sum()

        cors[i] = pred.argmax() == label

        write_jsonl(fs, [i + 1, pred.argmax().item(), str(pred.tolist()), str((pred.argmax() == label).item())])

    return {"accuracy": cors.mean().item()}


def concat_acc_norm_evaluator(dataset, generator, batch_size, fs):
    cors = torch.zeros(len(dataset))

    for i, (question, choices, label) in tqdm(enumerate(dataset)):
        len_ch = len(choices)
        pred = torch.ones(len_ch) * -1e6
        for j in range(len_ch):
            sample = " ".join([question, choices[j]])
            res1 = get_logprob_by_tokens(sample, generator, choices[j]).sum()

            sample = " ".join(["Answer:", choices[j]])
            res2 = get_logprob_by_tokens(sample, generator, choices[j]).sum()
            pred[j] = res1 - res2

        cors[i] = pred.argmax() == label

        write_jsonl(fs, [i + 1, pred.argmax().item(), str(pred.tolist()), str((pred.argmax() == label).item())])

    return {"accuracy": cors.mean().item()}


def short_generation_evaluator(dataset, generator, batch_size, fs):
    cors, incs, f1s = [], [], []
    questions, labels = [], []
    count = 1

    for i, (question, label) in tqdm(enumerate(dataset)):
        questions.append(question)
        labels.append(label)
        if i < (len(dataset) - 1) and (i + 1) % batch_size != 0:
            continue

        raw_results = generator.generate(questions, 100, temperature=0)
        if "\n\nAssistant:" in questions[0]:
            results = [i.split("Assistant:")[0].strip() for i in raw_results]
        else:
            results = [i.split("Answer:")[1].split("Question:")[0].strip() for i in raw_results]
        cor = [exact_match(result, label) for result, label in zip(results, labels)]
        inc = [include_answer(result, label) for result, label in zip(results, labels)]
        f1s.extend([f1_score(result, label) for result, label in zip(results, labels)])
        cors.extend(cor)
        incs.extend(inc)
        questions.clear()
        labels.clear()

        for raw_result, result, c, i in zip(raw_results, results, cor, inc):
            write_jsonl(fs, [count, result, str(c), str(i), raw_result])
            count += 1

    return {"include": np.mean(incs), "accuracy": np.mean(cors), "F1 score": np.mean(f1s)}


def short_generation_cn_evaluator(dataset, generator, batch_size, fs):
    cors, incs, f1s = [], [], []
    questions, labels = [], []
    count = 1
    for i, (question, label) in tqdm(enumerate(dataset)):
        questions.append(question)
        labels.append(label)
        if i < (len(dataset) - 1) and (i + 1) % batch_size != 0:
            continue

        raw_results = generator.generate(questions, 100, temperature=0)
        if "\n\nAssistant:" in questions[0]:
            results = [i.split("Assistant:")[0].strip() for i in raw_results]
        else:
            results = [i.split("回答：")[1].split("问题：")[0].strip() for i in raw_results]
        cor = [exact_match(result, label, en=False) for result, label in zip(results, labels)]
        inc = [include_answer(result, label, en=False) for result, label in zip(results, labels)]
        f1s.extend([f1_score(result, label, en=False) for result, label in zip(results, labels)])
        cors.extend(cor)
        incs.extend(inc)
        questions.clear()
        labels.clear()

        for raw_result, result, c, i in zip(raw_results, results, cor, inc):
            write_jsonl(fs, [count, result, str(c), str(i), raw_result])
            count += 1

    return {"include": np.mean(incs), "accuracy": np.mean(cors), "F1 score": np.mean(f1s)}


def generation_for_choice_evaluator(dataset, generator, batch_size, fs):
    cors = []
    questions, labels = [], []
    count = 1
    choices = dataset.labels
    for i, (question, label) in tqdm(enumerate(dataset)):
        questions.append(question)
        labels.append(choices[label])
        if i < (len(dataset) - 1) and (i + 1) % batch_size != 0:
            continue

        results = generator.generate(questions, 5, temperature=0)
        res = [post_process(i.split("Assistant:")[1], choices) for i in results]
        cors.extend([r.lower() == l.lower() for r, l in zip(res, labels)])
        questions.clear()
        labels.clear()

        for raw_result, result, label in zip(results, res, labels):
            write_jsonl(fs, [count, result, raw_result, str(result.lower() == label.lower())])
            count += 1

    return {"accuracy": np.mean(cors)}
