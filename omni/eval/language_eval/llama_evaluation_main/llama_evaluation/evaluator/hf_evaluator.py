import torch
import numpy as np

from tqdm import tqdm
from .evaluator_utils import hf_forward, get_logprob_by_tokens_hf, exact_match, include_answer, f1_score, post_process, write_jsonl


__all__ = [
    "hf_multiple_choice_evaluator",
    "hf_concat_logprob_evaluator",
    "hf_wsc_concat_logprob_evaluator",
    "hf_predict_last_evaluator",
    "hf_concat_acc_norm_evaluator",
    "hf_short_generation_evaluator",
    "hf_short_generation_cn_evaluator",
    "hf_generation_for_choice_evaluator",
]


def hf_multiple_choice_evaluator(dataset, tokenizer, model, fs):
    cors = torch.zeros(len(dataset))

    target_ids = [tokenizer(i, return_tensors="pt").input_ids for i in dataset.labels]
    target_ids = torch.cat(target_ids)

    for i, (prompt, label) in tqdm(enumerate(dataset)):
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        logits = hf_forward(tokens, model)
        logits = logits[:, -1].flatten()
        logits = torch.nn.functional.log_softmax(logits, dim=0).cpu()
        pred = torch.argmax(logits[target_ids].sum(dim=-1), dim=0).cpu()
        cors[i] = pred == label

        write_jsonl(fs, [i + 1, pred.item(), str(logits.tolist()), str((pred == label).item())])

    return {"accuracy": cors.mean().item()}


def hf_concat_logprob_evaluator(dataset, tokenizer, model, fs):
    cors = torch.zeros(len(dataset))
    for i, (question, options, label) in tqdm(enumerate(dataset)):
        question_tokens = tokenizer(question, return_tensors="pt").input_ids
        len_ch = len(options)
        pred = torch.ones(len_ch) * -1e6
        for j in range(len_ch):
            tokens = tokenizer(options[j], return_tensors="pt").input_ids
            logits = hf_forward(torch.cat((question_tokens, tokens), dim=1), model)
            logits = get_logprob_by_tokens_hf(logits, tokens)
            pred[j] = logits.sum() / len(options[j])
        cors[i] = pred.argmax() == label

        write_jsonl(fs, [i + 1, pred.argmax().item(), str(pred.tolist()), str((pred.argmax() == label).item())])

    return {"accuracy": cors.mean().item()}


def hf_wsc_concat_logprob_evaluator(dataset, tokenizer, model, fs):
    cors = torch.zeros(len(dataset))
    for i, (question, options, label) in tqdm(enumerate(dataset)):
        question_tokens = tokenizer(question, return_tensors="pt").input_ids
        len_ch = len(options)
        pred = torch.ones(len_ch) * -1e6
        for j in range(len_ch):
            tokens = tokenizer(options[j], return_tensors="pt").input_ids
            logits = hf_forward(torch.cat((question_tokens, tokens), dim=1), model)
            logits = get_logprob_by_tokens_hf(logits, tokens)
            pred[j] = logits.sum()
        cors[i] = pred.argmax() % 2 == label

        write_jsonl(fs, [i + 1, pred.argmax().item() % 2, str(pred.tolist()), str((pred.argmax() == label).item())])

    return {"accuracy": cors.mean().item()}


def hf_predict_last_evaluator(dataset, tokenizer, model, fs):
    cors = torch.zeros(len(dataset))
    for i, (question, options, label) in tqdm(enumerate(dataset)):
        post_ctx = options.pop()
        len_ch = len(options)
        pred = torch.ones(len_ch) * -1e6
        question_tokens = tokenizer(question, return_tensors="pt").input_ids
        post_ctx_tokens = tokenizer(post_ctx, return_tensors="pt").input_ids
        for j in range(len_ch):
            tokens = tokenizer(options[j], return_tensors="pt").input_ids
            logits = hf_forward(torch.cat((question_tokens, tokens, post_ctx_tokens), dim=1), model)
            logits = get_logprob_by_tokens_hf(logits, post_ctx_tokens)
            pred[j] = logits.sum()
        cors[i] = pred.argmax() == label

        write_jsonl(fs, [i + 1, pred.argmax().item(), str(pred.tolist()), str((pred.argmax() == label).item())])

    return {"accuracy": cors.mean().item()}


def hf_concat_acc_norm_evaluator(dataset, tokenizer, model, fs):
    cors = torch.zeros(len(dataset))
    for i, (question, options, label) in tqdm(enumerate(dataset)):
        len_ch = len(options)
        pred = torch.ones(len_ch) * -1e6
        question_tokens = tokenizer(question, return_tensors="pt").input_ids
        for j in range(len_ch):
            tokens = tokenizer(options[j], return_tensors="pt").input_ids
            answer_tokens = tokenizer("Answer:", return_tensors="pt").input_ids

            logits = hf_forward(torch.cat((question_tokens, tokens), dim=1), model)
            res1 = get_logprob_by_tokens_hf(logits, tokens).sum()

            logits = hf_forward(torch.cat((answer_tokens, tokens), dim=1), model)
            res2 = get_logprob_by_tokens_hf(logits, tokens).sum()
            pred[j] = res1 - res2
        cors[i] = pred.argmax() == label

        write_jsonl(fs, [i + 1, pred.argmax().item(), str(pred.tolist()), str((pred.argmax() == label).item())])

    return {"accuracy": cors.mean().item()}


def hf_short_generation_evaluator(dataset, tokenizer, model, fs):
    cors = []
    incs = []
    f1s = []
    for i, (question, labels) in tqdm(enumerate(dataset)):
        question_tokens = tokenizer(question, return_tensors="pt").input_ids
        result = hf_forward(question_tokens, model, generate=True)
        result = tokenizer.batch_decode(result, skip_special_tokens=True)[0][len(question) :]
        result = result.split("Question:")[0].strip()
        cors.append(exact_match(result, labels))
        incs.append(include_answer(result, labels))
        f1s.append(f1_score(result, labels))

        write_jsonl(fs, [(i + 1), result, str(exact_match(result, labels)), str(include_answer(result, labels)), result])

    return {"include": np.mean(incs), "accuracy": np.mean(cors), "F1 score": np.mean(f1s)}


def hf_short_generation_cn_evaluator(dataset, tokenizer, model, fs):
    cors = []
    incs = []
    f1s = []
    for i, (question, labels) in tqdm(enumerate(dataset)):
        question_tokens = tokenizer(question, return_tensors="pt").input_ids
        result = hf_forward(question_tokens, model, generate=True)
        result = tokenizer.batch_decode(result, skip_special_tokens=True)[0][len(question) :]
        result = result.split("回答：")[0].strip()
        cors.append(exact_match(result, labels, en=False))
        incs.append(include_answer(result, labels, en=False))
        f1s.append(f1_score(result, labels, en=False))

        write_jsonl(fs, [(i + 1), result, str(exact_match(result, labels, en=False)), str(include_answer(result, labels, en=False)), result])

    return {"include": np.mean(incs), "accuracy": np.mean(cors), "F1 score": np.mean(f1s)}


def hf_generation_for_choice_evaluator(dataset, tokenizer, model, fs):
    cors = []
    incs = []
    f1s = []
    choices = dataset.labels
    for i, (question, label) in tqdm(enumerate(dataset)):
        question_tokens = tokenizer(question, return_tensors="pt").input_ids
        result = hf_forward(question_tokens, model, generate=True, max_gen_length=10)
        result = tokenizer.batch_decode(result)[0][len(question) :]
        res = post_process(result.split("\n\n")[0], choices)
        label = choices[label]
        cors.append(res.lower() == label.lower())

        write_jsonl(fs, [(i + 1), res, result, str(res.lower() == label.lower())])

    return {"accuracy": np.mean(cors)}
