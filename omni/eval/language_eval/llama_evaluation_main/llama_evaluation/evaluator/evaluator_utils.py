import os
import re
import json
import time
import torch
import openai
import string
import requests
import numpy as np
import Levenshtein
from collections import Counter, defaultdict

openai.api_key = "EMPTY"
openai.proxy = ""

__all__ = [
    "local_forward", "online_forward", "hf_forward",
    "get_logprob_by_tokens", "get_logprobs_by_labels",
    "get_logprob_by_tokens_hf",
    "exact_match", "include_answer", "f1_score",
    "per_sentence_exact_match", "post_process", "write_jsonl", 
]


def local_forward(prompt, generator):
    tokens = generator.tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor(tokens).unsqueeze(0).long()
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    logits = generator.model.forward_all(tokens, 0)
    return tokens[0], logits


def tgi_forward(args, prompt, ip_port, max_gen_length, retry=5):
    url = f"http://{ip_port}/generate"
    args.pop("model", None)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_gen_length,
            "details": True,
            "decoder_input_details": True,
            "seed": None if args["do_sample"] else 0,
        }
    }
    payload["parameters"].update(args)

    headers = {
        "Content-Type": "application/json"
    }

    proxies = {"http": [], "https": []}
    response = requests.post(url, json=payload, headers=headers, proxies=proxies)
    while(response.status_code != 200 and retry > 0):
        retry -= 1
        time.sleep(1)
        response = requests.post(url, json=payload, headers=headers, proxies=proxies)
    assert response.status_code == 200
    response = response.json()
    generated_text = response["generated_text"]
    return generated_text, response["details"]["prefill"]


def convert_vllm_dict(prompt, completion):
    if completion["logprobs"] is None:
        return completion["text"], None
    res = []
    for text, logprob in zip(completion["logprobs"]["tokens"], completion["logprobs"]["token_logprobs"]):
        res.append({"text": text, "logprob": logprob})
    res.pop()
    return completion["text"].strip(" ")[len(prompt):].replace("</s>", ""), res


def vllm_forward(args, prompt, ip_port, max_gen_length, retry=5):
    openai.api_base = f"http://{ip_port}/v1"
    args.pop("top_k", None)
    do_sample = args.pop("do_sample", None)
    if do_sample == False:
        args["temperature"] = 0
        args.pop("top_p", None)

    if max_gen_length == 1:
        args["echo"] = True
        args["logprobs"] = 1

    while(retry > 0):
        try:
            completion = openai.Completion.create(
                  prompt=prompt, max_tokens=max_gen_length, **args)
            return (convert_vllm_dict(prompt, completion["choices"][0]))
        except:
            retry -= 1
    return ("Error.", None)


def online_forward(args, prompt, ip_port, max_gen_length):
    response = requests.get(f"http://{ip_port}/info", proxies={"http": [], "https": []})
    if response.status_code == 200:
        return tgi_forward(args, prompt, ip_port, max_gen_length)
    else:
        return vllm_forward(args, prompt, ip_port, max_gen_length)


@torch.no_grad()
def hf_forward(tokens, model, generate=False, max_gen_length=100):
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    if generate:
        res = model.generate(tokens, max_new_tokens=max_gen_length).cpu()
    else:
        res = model(input_ids=tokens).logits.cpu().float()
    return res


def get_choice_tokens(token_a, token_b):
    cur = len(token_a) - len(token_b)
    for i in range(cur, len(token_a)):
        if token_a[i] == token_b[i-cur]:
            return torch.tensor(token_b[i-cur:]).unsqueeze(0)
    return ""


def get_logprob_by_tokens(sample, generator, token):
    tokens, logits = local_forward(sample, generator)
    token = get_choice_tokens(tokens, 
            generator.tokenizer.encode(token, bos=False, eos=False))
    logits = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
    logits = torch.gather(logits[:, -token.size(1)-1:-1], dim=-1, index=token.unsqueeze(-1))
    return logits


def get_logprob_by_tokens_hf(logits, token):
    logits = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
    logits = torch.gather(logits[:, -token.size(1)-1:-1], dim=-1, index=token.unsqueeze(-1))
    return logits


def get_logprobs_by_labels(result_list, option):
    logprob = 0.
    from llama import Tokenizer
    weight_path = os.environ.get("EVAL_ROOT", "data/weights/")
    tok = Tokenizer(os.path.join(weight_path, "tokenizer.model"))
    option = tok.encode(option, bos=False, eos=False)
    for i in range(-1, -len(option)-1, -1):
        logprob += result_list[i]["logprob"]
    return logprob
        
    subs = ""
    option = "".join(option.split())
    for i in range(len(result_list)-1, 0, -1):
        subs = result_list[i]["text"] + subs
        logprob += result_list[i]["logprob"]
        if Levenshtein.distance(subs.strip(), option.strip()) == 0:
            return logprob
    return -1000


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def normalize_answer_cn(s):

    def white_space_fix(text):
        return ''.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + 
                "".join([u"‘", u"’", u"´", u"`", u"《", u"》", u"，", u"。", u"？", u"！"]))
        return ''.join(ch if ch not in exclude else '' for ch in text)

    def replace_underscore(text):
        return text.replace('_', '')

    return white_space_fix(handle_punc(replace_underscore(s))).strip()


def exact_match(res, labels, en=True):
    if not isinstance(labels, list):
        labels = [labels]
    norm_func = normalize_answer if en else normalize_answer_cn
    res = norm_func(res)
    for label in labels:
        if isinstance(label, dict) and "text" in label:
            label = label["text"]
        if res == norm_func(label):
            return True
    return False


def include_answer(res, labels, en=True):
    if not isinstance(labels, list):
        labels = [labels]
    norm_func = normalize_answer if en else normalize_answer_cn
    res = norm_func(res)
    for label in labels:
        if isinstance(label, dict) and "text" in label:
            label = label["text"]
        if norm_func(label) in res:
            return True
    return False


def f1_score_per_sample(prediction, ground_truth, en=True):
    norm_func = normalize_answer if en else normalize_answer_cn
    prediction_tokens = norm_func(prediction)
    ground_truth_tokens = norm_func(ground_truth)
    if en:
        prediction_tokens = prediction_tokens.split()
        ground_truth_tokens = ground_truth_tokens.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score(res, labels, en=True):
    if not isinstance(labels, list):
        labels = [labels]
    f1 = 0
    for label in labels:
        if res == "" and label == "":
            f1 = 1
            break
        if isinstance(label, dict) and "text" in label:
            label = label["text"]
        f1 = max(f1, f1_score_per_sample(res, label, en))
    return f1

    norm_func = normalize_answer if en else normalize_answer_cn
    res = norm_func(res)
    if not isinstance(labels, list):
        labels = [labels]
    if len(res) == 0:
        return 0

    if en:
        res = res.split()
    precision = recall = count = 0
    for label in labels:
        label = norm_func(label)
        if en:
            label = label.split()
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


def per_sentence_exact_match(result, label):
    label = normalize_answer_cn(label)
    def _split(res):
        tmp = []
        pre = cur = 0
        for cur in range(len(res)):
            if res[cur]in ["\n", "，", "。", "？", "！", "：", "_"]:
                if cur - pre > 1:
                    tmp.append(res[pre: cur])
                pre = cur + 1
        if cur - pre > 1:
            tmp.append(res[pre: cur])
        return tmp
    results = _split(result)
                
    for result in results:
        if label == normalize_answer_cn(result):
            return True
    return False


def post_process(result, labels):
    pattern = "|".join(labels)
    result = re.findall(r'(?<![a-zA-Z0-9_])(%s)(?![a-zA-Z0-9_])'%pattern, result)
    if len(result) == 0:
        result = ""
    else:
        result = result[0]
    return result


def write_jsonl(fs, data):
    if fs is not None:
        data = json.dumps(data)
        fs.write(data+'\n')
