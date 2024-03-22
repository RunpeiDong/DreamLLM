import argparse
import os
import sys
import torch
import numpy as np
import time
import json

from omni.utils.loguru import logger
from pathlib import Path

from categories import subcategories, categories
from eval_utils import add_common_args, sft_prompt
from omni.eval.language_eval.modeling_dreamllm import setup_model_parallel, load


CHOICES = ["A", "B", "C", "D"]


@torch.no_grad()
def eval(args, subject, generator, tokenizer, valset, train_prompt):
    cors = []
    all_probs = []
    start_time = time.time()
    for sample in valset:
        label = sample[-1]
        sample = train_prompt + sample[:-1]
        sample = sample.strip()
        assert label in CHOICES
        if "翻译成中文" in sample:
            continue
        # get prompt and make sure it fits
        if args.sft:
            sample = sft_prompt(sample)
        # tokens = tokenizer.encode(sample, bos=True, eos=False)
        tokens = tokenizer(sample, return_tensors="pt").input_ids.cuda()
        while len(tokens) > args.max_seq_len:
            sample = sample.split("\n\n")
            sample = "\n\n".join(sample[0:-2] + sample[-1:])
            # tokens = tokenizer.encode(sample, bos=True, eos=False)
            tokens = tokenizer(sample, return_tensors="pt", padding="longest", max_length=args.max_seq_len, truncation=True).input_ids.cuda()
        # tokens = torch.tensor(tokens).long().cuda().unsqueeze(0)
        logits = generator.forward(input_ids=tokens).logits[:, -1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
    end_time = time.time()

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    logger.info("Average accuracy {:.3f}, total time {:.2f} - {}".format(acc, end_time - start_time, subject))

    return cors, acc, all_probs


@logger.catch(reraise=True)
def main(args):
    # if args.model is None:
    #    args.model = modelname_by_path(args.ckpt_dir)
    assert args.model is not None
    if "_sft_" in args.model:
        args.sft = True
    if args.logfile is None:
        args.logfile = os.path.join(args.save_dir, args.model + "_" + args.language + ".log")

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        logger.remove()
        sys.stdout = open(os.devnull, "w")
    else:
        # save logger info to file
        logger.add(args.logfile, enqueue=True, backtrace=True, diagnose=True)

    logger.info(args)
    generator, tokenizer = load(args.ckpt_dir, world_size)

    # load dataset from json
    train_set = json.load(open(os.path.join(args.data_dir, "mmlu_encn_dev.json")))
    val_set = json.load(open(os.path.join(args.data_dir, "mmlu_encn_val.json")))
    name_map = val_set["en2cn"]

    all_cors = []
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}

    subjects = list(name_map.keys())
    subjects.sort()

    for subject in subjects:
        train_subset = train_set[subject][args.language]
        val_subset = val_set[subject][args.language]

        # generate train prompt
        if args.language == "en":
            prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(" ".join(subject.split("_")))
        elif args.language == "cn":
            prompt = "以下是关于“{}”的多选题（附答案）。\n\n".format(name_map[subject])
        train_prompt = prompt + "\n\n".join(train_subset) + "\n\n"

        cors, acc, probs = eval(args, subject, generator, tokenizer, val_subset, train_prompt)
        subcats = subcategories[subject]

        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        logger.info("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        logger.info("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    logger.info("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("--data_dir", "-d", type=str, default="/data/raw_data/mmlu", help="Path to the dataset directory")
    parser.add_argument("--ntrain", "-k", type=int, default=5, help="Number of samples for few-shot, default: 5")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the generation task, default: 1")
    parser.add_argument("--language", type=str, default="en", choices=["en", "cn"], help='Choose "cn" to evluate in Chinese and "en" in English, default: en')
    parser.add_argument("--save_dir", type=str, default="./results/mmlu", help="Path to save log files")
    args = parser.parse_args()
    main(args)
