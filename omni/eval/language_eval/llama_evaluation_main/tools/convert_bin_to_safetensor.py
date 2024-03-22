
from safetensors.torch import save_file
import torch
import glob

import os
import argparse
import torch
import json
from collections import defaultdict
from transformers import LlamaModel


# def parse_args():
#     parser = argparse.ArgumentParser(description='Extract MMProjector weights')
#     parser.add_argument('--model_name_or_path', type=str, help='model folder')
#     parser.add_argument('--output', type=str, help='output file')
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args()

#     keys_to_match = ['mm_projector', 'transformer.wte', "condition_projector"]
#     ckpt_to_key = defaultdict(list)
#     try:
#         model_indices = json.load(open(os.path.join(args.model_name_or_path, 'pytorch_model.bin.index.json')))
#         for k, v in model_indices['weight_map'].items():
#             if any(key_match in k for key_match in keys_to_match):
#                 ckpt_to_key[v].append(k)
#     except FileNotFoundError:
#         # Smaller models or model checkpoints saved by DeepSpeed.
#         v = 'pytorch_model.bin'
#         for k in torch.load(os.path.join(args.model_name_or_path, v), map_location='cpu').keys():
#             if any(key_match in k for key_match in keys_to_match):
#                 ckpt_to_key[v].append(k)

#     loaded_weights = {}

#     for ckpt_name, weight_keys in ckpt_to_key.items():
#         ckpt = torch.load(os.path.join(args.model_name_or_path, ckpt_name), map_location='cpu')
#         for k in weight_keys:
#             loaded_weights[k] = ckpt[k]

#     torch.save(loaded_weights, args.output)

def convert(source_path):
    bin_files = glob.glob(source_path + "/*.bin")
    print("bin_files: ", bin_files)
    for bin_file in bin_files:
        file_name = bin_file.split("/")[-1].split(".")[0]
        if file_name == "training_args":
            continue
        print("file_name: ", file_name)
        pt_state_dict = torch.load(bin_file, map_location=torch.device("cpu"))
        new_state_dict = {}
        for p in pt_state_dict.keys():
            if p.startswith("vae"):
                continue
            elif p.rfind("vision_tower") != -1:
                continue
            elif p.rfind("mm_projector") != -1:
                continue
            elif p.rfind("condition_projector") != -1:
                continue
            elif p.rfind("dream_queries") != -1:
                continue
            else:
                new_state_dict[p] = pt_state_dict[p]
        print(new_state_dict.keys())
        save_file(new_state_dict, source_path + "/" + file_name + ".safetensors", metadata={"format": "pt"})

if __name__ == "__main__":
    source_path = "path2model"
    #target_path = "**.safetensors"
    convert(source_path)
