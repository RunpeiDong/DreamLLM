import json
import time

import megfile
from joblib import Parallel, delayed

from omni.utils.loguru import logger


def _load_json_list(file_path: str, keys: tuple = None):
    data = []
    with megfile.smart_open(file_path, "r") as f:
        try:
            data.extend(json.load(f))
        except:
            pass
    if keys is None:
        return data
    new_data = []
    for d in data:
        assert isinstance(d, dict), f"data should be list[dict], but got list[{type(d)}]"
        new_d = {}
        for key in keys:
            if key in d.keys():
                new_d[key] = d[key]
        new_data.append(new_d)
    return new_data


def load_json_list(file_paths: str | list[str], *, keys: tuple = None):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    tic = time.time()
    list_data_dict = []
    for file_path in file_paths:
        list_data_dict.extend(_load_json_list(file_path, keys=keys))
    time_cost = time.time() - tic
    logger.info(f"Loading json files takes {time_cost} seconds")
    return list_data_dict


def load_json_list_parallel(file_paths: str | list[str], *, keys: tuple = None, max_workers=32):
    # FIXME If the file is too large or too many, it may cause memory leakage
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    tic = time.time()
    list_data_dict = []
    results = Parallel(n_jobs=max_workers)(delayed(_load_json_list)(file_path, keys=keys) for file_path in file_paths)
    for _, _r in enumerate(results):
        list_data_dict.extend(_r)
    time_cost = time.time() - tic
    logger.info(f"Loading json files takes {time_cost} seconds")
    return list_data_dict
