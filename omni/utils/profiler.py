import argparse
import datetime
import json
import os
import re
import signal
import time
from functools import wraps

import pytz
import yaml
from pyinstrument import Profiler
from transformers import PreTrainedTokenizerBase

from omni.utils.import_utils import is_torch_available
from omni.utils.loguru import logger


def timestamp_str(*, time_value=None, zone="Asia/Shanghai") -> str:
    """format given timestamp, if no timestamp is given, return a call time string"""
    if time_value is None:
        time_value = datetime.datetime.now(pytz.timezone(zone))
    return time_value.strftime("%Y-%m-%d_%H-%M-%S")


class FunctionProfiler:
    def __init__(self, func_name=""):
        self.func_name = func_name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Entering function {self.func_name}...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        logger.info(f"Execution time of {self.func_name}: {elapsed_time:.2f} seconds")


def auto_profiler(dir="./", seconds=None, use_torch_profiler=False, record_cuda=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timestamp = timestamp_str()

            # Start the pyinstrument Profiler
            profiler = Profiler()
            profiler.start()

            # Start the torch Profiler
            torch_profiler = None
            if use_torch_profiler:
                if is_torch_available(">=", "1.9.0"):
                    import torch
                else:
                    raise ImportError("Torch profiler is only available in PyTorch >= 1.9.0.")
                # Define the activities based on whether CUDA is used or not
                activities = [torch.profiler.ProfilerActivity.CPU]
                if record_cuda:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)

                torch_profiler = torch.profiler.profile(
                    activities=activities,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(dir, f"profiler_torch_{timestamp}")),
                    record_shapes=True,
                    profile_memory=True,
                )
                torch_profiler.__enter__()

            def timeout_handler(signum, frame):
                raise TimeoutError

            if seconds is not None:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)

            result = None
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                logger.info("Function timed out, stopping profiler...")
            finally:
                if seconds is not None:
                    signal.alarm(0)

                # Stop the pyinstrument Profiler
                profiler.stop()
                # Save the pyinstrument profile result
                pyinstrument_filename = f"profiler_pyinstrument_{timestamp}.html"
                with open(os.path.join(dir, pyinstrument_filename), "w") as f:
                    f.write(profiler.output_html())
                logger.info(f"Pyinstrument profiler output saved to {os.path.join(dir, pyinstrument_filename)}.")

                # Exit the torch profiler context
                if torch_profiler is not None:
                    torch_profiler.__exit__(None, None, None)
                    logger.info(f"Torch profiler output saved to {os.path.join(dir, f'profiler_torch_{timestamp}')}.")
            return result

        return wrapper

    return decorator


def find_matching_parenthesis(expression, opening_index):
    if expression[opening_index] != "(":
        raise ValueError("The character at the provided index is not '('.")

    stack = 0

    for index in range(opening_index + 1, len(expression)):
        char = expression[index]
        if char == "(":
            stack += 1
        elif char == ")":
            if stack == 0:
                return index
            stack -= 1

    raise ValueError("No matching ')' found for '(' at index {}.".format(opening_index))


def pretty_format(obj, indent: int = 4) -> str:
    if isinstance(obj, dict):
        return yaml.dump(obj, sort_keys=True, indent=indent)
    elif isinstance(obj, PreTrainedTokenizerBase):
        repr_str = obj.__repr__()
        class_name, rest = repr_str.split("(", 1)
        idx = find_matching_parenthesis(f"({rest}", 0)
        other = rest[idx:]
        other = other.strip(",").strip(" ")
        rest = rest[:idx]
        rest = rest.rstrip(")")

        pairs = re.findall(r"(\w+)=({[^}]*}|[^,]*),?", rest)

        formatted_pairs = []
        for k, v in pairs:
            if v.startswith("{") and v.endswith("}"):
                try:
                    v_dict = json.loads(v.replace("'", '"'))
                    v_formatted = json.dumps(v_dict, indent=indent).replace("\n", "\n" + " " * indent)
                except json.JSONDecodeError:
                    v_formatted = v
            else:
                v_formatted = v

            formatted_pairs.append(f"{' ' * indent}{k}={v_formatted},")

        return f"{class_name}(\n" + "\n".join(formatted_pairs) + "\n),\n" + other.replace("\t", " " * indent)
    elif isinstance(obj, argparse.Namespace):
        args_dict = vars(obj)
        return yaml.dump(args_dict, sort_keys=True, indent=indent)
    else:
        return obj
