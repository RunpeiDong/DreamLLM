import argparse
import copy
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, NewType

import dacite
import pygments
from colorama import Fore, Style, init
from omegaconf import DictConfig, OmegaConf
from pygments.formatters import Terminal256Formatter
from pygments.lexers import Python3Lexer

from omni.config.lazy import CONFIG_KEY, LazyConfig
from omni.utils.comm import get_rank, get_world_size, is_main_process
from omni.utils.import_utils import (
    is_basemind_platform_available,
    is_shaipower_platform_available,
    is_volc_mlplatform_available,
)
from omni.utils.loguru import logger, setup_logger
from omni.utils.omegaconf_utils import omageconf_safe_update
from omni.utils.profiler import auto_profiler, pretty_format, timestamp_str
from omni.utils.training_utils import set_seed

init(autoreset=True)


@dataclass
class LazyAguments:
    config_file: str = field(default="", metadata={"help": "path to config file"})
    run_dir: str = field(default="", metadata={"help": "path to save logs, config and profiler"})
    use_profiler: bool = field(
        default=True, metadata={"help": "use pyinstrument (and torch profiler) to profile, recommend to set to True"}
    )
    use_torch_profiler: bool = field(default=False, metadata={"help": "use torch profiler to profile"})
    record_cuda: bool = field(default=False, metadata={"help": "torch profiler record cuda"})
    time_out: int | None = field(
        default=None, metadata={"help": "time out, useful when debugging and performance optimization."}
    )
    logger_rank: str = field(default="main", metadata={"help": "logger rank, choose from `main` or `all`."})
    local_rank: int = field(default=0, metadata={"help": "local rank, used for deepspeed."})


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--logger_rank", default="main", choices=["main", "all"], help="logger rank, choose from `main` or `all`.")
    parser.add_argument("--local_rank", default=0, help="local rank, used for deepspeed.")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER, help="The remaining parameters will override the configuration file."
    )
    return parser


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def _try_set_key(cfg, *keys, value=None):
    """
    Try set keys from cfg until the first key that exists.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            omageconf_safe_update(cfg, k, value)


def _highlight(code):
    lexer = Python3Lexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def compare_dicts(d1, d2, parent_key="", diff_keys=None):
    if diff_keys is None:
        diff_keys = []

    for key in d1:
        if key not in d2:
            diff_keys.append(f"{parent_key}.{key}" if parent_key else key)
        else:
            if isinstance(d1[key], (dict, DictConfig)) and isinstance(d2[key], (dict, DictConfig)):
                compare_dicts(d1[key], d2[key], f"{parent_key}.{key}" if parent_key else key, diff_keys)
            elif d1[key] != d2[key]:
                diff_keys.append(f"{parent_key}.{key}" if parent_key else key)
    return diff_keys


def add_color(yaml_str, keys_to_color):
    for key in keys_to_color:
        key = key.split(".")[-1]
        pattern = re.compile(rf"(\b{key}\b: [^\n]+)")
        yaml_str = pattern.sub(Fore.RED + r"\1" + Style.RESET_ALL, yaml_str)
    return yaml_str


def default_setup(config: DictConfig, args, config_before_override: DictConfig):
    if is_volc_mlplatform_available():
        os.environ["TORCH_HOME"] = "~/.cache/torch/hub"
    elif is_basemind_platform_available():
        os.environ["ENDPOINT_URL"] = "http://oss.i.basemind.com"
    elif is_shaipower_platform_available():
        os.environ["ENDPOINT_URL"] = "http://oss.i.basemind.com"
    else:
        logger.warning("Make sure you are training on the correct platform.")

    output_dir = _try_get_key(config, "OUTPUT_DIR", "output_dir", "train.output_dir", "training.output_dir")
    if "debug" in output_dir:
        _try_set_key(config, "REPORT_TO", "report_to", "train.report_to", "training.report_to", value=[])

    current_time = timestamp_str()

    run_dir = os.path.join(output_dir, current_time)
    config.run_dir = run_dir
    if is_main_process():
        os.makedirs(config.run_dir, exist_ok=True)

    rank = get_rank()

    config.logger_rank = args.logger_rank
    # setup logger and warnings, only first process in distributed training will print info
    setup_logger(config.run_dir, f"training.log", distributed_rank=rank, logger_rank=config.logger_rank)
    if not is_main_process():
        warnings.filterwarnings("ignore")

    logger.info(f"World size: {get_world_size()}")

    # command line arguments and content of `args.config_file`
    logger.info("Command line arguments:\n" + pretty_format(args))
    logger.info(f"Config file: {args.config_file}")
    keys = compare_dicts(config_before_override, config)
    config_yaml_str = add_color(OmegaConf.to_yaml(config, sort_keys=True), keys)
    logger.info("Full config:\n" + config_yaml_str)

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(config, "SEED", "seed", "train.seed", "training.seed", default=42)
    set_seed(seed + rank)

    config.config_file = args.config_file

    if is_main_process() and config.run_dir:
        path = os.path.join(config.run_dir, f"{CONFIG_KEY}.py")
        LazyConfig.save(config, path)
        logger.info("Full config saved to {}".format(path))


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


def LazyArgumentParser(dataclass_type: DataClassType) -> DataClass:
    """
    Initialize an instance from the `CONFIG_KEY` dict in the config file, and accept modifications from the command line.

    Args:
        dataclass_type (DataClassType): Pass a dataclass type, which contains all arguments you need, including nested dataclass.

    Returns:
        DataClass: Return an instance of dataclass from config file.

    Example:
    ```
    torchrun --nproc-per-node=8 train.py --config_file path_to_config.py \
    "mix_precision=True" \
    "model.max_length=77"
    ```
    """
    args = default_parser().parse_args()

    # CONFIG_KEY is an Omegaconf variable name in file
    config: DictConfig = LazyConfig.load(args.config_file, CONFIG_KEY)
    config_before_override = copy.deepcopy(config)
    config: DictConfig = LazyConfig.apply_overrides(config, args.opts)
    default_setup(config, args, config_before_override)

    config: dict = OmegaConf.to_container(config, resolve=True)

    dacite_config = dacite.Config(check_types=True, strict=True)
    dataclass_args = dacite.from_dict(data_class=dataclass_type, data=config, config=dacite_config)
    return dataclass_args


@logger.catch
def LazyLaunch(main_func: Callable, dataclass_type: DataClassType, *args, **kwargs):
    """`main_func` must accept argument `config` of type `dataclass_type`."""
    config = LazyArgumentParser(dataclass_type)

    if config.use_profiler:
        main_func = auto_profiler(
            dir=config.run_dir,
            seconds=config.time_out,
            use_torch_profiler=config.use_torch_profiler,
            record_cuda=config.record_cuda,
        )(main_func)

    main_func(config, *args, **kwargs)
