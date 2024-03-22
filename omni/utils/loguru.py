import atexit as _atexit
import inspect
import os
import sys as _sys
import torch

import pendulum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger


class CustomLogger(_Logger):
    def __init__(self, core, exception, depth, record, lazy, colors, raw, capture, patchers, extra):
        self._core = core
        self._options = (exception, depth, record, lazy, colors, raw, capture, patchers, extra)

        # Newly added attributes.
        self._warned_messages = set()

    def warning_once(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'WARNING'``."""
        if __message not in __self._warned_messages:
            __self._warned_messages.add(__message)
            __self._log("WARNING", False, __self._options, __message, args, kwargs)


# from loguru
logger = CustomLogger(
    core=_Core(),
    exception=None,
    depth=0,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)


def set_datetime(record):
    record["extra"]["datetime"] = str(pendulum.now("Asia/Shanghai")).split(".")[0]


FORMAT = "<green>{extra[datetime]}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# make sure only rank0 process write log
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    if _defaults.LOGURU_AUTOINIT and _sys.stderr:
        logger.configure(patcher=set_datetime)
        logger.add(_sys.stderr, format=FORMAT, enqueue=True)

_atexit.register(logger.remove)


# from yolox
def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


CALLER_NAMES = "warnings"


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO", caller_names=CALLER_NAMES):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            _sys.__stdout__.write(buf)

    def flush(self):
        # flush is related with CPR(cursor position report) in terminal
        return _sys.__stdout__.flush()

    def isatty(self):
        # when using colab, jax is installed by default and issue like
        # https://github.com/Megvii-BaseDetection/YOLOX/issues/1437 might be raised
        # due to missing attribute like`isatty`.
        # For more details, checked the following link:
        # https://github.com/google/jax/blob/10720258ea7fb5bde997dfa2f3f71135ab7a6733/jax/_src/pretty_printer.py#L54  # noqa
        return _sys.__stdout__.isatty()

    def fileno(self):
        # To solve the issue when using debug tools like pdb
        return _sys.__stdout__.fileno()


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    _sys.stderr = redirect_logger
    _sys.stdout = redirect_logger


def setup_logger(save_dir, filename, distributed_rank=0, logger_rank="main", mode="a", is_redirect_sys_output=False):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        filename (string): log save name.
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)

    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)

    if logger_rank == "main":
        if distributed_rank == 0:
            logger.configure(patcher=set_datetime)
            logger.add(save_file, format=FORMAT)
    elif logger_rank == "all":
        save_file_no_ext, ext = os.path.splitext(save_file)
        save_file = f"{save_file_no_ext}_{distributed_rank}{ext}"
        torch.cuda.synchronize()
        logger.configure(patcher=set_datetime)
        torch.cuda.synchronize()
        logger.add(save_file, format=FORMAT)

    if is_redirect_sys_output:
        # redirect stdout/stderr to loguru
        redirect_sys_output("INFO")
