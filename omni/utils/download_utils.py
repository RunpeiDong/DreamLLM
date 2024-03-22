import os
import time
from functools import wraps

import timm.models.hub as timm_hub
import torch
from huggingface_hub import snapshot_download

from omni.utils.comm import is_main_process, synchronize


def retry(total_tries=5, initial_wait=1, backoff_factor=2, max_wait=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            for i in range(total_tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # If this was the last attempt
                    if i == total_tries - 1:
                        raise ValueError(f"Function failed after {total_tries} attempts") from e
                    time.sleep(wait_time)
                    # Exponential backoff
                    wait_time *= backoff_factor
                    if wait_time > max_wait and max_wait is not None:
                        wait_time = max_wait

        return wrapper

    return decorator


@retry(total_tries=999, initial_wait=1, backoff_factor=2, max_wait=60)
def volce_snapshot_download(repo_id, *, local_dir=None, resume_download=True, max_workers=8):
    os.environ["http_proxy"] = "100.66.27.151:3128"
    os.environ["https_proxy"] = "100.66.27.151:3128"
    snapshot_download(
        repo_id,
        local_dir=local_dir,
        resume_download=resume_download,
        max_workers=max_workers,
    )


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)
    synchronize()

    return get_cached_file_path()
