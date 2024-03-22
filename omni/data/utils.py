from typing import TypeAlias

import numpy as np
import PIL.Image
import torch

ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor
VideoType: TypeAlias = list[ImageType] | np.ndarray | torch.Tensor


class LargeInt(int):
    def __new__(cls, value):
        if isinstance(value, str):
            units = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
            last_char = value[-1].upper()
            if last_char in units:
                num = float(value[:-1]) * units[last_char]
                return super(LargeInt, cls).__new__(cls, int(num))
            else:
                return super(LargeInt, cls).__new__(cls, int(value))
        else:
            return super(LargeInt, cls).__new__(cls, value)

    def __str__(self):
        value = int(self)
        if abs(value) < 1000:
            return f"{value}"
        for unit in ["", "K", "M", "B", "T"]:
            if abs(value) < 1000:
                return f"{value:.1f}{unit}"
            value /= 1000
        return f"{value:.1f}P"  # P stands for Peta, or 10^15

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, int):
            return LargeInt(super().__add__(other))
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)  # This ensures commutativity
