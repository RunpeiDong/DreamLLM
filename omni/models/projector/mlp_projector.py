import json
import os

import torch
import torch.nn as nn

from omni.models.projector.base_projector import BaseProjector
from omni.utils.loguru import logger


class LinearProjector(BaseProjector):
    def __init__(self, args, in_hidden_size, out_hidden_size, bias=True):
        super().__init__()

        self.freeze_projector = args.freeze_projector
        self.depth = args.depth
        assert self.depth == 1, "LinearProjector now only supports depth=1"
        assert bias is not None, "bias should be set as True or False"
        self.projector = nn.Linear(in_hidden_size, out_hidden_size, bias=bias)
        if args.model_name_or_path is not None:
            self.load_weights(args.model_name_or_path)

    def forward(self, features):
        if not isinstance(features, list):
            features = [features]
        with torch.set_grad_enabled(not self.freeze_projector):
            return [self.projector(feature) for feature in features]


class MLPProjector(BaseProjector):
    def __init__(self, args, in_hidden_size, out_hidden_size, bias=False):
        super().__init__()

        self.freeze_projector = args.freeze_projector
        self.depth = args.depth
        assert self.depth > 1, "MLPProjector now only supports depth > 1, use linear if depth is 1"
        assert bias is not None, "bias should be set as True or False"
        modules = [nn.Linear(in_hidden_size, out_hidden_size, bias=bias)]
        for _ in range(1, self.depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_hidden_size, out_hidden_size, bias=bias))
        self.projector = nn.Sequential(*modules)
        if args.model_name_or_path is not None:
            self.load_weights(args.model_name_or_path)

    def forward(self, features):
        if not isinstance(features, list):
            features = [features]
        with torch.set_grad_enabled(not self.freeze_projector):
            return [self.projector(feature) for feature in features]
