import math

import torch
import torch.nn as nn

from omni.utils.loguru import logger


class SAMProjector(nn.Module):
    def __init__(self, args, in_hidden_size=256, out_hidden_size=4096):
        super().__init__()

        self.freeze_projector = args.freeze_projector
        self.depth = args.depth
        assert self.depth == 2, "SAMProjector now only supports depth=2"

        self.projector = nn.Sequential(
            nn.Conv2d(in_channels=in_hidden_size, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.mlp = nn.Linear(1024, out_hidden_size, bias=False)

    def forward(self, features):
        if not isinstance(features, list):
            features = [features]
        with torch.set_grad_enabled(not self.freeze_projector):
            projected_features = []
            for feature in features:
                B, C, H, W = feature.shape
                projected_feature = self.projector(feature)
                projected_feature = projected_feature.view(B, -1, H * W // 4 // 4).permute(0, 2, 1)
                projected_feature = self.mlp(projected_feature)
                projected_features.append(projected_feature)
            return projected_features

    @property
    def dtype(self):
        return self.projector.dtype

    @property
    def device(self):
        return self.projector.device
