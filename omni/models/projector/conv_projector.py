import math

import torch
import torch.nn as nn

from omni.models.projector.base_projector import BaseProjector
from omni.utils.loguru import logger


class ConvProjector(BaseProjector):
    def __init__(self, args, in_hidden_size, out_hidden_size, conv_stride=1):
        super().__init__()

        self.conv_stride = conv_stride
        self.freeze_projector = args.freeze_projector
        self.depth = args.depth
        assert self.depth == 1, "ConvProjector now only supports depth=1"
        self.projector = nn.Conv2d(
            in_channels=in_hidden_size, out_channels=out_hidden_size, kernel_size=3, stride=conv_stride, padding=1
        )
        if args.model_name_or_path is not None:
            self.load_weights(args.model_name_or_path)

    def forward(self, features):
        if not isinstance(features, list):
            features = [features]
        with torch.set_grad_enabled(not self.freeze_projector):
            projected_features = []
            for feature in features:
                if len(feature.shape) == 1:
                    feature = feature.view(1, 1, -1).repeat(1, 256, 1)

                B, P, C = feature.shape
                HW = int(math.sqrt(P))

                # B, P, C -> B, C, P -> B, C, HW, HW
                feature = feature.permute(0, 2, 1).view(B, C, HW, HW)
                projected_feature = self.projector(feature)
                projected_feature = projected_feature.view(B, -1, P // (self.conv_stride**2)).permute(0, 2, 1)

                projected_features.append(projected_feature)
            return projected_features
