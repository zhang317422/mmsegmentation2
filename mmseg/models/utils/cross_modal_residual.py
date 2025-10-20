# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class CrossModalResidualFusion(BaseModule):
    """Fuse multi-modal features through residual refinement.

    The module receives feature hierarchies from two modalities (e.g. RGB and
    disparity) and learns a residual correction that is added to the primary
    modality to obtain fused representations.

    Args:
        channels (Sequence[int]): Number of channels for each feature level.
        conv_cfg (dict, optional): Config dict for convolution layers.
            Defaults to None.
        norm_cfg (dict, optional): Config dict for normalization layers.
            Defaults to dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layers.
            Defaults to dict(type='ReLU').
    """

    def __init__(self,
                 channels: Sequence[int],
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        if not isinstance(channels, Iterable):
            raise TypeError('`channels` should be a sequence of integers.')
        self.channels = list(channels)
        self.fusion_convs = nn.ModuleList()
        for channel in self.channels:
            self.fusion_convs.append(
                nn.Sequential(
                    ConvModule(
                        channel * 2,
                        channel,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        channel,
                        channel,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)))

    def forward(self, rgb_feats: Sequence[torch.Tensor],
                disp_feats: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(rgb_feats) != len(disp_feats):
            raise AssertionError('RGB and disparity feature sequences must '
                                 'have the same length.')
        if len(rgb_feats) != len(self.fusion_convs):
            raise AssertionError('Feature hierarchy depth does not match the '
                                 'number of fusion blocks.')

        fused_feats = []
        for fusion, rgb_feat, disp_feat in zip(self.fusion_convs, rgb_feats,
                                               disp_feats):
            residual = fusion(torch.cat([rgb_feat, disp_feat], dim=1))
            fused_feats.append(rgb_feat + residual)
        return tuple(fused_feats)
