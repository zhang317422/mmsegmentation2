# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from .base import BaseSegmentor


@MODELS.register_module()
class CrossModalEncoderDecoder(BaseSegmentor):
    """Encoder-Decoder architecture with cross-modal residual fusion."""

    def __init__(self,
                 rgb_backbone: ConfigType,
                 disp_backbone: ConfigType,
                 fusion_module: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.rgb_backbone = MODELS.build(rgb_backbone)
        self.disp_backbone = MODELS.build(disp_backbone)
        self.fusion_module = MODELS.build(fusion_module)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        rgb_inputs = inputs['rgb']
        disp_inputs = inputs['disp']

        rgb_feats = self.rgb_backbone(rgb_inputs)
        disp_feats = self.disp_backbone(disp_inputs)

        if isinstance(rgb_feats, tuple):
            rgb_feats = list(rgb_feats)
        if isinstance(disp_feats, tuple):
            disp_feats = list(disp_feats)

        fused_feats = self.fusion_module(rgb_feats, disp_feats)
        if self.with_neck:
            fused_feats = self.neck(fused_feats)
        return list(fused_feats)

    def encode_decode(self, inputs: Dict[str, torch.Tensor],
                      batch_img_metas: List[dict]) -> torch.Tensor:
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs: List[torch.Tensor],
                                   data_samples: SampleList) -> dict:
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[torch.Tensor],
                                      data_samples: SampleList) -> dict:
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def loss(self, inputs: Dict[str, torch.Tensor],
             data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs)

        losses = self._decode_head_forward_train(x, data_samples)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            rgb = inputs['rgb']
            batch_img_metas = [
                dict(
                    ori_shape=rgb.shape[2:],
                    img_shape=rgb.shape[2:],
                    pad_shape=rgb.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * rgb.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Dict[str, torch.Tensor],
                 data_samples: OptSampleList = None) -> torch.Tensor:
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def inference(self, inputs: Dict[str, torch.Tensor],
                  batch_img_metas: List[dict]) -> torch.Tensor:
        if self.test_cfg is None:
            self.test_cfg = dict(mode='whole')
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole']
        if self.test_cfg['mode'] == 'slide':
            seg_logits = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logits = self.whole_inference(inputs, batch_img_metas)
        return seg_logits

    def whole_inference(self, inputs: Dict[str, torch.Tensor],
                        batch_img_metas: List[dict]) -> torch.Tensor:
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs: Dict[str, torch.Tensor],
                        batch_img_metas: List[dict]) -> torch.Tensor:
        rgb = inputs['rgb']
        disp = inputs['disp']

        h_stride, w_stride = self.test_cfg['stride']
        h_crop, w_crop = self.test_cfg['crop_size']
        batch_size, _, h_img, w_img = rgb.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = rgb.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = rgb.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_rgb = rgb[:, :, y1:y2, x1:x2]
                crop_disp = disp[:, :, y1:y2, x1:x2]

                crop_inputs = dict(rgb=crop_rgb, disp=crop_disp)

                batch_img_metas[0]['img_shape'] = crop_rgb.shape[2:]
                crop_seg_logit = self.encode_decode(crop_inputs, batch_img_metas)

                preds[:, :, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds
