# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import SampleList, stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        if batch_augments is not None:
            raise NotImplementedError(
                'Batch augmentations are not supported for RGB-D inputs yet.')
        self.batch_augments = None

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)


@MODELS.register_module()
class RGBDSegDataPreProcessor(BaseDataPreprocessor):
    """Pre-processor for RGB-D semantic segmentation tasks.

    This data pre-processor extends :class:`SegDataPreProcessor` to handle
    paired RGB images and disparity/depth maps. It normalizes and pads each
    modality independently before stacking them into batched tensors while
    keeping the accompanying :class:`SegDataSample` instances in sync.

    Args:
        rgb_mean (Sequence[Number], optional): Pixel mean of the RGB modality.
            Defaults to None.
        rgb_std (Sequence[Number], optional): Pixel standard deviation of the
            RGB modality. Defaults to None.
        disp_mean (Sequence[Number], optional): Pixel mean of the disparity
            modality. Defaults to None.
        disp_std (Sequence[Number], optional): Pixel standard deviation of the
            disparity modality. Defaults to None.
        size (tuple, optional): Fixed padding size. Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        rgb_pad_val (Number): Padding value for RGB images. Defaults to 0.
        disp_pad_val (Number): Padding value for disparity images. Defaults to
            0.
        seg_pad_val (Number): Padding value for segmentation maps. Defaults to
            255.
        bgr_to_rgb (bool): Whether to convert RGB images from BGR to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations.
            Defaults to None.
        test_cfg (dict, optional): Padding config used during testing. Defaults
            to None.
    """

    def __init__(
        self,
        rgb_mean: Sequence[Number] = None,
        rgb_std: Sequence[Number] = None,
        disp_mean: Sequence[Number] = None,
        disp_std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        rgb_pad_val: Number = 0,
        disp_pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.rgb_pad_val = rgb_pad_val
        self.disp_pad_val = disp_pad_val
        self.seg_pad_val = seg_pad_val

        self.rgb_channel_conversion = bgr_to_rgb

        if rgb_mean is not None:
            assert rgb_std is not None, (
                'To enable normalization for RGB inputs, please provide both '
                '`rgb_mean` and `rgb_std`.')
            self._enable_rgb_normalize = True
            self.register_buffer('rgb_mean',
                                 torch.tensor(rgb_mean).view(-1, 1, 1), False)
            self.register_buffer('rgb_std',
                                 torch.tensor(rgb_std).view(-1, 1, 1), False)
        else:
            self._enable_rgb_normalize = False

        if disp_mean is not None:
            assert disp_std is not None, (
                'To enable normalization for disparity inputs, please provide '
                'both `disp_mean` and `disp_std`.')
            self._enable_disp_normalize = True
            self.register_buffer(
                'disp_mean', torch.tensor(disp_mean).view(-1, 1, 1), False)
            self.register_buffer(
                'disp_std', torch.tensor(disp_std).view(-1, 1, 1), False)
        else:
            self._enable_disp_normalize = False

        self.batch_augments = batch_augments
        self.test_cfg = test_cfg

    def _stack_inputs(self, inputs: List[torch.Tensor], *, pad_val: Number,
                      data_samples: Optional[SampleList] = None):
        if self.size is not None or self.size_divisor is not None:
            return stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=pad_val,
                seg_pad_val=self.seg_pad_val)
        stacked = torch.stack(inputs, dim=0)
        return stacked, data_samples

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        assert isinstance(inputs, dict) and 'rgb' in inputs and 'disp' in inputs, \
            'RGBDSegDataPreProcessor expects dict inputs with ``rgb`` and ``disp``.'

        rgb_list = inputs['rgb']
        disp_list = inputs['disp']
        data_samples = data.get('data_samples', None)

        if self.rgb_channel_conversion and rgb_list[0].size(0) == 3:
            rgb_list = [_input[[2, 1, 0], ...] for _input in rgb_list]

        rgb_list = [_input.float() for _input in rgb_list]
        disp_list = [_input.float() for _input in disp_list]

        if self._enable_rgb_normalize:
            rgb_list = [(_input - self.rgb_mean) / self.rgb_std
                        for _input in rgb_list]
        if self._enable_disp_normalize:
            disp_list = [(_input - self.disp_mean) / self.disp_std
                         for _input in disp_list]

        if training:
            assert data_samples is not None, (
                'During training, `data_samples` must be provided for padding.')
            rgb_batch, data_samples = self._stack_inputs(
                rgb_list, pad_val=self.rgb_pad_val, data_samples=data_samples)
            disp_batch, _ = self._stack_inputs(
                disp_list, pad_val=self.disp_pad_val, data_samples=None)

        else:
            if self.test_cfg:
                rgb_batch, padded_samples = stack_batch(
                    inputs=rgb_list,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.rgb_pad_val,
                    seg_pad_val=self.seg_pad_val)
                disp_batch, _ = stack_batch(
                    inputs=disp_list,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.disp_pad_val,
                    seg_pad_val=self.seg_pad_val)
                if data_samples is not None:
                    for data_sample, pad_info in zip(data_samples,
                                                     padded_samples):
                        data_sample.set_metainfo({**pad_info})
            else:
                rgb_batch = torch.stack(rgb_list, dim=0)
                disp_batch = torch.stack(disp_list, dim=0)

        batched_inputs = dict(rgb=rgb_batch, disp=disp_batch)
        return dict(inputs=batched_inputs, data_samples=data_samples)
