_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='RGBDSegDataPreProcessor',
    rgb_mean=[123.675, 116.28, 103.53],
    rgb_std=[58.395, 57.12, 57.375],
    disp_mean=[0.0],
    disp_std=[1.0],
    bgr_to_rgb=True,
    size_divisor=32,
    rgb_pad_val=0,
    disp_pad_val=0,
    seg_pad_val=255)
model = dict(
    type='CrossModalEncoderDecoder',
    data_preprocessor=data_preprocessor,
    rgb_backbone=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')),
    disp_backbone=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=1,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=None),
    fusion_module=dict(
        type='CrossModalResidualFusion',
        channels=[256, 512, 1024, 2048],
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU')),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'CityscapesRGBDDataset'
data_root = 'data/cityscapes/'
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDisparityFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackRGBDInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDisparityFromFile', to_float32=True),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackRGBDInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDisparityFromFile', to_float32=True),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackRGBDInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            img_path2='disparity/train',
            seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            img_path2='disparity/val',
            seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
