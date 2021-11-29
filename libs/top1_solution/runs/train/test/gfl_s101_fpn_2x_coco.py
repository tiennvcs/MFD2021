dataset_type = 'CocoDataset'
data_root = '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/'
classes = ('embedded', 'isolated')
img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1600, 1440)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1583, 2048),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type='CocoDataset',
            ann_file=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Tr00/train_coco_sdk4.json',
            img_prefix=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Tr00/img/',
            classes=('embedded', 'isolated'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
                dict(type='RandomCrop', crop_size=(1600, 1440)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Tr01/train_coco_sdk4.json',
            img_prefix=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Tr01/img/',
            classes=('embedded', 'isolated'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
                dict(type='RandomCrop', crop_size=(1600, 1440)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Tr10/train_coco_sdk4.json',
            img_prefix=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Tr10/img/',
            classes=('embedded', 'isolated'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
                dict(type='RandomCrop', crop_size=(1600, 1440)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Va00/train_coco_sdk4.json',
            img_prefix=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Va00/img/',
            classes=('embedded', 'isolated'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
                dict(type='RandomCrop', crop_size=(1600, 1440)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Va01/train_coco_sdk4.json',
            img_prefix=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Va01/img/',
            classes=('embedded', 'isolated'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
                dict(type='RandomCrop', crop_size=(1600, 1440)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Ts00/train_coco_sdk4.json',
            img_prefix=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Ts00/img/',
            classes=('embedded', 'isolated'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
                dict(type='RandomCrop', crop_size=(1600, 1440)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Ts01/train_coco_sdk4.json',
            img_prefix=
            '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Ts01/img/',
            classes=('embedded', 'isolated'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
                dict(type='RandomCrop', crop_size=(1600, 1440)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])
    ],
    val=dict(
        type='CocoDataset',
        ann_file=
        '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Ts01/train_coco_sdk4.json',
        img_prefix=
        '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Ts01/img/',
        classes=('embedded', 'isolated'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1583, 2048),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Ts10/train_coco_sdk4.json',
        img_prefix=
        '/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/top1_solution/IBEM_dataset/Ts10/img/',
        classes=('embedded', 'isolated'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1583, 2048),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='GFL',
    pretrained='open-mmlab://resnest101',
    backbone=dict(
        type='ResNeSt',
        stem_channels=128,
        depth=101,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        with_cp=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=24,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)
optimizer = dict(type='Ranger', lr=0.001)
fp16 = dict(loss_scale='dynamic')
work_dir = './runs/train/test/'
gpu_ids = range(0, 1)
