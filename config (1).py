train_root = '/content/dataset/img_dir/ISIC2018_Task3_Training_Input'
val_root = '/content/dataset/img_dir/ISIC2018_Task3_Validation_Input'
test_root = '/content/dataset/img_dir/ISIC2018_Task3_Test_Input'

train_csv = '/content/train.txt'
val_csv = '/content/val.txt'
test_csv = '/content/test.txt'
_base_ = [
    '/content/mmsegmentation/mmpretrain/configs/_base_/models/vgg11.py',
    '/content/mmsegmentation/mmpretrain/configs/_base_/datasets/imagenet_bs32_pil_resize.py',
    '/content/mmsegmentation/mmpretrain/configs/_base_/schedules/imagenet_bs256.py',
    '/content/mmsegmentation/mmpretrain/configs/_base_/default_runtime.py',
]

# model settings
model = dict(
        backbone=dict(type='VGG', depth=11, num_classes=7),)
# dataset settings
dataset_type = 'ImageNet'
data_preprocessor = dict(
    # RGB format normalization parameters
   /
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(
            type=dataset_type,
            data_prefix=train_root,
            ann_file=train_csv,
            classes=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
            pipeline=train_pipeline,
            ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
            type=dataset_type,
            data_prefix=val_root,
            ann_file=val_csv,
            classes=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
            pipeline=test_pipeline,
            ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
            type=dataset_type,
            data_prefix=test_root,
            ann_file=test_csv,
            classes=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
            pipeline=test_pipeline,
            ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=1)
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=0, deterministic=False)

# If you want standard test, please manually configure the test dataset
test_evaluator = val_evaluator
# schedule settings
optim_wrapper = dict(optimizer=dict(lr=0.01))