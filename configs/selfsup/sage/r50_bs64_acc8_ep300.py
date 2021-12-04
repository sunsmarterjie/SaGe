import copy

_base_ = '../../base.py'
# model settings
model = dict(
    type='SaGe_Net',
    pretrained=None,
    base_momentum=0.99,
    pre_conv=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[1, 2, 3, 4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        sync_bn=True,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckSimCLR',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, num_layers=2, sync_bn=True,
                             with_bias=True, with_last_bn=False, with_avg_pool=False)))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/imagenet/meta/train.txt'
data_train_root = 'data/imagenet/train'
dataset_type = 'SaGeDataset'
#img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)
train_pipeline2[4]['p'] = 0.1  # gaussian blur
train_pipeline2[5]['p'] = 0.2  # solarization

data = dict(
    imgs_per_gpu=64,  # total 32*8(gpu)*16(interval)=4096
    # imgs_per_gpu=8,  # total 64*4(gpu)*16(interval)=4096
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
    ))
# additional hooks
update_interval = 2  # interval for accumulate gradient
custom_hooks = [
    dict(type='SaGeHook', end_momentum=1., update_interval=update_interval)
]

decoder_lr_updater_hooks = dict(min_lr=0.0,
                                by_epoch=True,
                                warmup='linear',
                                warmup_iters=10,
                                warmup_ratio=0.0001,
                                warmup_by_epoch=True)
# optimizer
optimizer = dict(type='LARS', lr=4.8, weight_decay=0.000001, momentum=0.9,
                 paramwise_options={
                     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                     'bias': dict(weight_decay=0., lars_exclude=True)})
optimizer_decoder = dict(type='Adam', lr=0.048, weight_decay=0.000001, momentum=0.9,
                         paramwise_options={
                             '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                             'bias': dict(weight_decay=0., lars_exclude=True)})
# apex
use_fp16 = False
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)
optimizer_decoder_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001,  # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 300
