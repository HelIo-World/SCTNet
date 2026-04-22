_base_ = ['../_base_/datasets/mars-seg.py']

# model
checkpoint_backbone = 'pretrain/SCT-B_Pretrain.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 8
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SCTNet',
        init_cfg=dict(
            type='Pretrained',
            checkpoint= checkpoint_backbone
        ),
        base_channels=64,
        spp_channels=128),
    decode_head=dict(
        type='SCTHead',
        in_channels=256,
        channels=256,
        dropout_ratio=0.0,
        in_index=0,
        num_classes=num_classes,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



# optimizer
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            teacher_backbone=dict(lr_mult=0.0),
            teacher_head=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
find_unused_parameters = True
auto_resume = False
seed = 1440161127