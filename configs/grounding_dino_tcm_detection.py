# Grounding DINO fine-tuned as a 4-class acupoint DETECTOR (prompt = all 4 names),
# so training matches the COCO mAP evaluation. Trains directly on the original COCO json
# (no MDETR / ODVG files needed). Based on mmdet's grounding_dino_swin-t_finetune_8xb2_20e_cat.py.
#
# Train:  python tools/train.py /path/to/grounding_dino_tcm_detection.py
# Test:   python tools/test.py  /path/to/grounding_dino_tcm_detection.py work_dirs/<run>/best_*.pth

# --- edit these paths ---------------------------------------------------------
_base_ = '/home/suhel.khan/TCM/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
data_root = '/home/suhel.khan/TCM/dataset/new_data_with_captions/New_DATA_COCO/New_DATA_with_subject_level/'
train_ann, train_img = 'train/train_anno.json', 'train/'
val_ann, val_img = 'valid/valid_anno.json', 'valid/'
test_ann, test_img = 'test/test_anno.json', 'test/'
# ------------------------------------------------------------------------------

# MUST be in the same order as the categories in the json
class_name = ('yuji', 'laogong', 'zhongchong', 'shaofu')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (0, 128, 0), (0, 0, 230), (255, 165, 0)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann,
        data_prefix=dict(img=train_img)))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann,
        data_prefix=dict(img=val_img)))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=test_ann,
        data_prefix=dict(img=test_img)))

val_evaluator = dict(ann_file=data_root + val_ann, classwise=True)
test_evaluator = dict(ann_file=data_root + test_ann, classwise=True)

max_epoch = 20
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='coco/bbox_mAP_50', rule='greater'),
    logger=dict(type='LoggerHook', interval=50))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epoch, by_epoch=True, milestones=[15], gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1)
    }))

auto_scale_lr = dict(base_batch_size=16)
