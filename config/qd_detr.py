_base_ = ["mmdet::conditional_detr/conditional-detr_r50_8xb2-50e_coco.py"]

custom_imports = dict(imports=["models"], allow_failed_imports=False)

model = dict(
    type="QueryDsitillationDETR",
    bbox_head=dict(
        type="QueryDistallationDETRHead",
    ),
    # training and testing settings
)

# learning policy
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
param_scheduler = [dict(type="MultiStepLR", end=12, milestones=[11])]

train_dataloader = dict(batch_size=4)  # 4 gpu total batchsize = 16
