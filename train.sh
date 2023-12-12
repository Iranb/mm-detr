GPU_NUM=4
# CONFIG_FILE=./config/deqd_detr.py
CONFIG_FILE=./config/deqd_detr_50e.py
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH="/home/disk3/hyq/newcode/QueryDistillationDETR/models":$PYTHONPATH
# python tools/train.py \
#     ${CONFIG_FILE} \
#     --work-dir ./log/test --cfg-options randomness.seed=850404816
#     # ${CHECKPOINT_FILE} \
    # --out ${RESULT_FILE} \

bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --work-dir ./log/decoupled_rollback_1_iou_fusion_50e --cfg-options randomness.seed=850404816 #--resume