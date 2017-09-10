DATASET_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/tf_records/plate
TRAIN_DIR=/home/thn2079/train_logs_plate_June14
python training_plate.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=stamp_plate \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v2_50_multiple