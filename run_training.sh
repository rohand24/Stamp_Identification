DATASET_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/tf_records/row_column
TRAIN_DIR=/home/thn2079/train_logs
python training.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=stamp \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v2_50