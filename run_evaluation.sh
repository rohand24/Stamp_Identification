

# CHECKPOINT_DIR=/home/thn2079/train_logs
# # CHECKPOINT_FILE=${CHECKPOINT_DIR}/model.ckpt-15090 # Example
# CHECKPOINT_FILE=${CHECKPOINT_DIR}/model.ckpt-14754 # Example
# DATASET_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/tf_records/row_column
# python eval_image_classifier.py \
    # --alsologtostderr \
    # --checkpoint_path=${CHECKPOINT_FILE} \
    # --dataset_dir=${DATASET_DIR} \
    # --dataset_name=stamp \
    # --dataset_split_name=validation \
    # --model_name=resnet_v2_50

CHECKPOINT_DIR=/home/thn2079/train_logs_plate_June14
# CHECKPOINT_FILE=${CHECKPOINT_DIR}/model.ckpt-15090 # Example
CHECKPOINT_FILE=${CHECKPOINT_DIR}/model.ckpt-57019 # Example
DATASET_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/tf_records/plate
EVAL_DIR=/home/thn2079/git/stamp_project/save_evaluation_results_plate_June14/
python inference_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=stamp_plate \
    --dataset_split_name=validation \
    --model_name=resnet_v2_50_multiple