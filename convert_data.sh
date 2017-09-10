# DATA_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/row_column
# OUTPUT_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/tf_records/row_column

# DATA_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/plate
# OUTPUT_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/tf_records/plate

# python download_and_convert_data.py \
    # --dataset_name=stamp_plate \
    # --dataset_dir="${DATA_DIR}" \
    # --output_dir="${OUTPUT_DIR}"
DATA_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/plate
OUTPUT_DIR=/shared/kgcoe-research/mil/stamp_stamp/data/tf_records/plate_150_classes

python download_and_convert_data.py \
    --dataset_name=stamp_plate_151 \
    --dataset_dir="${DATA_DIR}" \
    --output_dir="${OUTPUT_DIR}"