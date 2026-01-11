#!/bin/bash
# scripts/pipeline.sh

DATASET_NAME=$1
OUTPUT_DIR=$2
SEED=${3:-3}  # 默认seed为3

echo "Running pipeline for dataset: $DATASET_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Seed: $SEED"

# 检查参数
if [ -z "$DATASET_NAME" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash scripts/pipeline.sh <dataset_name> <output_dir> [seed]"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "========== 1. Training Phase =========="
python train.py \
    --config-file configs/${DATASET_NAME}.yaml \
    --output-dir $OUTPUT_DIR \
    --seed $SEED

if [ $? -ne 0 ]; then
    echo "Training phase failed!"
    exit 1
fi

echo "Training phase completed!"

echo "========== 2. Inference Phase =========="
python inference.py \
    --config-file configs/${DATASET_NAME}.yaml \
    --output-dir $OUTPUT_DIR \
    --seed $SEED

if [ $? -ne 0 ]; then
    echo "Inference phase failed!"
    exit 1
fi

echo "Inference phase completed!"

echo "========== 3. Evaluation Phase =========="
python evaluation/eval.py \
    --config-file configs/${DATASET_NAME}.yaml \
    --output-dir $OUTPUT_DIR \
    --seed $SEED

if [ $? -ne 0 ]; then
    echo "Evaluation phase failed!"
    exit 1
fi

echo "Evaluation phase completed!"

echo "========== Pipeline Complete! =========="
echo "Results saved in: $OUTPUT_DIR/$DATASET_NAME/"
echo "Metrics saved in: $OUTPUT_DIR/$DATASET_NAME/trained_models/seed${SEED}/"
