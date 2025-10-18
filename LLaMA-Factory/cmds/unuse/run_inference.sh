#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_size> [data_setting] [seed]"
    echo "Example: $0 7b html 3407"
    exit 1
fi

# 参数设置
MODEL_SIZE=$1
DATA_SETTING=${2:-html}  # 如果没有提供第二个参数，默认使用 "html"
SEED=${3:-3407}         # 如果没有提供第三个参数，默认使用 3407

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/fs-computility/llmit_d/shared/zhuangxinlin/html-alg-lib:$PYTHONPATH

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 运行推理脚本，确保传递所有必要参数
python /fs-computility/llmit_d/shared/liumengjie/SFT/inference_sft.py \
    --model_size "$MODEL_SIZE" \
    --data_setting "$DATA_SETTING" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/inference_${MODEL_SIZE}_${DATA_SETTING}_${TIMESTAMP}.log"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo "Inference completed successfully!"
else
    echo "Inference failed! Check logs for details."
    exit 1
fi
