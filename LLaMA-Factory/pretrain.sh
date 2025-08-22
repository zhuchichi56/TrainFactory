#!/bin/bash

START_TIME=`date +%Y%m%d-%H:%M:%S`
export HF_DATASETS_CACHE="/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data/cache"

### 基本参数设置
# DATA_SETTING=${1:-"slimpajama_subset"}
# DATA_SETTING=${1:-"c4_demo"}
DATA_SETTING=${1:-"industrycn"}
SEED=${2:-42}
GPUS_PER_NODE=8     
TARGET_MODEL=Qwen2.5-7B-Instruct
TOTAL_BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=1
CUTOFF_LEN=4096
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * GPUS_PER_NODE)))
LEARNING_RATE=1.0e-4
OUTPUT_DIR=/fs-computility/llmit_d/shared/zhuhe/trained_model/saves/${TARGET_MODEL}/pretrain-full
# 计算MAX_STEPS
DATASET_SIZE=$(wc -l < "/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data/merged_sample.jsonl")
echo "DATASET_SIZE=$DATASET_SIZE"
MAX_STEPS=$((DATASET_SIZE * NUM_TRAIN_EPOCHS / TOTAL_BATCH_SIZE))
echo "MAX_STEPS=$MAX_STEPS"


echo "总批次大小: ${TOTAL_BATCH_SIZE}"
echo "梯度累积步数: ${GRAD_ACCUM_STEPS}"

TRAIN_FULL_DIR="/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/pretrain"
mkdir -p ${TRAIN_FULL_DIR}
mkdir -p ${TRAIN_FULL_DIR}/logs
mkdir -p ${TRAIN_FULL_DIR}/saves

# 配置文件和日志文件路径
output_file="${TRAIN_FULL_DIR}/${TARGET_MODEL}_pretrain_${START_TIME}.yaml"
LOG_FILE=${TRAIN_FULL_DIR}/logs/${START_TIME}_pretrain_${SEED}.log

echo "正在创建配置文件: $output_file"

cat <<EOL > $output_file
### model
model_name_or_path: /fs-computility/llmit_d/shared/models/${TARGET_MODEL}

### method
stage: pt
do_train: true
finetuning_type: full

### dataset
dataset: ${DATA_SETTING}
cutoff_len: 4096
# max_samples: 1000
# streaming: true
preprocessing_batch_size: 10000
buffer_size: 10000
max_steps: ${MAX_STEPS}
overwrite_cache: true
preprocessing_num_workers: 1
deepspeed: /fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/examples/deepspeed/ds_z3_config.json

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 10
save_steps: 500
plot_loss: true
report_to: wandb
overwrite_output_dir: true

### train
per_device_train_batch_size: ${PER_DEVICE_BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM_STEPS}
learning_rate: ${LEARNING_RATE}

num_train_epochs: ${NUM_TRAIN_EPOCHS}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
seed: ${SEED}
EOL

echo "YAML CONFIG FILE CREATED: $output_file"

# 确保配置文件已创建
if [ ! -f "$output_file" ]; then
    echo "错误：配置文件未能成功创建: $output_file"
    exit 1
fi

# 打印训练信息
echo "======================================================"
echo "启动训练:"
echo "种子: ${SEED}"
echo "GPU数: ${GPUS_PER_NODE}"
echo "总批次大小: ${TOTAL_BATCH_SIZE}"
echo "日志文件: ${LOG_FILE}"
echo "======================================================"

# 运行训练命令
echo "开始训练..."
llamafactory-cli train ${output_file} 2>&1 | tee ${LOG_FILE}

echo "训练完成。查看日志: ${LOG_FILE}"

