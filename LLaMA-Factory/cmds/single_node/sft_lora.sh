#!/bin/bash

export HF_DATASETS_CACHE="/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data/cache"

TARGET_MODEL=${1}
TEMPLATE=${2}
NAME=${3}
PACKING=${4}
GPUS_PER_NODE=${5}
TOTAL_BATCH_SIZE=128
# GPUS_PER_NODE=8     # 使用平台注入的GPU数量
N_NODE=1            # 使用平台注入的节点总数
PER_DEVICE_BATCH_SIZE=2
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * GPUS_PER_NODE * N_NODE)))

OUTPUT_DIR=/fs-computility/llmit_d/shared/zhuhe/sft_model/${TARGET_MODEL}-${NAME}

TRAIN_FULL_DIR="/fs-computility/llmit_d/shared/zhuhe/sft_model"
mkdir -p ${TRAIN_FULL_DIR}
mkdir -p ${TRAIN_FULL_DIR}/logs
output_file="${TRAIN_FULL_DIR}/${TARGET_MODEL}-${NAME}.yaml"
LOG_FILE=${TRAIN_FULL_DIR}/logs/${TARGET_MODEL}-${NAME}.log

cat <<EOL > $output_file
### model
model_name_or_path: /fs-computility/llmit_d/shared/models/${TARGET_MODEL}
# trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 16
lora_dropout: 0.05
lora_target: all
deepspeed: /fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/examples/deepspeed/ds_z3_config.json

### dataset
dataset: select_data
template: ${TEMPLATE}
cutoff_len: 2048
overwrite_cache: false
preprocessing_num_workers: 32

### output
output_dir: ${OUTPUT_DIR}
save_only_model: true
logging_steps: 1
save_strategy: epoch
plot_loss: true
overwrite_output_dir: false
report_to: wandb

### train
per_device_train_batch_size: ${PER_DEVICE_BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM_STEPS}
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
# disable_gradient_checkpointing: false
seed: 42
packing: ${PACKING}
EOL

llamafactory-cli train ${output_file}