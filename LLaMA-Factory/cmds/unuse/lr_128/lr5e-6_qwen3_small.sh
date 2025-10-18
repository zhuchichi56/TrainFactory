#!/bin/bash

if [ $# -lt 2 ]; then
  echo "用法: bash train_bash.sh <数据集名称> <随机种子>"
  echo "例如: bash train_bash.sh html 42"
  echo "数据集请在LLaMA-Factory/data/dataset_info.json 中注册"
  exit 1
fi


echo "===== 火山引擎环境变量信息 ====="
echo "MLP_WORKER_0_HOST: ${MLP_WORKER_0_HOST}"
echo "MLP_WORKER_0_PORT: ${MLP_WORKER_0_PORT}"
echo "MLP_ROLE_INDEX: ${MLP_ROLE_INDEX}"
echo "MLP_WORKER_NUM: ${MLP_WORKER_NUM}"
echo "MLP_WORKER_GPU: ${MLP_WORKER_GPU}"
echo "========================="

START_TIME=`date +%Y%m%d-%H:%M:%S`

export HF_DATASETS_CACHE="/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data/cache"

### 基于 new.sh 的参数设置
DATA_SETTING=$1
SEED=$2
NODE_RANK=${MLP_ROLE_INDEX}         # 使用平台注入的节点编号
EXP_NAME=qwen_${DATA_SETTING}
PARITION=belt_road
QUOTA_TYPE=reserved # auto | spot | reserved
GPUS_PER_NODE=${MLP_WORKER_GPU}     # 使用平台注入的GPU数量
N_NODE=${MLP_WORKER_NUM}            # 使用平台注入的节点总数
DEEPSPEED_LEVEL=z3_offload          # 与 new.sh 保持一致
SEQ_PARALLEL_SIZE=8                 # 从 new.sh 添加
CUTOFF_LEN=32000                   # 与 new.sh 保持一致
TEMPLATE=qwen3                      # 与 new.sh 保持一致
TARGET_MODEL=Qwen3-0.6B             # 与 new.sh 保持一致
TOTAL_BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * GPUS_PER_NODE * N_NODE)))
LEARNING_RATE=5.0e-6
OUTPUT_DIR=/fs-computility/llmit_d/shared/zhuhe/trained_model/saves/${TARGET_MODEL}/full/sft-${DATA_SETTING}/${TARGET_MODEL}_${START_TIME}_CUTOFF_LEN${CUTOFF_LEN}_LR${LEARNING_RATE}


echo "总批次大小: ${TOTAL_BATCH_SIZE}"
echo "梯度累积步数: ${GRAD_ACCUM_STEPS}"

# 分布式训练参数 - 使用火山引擎平台注入的环境变量
MASTER_ADDR="${MLP_WORKER_0_HOST}"  # 使用worker0的地址作为主节点
MASTER_PORT="${MLP_WORKER_0_PORT}"  # 使用worker0的端口

TRAIN_FULL_DIR="/fs-computility/llmit_d/shared/zhuhe/trained_model"
mkdir -p ${TRAIN_FULL_DIR}
mkdir -p ${TRAIN_FULL_DIR}/logs

# 配置文件和日志文件路径
output_file="${TRAIN_FULL_DIR}/${TARGET_MODEL}_full_sft-${DATA_SETTING}_${START_TIME}.yaml"
LOG_FILE=${TRAIN_FULL_DIR}/logs/${START_TIME}_${EXP_NAME}_${SEED}_node${NODE_RANK}.log

echo "正在创建配置文件: $output_file"
echo "节点等级: ${NODE_RANK} / ${N_NODE}"
echo "主节点地址: ${MASTER_ADDR}:${MASTER_PORT}"



#WARNING: YOU SHOULD NOT MODIFY THE FOLLOWING PARAMETERS !!!!!!
# 你不能修改以下参数
cat <<EOL > $output_file
### model
model_name_or_path: /fs-computility/llmit_d/shared/models/${TARGET_MODEL}
# trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: /fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/examples/deepspeed/ds_${DEEPSPEED_LEVEL}_config.json

### dataset
dataset: ${DATA_SETTING}
template: ${TEMPLATE}
cutoff_len: ${CUTOFF_LEN}
# sequence_parallel_size: ${SEQ_PARALLEL_SIZE}
overwrite_cache: false
preprocessing_num_workers: 32

### output
output_dir: ${OUTPUT_DIR}
save_only_model: true
logging_steps: 1
save_strategy: epoch
# save_steps: 5 
plot_loss: true
overwrite_output_dir: false
report_to: wandb

### train
per_device_train_batch_size: ${PER_DEVICE_BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM_STEPS}
learning_rate: ${LEARNING_RATE}
num_train_epochs: 3.0 
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
# disable_gradient_checkpointing: false
seed: ${SEED}
packing: true
EOL

echo "YAML CONFIG FILE CREATED: $output_file"

# 确保配置文件已创建
if [ ! -f "$output_file" ]; then
    echo "错误：配置文件未能成功创建: $output_file"
    exit 1
fi

# 打印分布式训练信息
echo "======================================================"
echo "启动分布式训练:"
echo "数据集: ${DATA_SETTING}"
echo "种子: ${SEED}"
echo "节点数: ${N_NODE}"
echo "当前节点等级: ${NODE_RANK}"
echo "每节点GPU数: ${GPUS_PER_NODE}"
echo "总批次大小: ${TOTAL_BATCH_SIZE}"
echo "日志文件: ${LOG_FILE}"
echo "======================================================"

# 运行分布式训练命令 - 使用火山引擎环境变量
echo "开始训练..."
FORCE_TORCHRUN=1 NNODES=${N_NODE} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} llamafactory-cli train ${output_file} 2>&1 | tee ${LOG_FILE}

echo "节点 ${NODE_RANK} 训练完成。查看日志: ${LOG_FILE}"
