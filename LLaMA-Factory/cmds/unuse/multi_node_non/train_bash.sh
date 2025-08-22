#!/bin/bash

# 多节点分布式训练脚本 - 适用于火山引擎机器学习平台
# 使用平台自动注入的环境变量进行分布式配置

if [ $# -lt 2 ]; then
  echo "用法: bash run_distributed_mlp.sh <数据集名称> <随机种子>"
  echo "例如: bash run_distributed_mlp.sh gsm8k 42"
  echo "数据集请在LLaMA-Factory/data/dataset_info.json 中注册"
  exit 1
fi

# 显示火山引擎机器学习平台注入的环境变量
echo "===== 火山引擎环境变量信息 ====="
echo "MLP_WORKER_0_HOST: ${MLP_WORKER_0_HOST}"
echo "MLP_WORKER_0_PORT: ${MLP_WORKER_0_PORT}"
echo "MLP_ROLE_INDEX: ${MLP_ROLE_INDEX}"
echo "MLP_WORKER_NUM: ${MLP_WORKER_NUM}"
echo "MLP_WORKER_GPU: ${MLP_WORKER_GPU}"
echo "========================="

# 基本参数
export HF_DATASETS_CACHE="/fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/data/cache"
START_TIME=`date +%Y%m%d-%H:%M:%S`
DATA_SETTING=$1
SEED=$2
NODE_RANK=${MLP_ROLE_INDEX}         # 使用平台注入的节点编号
EXP_NAME=qwen_${DATA_SETTING}_distributed
PARITION=belt_road
QUOTA_TYPE=reserved # auto | spot | reserved
GPUS_PER_NODE=${MLP_WORKER_GPU}     # 使用平台注入的GPU数量
N_NODE=${MLP_WORKER_NUM}            # 使用平台注入的节点总数
DEEPSPEED_LEVEL=z3
CUTOFF_LEN=32000
TEMPLATE=default

# 分布式训练参数 - 使用火山引擎平台注入的环境变量
MASTER_ADDR="${MLP_WORKER_0_HOST}"  # 使用worker0的地址作为主节点
MASTER_PORT="${MLP_WORKER_0_PORT}"  # 使用worker0的端口

# 创建目录（如果不存在）
TRAIN_FULL_DIR="/fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/examples/train_full"
mkdir -p ${TRAIN_FULL_DIR}

# 配置文件路径
output_file="${TRAIN_FULL_DIR}/qwen_3_full_sft-${DATA_SETTING}_${START_TIME}.yaml"
LOG_DIR="/fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/logs/distributed"
LOG_FILE=${LOG_DIR}/${START_TIME}_${EXP_NAME}_${SEED}_node${NODE_RANK}.log

# 确保日志目录存在
mkdir -p ${LOG_DIR}

echo "正在创建配置文件: $output_file"
echo "节点等级: ${NODE_RANK} / ${N_NODE}"
echo "主节点地址: ${MASTER_ADDR}:${MASTER_PORT}"

# 写入配置文件
cat <<EOL > $output_file
### model
model_name_or_path: /fs-computility/llmit_d/shared/models/Qwen3-8B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: /fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/examples/deepspeed/ds_${DEEPSPEED_LEVEL}_offload_config.json

### dataset
dataset: ${DATA_SETTING}
template: ${TEMPLATE}
cutoff_len: ${CUTOFF_LEN}
overwrite_cache: false
preprocessing_num_workers: 32

### output
output_dir: /fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/saves/qwen3-0.6b/full/sft-${DATA_SETTING}/${START_TIME}_NNODE${N_NODE}_SEED${SEED}
save_only_model: true
logging_steps: 1
save_steps: 5 
plot_loss: true
overwrite_output_dir: true
report_to: wandb

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
seed: ${SEED}
packing: true
EOL

echo "YAML 配置文件已成功创建：$output_file"

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
echo "日志文件: ${LOG_FILE}"
echo "======================================================"

# 运行分布式训练命令 - 使用火山引擎环境变量
echo "开始训练..."
FORCE_TORCHRUN=1 NNODES=${N_NODE} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} llamafactory-cli train ${output_file} 2>&1 | tee ${LOG_FILE}

echo "节点 ${NODE_RANK} 训练完成。查看日志: ${LOG_FILE}"