# source activate llama_factory
# sleep 2

if [ $# -eq 0 ]; then
  echo "请在命令行中提供训练集名称，例如：gsm8k ,数据集请在LLaMA-Factory/data/dataset_info.json 中注册"
  exit 1
fi

START_TIME=`date +%Y%m%d-%H:%M:%S`
DATA_SETTING=$1
SEED=$2
EXP_NAME=llama_${DATA_SETTING}
PARITION=belt_road
QUOTA_TYPE=reserved # auto | spot | reserved
GPUS_PER_NODE=8
N_NODE=1

# 创建目录（如果不存在）
TRAIN_FULL_DIR="/fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/examples/train_full"
mkdir -p ${TRAIN_FULL_DIR}

# 使用完整路径
output_file="${TRAIN_FULL_DIR}/llama_3_full_sft-${DATA_SETTING}_${START_TIME}.yaml"
LOG_FILE=/fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/logs/${START_TIME}_${EXP_NAME}_${SEED}.log

# 确保日志目录存在
mkdir -p $(dirname ${LOG_FILE})

# 写入内容
cat <<EOL > $output_file
### model
model_name_or_path: /fs-computility/llmit_d/shared/models/Llama-3.2-3B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: /fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/examples/deepspeed/ds_z2_config.json

### dataset
dataset: ${DATA_SETTING}
template: default
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/saves/llama3.2-3b/full/sft-${EXEP_NAME}/seed-${SEED}
save_only_model: true
logging_steps: 10
save_steps: 999999
### save_strategy: epoch
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000
seed: ${SEED}
EOL

echo "YAML 配置文件已成功创建：$output_file"

# srun -p ${PARITION} --job-name=${EXP_NAME} --quotatype=${QUOTA_TYPE} \
#     --gres=gpu:${GPUS_PER_NODE}  --cpus-per-task=16 --kill-on-bad-exit=1 \
#     --exclude=SH-IDC1-10-140-24-[114,96,51,128] \
#     --async -o ${LOG_FILE} llamafactory-cli train ${output_file} 
llamafactory-cli train ${output_file} 