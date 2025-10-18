START_TIME=`date +%Y%m%d-%H:%M:%S`

export HF_DATASETS_CACHE="/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data/cache"

### YOU SHOULD MODIFY THE FOLLOWING PARAMETERS 
DATA_SETTING=html
SEED=2
EXP_NAME=qwen_${DATA_SETTING}
PARITION=belt_road
QUOTA_TYPE=reserved # auto | spot | reserved
GPUS_PER_NODE=8
N_NODE=1
DEEPSPEED_LEVEL=z3_offload
SEQ_PARALLEL_SIZE=8
CUTOFF_LEN=128000
TEMPLATE=qwen3
TRAIN_FULL_DIR="/fs-computility/llmit_d/shared/zhuhe/trained_model"
output_file="${TRAIN_FULL_DIR}/qwen_3_full_sft-${DATA_SETTING}_${START_TIME}.yaml"
LOG_FILE=/fs-computility/llmit_d/shared/zhuhe/trained_model/logs/${START_TIME}_${EXP_NAME}_${SEED}.log



#WARNING: YOU SHOULD NOT MODIFY THE FOLLOWING PARAMETERS !!!!!!
# 你不能修改以下参数
cat <<EOL > $output_file
### model
model_name_or_path: /fs-computility/llmit_d/shared/models/Qwen3-8B
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
sequence_parallel_size: ${SEQ_PARALLEL_SIZE}
overwrite_cache: false
preprocessing_num_workers: 32


### output
output_dir: /fs-computility/llmit_d/shared/zhuhe/trained_model/saves/qwen3-8b/full/sft-${DATA_SETTING}/${START_TIME}_CUTOFF_LEN${CUTOFF_LEN}_NNODE${N_NODE}_SEED${SEED}
save_only_model: true
logging_steps: 1
# save_strategy: epoch
save_steps: 5 
plot_loss: true
overwrite_output_dir: true
report_to: wandb



### train
# per_device_train_batch_size: 4
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
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

# 运行训练命令
llamafactory-cli train ${output_file}



