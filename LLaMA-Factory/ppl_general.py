#!/usr/bin/env python3

import subprocess
import json
import os
import shutil
import glob


gpus = [0,1,2,3]
os.environ["HF_ENDPOINT"] = "https://huggingface.cn"
os.environ["WANDB_MODE"] = "offline"
# export FORCE_TORCHRUN=1


def load_jsonl(data_path):
    with open(data_path, "r") as f:
        return [json.loads(line) for line in f]


def create_yaml_config(target_model, template, info, packing, data_size):
    """创建训练配置yaml文件"""
    TOTAL_BATCH_SIZE = 128
    N_NODE = 1
    PER_DEVICE_BATCH_SIZE = 8
    GPUS_PER_NODE = len(gpus)
    GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // (PER_DEVICE_BATCH_SIZE * GPUS_PER_NODE * N_NODE)
    
    # 计算每0.5个epoch的步数
    steps_per_epoch = data_size // TOTAL_BATCH_SIZE
    save_steps = max(1, steps_per_epoch // 2)  # 每0.5个epoch保存
    
    model_name = os.path.basename(target_model)
    OUTPUT_DIR = f"/volume/pt-train/users/wzhang/ghchen/zh/saves/sft/{model_name}-{info}"
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
    
    output_file = f"{OUTPUT_DIR}/{model_name}-{info}.yaml"
    
    yaml_content = f"""### model
model_name_or_path: {target_model}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: select_data
template: {template}
cutoff_len: 2048
overwrite_cache: false
preprocessing_num_workers: 32

### output
output_dir: {OUTPUT_DIR}
save_only_model: true
logging_steps: 1
save_strategy: steps
save_steps: {save_steps}
plot_loss: true
overwrite_output_dir: false
report_to: wandb

### train
per_device_train_batch_size: {PER_DEVICE_BATCH_SIZE}
gradient_accumulation_steps: {GRAD_ACCUM_STEPS}
learning_rate: 2.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
seed: 42
packing: {packing}
"""
    
    # 写入yaml文件
    with open(output_file, 'w') as f:
        f.write(yaml_content)
    
    return output_file


def sft(target_model, template, data_path, info, mode="sft", packing=False):
    
    target_path = "./data/sft.jsonl"
    shutil.copyfile(data_path, target_path)
    data = load_jsonl(target_path)
    print(f"Copied data to {target_path}")
    print("Sample data:")
    print(data[:1])
    
    # 创建yaml配置文件
    packing_str = "false"  # 默认packing为false
    yaml_file = create_yaml_config(target_model, template, info, packing_str,  len(data))
    
    print(f"Created config file: {yaml_file}")
    print(f"Training model: {target_model} with data: {data_path}")
    
    # 执行训练 - 指定使用GPU 2,3,6,7
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    cmd = f"llamafactory-cli train {yaml_file}"
    print(f"Running command: {cmd}")
    print("=" * 50)
    
    try:
        # 使用 subprocess.Popen 来实时显示输出
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时读取并显示输出
        for line in process.stdout:
            print(line.rstrip())  # 移除行尾换行符，避免双重换行
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print("=" * 50)
            print("Training completed successfully!")
        else:
            print("=" * 50)
            print(f"Training failed with return code: {return_code}")
            raise subprocess.CalledProcessError(return_code, cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        raise
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        process.terminate()
        raise



def eval_model(model_path):
    
    print(f"评测: {model_path}")
    cmd = f"""bash -c " 
    export HF_ENDPOINT=https://huggingface.cn && \
    source /volume/pt-train/users/wzhang/ghchen/laip/miniconda3/etc/profile.d/conda.sh && conda activate lm-eval-harness && \
    accelerate launch -m --main_process_port=8189 --num_processes={len(gpus)} lm_eval \
        --model hf --model_args pretrained={model_path} --trust_remote_code \
        --tasks arc_challenge,hellaswag,mmlu,truthfulqa \
        --batch_size 16 --write_out --output_path output/ --seed 42
    " """
    subprocess.run(cmd, shell=True)








if __name__ == "__main__":
    
    template = "alpaca"
    
    
    # /volume/pt-train/users/wzhang/ghchen/zh/rewrite/rewrite_5k.jsonl
    # /volume/pt-train/users/wzhang/ghchen/zh/rewrite/alpaca_rewrite_5k.jsonl
    # general_configs = [
    #     ("/volume/pt-train/users/wzhang/ghchen/zh/rewrite/alpaca_orig_5k.jsonl", "alpaca_orig_5k"),
    #     ("/volume/pt-train/users/wzhang/ghchen/zh/rewrite/alpaca_rewrite_5k.jsonl", "alpaca_rewrite_5k"),
    # ]
    # Base model path
    # model_path = "/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b"
    model_configs = [
        ("/volume/pt-train/users/wzhang/ghchen/zh/rewrite/result/math/math_orig_5k.jsonl", "math_orig_5k"),
        ("/volume/pt-train/users/wzhang/ghchen/zh/rewrite/result/math/math_rewrite_5k.jsonl", "math_rewrite_5k"),
    ]
    model_path = "/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-7B"
    
    print("=== Starting Math Domain Fine-tuning ===")
    
    for i, (data_path, info) in enumerate(model_configs):
        model_name = os.path.basename(model_path)
        print(f"\nTraining with math data {i+1}/{len(model_configs)}: {model_name}-{info}")
        print(f"Data path: {data_path}")
        
        try:
            sft(model_path, template, data_path, info)
            print(f"✓ Completed training: {info}")
        except Exception as e:
            print(f"✗ Error in training {info}: {e}")
            continue

    print("\n=== General SFT Fine-tuning Experiment Completed ===")

    # # 评测所有模型
    # for _, info in general_configs:
    #     model_name = os.path.basename(model_path)
    #     model_path = f"/volume/pt-train/users/wzhang/ghchen/zh/saves/sft/cross_ana/{model_name}-{info}"
    #     eval_model(model_path, gpus_per_node=GPUS_PER_NODE)
        
    

    


# if __name__ == "__main__":
#     model_name = "Llama-2-7b"
#     template = "alpaca"
#     configs = [
#         ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/data/train/alpaca_10k.jsonl", "alpaca_10k", "Llama-2-7b"),
#         ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/data/train/metamath_10k.jsonl", "metamath_10k", "Llama-2-7b"),
#         ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/data/train/magicoder_10k.jsonl", "magicoder_10k", "Llama-2-7b"),
#     ]

#     GPUS_PER_NODE = 4
#     for data_path, info, model_name in configs:
#         try: 
#             sft(model_name, template, data_path, info, gpus_per_node=GPUS_PER_NODE)
#         except Exception as e:
#             print(f"Error: {e}")
#             continue
#     # 评测所有模型
#     for _, info, model_name in configs:
#         model_path = f"/volume/pt-train/users/wzhang/ghchen/zh/saves/sft/cross_ana/{model_name}-{info}"
#         eval_model(model_path, gpus_per_node=GPUS_PER_NODE)
        
    