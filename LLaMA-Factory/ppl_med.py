#!/usr/bin/env python3

import subprocess
import json
import os
import shutil
import glob

os.environ["HF_ENDPOINT"] = "https://huggingface.cn"
os.environ["WANDB_MODE"] = "offline"


def load_jsonl(data_path):
    with open(data_path, "r") as f:
        return [json.loads(line) for line in f]


def create_yaml_config(target_model, template, info, packing, gpus_per_node):
    """创建训练配置yaml文件"""
    TOTAL_BATCH_SIZE = 128
    N_NODE = 1
    PER_DEVICE_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // (PER_DEVICE_BATCH_SIZE * gpus_per_node * N_NODE)
    
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
save_strategy: epoch
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


def sft(target_model, template, data_path, info, mode="sft", packing=False, gpus_per_node=4):
    """简化的SFT训练函数"""
    # 复制数据文件
    target_path = "./data/sft.jsonl"
    shutil.copyfile(data_path, target_path)
    data = load_jsonl(target_path)
    print(f"Copied data to {target_path}")
    print("Sample data:")
    print(data[:2])
    
    # 创建yaml配置文件
    packing_str = "false"  # 根据math.sh中的设置，packing设为false
    yaml_file = create_yaml_config(target_model, template, info, packing_str, gpus_per_node)
    
    print(f"Created config file: {yaml_file}")
    print(f"Training model: {target_model} with data: {data_path}")
    
    # 执行训练 - 指定使用GPU 2,3,6,7
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
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


if __name__ == "__main__":
    
    template = "alpaca"
    GPUS_PER_NODE = 2
    
    # Math training configurations
    math_configs = [
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/math/Numina_Math_random_20pct.jsonl", "Numina_Math_random_20pct"),
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/math/Numina_Math_rank_fusion_20pct.jsonl", "Numina_Math_rank_fusion_20pct"),
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/math/Numina_Math_flatness_high_20pct.jsonl", "Numina_Math_flatness_high_20pct"),
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/math/Numina_Math_flatness_low_20pct.jsonl", "Numina_Math_flatness_low_20pct"),
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/math/Numina_Math_loss_high_20pct.jsonl", "Numina_Math_loss_high_20pct"),
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/math/Numina_Math_loss_low_20pct.jsonl", "Numina_Math_loss_low_20pct"),
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/math/Numina_Math_grad_high_20pct.jsonl", "Numina_Math_grad_high_20pct"),
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/loss-landscape/math/Numina_Math_grad_low_20pct.jsonl", "Numina_Math_grad_low_20pct"),
    ]
    # Base model path
    # model_path = "/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b"
    model_path = "/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-7B"
    
    print("=== Starting Math Domain Fine-tuning ===")
    
    for i, (data_path, info) in enumerate(math_configs):
        model_name = os.path.basename(model_path)
        print(f"\nTraining with math data {i+1}/{len(math_configs)}: {model_name}-{info}")
        print(f"Data path: {data_path}")
        
        try:
            sft(model_path, template, data_path, info, gpus_per_node=GPUS_PER_NODE)
            print(f"✓ Completed training: {info}")
        except Exception as e:
            print(f"✗ Error in training {info}: {e}")
            continue
    
    # print("\n=== Math Domain Fine-tuning Experiment Completed ===")
    
    # # 显示所有训练完成的模型
    # math_model_dir = "/volume/pt-train/users/wzhang/ghchen/zh/saves/sft"
    # math_model_paths = glob.glob(os.path.join(math_model_dir, "Llama-2-7b-math_*"))
    # math_model_paths = [path for path in math_model_paths if os.path.isdir(path)]
    
    # print(f"\nFound {len(math_model_paths)} trained math models:")
    # for path in math_model_paths:
    #     print(f"  - {os.path.basename(path)}")
    
    # print("\nAll math models are ready for evaluation and analysis.")