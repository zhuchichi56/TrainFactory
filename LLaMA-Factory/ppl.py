#!/usr/bin/env python3

import subprocess
import json
import os
os.environ["HF_ENDPOINT"] = "https://huggingface.cn"
os.environ["WANDB_MODE"] ="offline"

# alias proxy_on='export http_proxy=http://100.68.170.107:3128 ; export https_proxy=http://100.68.170.107:3128 ; export HTTP_PROXY=http://100.68.170.107:3128 ; export HTTPS_PROXY=http://100.68.170.107:3128'
# os.environ["HTTP_PROXY"] = "http://100.68.170.107:3128"
# os.environ["HTTPS_PROXY"] = "http://100.68.170.107:3128"
# os.environ["http_proxy"] = "http://100.68.170.107:3128"
# os.environ["https_proxy"] = "http://100.68.170.107:3128"

# def test_connect_google():
#     import requests
#     response = requests.get("https://www.google.com")
#     if response.status_code == 200:
#         print("Successfully connected to Google")
#     else:
#         print("Failed to connect to Google")


def load_jsonl(data_path):
    with open(data_path, "r") as f:
        return [json.loads(line) for line in f]

# 这个使用conda activate llf 环境
def sft(target_model, template, data_path, info, mode="sft", packing=False, gpus_per_node=4):
    import shutil
    target_path = "./data/sft.jsonl"
    shutil.copyfile(data_path, target_path)
    print(f"Copied data to {target_path}")
    # show 2 lines of data
    data = load_jsonl(target_path)
    print(data[:2])
    packing = "true" if packing else "false"   
    
    if mode == "lora":
        cmd = f"./cmds/single_node/sft_lora.sh {target_model} {template} {info}_lora {packing} {gpus_per_node}"
    elif mode == "debug":
        cmd = f"./cmds/single_node/sft_debug.sh {target_model} {template} {info} {packing} 1"
    else:
        cmd = f"./cmds/single_node/sft.sh {target_model} {template} {info} {packing} {gpus_per_node}"
    subprocess.run(cmd, shell=True)
    

# export HTTP_PROXY=http://100.68.170.107:3128 && \
# export HTTPS_PROXY=http://100.68.170.107:3128 && \
# export http_proxy=http://100.68.170.107:3128 && \
# export https_proxy=http://100.68.170.107:3128 && \

def eval_model(model_path, gpus_per_node=4):
    print(f"评测: {model_path}")
    cmd = f"""bash -c " 
    export HF_ENDPOINT=https://huggingface.cn && \
    source activate /fs-computility/llmit_d/shared/zhuangxinlin/envs/lm-eval-harness && \
    accelerate launch -m --main_process_port=8189 --num_processes={gpus_per_node} lm_eval \
        --model hf --model_args pretrained={model_path} --trust_remote_code \
        --tasks arc_challenge,hellaswag,mmlu,truthfulqa \
        --batch_size 16 --write_out --output_path output/ --seed 42
    " """
    subprocess.run(cmd, shell=True)




if __name__ == "__main__":
    model_name = "Llama-2-7b"
    template = "alpaca"
    configs = [
        ("/volume/pt-train/users/wzhang/ghchen/zh/code/TrainFactory/data/alpaca_sft.jsonl", "alpaca_full", "Llama-2-7b"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/experiment_data/alpaca_sft.jsonl", "alpaca_5k_full", "Llama-2-7b"),
    ]

    GPUS_PER_NODE = 8
    for data_path, info, model_name in configs:
        try: 
            sft(model_name, template, data_path, info, gpus_per_node=GPUS_PER_NODE)
        except Exception as e:
            print(f"Error: {e}")
            continue
    # 评测所有模型
    # for _, info, model_name in configs:
    #     model_path = f"/fs-computility/llmit_d/shared/zhuhe/sft_model/{model_name}-{info}"
    #     eval_model(model_path, gpus_per_node=GPUS_PER_NODE)
        
    