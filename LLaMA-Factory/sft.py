import subprocess
import json
import os
#

def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def load_jsonl(data_path):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data






# 这个使用conda activate llf 环境
def sft(target_model, template, data_path, info, lora=False):
    import shutil
    target_path = "/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data/sft.jsonl"
    shutil.copyfile(data_path, target_path)
    print(f"Copied data to {target_path}")
    # show 2 lines of data
    data = load_jsonl(target_path)
    print(data[:2])
    
    # 方法1：使用字符串格式，适用于 shell=True
    if lora:
        cmd = f"/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/cmds/single_node/sft_lora.sh {target_model} {template} {info}_lora"
    else:
        cmd = f"/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/cmds/single_node/sft.sh {target_model} {template} {info}"
    subprocess.run(cmd, shell=True)
    

    


if __name__ == "__main__":  
    model_name = "Llama-2-7b-ms"
    template = "llama2"
    
    data_path_list = [
        # "/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/alpaca_top100_landscape_scores_512_sft.jsonl",
        # "/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/alpaca_bottom100_landscape_scores_512_sft.jsonl"
        # "/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/selection/alpaca_top100_gradient_scores_512_sft.jsonl",
        # "/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/selection/alpaca_top100_landscape_scores_512_sft.jsonl",
        # "/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/selection/alpaca_5k_longest.jsonl",
        # "/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/selection/alpaca_5k_random.jsonl"
        # "/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/data/alpaca_sft_ifd_ray1_top10.jsonl"
        "/fs-computility/llmit_d/shared/zhuhe/sft_model/Llama-2-7b-ms-5k_longest",
        "/fs-computility/llmit_d/shared/zhuhe/sft_model/Llama-2-7b-ms-5k_random",
        "/fs-computility/llmit_d/shared/zhuhe/sft_model/Llama-2-7b-ms-5k_gradient",
        "/fs-computility/llmit_d/shared/zhuhe/sft_model/Llama-2-7b-ms-5k_landscape",
        "/fs-computility/llmit_d/shared/zhuhe/sft_model/Llama-2-7b-ms-5k_ifd",
        

    ]
    
    data_info_list = [
        # "5k_gradient",
        # "5k_landscape",
        # "5k_longest",
        # "5k_random"
        "5k_ifd"
    ]
    for data_path, info in zip(data_path_list, data_info_list):
        sft(model_name, template, data_path, info)