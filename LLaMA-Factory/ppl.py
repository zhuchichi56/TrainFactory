#!/usr/bin/env python3

import subprocess
import json
import shutil

import os
# alias proxy_on='export http_proxy=http://100.68.170.107:3128 ; export https_proxy=http://100.68.170.107:3128 ; export HTTP_PROXY=http://100.68.170.107:3128 ; export HTTPS_PROXY=http://100.68.170.107:3128'

os.environ["HTTP_PROXY"] = "http://100.68.170.107:3128"
os.environ["HTTPS_PROXY"] = "http://100.68.170.107:3128"
os.environ["http_proxy"] = "http://100.68.170.107:3128"
os.environ["https_proxy"] = "http://100.68.170.107:3128"
os.environ["HF_ENDPOINT"] = "https://huggingface.cn"


def test_connect_google():
    import requests
    response = requests.get("https://www.google.com")
    if response.status_code == 200:
        print("Successfully connected to Google")
    else:
        print("Failed to connect to Google")


def load_jsonl(data_path):
    with open(data_path, "r") as f:
        return [json.loads(line) for line in f]

# 这个使用conda activate llf 环境
def sft(target_model, template, data_path, info, mode="sft", packing=False, gpus_per_node=4):
    import shutil
    target_path = "/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data/sft.jsonl"
    shutil.copyfile(data_path, target_path)
    print(f"Copied data to {target_path}")
    # show 2 lines of data
    data = load_jsonl(target_path)
    print(data[:2])
    packing = "true" if packing else "false"   
    
    if mode == "lora":
        cmd = f"/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/cmds/single_node/sft_lora.sh {target_model} {template} {info}_lora {packing} {gpus_per_node}"
    elif mode == "debug":
        cmd = f"/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/cmds/single_node/sft_debug.sh {target_model} {template} {info} {packing} 1"
    else:
        cmd = f"/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/cmds/single_node/sft.sh {target_model} {template} {info} {packing} {gpus_per_node}"
    subprocess.run(cmd, shell=True)
    


def eval_model(model_path, gpus_per_node=4):
    print(f"评测: {model_path}")
    cmd = f"""bash -c " 
    export HTTP_PROXY=http://100.68.170.107:3128 && \
    export HTTPS_PROXY=http://100.68.170.107:3128 && \
    export http_proxy=http://100.68.170.107:3128 && \
    export https_proxy=http://100.68.170.107:3128 && \
    export HF_ENDPOINT=https://huggingface.cn && \
    source activate /fs-computility/llmit_d/shared/zhuangxinlin/envs/lm-eval-harness && \
    accelerate launch -m --main_process_port=8189 --num_processes={gpus_per_node} lm_eval \
        --model hf --model_args pretrained={model_path} --trust_remote_code \
        --tasks arc_challenge,hellaswag,mmlu,truthfulqa \
        --batch_size 16 --write_out --output_path output/ --seed 42
    " """
    subprocess.run(cmd, shell=True)


# 

# import os

# paths = [
#        ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/landscape_bottom_10_percent_Qwen2.5-1.5B.jsonl", "landscape_bottom_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
#         ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/landscape_middle_10_percent_Qwen2.5-1.5B.jsonl", "landscape_middle_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
#         ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/landscape_top_10_percent_Qwen2.5-1.5B.jsonl", "landscape_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
#         ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/ppl_bottom_10_percent_Qwen2.5-1.5B.jsonl", "ppl_bottom_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
#         ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/ppl_middle_10_percent_Qwen2.5-1.5B.jsonl", "ppl_middle_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
#         ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/ppl_top_10_percent_Qwen2.5-1.5B.jsonl", "ppl_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
#         ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/longest_instruction_top_10_percent_Qwen2.5-1.5B.jsonl", "longest_instruction_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
#          ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/longest_ratio_top_10_percent_Qwen2.5-1.5B.jsonl", "longest_ratio_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
#          ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/longest_response_top_10_percent_Qwen2.5-1.5B.jsonl", "longest_response_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B")
# ]

# for path, name,infod in paths:
#     if os.path.exists(path):
#         print(f"✓ {name}")
#     else:
#         print(f"✗ {name}")
        
        


if __name__ == "__main__":

    # model_name = "Llama-3-8B"
    # model_name = "Llama-2-
    # template = "llama3"
    
    model_name = "Llama-2-7b-ms"
    template = "llama2"
    save_path = "/fs-computility/llmit_d/shared/zhuhe/sft_model"
#     /fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B
#     (base) root@di-20250509113709-5jp9k:/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B# ls 
# landscape_bottom_10_percent_Llama-3-8B.jsonl         middle_10_percent_llama3-8b.jsonl
# landscape_middle_10_percent_Llama-3-8B.jsonl         ppl_bottom_10_percent_Llama-3-8B.jsonl
# landscape_top_10_percent_Llama-3-8B.jsonl            ppl_middle_10_percent_Llama-3-8B.jsonl
# longest_instruction_top_10_percent_Llama-3-8B.jsonl  ppl_top_10_percent_Llama-3-8B.jsonl
# longest_ratio_top_10_percent_Llama-3-8B.jsonl

    configs = [
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/selection/alpaca_top100_gradient_scores_512_sft.jsonl", "5k_gradient"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/selection/alpaca_top100_landscape_scores_512_sft.jsonl", "5k_landscape"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/selection/alpaca_5k_longest.jsonl", "5k_longest"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/selection/selection/alpaca_5k_random.jsonl", "5k_random"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/data/alpaca_sft_ifd_ray1_top10.jsonl", "5k_ifd"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/bug_2048.jsonl", "bug_2048")
        # /fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/top_10_landscape_ppl.jsonl
        # /fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/highest_gradient.jsonl
        # /fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/lowest_gradient.jsonl
        # /fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/highest_flatness.jsonl
        # /fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/lowest_flatness.jsonl
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/top_10_landscape_ppl.jsonl", "top_10_landscape_ppl"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/highest_flatness.jsonl", "highest_flatness"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/highest_gradient.jsonl", "highest_gradient"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/lowest_gradient.jsonl", "lowest_gradient"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/lowest_flatness.jsonl", "lowest_flatness"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/top_10percent_ppl_length.jsonl", "top_10percent_ppl_length"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/top_10percent_ifd.jsonl", "top_10percent_ifd"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/middle_10_percent.jsonl", "middle_10percent_ppl"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/bottom_10_percent.jsonl", "bottom_10percent_ppl"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/top_10_percent.jsonl", "top_10_percent_ppl"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/experiment_data/alpaca_sft_ifd.jsonl", "alpaca_sft_ifd"),
        
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/landscape_bottom_10_percent_Llama-3-8B.jsonl", "landscape_bottom_10_percent_Llama-3-8B", "Llama-3-8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/landscape_middle_10_percent_Llama-3-8B.jsonl", "landscape_middle_10_percent_Llama-3-8B", "Llama-3-8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/landscape_top_10_percent_Llama-3-8B.jsonl", "landscape_top_10_percent_Llama-3-8B", "Llama-3-8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/longest_instruction_top_10_percent_Llama-3-8B.jsonl", "longest_instruction_top_10_percent_Llama-3-8B", "Llama-3-8B"),
        # # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/middle_10_percent_llama3-8b.jsonl", "middle_10_percent_llama3-8b", "Llama-3-8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/ppl_bottom_10_percent_Llama-3-8B.jsonl", "ppl_bottom_10_percent_Llama-3-8B", "Llama-3-8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/ppl_middle_10_percent_Llama-3-8B.jsonl", "ppl_middle_10_percent_Llama-3-8B", "Llama-3-8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/ppl_top_10_percent_Llama-3-8B.jsonl", "ppl_top_10_percent_Llama-3-8B", "Llama-3-8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/llama3-8B/longest_ratio_top_10_percent_Llama-3-8B.jsonl", "longest_ratio_top_10_percent_Llama-3-8B", "Llama-3-8B"),
        
        
    
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/landscape_bottom_10_percent_Qwen2.5-7B.jsonl", "landscape_bottom_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/landscape_middle_10_percent_Qwen2.5-7B.jsonl", "landscape_middle_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/landscape_top_10_percent_Qwen2.5-7B.jsonl", "landscape_top_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/ppl_bottom_10_percent_Qwen2.5-7B.jsonl", "ppl_bottom_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/ppl_middle_10_percent_Qwen2.5-7B.jsonl", "ppl_middle_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/ppl_top_10_percent_Qwen2.5-7B.jsonl", "ppl_top_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/longest_instruction_top_10_percent_Qwen2.5-7B.jsonl", "longest_instruction_top_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
        #  ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/longest_ratio_top_10_percent_Qwen2.5-7B.jsonl", "longest_ratio_top_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
        #  ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-7B/longest_response_top_10_percent_Qwen2.5-7B.jsonl", "longest_response_top_10_percent_Qwen2.5-7B", "Qwen2.5-7B"),
         
        
        
        
        # /fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/landscape_bottom_10_percent_pythia-2.8B.jsonl", "landscape_bottom_10_percent_pythia-2.8B", "pythia-2.8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/landscape_middle_10_percent_pythia-2.8B.jsonl", "landscape_middle_10_percent_pythia-2.8B", "pythia-2.8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/landscape_top_10_percent_pythia-2.8B.jsonl", "landscape_top_10_percent_pythia-2.8B", "pythia-2.8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/longest_instruction_top_10_percent_pythia-2.8B.jsonl", "longest_instruction_top_10_percent_pythia-2.8B", "pythia-2.8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/longest_ratio_top_10_percent_pythia-2.8B.jsonl", "longest_ratio_top_10_percent_pythia-2.8B", "pythia-2.8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/longest_response_top_10_percent_pythia-2.8B.jsonl", "longest_response_top_10_percent_pythia-2.8B", "pythia-2.8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/ppl_bottom_10_percent_pythia-2.8B.jsonl", "ppl_bottom_10_percent_pythia-2.8B", "pythia-2.8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/ppl_middle_10_percent_pythia-2.8B.jsonl", "ppl_middle_10_percent_pythia-2.8B", "pythia-2.8B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/pythia-2.8B/ppl_top_10_percent_pythia-2.8B.jsonl", "ppl_top_10_percent_pythia-2.8B", "pythia-2.8B"),
        
        # /fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/landscape_bottom_10_percent_Qwen2.5-1.5B.jsonl", "landscape_bottom_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/landscape_middle_10_percent_Qwen2.5-1.5B.jsonl", "landscape_middle_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/landscape_top_10_percent_Qwen2.5-1.5B.jsonl", "landscape_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/ppl_bottom_10_percent_Qwen2.5-1.5B.jsonl", "ppl_bottom_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/ppl_middle_10_percent_Qwen2.5-1.5B.jsonl", "ppl_middle_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/ppl_top_10_percent_Qwen2.5-1.5B.jsonl", "ppl_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/longest_instruction_top_10_percent_Qwen2.5-1.5B.jsonl", "longest_instruction_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        #  ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/longest_ratio_top_10_percent_Qwen2.5-1.5B.jsonl", "longest_ratio_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        #  ("/fs-computility/llmit_d/shared/zhuhe/Gap/sft-selection/data/Qwen2.5-1.5B/longest_response_top_10_percent_Qwen2.5-1.5B.jsonl", "longest_response_top_10_percent_Qwen2.5-1.5B", "Qwen2.5-1.5B"),
         
         

        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/random_sampled_10k.jsonl", "random_sampled_10k", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_response_len_desc.jsonl", "sorted_by_response_len_desc", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_response_len_asc.jsonl", "sorted_by_response_len_asc", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_flatness_score_asc.jsonl", "sorted_by_flatness_score_asc", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_flatness_score_desc.jsonl", "sorted_by_flatness_score_desc", "Llama-2-7b-ms"),
        # # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_gradient_consistency_score_asc.jsonl", "sorted_by_gradient_consistency_score_asc", "Llama-2-7b-ms"),
        # # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_gradient_consistency_score_desc.jsonl", "sorted_by_gradient_consistency_score_desc", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_ifd_score_asc.jsonl", "sorted_by_ifd_score_asc", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_ifd_score_desc.jsonl", "sorted_by_ifd_score_desc", "Llama-2-7b-ms"),
        # # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_length_ratio_score_asc.jsonl", "sorted_by_length_ratio_score_asc", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_ppl_score_desc.jsonl", "sorted_by_ppl_score_desc", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/research/Gap/landsacpe/culearning/results/sorted_by_ppl_score_asc.jsonl", "sorted_by_ppl_score_asc", "Llama-2-7b-ms"),
        
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/filtered_rank_fusion.jsonl", "filtered_rank_fusion", "Llama-2-7b-ms")
        
        # math:
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/result_0818/math_with_loss_random_selected.jsonl", "math_with_loss_random_selected", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/result_0818/math_with_loss_sorted_by_rank_fusion.jsonl", "math_with_loss_sorted_by_rank_fusion", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/result_0818/math_with_loss_sorted_by_loss.jsonl", "math_with_loss_sorted_by_loss", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/result_0818/math_with_loss_sorted_by_flatness.jsonl", "math_with_loss_sorted_by_flatness", "Llama-2-7b-ms"),
        # ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/result_0818/math_with_loss.jsonl", "math_with_loss", "Llama-2-7b-ms"),
        ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/experiment_data/alpaca_5k_random.jsonl", "alpaca_5k_random", "Llama-2-7b-ms"),
        ("/fs-computility/llmit_d/shared/zhuhe/Gap/landsacpe/experiment_data/alpaca_sft.jsonl", "alpaca_5k_full", "Llama-2-7b-ms"),
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
        
    