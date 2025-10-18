from datasets import load_dataset
import json
import random
from transformers import AutoTokenizer

# Step 1: Load dataset from HuggingFace
dataset_name = "Magpie-Align/Magpie-Llama-3.3-Pro-500K-Filtered"
data = load_dataset(dataset_name)
print(data['train'][0])

# Step 2: Convert ShareGPT format to instruction-response format
output_file = "/home/zhe/toolkits/magpie_pro_3.3_500k.jsonl"
processed_data = []

import tqdm
from tqdm import tqdm
for conversation in tqdm(data['train']):
    dialogue = conversation.get("conversations", [])
    # print(len(dialogue))
    if len(dialogue) == 2:
        instruction = dialogue[0].get("value", "")
        response = dialogue[1].get("value", "")
        if dialogue[0].get("from") == "human" and dialogue[1].get("from") == "gpt":
            processed_data.append({"instruction": instruction, "response": response})
            
def save_jsonlines(data, file_path):
    print(f"Saving the processed data to file {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    

save_jsonlines(processed_data, output_file)
# # Step 3: Tokenize instructions and select top 10K by token length
# tokenizer_path = "/home/admin/data/huggingface_model/LLaMA/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# tokenized_lengths = [
#     (idx, len(tokenizer.encode(entry['instruction'], truncation=True)))
#     for idx, entry in enumerate(processed_data)
# ]
# top_10k_indices = sorted(tokenized_lengths, key=lambda x: x[1], reverse=True)[:10000]
# top_10k_data = [processed_data[idx] for idx, _ in top_10k_indices]

# top_10k_file = "magpile_longest10k.jsonl"
# with open(top_10k_file, "w", encoding="utf-8") as f:
#     for entry in top_10k_data:
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Step 4: Randomly select 10K samples
# random_10k_data = random.sample(processed_data, 10000)

# random_10k_file = "magpile_random_10k_air.jsonl"
# with open(random_10k_file, "w", encoding="utf-8") as f:
#     for entry in random_10k_data:
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# print("Data processing completed. Files saved:")
# print(f"1. {output_file}")
# # print(f"2. {top_10k_file}")
# print(f"3. {random_10k_file}")



# # parquet to jsonl 
# import pandas as pd
# parquet_file = '/home/admin/advance-pipeline/data/cherry_data_v1/alpaca/train-00000-of-00001-6ef3991c06080e14.parquet'
# df = pd.read_parquet(parquet_file)
# jsonl_file = '/home/admin/advance-pipeline/data/cherry_data_v1/alpaca/alpaca_gpt4.jsonl'
# df.to_json(jsonl_file, orient='records', lines=True, force_ascii=False)
# print(f"转换完成：{parquet_file} -> {jsonl_file}")



# # alpaca2sharegpt
# def alpaca_to_sharegpt(alpaca_data: list) -> dict:
#     sharegpt_data = {"conversations": []}
    
#     for entry in alpaca_data:
#         user_question = entry.get("instruction", "")
#         input_content = entry.get("input", "")
#         sharegpt_data["conversations"].append({
#             "from": "user",
#             "value": user_question if input_content == "" else f"{user_question}\n{input_content}" 
#         })
        
#         assistant_answer = entry.get("output", "")
#         sharegpt_data["conversations"].append({
#             "from": "assistant",
#             "value": assistant_answer
#         })
    
#     return sharegpt_data


# def sharegpt_to_alpaca(sharegpt_data: dict) -> list:
#     alpaca_data = []
#     conversation = sharegpt_data.get("conversations", [])

#     for i in range(len(conversation)):
#         if conversation[i]["from"] == "user":  
#             instruction = conversation[i]["value"]

#             if i + 1 < len(conversation) and conversation[i + 1]["from"] == "assistant":
#                 output = conversation[i + 1]["value"]
#                 alpaca_data.append({
#                     "instruction": instruction,
#                     "input": "",  
#                     "output": output
#                 })
    
#     return alpaca_data


