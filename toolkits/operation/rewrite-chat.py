import json
from inference_utils import parallel_inference
hf_model_path = '/home/admin/data/huggingface_model/LLaMA/llama2-7b'

def load_jsonl(input_path):
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, output_path):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
def load_json(input_path):
    with open(input_path, 'r') as f:
        return json.load(f)
    
    # 
    
            
data_pth =  "/home/admin/data_cookbook/ppl-experiment/lima.json"

# 都是7b
# base 模型：mistral,llama2,llama3

# experiment1: 使用这几个模型对lima 直接回答问题，测试ppl; 
mistral_pth_v3 = "/home/admin/data/huggingface_model/mistral/Mistral-7B-Instruct-v0.3"
mistral_pth_v2 = "/home/admin/data/huggingface_model/mistral/Mistral-7B-Instruct-v0.2"
llama2_7b_chat = "/home/admin/data/huggingface_model/LLaMA/llama2-7b-chat"
llama3_8b_instruct = "/home/admin/data/huggingface_model/LLaMA/llama3-8b-instruct"
qwen_15_14b_chat = "/home/admin/data/huggingface_model/qwen/Qwen1.5-14B-Chat"
qwen_15_32b_chat = "/home/admin/data/huggingface_model/qwen/Qwen1.5-32B-Chat"
qwen_2_72B_instruct = "/home/admin/data/huggingface_model/qwen/Qwen2.0-72B-Instruct"


def rewrite_prompt_template(response):
    # 将这个response 进行等意改写
   TEMPLATE = """You're a skilled writer. Please rewrite the following text while preserving its original meaning.
### Original text:
{response}

### Rewritten text:
"""
   return TEMPLATE.format(response=response)


data = load_json(data_pth)
data = [{"instruction": item["instruction"] + '\n' + item["input"] if item["input"] else item["instruction"], "output": item["output"]} for item in data]
rewrite_prompts  = [rewrite_prompt_template(item["output"]) for item in data]
outputs = parallel_inference(prompt_list=rewrite_prompts, model_path=mistral_pth_v3, max_tokens=4096, temperature=0.8, top_p=0.9, skip_special_tokens=True, score=False)
for idx, item in enumerate(data):
    item["rewrite"] = outputs[idx]
save_jsonl(data, f"/home/admin/data_cookbook/ppl-experiment/lima-rewrite_mistral-v0.3.jsonl")
    