import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from loguru import logger
from typing import List, Dict, Any


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        f.writelines(json.dumps(item) + '\n' for item in data)


@dataclass
class Config:
    hf_model_path: str = '/home/admin/data/huggingface_model/qwen/Qwen2-7B-Instruct'
    max_new_tokens: int = 512
    max_input_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    gpus: list = field(default_factory=lambda: ["0", "1", "2", "3"])
    pad_token: str = None
    batch_size = 8

config = Config()

# 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.hf_model_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    

# 模板生成
def get_template(prompt, template_type="default"):
    if template_type == "alpaca":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def split_list(lst, n):
    return (lst[i:i + n] for i in range(0, len(lst), n))


def get_output_on_gpu_batch(args):
    chunk, gpu_id, max_tokens, temperature, top_p, top_k = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = AutoModelForCausalLM.from_pretrained(config.hf_model_path, torch_dtype="auto").to('cuda')
    outputs = []
    for text in tqdm(list(split_list(chunk, config.batch_size)), desc=f"GPU-{gpu_id}", leave=False):
        logger.info(f"Processing text length: {len(text)}")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=config.max_input_length).to('cuda')
        generated_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        responses = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
        outputs.extend(responses)
    return outputs


def parallel_inference(prompt_list: List[str], max_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, template_type="default"):
    data = [get_template(item, template_type=template_type) for item in prompt_list]
    n_chunks = len(config.gpus)
    chunk_size = len(data) // n_chunks
    chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks)]
    if len(data) % n_chunks != 0:
        chunks[-1].extend(data[n_chunks * chunk_size:])
    for i in range(n_chunks):
        logger.info(f"Chunk {i} size: {len(chunks[i])}")
    args = [(chunk,
             config.gpus[i % len(config.gpus)],
             max_tokens,
             temperature,
             top_p,
             top_k ) for i, chunk in enumerate(chunks)]
    with ProcessPoolExecutor(max_workers=len(config.gpus)) as executor:
        results = list(executor.map(get_output_on_gpu_batch, args))
    return [res for sublist in results for res in sublist]


def run_inference(input_path, output_path, batch_size):
    data = load_jsonl(input_path)
    data = [item['prompt'] for item in data]
    if not os.path.exists(output_path):
        results = parallel_inference(data, max_tokens=config.max_new_tokens, temperature=config.temperature, top_p=config.top_p, top_k=config.top_k)
        save_jsonl(results, output_path)



# if __name__ == "__main__":
#     input_paths = ["/home/admin/Tag-instruct/results/auto_evol1/auto_evol_0.jsonl"]
#     output_paths = ["/home/admin/output2.jsonl"]
    
    
#     for inp, outp in tqdm(zip(input_paths, output_paths), desc="Processing Files"):
#         run_inference(inp, outp, batch_size)

