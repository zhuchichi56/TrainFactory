import json
import os
from re import template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from tqdm import tqdm
import ray

gpus = ["2", "3", "4", "5", "6", "7"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

hf_model_path = '/home/admin/data/huggingface_model/LLaMA/llama2-7b'




def get_perplexity(text, tokenizer, model):
    input = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')
    labels = input['input_ids'].clone()
    with torch.no_grad():
        outputs = model(**input)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1).to(shift_logits.device))
        loss = loss.view(-1, input['input_ids'].size(-1) - 1).mean(-1)
    return torch.exp(loss).detach().cpu().numpy().tolist()

@ray.remote(num_gpus=1)
def perplexity_on_gpu(batch, hf_model_path):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype="auto").to('cuda')
    tokenizer.pad_token = tokenizer.eos_token
    return [get_perplexity(text, tokenizer, model) for text in tqdm(batch)]


def get_ppl(data, config):
    def get_template(item):
        return f"Question:{item['instruction']}\nAnswer:{item['response']}"
    templates = [get_template(item) for item in data]
    ray.init()
    chunk_size = len(templates) // len(gpus)
    chunks = [templates[i:i + chunk_size] for i in range(0, len(templates), chunk_size)]
    if len(templates) % len(gpus) != 0:
        chunks[-1].extend(templates[len(gpus) * chunk_size:])
        
    
    result_ids = [perplexity_on_gpu.remote(chunk, config.hf_model_path) for chunk in chunks]
    results = ray.get(result_ids)
    
    ray.shutdown()
    return [ppl for result in results for ppl in result]