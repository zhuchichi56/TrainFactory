
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
import ray
from loguru import logger
from tqdm import tqdm
import benepar
import spacy
from config import InstaggerConfig, NVConfig, DeitaConfig, DebertaConfig, TokenConfig, pplConfig
from typing import List, Dict
from inference.inference_utils import parallel_inference, parallel_inference_logprobs
from scipy.special import softmax
import numpy as np
import re
import torch.nn as nn
import os
import json



# Download necessary models
benepar.download('benepar_en3')




##################### DeBERTa #####################
@ray.remote(num_gpus=1)
def process_batch(batch, config):
    
    logger.info(f"Loading model on GPU {ray.get_gpu_ids()}")
    tokenizer = AutoTokenizer.from_pretrained(config.reward_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.reward_name).to("cuda").eval()
    logger.info(f"Model loaded on GPU {ray.get_gpu_ids()}")

    deberta_score = []
    for item in tqdm(batch, desc="Processing batch"):
        inputs = tokenizer(item['instruction'], item['response'], return_tensors='pt', padding=True, truncation=True, max_length=config.max_length).to("cuda")
        with torch.no_grad():
            output = model(**inputs)
        score = output.logits.cpu().numpy().flatten().tolist()[0]  # 假设模型输出单个得分
        deberta_score.append(score)
    return deberta_score

def get_deberta_quality(data, config):
    ray.init()
    gpu_num = config.gpu_num
    batch_size = len(data) // gpu_num
    data_batches = [data[i*batch_size:(i+1)*batch_size] for i in range(gpu_num)]
    if len(data) % gpu_num != 0:
        data_batches[-1].extend(data[gpu_num*batch_size:])
    
    futures = [process_batch.remote(batch,config) for batch in data_batches]
    deberta_scores = []
    for result in ray.get(futures):
        deberta_scores.extend(result)

    ray.shutdown()  # 在这里关闭Ray实例
    return deberta_scores

##################### DeBERTa #####################



##################### instagger #####################

def get_tagger_complexity_quality(data: List[dict], config: InstaggerConfig) -> List[dict]:
        
    instructions = [entry["instruction"] for entry in data]
    prompts = [f"""You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{"tag": "str", "explanation": "str"}}.
    Query: {instruction}
    Assistant:""" for instruction in instructions]
    
    responses = parallel_inference(prompts, **vars(config))
    logger.debug(f"responses: {responses[0]}")
    pattern = r'"tag":\s*"([^"]*)",\s*"explanation":\s*"([^"]*)"'
    tags_list = []
    for i, response in enumerate(responses):
        matches = re.findall(pattern, response)
        tags_list.append([{"tag": tag if tag else None, "explanation": explanation if explanation else None} for tag, explanation in matches])

    complexity = [len(tag) for tag in tags_list]
    avg_complexity = sum(complexity) / len(complexity) if complexity else 0
    
    # 计算多样性
    try:
        diversity = len(set(tag["tag"] for tags in tags_list for tag in tags))
    except KeyError as e:
        logger.error(f"Missing 'tag' key in one of the tags. Error: {e}")
        diversity = 0
    
    return complexity, avg_complexity, diversity, len(data), tags_list


##################### instagger #####################

##################### NV #####################
class CustomParser():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        if spacy.__version__.startswith('2'):
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    def parse(self, text):
        if '\n' in text:
            text = text.replace('\n', ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')
        doc = self.nlp(text.strip())
        return doc
    
    
    def parse_map(self, text):
        doc = self.parse(text)
        words_map = {}
        for token in doc:
            if token.dep_ not in words_map:
                words_map[token.dep_] = []
            words_map[token.dep_].append(token.text)
        return words_map
    
    def parse_verb_nouns_pair(self, text):
        doc = self.parse(text)
        pairs = []
        for token in doc:
            found = False
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.pos_ == "NOUN":
                        pairs.append((token.lemma_, child.text))
                        found = True
                        break  # Stop searching for nouns after finding one
                if found:
                    break
        return pairs
    
    
@ray.remote(num_gpus=1)
def nv_process_batch(batch: List[Dict]) -> List:
    parser = CustomParser() 
    pairs = []
    for d in tqdm(batch):
        try:
            pair = parser.parse_verb_nouns_pair(d['instruction'])
            pairs.extend(pair)
        except Exception:
            continue
    return pairs


def get_nv_pairs(data, config):
    ray.init()
    num_gpus = config.gpu_num
    batch_size = len(data) // num_gpus
    data_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    results = ray.get([nv_process_batch.remote(batch) for batch in data_batches])
    output_data = [pair for result in results for pair in result]
    ray.shutdown()
    return output_data

##################### NV #####################
    
    

#     return func(*args, **kwargs)
#   File "/home/admin/anaconda3/envs/llama_factory/lib/python3.10/site-packages/ray/_private/worker.py", line 2624, in get
#     raise value.as_instanceof_cause()
# ray.exceptions.RayTaskError(IndexError): ray::vllm_logprobs() (pid=2475007, ip=192.168.2.69)
#   File "/home/admin/scorer/inference/inference_utils.py", line 84, in vllm_logprobs
#     return [output.outputs[0].logprobs[0] for output in outputs]
#   File "/home/admin/scorer/inference/inference_utils.py", line 84, in <listcomp>
#     return [output.outputs[0].logprobs[0] for output in outputs]
# IndexError: list index out of range


##################### Deita #####################
def infer_score(user_inputs: list, model_name_or_path: str, config: DeitaConfig):
    logprobs_list = parallel_inference_logprobs(user_inputs, model_name_or_path)
    scores = []
    for logprobs in logprobs_list:
        if logprobs is None:
            scores.append(0)
            continue
        score_logits = [logprobs.get(k, 0) for k in config.id2score]
        score_npy = softmax(score_logits) * np.array([1, 2, 3, 4, 5, 6])
        scores.append(np.sum(score_npy))
    return scores

def get_deita_complexity(input_text: list, config: DeitaConfig):
    complexity_model_path  = config.complexity_model_path
    complexity_template = (
        "You are a helpful assistant. Please identify the complexity score of the "
        "following user query. \n##Query: {instruction}  \n##Complexity: "
    )
    user_inputs = [complexity_template.format(instruction=i) for i in input_text]
    return infer_score(user_inputs, complexity_model_path, config)

def get_deita_quality(input_text: list, resp_text: list, config: DeitaConfig):
    quality_model_path = config.quality_model_path
    quality_template = (
        "You are a helpful assistant. Please identify the quality score of the Response "
        "corresponding to the Question. \n#Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: "
    )
    user_inputs = [quality_template.format(instruction=input_text[i], output=resp_text[i]) for i in range(len(input_text))]
    return infer_score(user_inputs, quality_model_path, config)

##################### Deita #####################



##################### PPL #####################

def get_perplexity(text, tokenizer, model):
    logger.info(f"Calculating perplexity for: {text}")
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

def get_template(item):
        return f"Question: {item['instruction']}\nAnswer: {item['response']}"

def get_ppl(data, config):
    ray.init()
    
    templates = [get_template(item) for item in data]
    
    gpu_num = config.gpu_num
    batch_size = len(data) // gpu_num
    data_batches = [templates[i*batch_size:(i+1)*batch_size] for i in range(gpu_num)]
    if len(data) % gpu_num != 0:
        data_batches[-1].extend(templates[gpu_num*batch_size:])
  
        

    result_ids = [perplexity_on_gpu.remote(chunk, config.hf_model_path) for chunk in  data_batches]
    results = ray.get(result_ids)
    
    ray.shutdown()
    return [ppl for result in results for ppl in result]

# ValueError: text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).



##################### PPL #####################


# 计算token分布;
def get_token_length(data, model_name):
    data = pd.DataFrame(data)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data['instruction_length'] = data['instruction'].apply(lambda x: len(x.split()))
    data['response_length'] = data['response'].apply(lambda x: len(x.split()))
    data['instruction_tokens'] = data['instruction'].apply(lambda x: len(tokenizer(x)['input_ids']))
    data['response_tokens'] = data['response'].apply(lambda x: len(tokenizer(x)['input_ids']))
    data['total_length'] = data['instruction_length'] + data['response_length']
    data['total_tokens'] = data['instruction_tokens'] + data['response_tokens']
    return data


def load_jsonlines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonlines(data, file_path):
    if os.path.exists(file_path):
        logger.info(f"Skipping operation because {file_path} already exists.")
        return
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def load_json(file_path):

    if file_path.endswith(".jsonl"):
        return load_jsonlines(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        
        
    
if __name__ == "__main__":
    # 加载一个数据, 依次 nv_pairs, instagger, deita, deberta_score;
    # data = [{"instruction": "The quick brown fox jumps over the lazy dog.", 
    #          "response": "The quick brown fox jumps over the lazy dog."}, 
    #         {"instruction": "How do you make a cake?", "response": "How do you make a cake?"}, 
    #         {"instruction": "What is the capital of France?", "response": "What is the capital of France."}]
    
    
    # test_list =["/home/admin/research/FANNO/Fanno/compare/self-instruct/data/seeds_new.jsonl"]
    
    # files_list = ["/home/admin/toolkits/scorer/test_data/ablation_11_25_change3_20000_data.jsonl",
    #                   "/home/admin/arixv_data/fanno_rebuttal_naacl/fanno-human-seed.jsonl",
    #                  "/home/admin/research/FANNO/experiment/ablation_11_25_change3_20000/initial_seed.jsonl"]
    files_list = ["/home/admin/research/FANNO/experiment/ablation_11_25_change3_20000/initial_seed.jsonl"]
        
    

    
    instagger_config = InstaggerConfig()
    # nv_config = NVConfig()
    deita_config = DeitaConfig()
    deberta_config = DebertaConfig()
    token_config = TokenConfig()
    ppl_config = pplConfig()

    # nv_pairs = get_nv_pairs(data, nv_config)
    # print(nv_pairs)
    # for test in test_list:
    #     data = load_json(test)
    #     # instagger
    #     it_complexity, it_avg_complexity, it_diversity, it_num_samples, it_tags_list = get_tagger_complexity_quality(data, instagger_config)
    #     # deita
    #     deita_quality = get_deita_quality([d["instruction"] for d in data], [d["response"] for d in data], deita_config)
    #     deita_complexity = get_deita_complexity([d["instruction"] for d in data], deita_config)
    #     # deberta
    #     deberta_quality = get_deberta_quality(data, deberta_config)
    #     # ppl 
    #     ppl_list = get_ppl(data, ppl_config)


    #     for item, ppl, deita_q, deita_c, deberta_q , it_tag , it_complexity in zip(data , ppl_list, deita_quality, deita_complexity, deberta_quality, it_tags_list, it_complexity):
    #         item["ppl"] = ppl[0]
    #         item["deita_quality"] = deita_q
    #         item["deita_complexity"] = deita_c
    #         item["deberta_quality"] = deberta_q
    #         item["instagger_tags"] = it_tag
    #         item["instagger_complexity"] = it_complexity
    #         item["instagger_avg_complexity"] = it_avg_complexity
    #         item["instagger_diversity"] = it_diversity
        
    #     # data = get_token_length(data, token_config.model_name)
    #     save_pth = "/home/admin/scorer/result/" + test + "_scored.json"
    #     write_jsonlines(data, save_pth)
        
    for test in files_list:
        data = load_json(test)
        
        deita_quality = get_deita_quality([d["instruction"] for d in data], [d["response"] for d in data], deita_config)
        # deita_complexity = get_deita_complexity([d["instruction"] for d in data], deita_config)
        deberta_quality = get_deberta_quality(data, deberta_config)
        
        idx = 0 
        for item,  deita_q, deberta_q in zip(data , deita_quality, deberta_quality):
            item["deita_quality"] = deita_q
            item["deberta_quality"] = deberta_q
            
        save_pth = f"/home/admin/toolkits/scorer/result/humans_record{idx}.jsonl"
        idx+=1
        write_jsonlines(data, save_pth)
    
    
        
        
        
  
    
    
        

    
    
    
        
        
    
    