import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from dataclasses import dataclass
import json
import os
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
    if input_path.endswith(".jsonl"):
        return load_jsonl(input_path)
    with open(input_path, 'r') as f:
        return json.load(f)

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f)



# 这个能保证顺序吗，请附上idx，最后collect以后再通过idx 找到原有的item 
gpus = ["2", "3"]
num_gpus = len(gpus)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ModelConfig:
    hf_model_path: str = '/home/admin/data/huggingface_model/qwen/Qwen2-7B-Instruct'
    torch_dtype: str = "auto"
    max_new_tokens: int = 512
    max_input_length: int = 1024
    pad_token: str = None  # 将在初始化时设定为tokenizer.eos_token
    
config = ModelConfig()

class InferenceConfig:
    hf_model_path: str = '/home/admin/data/huggingface_model/qwen/Qwen2-7B-Instruct'
    max_new_tokens: int = 512
    max_input_length: int = 1024
    pad_token: str = None  # 将在初始化时设定为tokenizer.eos_token


#  decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.6 [00:16<01:22, 16.59s/it]
                                                                                               


# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.hf_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}






def get_template(prompt, template_type="default"):
    if template_type == "alpaca":
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text




def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_output_on_gpu_batch(args):
    batch, hf_model_path, gpu_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype="auto",
    ).to('cuda')

    output_list = []
    for text in tqdm(batch):
        input_data = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=config.max_input_length).to('cuda')
        generated_ids = model.generate(
            input_data['input_ids'],
            max_new_tokens=config.max_new_tokens
        )
        response_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_data['input_ids'], generated_ids)
        ]
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
        output_list.append(response)
    return output_list

def parallel_get_output_hf(templates, hf_model_path):
    chunk_size = len(templates) // num_gpus
    chunks = list(split_list(templates, chunk_size))

    if len(templates) % num_gpus != 0:
        chunks[-1].extend(templates[num_gpus * chunk_size:])

    results = []
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = executor.map(get_output_on_gpu_batch, [(chunks[i], hf_model_path, gpus[i]) for i in range(num_gpus)])
        for result in futures:
            results.extend(result)

    return results

def run_output_hf(input_path, output_path, batch_size):
    data = load_json(input_path)
    templates = [get_template(item['instruction']) for item in data]   
    templates = list(split_list(templates, batch_size)) #List[List[str]]
    
    if os.path.exists(output_path):
        print(f"File {output_path} exists")
        return
    else:
        results = parallel_get_output_hf(templates, config.hf_model_path)
        save_jsonl(output_path, results)

if __name__ == "__main__":
    input_path_list = ["/home/admin/Tag-instruct/results/auto_evol1/auto_evol_0.jsonl"]
    batch_size = 8
    output_path_list = ["/home/admin/output,jsonl"]
    
    for input_path, output_path in tqdm(zip(input_path_list, output_path_list)):
        run_output_hf(input_path, output_path, batch_size)
    
    