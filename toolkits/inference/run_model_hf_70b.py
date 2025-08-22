# How to compute this maps?
# answer= {}

# answer ['lm_head'] = 0
# answer ['model.embed_tokens'] = 0

# for i in range(80):
#     device = i // 10
#     answer['model.layers.{}'.format(i)] = device

# answer ['model.norm'] = 1

# import json
# print(json.dumps(answer, indent=4))

# Code here!

maps = {
    "lm_head": 0,
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 0,
    "model.layers.5": 0,
    "model.layers.6": 0,
    "model.layers.7": 0,
    "model.layers.8": 0,
    "model.layers.9": 0,
    "model.layers.10": 1,
    "model.layers.11": 1,
    "model.layers.12": 1,
    "model.layers.13": 1,
    "model.layers.14": 1,
    "model.layers.15": 1,
    "model.layers.16": 1,
    "model.layers.17": 1,
    "model.layers.18": 1,
    "model.layers.19": 1,
    "model.layers.20": 2,
    "model.layers.21": 2,
    "model.layers.22": 2,
    "model.layers.23": 2,
    "model.layers.24": 2,
    "model.layers.25": 2,
    "model.layers.26": 2,
    "model.layers.27": 2,
    "model.layers.28": 2,
    "model.layers.29": 2,
    "model.layers.30": 3,
    "model.layers.31": 3,
    "model.layers.32": 3,
    "model.layers.33": 3,
    "model.layers.34": 3,
    "model.layers.35": 3,
    "model.layers.36": 3,
    "model.layers.37": 3,
    "model.layers.38": 3,
    "model.layers.39": 3,
    "model.layers.40": 4,
    "model.layers.41": 4,
    "model.layers.42": 4,
    "model.layers.43": 4,
    "model.layers.44": 4,
    "model.layers.45": 4,
    "model.layers.46": 4,
    "model.layers.47": 4,
    "model.layers.48": 4,
    "model.layers.49": 4,
    "model.layers.50": 5,
    "model.layers.51": 5,
    "model.layers.52": 5,
    "model.layers.53": 5,
    "model.layers.54": 5,
    "model.layers.55": 5,
    "model.layers.56": 5,
    "model.layers.57": 5,
    "model.layers.58": 5,
    "model.layers.59": 5,
    "model.layers.60": 6,
    "model.layers.61": 6,
    "model.layers.62": 6,
    "model.layers.63": 6,
    "model.layers.64": 6,
    "model.layers.65": 6,
    "model.layers.66": 6,
    "model.layers.67": 6,
    "model.layers.68": 6,
    "model.layers.69": 6,
    "model.layers.70": 7,
    "model.layers.71": 7,
    "model.layers.72": 7,
    "model.layers.73": 7,
    "model.layers.74": 7,
    "model.layers.75": 7,
    "model.layers.76": 7,
    "model.layers.77": 7,
    "model.layers.78": 7,
    "model.layers.79": 7,
    "model.norm": 1
}
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
hf_model_path = '/home/admin/data/AI-ModelScope/Llama-2-70b-hf'
tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
model = LlamaForCausalLM.from_pretrained(hf_model_path, device_map=maps, torch_dtype=torch.float16)
model.eval()

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
 
# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)
