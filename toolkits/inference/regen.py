
import json
hf_model_path = '/home/admin/data/huggingface_model/LLaMA/llama2-7b'

from data_utils import *
# 仙子
data_pth =  "/home/admin/fanno-experiment/fanno/experiment/fanno-random/fix_respons_0.jsonl"



import re

def detect_string_quality(s: str, repeat_threshold: int = 5, special_symbol: str = "###"):
    result = {
        'large_repetition_detected': False,
        'non_ascii_detected': False,
        'sentence_repetition_detected': False,
        'special_symbol_detected': False
    }
    if re.search(r"(.)\1{" + str(repeat_threshold - 1) + ",}", s):
        result['large_repetition_detected'] = True
    
    if not all(ord(c) < 128 for c in s):
        result['non_ascii_detected'] = True
    
    sentences = re.split(r'[.!?。！？]', s)  
    sentence_counts = {}
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            if sentence in sentence_counts:
                sentence_counts[sentence] += 1
                if sentence_counts[sentence] > 4:
                    result['sentence_repetition_detected'] = True
            else:
                sentence_counts[sentence] = 1
    
    if special_symbol in s:
        result['special_symbol_detected'] = True
    
    return result


data = load_jsonl(data_pth)
print(data[0].keys())
model_response = [item["model_response"] for item in data]
texts = [{"idx": idx, "text": item} for idx, item in enumerate(model_response)]

bad_text_idx = []
for item in texts:
    result = detect_string_quality(item["text"])
    if result["large_repetition_detected"] or result["non_ascii_detected"] or result["sentence_repetition_detected"] or result["special_symbol_detected"]:
        # 0001 1101
        bad_text_idx.append({"idx": item["idx"], "result": "".join([str(int(result["large_repetition_detected"])), str(int(result["non_ascii_detected"])), str(int(result["sentence_repetition_detected"])), str(int(result["special_symbol_detected"]))])})
print(bad_text_idx)
print(len(bad_text_idx))

