import json
import re
import ast
from typing import List, Tuple
from functools import wraps
import os
from loguru import logger
import glob
from typing import Iterable, Sequence, TypeVar

import matplotlib.pyplot as plt
from transformers import LlamaTokenizer
from collections import Counter
import numpy as np

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
        



data1 = "/home/admin/tag_instruct_big_mt.jsonl"

data =load_jsonlines(data1)

from tqdm import tqdm



new_data = []
for idx,item in tqdm(enumerate(data)):
    item = item['conversations']
    user1 = item[0]['value']
    assistant1 = item[1]['value']
    user2 = item[2]['value']
    assistant2 = item[3]['value']
    # 判断是不是空
    # 判断user2 < 500
    if len(user2.split()) < 5:
        new_data.append({'conversations': [{'from': 'user', 'value': user1},
                                           {'from': 'assistant', 'value': assistant1}]})
    else:
        new_data.append(item)
        
write_jsonlines(new_data, "/home/admin/tag_instruct_big_mt_new.jsonl")
    
        









# 参考格式 data[0]
# {'conversations': [{'from': 'user', 'value': 'Considering the '},
#                    {'from': 'assistant', 'value': "Neurodegenerativemi"},  # 
#                    {'from': 'user', 'value': ' What aren'}, 
#                    {'from': 'assistant', 'value': ' Combinat.'}]}

# use llama
# llama:/home/admin/data/huggingface_model/LLaMA/llama2-7b
# 使用并行token

# 帮我进行数据分析:
# 一轮对话有N次对话
# 写一个def:统计第N次对话中user 和 assistant 的value的 token数量长度分布, 需要使用tokenizer, 并画出分布图， 同时output最短的10个和最长的3个对话
# 统计一轮对话： 两次对话按照 User: Assistant: User: Assistant: 的顺序，统计每一轮的token数量长度分布， 并画出分布图
# 统计这个数据集里每轮对话的平均长度，和总token数量
import matplotlib.pyplot as plt
from transformers import LlamaTokenizer
from collections import Counter
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 初始化tokenizer
tokenizer = LlamaTokenizer.from_pretrained('/home/admin/data/huggingface_model/LLaMA/llama2-7b')

def tokenize_conversation(conversation):
    """对每一轮对话进行tokenize，并返回统计信息"""
    user_len = []
    assistant_len = []
    total_tokens = 0
    for item in conversation:
        text = item['value']
        tokens = tokenizer.tokenize(text)  # Tokenize文本
        token_len = len(tokens)
        total_tokens += token_len

        if item['from'] == 'user':
            user_len.append(token_len)
        elif item['from'] == 'assistant':
            assistant_len.append(token_len)

    return user_len, assistant_len, total_tokens

def analyze_conversations_parallel(data):
    user_token_lens = []
    assistant_token_lens = []
    conv_lens = []
    conv_tokens = []
    indices_less_than_500 = []

    # 使用进程池进行并行化
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(tokenize_conversation, conv['conversations']): idx for idx, conv in enumerate(data)}

        # 使用tqdm来显示每个进程的实时进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing conversations", unit="conv"):
            idx = futures[future]
            user_len, assistant_len, total_tokens = future.result()

            user_token_lens.extend(user_len)
            assistant_token_lens.extend(assistant_len)
            conv_lens.append(len(user_len) + len(assistant_len))  # 统计对话的总次数
            conv_tokens.append(total_tokens)      # 每轮对话的总token数

            # 如果总token数小于500，记录下它的index
            if total_tokens < 500:
                indices_less_than_500.append(idx)

    # 计算每轮对话的平均长度和总token数量
    avg_conv_len = np.mean(conv_tokens)
    total_tokens_sum = np.sum(conv_tokens)

    return user_token_lens, assistant_token_lens, conv_lens, conv_tokens, avg_conv_len, total_tokens_sum, indices_less_than_500

def plot_token_distribution(user_token_lens, assistant_token_lens, conv_tokens):
    # 绘制用户和助手的token长度分布
    plt.figure(figsize=(10, 5))
    plt.hist(user_token_lens, bins=30, alpha=0.5, label='User')
    plt.hist(assistant_token_lens, bins=30, alpha=0.5, label='Assistant')
    plt.title('Token Length Distribution')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("token_length_distribution.png")

    # 绘制每轮对话的token数量分布
    plt.figure(figsize=(10, 5))
    plt.hist(conv_tokens, bins=30, alpha=0.7)
    plt.title('Conversation Token Length Distribution')
    plt.xlabel('Total Tokens per Conversation')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig("conversation_token_length_distribution.png")


def find_extreme_conversations(data, user_token_lens, assistant_token_lens):
    # 找到最短的10个和最长的3个对话
    combined_lens = user_token_lens + assistant_token_lens
    sorted_lens = sorted(combined_lens)
    
    shortest_10 = sorted_lens[:10]
    longest_3 = sorted_lens[-3:]
    
    return shortest_10, longest_3

# 使用数据进行分析并显示进度
data = load_jsonlines(data1)
none_number = sum([1 for conv in data if conv is None])


user_token_lens, assistant_token_lens, conv_lens, conv_tokens, avg_conv_len, total_tokens, indices_less_than_500 = analyze_conversations_parallel(data)

# 绘制token长度分布图
plot_token_distribution(user_token_lens, assistant_token_lens, conv_tokens)

# 输出平均长度和总token数量
print(f"每轮对话的平均长度: {avg_conv_len}")

# 转换成 M
print(f"总token数量: {total_tokens / 1e6:.2f}M")    


# 输出token数小于500的对话索引
print(f"Token 数量小于 500 的对话索引: {indices_less_than_500}")

# 从原数据中找出token数小于500的对话
less_than_500_conversations = [data[idx] for idx in indices_less_than_500]
write_jsonlines(less_than_500_conversations, "less_than_500_conversations.jsonl")


# 找出最短的10个和最长的3个对话
shortest_10, longest_3 = find_extreme_conversations(data, user_token_lens, assistant_token_lens)
print(f"最短的10个对话token长度: {shortest_10}")
print(f"最长的3个对话token长度: {longest_3}")