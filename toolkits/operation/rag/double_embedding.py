import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import fire
import json
from typing import List, Dict, Any
from matplotlib import pyplot as plt


encode_model_path = "/home/admin/data/huggingface_model/embedding_model/bge-large-zh-v1.5"

def encode_batch(instructions: List[str], model_path: str = encode_model_path, batch_size: int = 32) -> torch.Tensor:
    """
    编码输入的文本列表，并返回其嵌入向量。
    """
    model = SentenceTransformer(model_path).cuda()
    embeddings = model.encode(instructions, convert_to_tensor=True, show_progress_bar=True, device='cuda', batch_size=batch_size)
    return embeddings

def calculate_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> np.ndarray:
    """
    计算两个嵌入向量集之间的余弦相似度。
    """
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.cpu().numpy()


def load_jsonlines(input_path: str) -> List[Dict[str, Any]]:
    """
    加载JSONL文件，返回包含每行内容的列表。
    """
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def filter_outliers(similarities: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    根据给定的阈值过滤相似度分布中的异常值。
    """
    mean_similarity = np.mean(similarities)
    # std_dev = np.std(similarities)
    # filtered_indices = np.where((similarities > mean_similarity - threshold * std_dev) &
    #                             (similarities < mean_similarity + threshold * std_dev))
    # 过滤掉<0.9的
    filtered_indices = np.where(similarities > 0.9)
    return filtered_indices[0]  # 返回保留的索引

def main(input_path: str = "/home/admin/data_cookbook/ppl-experiment/lima-rewrite_mistral-v0.3.jsonl",
         output_path: str = "/home/admin/data_cookbook/ppl-experiment/lima-rewrite_mistral-v0.3-filtered.jsonl",
         threshold: float = 0.2):
    
    data = load_jsonlines(input_path)
    
    output1 = [item["rewrite"] for item in data]
    output2 = [item["output"] for item in data]
    

    embeddings1 = encode_batch(output1)
    embeddings2 = encode_batch(output2)
    

    similarities = calculate_similarity(embeddings1, embeddings2)
    
    # 统计一下 histogram
    plt.hist(similarities.diagonal(), bins=100)
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution')
    plt.show()
    
    #save plot
    plt.savefig('similarity_distribution.png')
    
    

    valid_indices = filter_outliers(similarities.diagonal(), threshold=threshold)
    
    # 过滤后的结果
    filtered_data = [data[idx] for idx in valid_indices]
    
    # 保存过滤后的数据
    with open(output_path, 'w') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    fire.Fire(main)