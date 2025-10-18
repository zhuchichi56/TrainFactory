
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import ray
from loguru import logger
from tqdm import tqdm  # 引入tqdm
from config import DebertaConfig

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
    
    futures = [process_batch.remote(batch) for batch in data_batches]
    deberta_scores = []
    for result in ray.get(futures):
        deberta_scores.extend(result)

    ray.shutdown()  # 在这里关闭Ray实例
    return deberta_scores

if __name__ == "__main__":
    data = [{"instruction": "The quick brown fox jumps over the lazy dog.", "response": "The quick brown fox jumps over the lazy dog."}, {"instruction": "How do you make a cake?", "response": "How do you make a cake?"}, {"instruction": "What is the capital of France?", "response": "What is the capital of France?"}]
    scores = get_deberta_quality(data, 2)
    print(scores)
    
