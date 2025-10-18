from typing import List
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from loguru import logger


def parallel_inference(
    prompt_list: List[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    batch_size: int = 8,
    gpus: List[str] = None
):
    """
    并行生成模型输出。

    Args:
        prompt_list (List[str]): 输入的提示列表。
        max_tokens (int): 最大生成的 token 数。
        temperature (float): 生成的随机性参数。
        top_p (float): 核采样的阈值。
        batch_size (int): 每个 GPU 上处理的 batch 大小。
        gpus (List[str]): 可用的 GPU 列表。

    Returns:
        List[str]: 模型生成的响应列表。
    """
    if gpus is None:
        gpus = config.gpus  # 默认使用全局配置中的 GPU 列表

    def get_output_on_gpu_batch(args):
        batch, gpu_id = args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        model = AutoModelForCausalLM.from_pretrained(config.hf_model_path, torch_dtype="auto").to('cuda')
        outputs = []
        for text in tqdm(batch, desc=f"GPU-{gpu_id}", leave=False):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=config.max_input_length).to('cuda')
            generated_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=config.top_k
            )
            responses = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
            outputs.extend(responses)
        return outputs

    # 分割 prompts
    prompts_batches = list(split_list(prompt_list, batch_size))
    chunks = list(split_list(prompts_batches, len(prompts_batches) // len(gpus) + 1))
    args = [(chunk, gpus[i % len(gpus)]) for i, chunk in enumerate(chunks)]

    # 并行推理
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        results = list(executor.map(get_output_on_gpu_batch, args))

    return [res for sublist in results for res in sublist]