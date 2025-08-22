import asyncio
import httpx
from typing import List
from loguru import logger
# from start import gpu_ids
# List of server URLs

# from config import gpu_ids
import re 
def parser_score(input_list: List[str]) -> List[int]:
    pattern = re.compile(r'score:\s*(\d)', re.IGNORECASE)
    scores = [int(match.group(1)) if (match := pattern.search(s)) else 0 for s in input_list]
    return scores


async def distribute_requests(prompt_list: List[str], max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.9, skip_special_tokens: bool = True, score = False, servers: List[str] = None) -> List[str]:
    n_chunks = len(servers)
    chunk_size = len(prompt_list) // n_chunks
    chunks = [prompt_list[i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks)]
    
    if len(prompt_list) % n_chunks != 0:
        chunks[-1].extend(prompt_list[n_chunks * chunk_size:])
    
    for i in range(n_chunks):
        logger.info(f"Chunk {i} size: {len(chunks[i])}")

    tasks = [fetch_results(servers[i], chunks[i], max_tokens, temperature, top_p, skip_special_tokens) for i in range(n_chunks)]
    results = await asyncio.gather(*tasks)
    results = sum(results, [])
    return results if not score else parser_score(results)


async def fetch_results(server_url: str, chunk: List[str], max_tokens: int, temperature: float, top_p: float, skip_special_tokens: bool):
    async with httpx.AsyncClient(timeout=180.0) as client:  # 将超时时间设置为30秒
        response = await client.post(f"{server_url}/inference", json={
            "input_data": chunk,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "skip_special_tokens": skip_special_tokens
        })
        response.raise_for_status()
        return response.json()["outputs"]
    
def parallel_inference(prompt_list: List[str], max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.9, skip_special_tokens: bool = True, score=False) -> List[str]:
    gpu_ids = [4, 5, 6, 7]
    servers = ["http://localhost:800{}".format(i) for i in range(len(gpu_ids))] 
    return asyncio.run(distribute_requests(prompt_list, max_tokens, temperature, top_p, skip_special_tokens, score, servers))


if __name__ == "__main__":
    model_path = "/home/admin/data/huggingface_model/mistral/Mistral-7B-Instruct-v0.2"
    prompt_list = ["Hello, how are you?", "What is the meaning of life?"] * 100
    result = parallel_inference(prompt_list, max_tokens=256, temperature=0, top_p=0.95, skip_special_tokens=True, score=False, gpu_ids=[0, 1, 2, 3])
    print(result)
    

    

    