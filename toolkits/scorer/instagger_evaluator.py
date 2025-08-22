from typing import List
from dataclasses import dataclass
from loguru import logger
import re
from scorer.inference.inference_utils_1 import parallel_inference
from config import InstaggerConfig 


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
    
    return complexity, avg_complexity, diversity, len(data)


if __name__ == "__main__":
    data = [{"instruction": "View tabular file such as CSV from command line, having horizontal and vertical scrolling would be great."}, {"instruction": "How do you make a cake?"}, {"instruction": "What is the capital of France?"}]
    complexity, avg_complexity, diversity, num_samples = get_tagger_complexity_quality(data)
    print(f"Complexity: {complexity}, Average Complexity: {avg_complexity}, Diversity: {diversity}, Number of Samples: {num_samples}")
    
