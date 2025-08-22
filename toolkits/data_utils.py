import json
from functools import wraps
import os
import glob
from loguru import logger
def load_jsonlines(input_path):
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonlines(data, output_path):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_json(input_path):
    if input_path.endswith(".jsonl"):
        return load_jsonlines(input_path)
    with open(input_path, 'r') as f:
        return json.load(f)

def write_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f)



def save_or_skip(file_pth):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Executing function: {func.__name__}")
            
            if os.path.exists(file_pth):
                logger.info(f"Skipping operation because {file_pth} already exists.")
                return load_jsonlines(file_pth)

            result = func(*args, **kwargs)
        
            os.makedirs(os.path.dirname(file_pth), exist_ok=True)
            write_jsonlines(result, file_pth)

            # Count the number of files in the directory
            num_files = len(glob.glob(os.path.join(os.path.dirname(file_pth), '*')))
            logger.info(f"Number of files in the directory: {num_files}")

            return result
        return wrapper
    return decorator




def save_or_skip_dynamic(parameter_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Executing function: {func.__name__}")

            # 获取 file_path 参数
            file_pth = kwargs.get(parameter_name)
            if not file_pth:
                raise ValueError(f"Parameter '{parameter_name}' not provided in kwargs.")
            
            if os.path.exists(file_pth):
                logger.info(f"Skipping operation because {file_pth} already exists.")
                return load_jsonlines(file_pth)
            
            result = func(*args, **kwargs)
        
            os.makedirs(os.path.dirname(file_pth), exist_ok=True)
            write_jsonlines(result, file_pth)

            # Count the number of files in the directory
            num_files = len(glob.glob(os.path.join(os.path.dirname(file_pth), '*')))
            logger.info(f"Number of files in the directory: {num_files}")

            return result
        return wrapper
    return decorator


