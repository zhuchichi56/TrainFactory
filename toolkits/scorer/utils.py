import json
from math import log
import re
import ast
from typing import List, Tuple
from functools import wraps
import os
from loguru import logger
import glob
from typing import Iterable, Sequence, TypeVar



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
        



def save_or_skip(file_pth_func):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 每次执行时获取最新的文件路径
            file_pth = file_pth_func()

            logger.info(f"Executing function: {func.__name__}")

            # 如果文件已经存在，加载并返回文件内容
            if os.path.exists(file_pth):
                logger.info(f"Skipping operation because {file_pth} already exists.")
                return load_jsonlines(file_pth)

            # 执行函数，获取结果
            result = func(*args, **kwargs)

            # 创建目录并保存结果到文件
            os.makedirs(os.path.dirname(file_pth), exist_ok=True)
            write_jsonlines(result, file_pth)

            # 统计文件夹中的文件数目
            num_files = len(glob.glob(os.path.join(os.path.dirname(file_pth), '*')))
            logger.info(f"Number of files in the directory: {num_files}")

            return result
        return wrapper
    return decorator


def calculate_avg_word_length(data):
    """统计每条 instruction 的单词长度，并返回平均值"""
    total_length = 0
    total_instructions = len(data)
    
    for item in data:
        instruction = item.get("instruction", "")
        word_count = len(instruction.split())  # 统计单词数
        total_length += word_count

    avg_length = total_length / total_instructions if total_instructions > 0 else 0    
    return avg_length



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
                data = load_jsonlines(file_pth)
            else:
                result = func(*args, **kwargs)
                os.makedirs(os.path.dirname(file_pth), exist_ok=True)
                write_jsonlines(result, file_pth)
                data = result

            # 统计单词长度，并计算平均值
            avg_length = calculate_avg_word_length(data)
            logger.warning(f"Number of results: {len(data)}")
            logger.warning(f"Average word count for instructions in {file_pth}: {avg_length}")

            # 记录目录中的文件数量
            num_files = len(glob.glob(os.path.join(os.path.dirname(file_pth), '*')))
            logger.info(f"Number of files in the directory: {num_files}")

            return data
        return wrapper
    return decorator





_T = TypeVar("_T")

def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def is_compilable(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def filter_non_python_data(data):
    new_data = []
    language_list = [
        "java", "javascript", "c#", "c++", "ruby", "go", "swift",
        "kotlin", "typescript", "php", "r", "perl", "scala", "shell", 
        "bash", "lua", "matlab", "haskell", "rust", "dart", "objective-c",
        "julia", "clojure", "f#", "groovy", "elixir", "erlang", "fortran",
        "vb.net", "powershell", "sql", "sas", "vhdl", "verilog", "assembly"
    ]
    
    for d in data:
        instruction = d["instruction"].lower().split()
        if any([lang in instruction for lang in language_list]):
            continue
        
        if d["instruction"] and d["instruction"].strip()[-1] == "." and len(d["instruction"].strip()) > 5:
            new_data.append(d)
    return new_data



def extract_code(response):
    pattern = r"^```python\s*\n(.*?)(?=^```)"
    result = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
    return "\n".join([x for x in result if is_compilable(x)])




def alpaca_to_sharegpt(alpaca_data: list) -> dict:
    sharegpt_data = {"conversations": []}
    
    for entry in alpaca_data:
        user_question = entry.get("instruction", "")
        input_content = entry.get("input", "")
        sharegpt_data["conversations"].append({
            "from": "user",
            "value": user_question if input_content == "" else f"{user_question}\n{input_content}" 
        })
        
        assistant_answer = entry.get("output", "")
        sharegpt_data["conversations"].append({
            "from": "assistant",
            "value": assistant_answer
        })
    
    return sharegpt_data


def sharegpt_to_alpaca(sharegpt_data: dict) -> list:
    alpaca_data = []
    conversation = sharegpt_data.get("conversations", [])

    for i in range(len(conversation)):
        if conversation[i]["from"] == "user":  
            instruction = conversation[i]["value"]

            if i + 1 < len(conversation) and conversation[i + 1]["from"] == "assistant":
                output = conversation[i + 1]["value"]
                alpaca_data.append({
                    "instruction": instruction,
                    "input": "",  
                    "output": output
                })
    
    return alpaca_data

