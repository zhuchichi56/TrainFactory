import glob
import os
import sys
import json
from tqdm import tqdm
import shutil

def check_keys(base_path):
   folders = ["commoncrawl", "c4", "wikipedia", "github", "stackexchange", "book", "arxiv"]
   
   for folder in folders:
       folder_path = os.path.join(base_path, folder)
       if os.path.exists(folder_path):
           files = sorted(glob.glob(f"{folder_path}/**/*.jsonl", recursive=True))
           if files:
               with open(files[0], "r") as f:
                   first_line = f.readline().strip()
                   if first_line:
                       data = json.loads(first_line)
                       print(f"{folder}: {list(data.keys())}")

def merge_files(base_path):
   folders = ["commoncrawl", "c4", "wikipedia", "github", "stackexchange", "book", "arxiv"]
   
   with open("merged_sample.jsonl", "w", buffering=8192*16) as out:
       for folder in folders:
           folder_path = os.path.join(base_path, folder)
           if os.path.exists(folder_path):
               files = sorted(glob.glob(f"{folder_path}/**/*.jsonl", recursive=True))[:10]
               for file in tqdm(files, desc=f"{folder}"):
                   with open(file, "r", buffering=8192*16) as f:
                       lines = f.readlines()
                       batch = []
                       for line in lines:
                           if line.strip():
                               data = json.loads(line)
                               batch.append('{"text":' + json.dumps(data["content"]) + '}\n')
                       out.writelines(batch)

if __name__ == "__main__":
   base_path = "/fs-computility/llmit_d/shared/zhuangxinlin/data/slimpajama_rps_v003/raw_data_merged/train/en"
   
   print("检查每个文件夹第一个文件的keys:")
   check_keys(base_path)
   
   print("\n开始合并文件:")
   merge_files(base_path)