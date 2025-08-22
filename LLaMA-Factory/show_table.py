import os
import json
import glob

def extract_metrics(output_dir):
    # 获取所有结果文件
    result_files = glob.glob(os.path.join(output_dir, "**", "results_*.json"), recursive=True)
    
    all_results = {}
    for file_path in result_files:
        # 从路径中提取模型名称
        model_name = file_path.split("__")[-1].split("/")[0]
        print(model_name)
        # 过滤不是Llama-2-7b-ms的
        if "Llama-2-7b-ms" not in file_path:
            continue
        
        # 读取JSON文件
        with open(file_path) as f:
            data = json.load(f)
            
        metrics = {
            "arc_challenge": data["results"]["arc_challenge"]["acc_norm,none"],
            "hellaswag": data["results"]["hellaswag"]["acc_norm,none"],
            "mmlu": data["results"]["mmlu"]["acc,none"],
            "truthfulqa": data["results"]["truthfulqa_mc2"]["acc,none"]
        }
        
        # 计算平均值
        metrics["average"] = sum(metrics.values()) / len(metrics)
        
        all_results[model_name] = metrics
        
    return all_results

def print_results_table(results):
    # 按平均值排序
    sorted_items = sorted(results.items(), key=lambda x: x[1]["average"], reverse=True)
    # 打印表头
    print("\nModel Name | ARC-c | HS | MMLU | TQA | AVG")
    print("-" * 60)
    
    # 打印每个模型的结果
    for model, metrics in sorted_items:
        print(f"{model} | {metrics['arc_challenge']:.3f} | {metrics['hellaswag']:.3f} | {metrics['mmlu']:.3f} | {metrics['truthfulqa']:.3f} | {metrics['average']:.3f}")

if __name__ == "__main__":
    output_dir = "./output"
    results = extract_metrics(output_dir)
    print_results_table(results)
    


