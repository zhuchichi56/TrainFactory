import numpy as np
from scipy.special import softmax
from sympy import comp
from inference.inference_utils import parallel_inference_logprobs
from config import DeitaConfig
import json
def load_json(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

config = DeitaConfig()

def infer_score(user_inputs: list, model_name_or_path: str, config: DeitaConfig):
    logprobs_list = parallel_inference_logprobs(user_inputs, model_name_or_path)
    scores = []
    for logprobs in logprobs_list:
        if not logprobs:
            scores.append(0)
            continue
        score_logits = [logprobs.get(k, 0) for k in config.id2score]
        score_npy = softmax(score_logits) * np.array([1, 2, 3, 4, 5, 6])
        scores.append(np.sum(score_npy))
    return scores

def infer_deita_complexity(input_text: list, config: DeitaConfig):
    complexity_model_path  = config.complexity_model_path
    complexity_template = (
        "You are a helpful assistant. Please identify the complexity score of the "
        "following user query. \n##Query: {instruction}  \n##Complexity: "
    )
    user_inputs = [complexity_template.format(instruction=i) for i in input_text]
    return infer_score(user_inputs, complexity_model_path, config)

def infer_deita_quality(input_text: list, resp_text: list, config: DeitaConfig):
    quality_model_path = config.quality_model_path
    quality_template = (
        "You are a helpful assistant. Please identify the quality score of the Response "
        "corresponding to the Question. \n#Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: "
    )
    user_inputs = [quality_template.format(instruction=input_text[i], output=resp_text[i]) for i in range(len(input_text))]
    return infer_score(user_inputs, quality_model_path, config)


def save_jsonlines(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
if __name__ == "__main__":
    # input_text = "View tabular file such as CSV from command line, having horizontal and vertical scrolling would be great." # Example Input
    # output_text = "Sure, please take a look at csvkit. It provides a set of tools that adhere to the UNIX philosophy (meaning they are small, simple, single-purposed and can be combined). \n\nHere is an example that extracts the ten most populated cities in Germany from the free Maxmind World Cities database and displays the result in a console-readable format:\n```$ csvgrep -e iso-8859-1 -c 1 -m \"de\" worldcitiespop | csvgrep -c 5 -r \"\\d+\"\n  | csvsort -r -c 5 -l | csvcut -c 1,2,4,6 | head -n 11 | csvlook\n-----------------------------------------------------\n|  line_number | Country | AccentCity | Population  |\n-----------------------------------------------------\n|  1           | de      | Berlin     | 3398362     |\n|  2           | de      | Hamburg    | 1733846     |\n|  3           | de      | Munich     | 1246133     |\n|  4           | de      | Cologne    | 968823      |\n|  5           | de      | Frankfurt  | 648034      |\n|  6           | de      | Dortmund   | 594255      |\n|  7           | de      | Stuttgart  | 591688      |\n|  8           | de      | Düsseldorf | 577139      |\n|  9           | de      | Essen      | 576914      |\n|  10          | de      | Bremen     | 546429      |\n-----------------------------------------------------\n```\n\nCsvkit is platform independent because it is written in Python. " # Example Output
    # print(infer_deita_complexity([input_text]))
    # print(infer_deita_quality([input_text], [output_text]))
    
    # files_list = ["/home/admin/research/FANNO/Fanno/compare/self-instruct/data/seeds_new.jsonl", # 3.7158651014436304
                    # "/home/admin/research/FANNO/experiment/fanno-human-seed/final_data.jsonl",  # 5.278466474025101
                    # "/home/admin/research/FANNO/experiment/ablation_11_25_change3_20000/initial_seed_response.jsonl", #4.80 
                    
                    
                    
                    
    # files_list = ["/home/admin/arixv_data/tag-instruct-related-data/ablation/auto_evol_10000/auto_evol_0_instruction_converted.jsonl", # 4.920464756083552 #3.79166813446355
    # files_list = ["/home/admin/arixv_data/tag-instruct-related-data/ablation/auto_evol_10000/auto_evol_1_instruction_converted.jsonl", # 5.062925137906419 #4.751079141247689
    #               "/home/admin/arixv_data/tag-instruct-related-data/ablation/auto_evol_10000/auto_evol_2_instruction_converted.jsonl", #5.083591893467691 #5.202325760036384
    #               "/home/admin/arixv_data/tag-instruct-related-data/ablation/icl_spilt_11_15/complexity_0_1w_data.jsonl", #5.126034491932021 #4.075679218432184
    #               "/home/admin/arixv_data/tag-instruct-related-data/ablation/icl_spilt_11_15/complexity_1_1w_data.jsonl", #5.277279239290697 #5.195688057069132
    #               "/home/admin/arixv_data/tag-instruct-related-data/ablation/icl_spilt_11_15/complexity_2_1w_data.jsonl"] #5.332319953160766 #5.614681545385524
    # files_list = ["/home/admin/arixv_data/tag-instruct-related-data/ablation/evol/evol_0.jsonl", # 4.848301431249212 3.288535280699645
    #               "/home/admin/arixv_data/tag-instruct-related-data/ablation/evol/evol_1.jsonl", # 5.034  4.208633262254153
    #                 "/home/admin/arixv_data/tag-instruct-related-data/ablation/evol/evol_2.jsonl", # 5.141866168254597 4.829677551355801
    #                 "/home/admin/arixv_data/tag-instruct-related-data/ablation/evol/evol_3.jsonl", # 5.210679134986119 #5.198895107828568
    #                 "/home/admin/arixv_data/tag-instruct-related-data/ablation/diversity_1w_data.jsonl", #4.515679695283203 2.198394438024931
    #                 "/home/admin/arixv_data/tag-instruct-related-data/ablation/self-instruct/final_data.jsonl"] #4.31032427488285 1.9187705435899178
    # files_list = [ "/home/admin/arixv_data/tag-instruct-related-data/ablation/instagger_split_11_15/complexity_0_1w_data.jsonl", # 5.128431205974354, 3.8345263929862576
    #                 "/home/admin/arixv_data/tag-instruct-related-data/ablation/instagger_split_11_15/complexity_1_1w_data.jsonl", # 5.274250490679867 5.041151881711874
    #                 "/home/admin/arixv_data/tag-instruct-related-data/ablation/instagger_split_11_15/complexity_2_1w_data.jsonl"] # 5.330427450202444 5.50767638363343
    
    
    # 结束再测测 instagg
    
    deita_config = DeitaConfig()
    for test in files_list:
        data = load_json(test)
        data = [{"instruction": d['instruction'] + "\n" + d.get('input', '') if d.get('input', '')!='' else d['instruction'], 'response': d["output"] if 'output' in d else d["response"]} for d in data]
        deita_quality = infer_deita_quality([d["instruction"] for d in data], [d["response"] for d in data], deita_config)
        # save jsonls
        save_jsonlines(deita_quality, test.split("/")[-1].split(".")[0] + "_deita_quality.jsonl")
        # 取avg
        print("This file is: ", test.split("/")[-1])
        print(np.mean(deita_quality))
        print("---------------------")
        
        deita_complexity = infer_deita_complexity([d["instruction"] for d in data], deita_config)
        # 取avg
        save_jsonlines(deita_quality, test.split("/")[-1].split(".")[0] + "_deita_quality.jsonl")
        
        print(np.mean(deita_complexity))
        print("---------------------")
        
        
        
        
    
    
