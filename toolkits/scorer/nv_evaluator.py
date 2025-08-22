from dataclasses import dataclass
from typing import List, Dict
import tqdm
import benepar, spacy
import ray
benepar.download('benepar_en3')
from config import NVConfig
import json
config = NVConfig()
    
class CustomParser():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        if spacy.__version__.startswith('2'):
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    def parse(self, text):
        if '\n' in text:
            text = text.replace('\n', ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')
        doc = self.nlp(text.strip())
        return doc
    
    
    def parse_map(self, text):
        doc = self.parse(text)
        words_map = {}
        for token in doc:
            if token.dep_ not in words_map:
                words_map[token.dep_] = []
            words_map[token.dep_].append(token.text)
        return words_map
    
    def parse_verb_nouns_pair(self, text):
        doc = self.parse(text)
        pairs = []
        for token in doc:
            found = False
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.pos_ == "NOUN":
                        pairs.append((token.lemma_, child.text))
                        found = True
                        break  # Stop searching for nouns after finding one
                if found:
                    break
        return pairs
    
    
@ray.remote(num_gpus=1)
def process_batch(batch: List[Dict]) -> List:
    parser = CustomParser() 
    pairs = []
    for d in tqdm.tqdm(batch):
        try:
            pair = parser.parse_verb_nouns_pair(d['instruction'])
            pairs.extend(pair)
        except Exception:
            continue
    return pairs


def get_nv_analysis(data, num_gpus):
    ray.init()
    num_gpus = config.gpu_num
    batch_size = len(data) // num_gpus
    data_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    results = ray.get([process_batch.remote(batch) for batch in data_batches])
    output_data = [pair for result in results for pair in result]
    ray.shutdown()
    return output_data
    

def load_jsonlines(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    "/home/admin/research/FANNO/experiment/ablation_11_25_change3_20000/initial_seed.jsonl"
    data = [{"instruction": "The quick brown fox jumps over the lazy dog."}, {"instruction": "How do you make a cake?"}, {"instruction": "What is the capital of France?"}]
    # print(get_nv_analysis(data, 2)) # [('jump', 'fox'), ('make', 'cake')]
    
    
    
    files_list = [
    # 第一部分
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/auto_evol_10000/auto_evol_0_instruction_converted.jsonl", # 10364 5675
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/auto_evol_10000/auto_evol_1_instruction_converted.jsonl", # 10469 5873
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/auto_evol_10000/auto_evol_2_instruction_converted.jsonl", # 10445 5859
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/icl_spilt_11_15/complexity_0_1w_data.jsonl", # 10509 5214
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/icl_spilt_11_15/complexity_1_1w_data.jsonl", # 10506 5141
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/icl_spilt_11_15/complexity_2_1w_data.jsonl", # 10464 4827

    # 第二部分
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/evol/evol_0.jsonl", # 10033 5449 
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/evol/evol_1.jsonl", # 10364 6020
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/evol/evol_2.jsonl", # 10448 6153
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/evol/evol_3.jsonl", # 10465 6295
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/diversity_1w_data.jsonl", # 8759 3911
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/self-instruct/final_data.jsonl", # 8771 1365

    # 第三部分
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/instagger_split_11_15/complexity_0_1w_data.jsonl", # 10502 4676
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/instagger_split_11_15/complexity_1_1w_data.jsonl", # 10513 5050
    "/home/admin/arixv_data/tag-instruct-related-data/ablation/instagger_split_11_15/complexity_2_1w_data.jsonl"  # 10494 4748
]
    
    # files_list = ["/home/admin/research/FANNO/Fanno/compare/self-instruct/data/seeds_new.jsonl", #154 ,139
    #                 "/home/admin/research/FANNO/experiment/fanno-human-seed/final_data.jsonl", #9727, 3028
    #                 "/home/admin/research/FANNO/experiment/ablation_11_25_change3_20000/initial_seed.jsonl", #729, 575
    #                 "/home/admin/research/FANNO/experiment/ablation_11_25_change3_20000/final_data.jsonl"] # #10097, 3105
    
    for file in files_list:
        data = load_jsonlines(file)
        pairs = get_nv_analysis(data, 8)
        # store the pairs in a file like: [('jump', 'fox'), ('make', 'cake')]
        print(len(pairs))
        print(len(set(pairs)))  
        print('-----------------------------------')
        
    
    
    
    
# import benepar, spacy
# import nltk
# benepar.download('benepar_en3')

# class CustomParser():
#     def __init__(self):
#         self.nlp = spacy.load('en_core_web_md')
#         if spacy.__version__.startswith('2'):
#             self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
#         else:
#             self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

#     def parse(self, text):
#         if '\n' in text:
#             text = text.replace('\n', ' ')
#         while '  ' in text:
#             text = text.replace('  ', ' ')
#         doc = self.nlp(text.strip())
#         return doc

#     def parse_map(self, text):
#         doc = self.parse(text)
#         words_map = {}
#         for token in doc:
#             if token.dep_ not in words_map:
#                 words_map[token.dep_] = []
#             words_map[token.dep_].append(token.text)
#         return words_map
    
#     def parse_verb_nouns_pair(self, text):
#         doc = self.parse(text)
#         pairs = []
#         for token in doc:
#             found = False
#             if token.pos_ == "VERB":
#                 # Initialize object for the verb
#                 verb_object = None
#                 # Iterate through children of the token
#                 for child in token.children:
#                     # Check if child is a noun
#                     if child.pos_ == "NOUN":
#                         # Append verb-noun pair to instruction_pairs list
#                         pairs.append((token.lemma_, child.text))
#                         found = True
#                         break  # Stop searching for nouns after finding one
#                 if found:
#                     break
#         return pairs
    

# if __name__ == "__main__":
#     parser = CustomParser()
#     print(parser.parse_verb_nouns_pair("The quick brown fox jumps over the lazy dog.")) #[('jump', 'fox')]
#     print(parser.parse_verb_nouns_pair("How do you make a cake?")) #[('make', 'cake')]
#     print(parser.parse_verb_nouns_pair("What is the capital of France?")) # []
    
