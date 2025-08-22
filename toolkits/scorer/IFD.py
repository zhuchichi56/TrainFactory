import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import csv

# Part 1: ppl_A_condition violin plot
folder_path = './data'
file_names = ['alpaca_data_gpt2_data_full.json', 'alpaca_data_gpt2-large_data_full.json', 
              'alpaca_data_gpt2-xl_data_full.json', 'alpaca_data_gpt-neo_data_full.json']
file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]
model_labels = ['gpt2', 'gpt2-large', 'gpt2-xl', 'gpt-neo']
filtered_ifd_ppl_dict = {}

for file_path, model_label in zip(file_paths, model_labels):
    try:
        with open(file_path, 'r') as file:
            alpaca_data = json.load(file)
            ifd_ppl_values = [item['ppl_A_condition'] for item in alpaca_data if 'ppl_A_condition' in item]
            
            upper_limit = np.quantile(ifd_ppl_values, 0.9)
            lower_limit = np.quantile(ifd_ppl_values, 0.01)
            
            filtered_values = [value for value in ifd_ppl_values if lower_limit <= value <= upper_limit]
            filtered_ifd_ppl_dict[model_label] = filtered_values
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")

plt.violinplot(dataset=[filtered_ifd_ppl_dict[model] for model in model_labels], showmedians=True)
plt.xticks(ticks=np.arange(1, len(model_labels) + 1), labels=model_labels, rotation=45)
plt.title('Filtered ppl Comparison')
plt.xlabel('Model')
plt.ylabel('ppl')
plt.savefig('./output/ppl.png')
plt.clf()

# Part 2: ifd_ppl violin plot
filtered_ifd_ppl_dict = {}
for file_path, model_label in zip(file_paths, model_labels):
    try:
        with open(file_path, 'r') as file:
            alpaca_data = json.load(file)
            ifd_ppl_values = [item['ifd_ppl'] for item in alpaca_data if 'ifd_ppl' in item]
            
            upper_limit = np.quantile(ifd_ppl_values, 0.99)
            filtered_values = [value for value in ifd_ppl_values if 0 <= value <= upper_limit]
            filtered_ifd_ppl_dict[model_label] = filtered_values
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")

plt.violinplot(dataset=[filtered_ifd_ppl_dict[model] for model in model_labels], showmedians=True)
plt.xticks(ticks=np.arange(1, len(model_labels) + 1), labels=model_labels, rotation=45)
plt.title('Filtered ifd_ppl Comparison')
plt.xlabel('Model')
plt.ylabel('ifd_ppl')
plt.savefig('./output/ifd_ppl.png')
plt.clf()

# Part 3: Spearman rank correlation
neo_file_path = 'data/alpaca_data_gpt-neo_data_full.json'
model_file_paths = {
    'gpt2': 'data/alpaca_data_gpt2_data_full.json',
    'gpt2-large': 'data/alpaca_data_gpt2-large_data_full.json',
    'gpt2-xl': 'data/alpaca_data_gpt2-xl_data_full.json'
}

with open(neo_file_path, 'r') as f:
    neo_data = json.load(f)
neo_ids = [item['output'] for item in neo_data]

spearman_results = []

for model_label, file_path in model_file_paths.items():
    with open(file_path, 'r') as f:
        model_data = json.load(f)
    
    model_ids = [item['output'] for item in model_data]
    
    common_ids = set(neo_ids) & set(model_ids)
    
    neo_ranks = [neo_ids.index(id_) + 1 for id_ in common_ids]
    model_ranks = [model_ids.index(id_) + 1 for id_ in common_ids]
    
    spearman_corr, p_value = spearmanr(neo_ranks, model_ranks)
    
    spearman_results.append([model_label, spearman_corr])

output_csv = './output/spearman_correlation.csv'
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Spearman Rank Correlation'])
    writer.writerows(spearman_results)

print(f"Finished!")