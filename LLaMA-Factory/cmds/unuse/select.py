
import json
import random
import os

# Load original data
with open('/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data/html_content_extraction_long.json', 'r') as f:
    data = json.load(f)

# Get base filename without extension
base_name = 'html_content_extraction_long'
base_path = '/fs-computility/llmit_d/shared/zhuhe/LLaMA-Factory/data'


# easy: 
# Ratios to select
ratios = [0.5, 0.25, 0.1, 0.05]

for ratio in ratios:
    # Calculate number of samples to select
    n_samples = int(len(data) * ratio)
    
    # Randomly sample data
    sampled_data = random.sample(data, n_samples)
    
    # Create output filename
    output_filename = f"{base_name}_{ratio}.json"
    output_path = os.path.join(base_path, output_filename)
    
    # Save sampled data
    with open(output_path, 'w') as f:
        json.dump(sampled_data, f, indent=2)
        
    print(f"Saved {n_samples} samples ({ratio*100}%) to {output_filename}")



    # Sort data by instruction length
    data_with_lengths = [(item, len(item['instruction'])) for item in data]
    sorted_data = sorted(data_with_lengths, key=lambda x: x[1])
    
    # Get just the data items without lengths
    sorted_items = [item[0] for item in sorted_data]
    
    # Select top and bottom 10k samples
    top_10k = sorted_items[-10000:]
    bottom_10k = sorted_items[:10000]
    
    # Save longest 10k samples
    with open(os.path.join(base_path, f"{base_name}_longest_10k.json"), 'w') as f:
        json.dump(top_10k, f, indent=2)
    print(f"Saved longest 10k samples to {base_name}_longest_10k.json")
    
    # Save shortest 10k samples  
    with open(os.path.join(base_path, f"{base_name}_shortest_10k.json"), 'w') as f:
        json.dump(bottom_10k, f, indent=2)
    print(f"Saved shortest 10k samples to {base_name}_shortest_10k.json")
    # Create balanced samples across length percentiles
    total_samples = len(sorted_items)
    target_total = 10000  # Target total samples to collect
    percentile_size = total_samples // 100  # 1% intervals
    samples_per_percentile = target_total // 100  # Samples to take from each percentile
    balanced_samples = []
    
    for i in range(0, total_samples, percentile_size):
        # Get all samples in this 1% percentile
        percentile_data = sorted_items[i:i + percentile_size]
        # Take all samples if less than target, otherwise sample randomly
        if len(percentile_data) <= samples_per_percentile:
            balanced_samples.extend(percentile_data)
        else:
            balanced_samples.extend(random.sample(percentile_data, samples_per_percentile))
            
    # Save balanced samples
    with open(os.path.join(base_path, f"{base_name}_balanced.json"), 'w') as f:
        json.dump(balanced_samples, f, indent=2)
    print(f"Saved {len(balanced_samples)} balanced samples to {base_name}_balanced.json")

