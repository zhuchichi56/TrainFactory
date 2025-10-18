

### Use vllm to inference
from numpy import save
from vllm import LLM, SamplingParams

# load data response_data.jsonl
data = None
instructions = [item["instruction"] for item in data]

get_response_template = '''Below is an instruction that describes a task, write a response that appropriately completes the request. Let's think step by step and provide a detailed response.

### Instruction:
{instruction}

### Response:
'''

prompts = [get_response_template.format(instruction=instruction) for instruction in instructions]

llm = LLM(
    model="/home/admin/data/huggingface_model/qwen/Qwen2-72B", #replace
    tensor_parallel_size=8,  #replace , 4 or 5?  test! 
    max_model_len=48000) #replace
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
answer_qwen = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    answer_qwen.append(generated_text)
    

for item, response in zip(data, answer_qwen):
    item["qwen_response"] = response
    
save("response_data.jsonl", data)
