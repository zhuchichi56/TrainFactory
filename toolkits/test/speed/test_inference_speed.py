import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


# 加载模型和tokenizer
model_name = "/home/admin/data/huggingface_model/LLaMA/llama2-7b-chat"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# 切换模型到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # 关闭训练模式

# 要测试的文本输入
test_text = "The future of AI is"
inputs = tokenizer(test_text, return_tensors="pt").to(device)

# 测试推理的次数
num_inferences = 10

# 记录推理时间和token数
total_time = 0
total_tokens = 0

print(f"Starting inference speed test for {num_inferences} runs...\n")

for i in range(num_inferences):
    start_time = time.time()
    
    # 执行推理
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=50,  # 控制生成序列的最大长度
            do_sample=False  # 关闭采样，使用贪心解码
        )
    
    # 计算推理时间
    inference_time = time.time() - start_time
    total_time += inference_time
    
    # 计算生成的 tokens 数量
    generated_tokens = outputs.shape[1]  # outputs 是 (batch_size, sequence_length)
    total_tokens += generated_tokens
    
    # 输出生成结果和推理时间
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Run {i+1}: Generated text: {generated_text}")
    print(f"Run {i+1}: Inference time: {inference_time:.4f} seconds")
    print(f"Run {i+1}: Generated {generated_tokens} tokens\n")

# 计算平均推理时间和每秒生成的tokens数量
average_time = total_time / num_inferences
tokens_per_second = total_tokens / total_time

print(f"Average inference time over {num_inferences} runs: {average_time:.4f} seconds")
print(f"Average tokens per second: {tokens_per_second:.2f} tokens/s")
