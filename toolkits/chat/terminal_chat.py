import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# 模型路径
model_path = "/home/admin/data/ppl_output/llama2-7b-sharegpt_gpt4"

# 加载tokenizer和模型
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

# 设置生成配置
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# 流式生成回答
def ask_model_stream(question):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    # 生成输出，逐步返回token
    with torch.no_grad():
        output_stream = model.generate(
            **inputs, 
            max_length=2048,
            do_sample=True,  # 开启随机采样
            temperature=0.7,  # 控制生成的多样性
            top_p=0.9,  # 使用nucleus采样策略
            num_return_sequences=1
        )
    
    # 逐步解码输出token
    answer = tokenizer.decode(output_stream[0], skip_special_tokens=True)
    
    return answer


# 交互式终端
def interactive_terminal():
    print("Welcome to the Llama2 interactive Q&A terminal!")
    print("Type 'exit' to quit at any time.")
    
    while True:
        # 获取用户输入
        question = input("\nEnter your question: ")
        
        # 检查是否退出
        if question.lower() == "exit":
            print("Exiting the terminal. Goodbye!")
            break
        
        # 流式输出回答
        print("Answer: ", end="", flush=True)
        answer = ask_model_stream(question)
        
        # 输出最终的完整答案
        print(answer)

# 运行交互模式
if __name__ == "__main__":
    interactive_terminal()