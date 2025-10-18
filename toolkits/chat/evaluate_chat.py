import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = "/home/admin/data/ppl_output/llama2-7b-sharegpt_gpt4"

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = LlamaForCausalLM.from_pretrained(
    model_path,                                
    torch_dtype="auto",
    device_map="auto",
)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

questions = [
    "What is 2 + 2?",  # 简单的数学问题
    "Who is the current president of the United States?",  # 事实性问题
    "What is the capital of France?",  # 常识性问题
    "Explain the theory of relativity in simple terms.",  # 简单的物理解释
    "What is the meaning of life?",  # 哲学问题
    "How does photosynthesis work?",  # 生物学概念
    "Can you summarize the plot of 'Pride and Prejudice'?",  # 文学问题
    "What are the pros and cons of machine learning?",  # 技术讨论
    "How do quantum computers differ from classical computers?",  # 高级技术问题
    "What are the ethical implications of artificial intelligence?"  # 哲学与伦理问题
]

def ask_model(question):
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=2048)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for i, question in enumerate(questions):
    print(f"Question {i + 1}: {question}")
    answer = ask_model(question)
    print(f"Answer: {answer}\n")
    
    
    
    
    