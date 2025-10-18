# from transformers import AutoTokenizer, AutoModel
# from huggingface_hub import HfApi, HfFolder, Repository

# # Step 1: 加载模型
# model_name = "shibing624/text2vec-base-multilingual"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Step 2: 定义你要上传到 Hugging Face 的路径
# new_model_name = "chichi56/planEmb"
# model.push_to_hub(new_model_name)
# tokenizer.push_to_hub(new_model_name)

# print(f"模型已成功上传到 Hugging Face Model Hub: https://huggingface.co/{new_model_name}")


from modelscope import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



# Step1: 
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import requests

# model_id = "/home/zhe/.cache/modelscope/hub/AI-ModelScope/Mixtral-8x7B-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')



# from datasets import load_dataset

# def download_tulu_v2(save_path="WizardLM_evol_instruct_V2_196k"):
#     dataset = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k")
    
#     # 将数据集保存到本地
#     for split, data in dataset.items():
#         save_file = f"{save_path}/{split}.jsonl"
#         data.to_json(save_file)
#         print(f"已保存 {split} 数据到 {save_file}")
    
#     print("数据集下载完成！")

# if __name__ == "__main__":
#     download_tulu_v2()


# from sentence_transformers import SentenceTransformer

# # This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
# # They are defined in `config_sentence_transformers.json`
# query_prompt_name = "s2p_query"
# queries = [
#     "What are some ways to reduce stress?",
#     "What are the benefits of drinking green tea?",
# ]
# # docs do not need any prompts
# docs = [
#     "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
#     "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
# ]

# # ！The default dimension is 1024, if you need other dimensions, please clone the model and modify `modules.json` to replace `2_Dense_1024` with another dimension, e.g. `2_Dense_256` or `2_Dense_8192` !
# # on gpu
# model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
# # you can also use this model without the features of `use_memory_efficient_attention` and `unpad_inputs`. It can be worked in CPU.
# # model = SentenceTransformer(
# #     "dunzhang/stella_en_400M_v5",
# #     trust_remote_code=True,
# #     device="cpu",
# #     config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
# # )
# query_embeddings = model.encode(queries, prompt_name=query_prompt_name)
# doc_embeddings = model.encode(docs)
# print(query_embeddings.shape, doc_embeddings.shape)
# # (2, 1024) (2, 1024)

# similarities = model.similarity(query_embeddings, doc_embeddings)
# print(similarities)
# # tensor([[0.8398, 0.2990],
# #         [0.3282, 0.8095]])

# import torch
# from transformers import pipeline

# model_id = "meta-llama/Llama-3.2-1B"

# pipe = pipeline(
#     "text-generation", 
#     model=model_id, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )

# pipe("The key to life is")