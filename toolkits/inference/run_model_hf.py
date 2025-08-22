
from modelscope import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
pth = "meta-llama/Llama-2-13b"
model = AutoModelForCausalLM.from_pretrained(
    pth,
    torch_dtype="auto",
    device_map="auto",
    # cache_dir="......" #  加载路径
)
tokenizer = AutoTokenizer.from_pretrained(pth)


### use template
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


# Batch Decode:
# prompt = ["You are a helpful assistant. Give me a short introduction to large language model.", "You are a not helpful assistant. Give me a short introduction to large language model."]
# model_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

# # A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
# print(generated_ids)
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(response)

