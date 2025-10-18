# # Step1: 
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import requests


# from huggingface_hub import login
# import os

# # Load token from environment variable
# # hf_token = os.getenv("HF_TOKEN")
# # if hf_token is None:
# #     raise ValueError("Hugging Face token not found in environment variables.")

# # Login using the token from environment variable
# # login(token=hf_token)
# login(token="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")

# # export HF_TOKEN=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

# try:
#     response = requests.get("https://www.youtube.com")
#     if response.status_code == 200:
#         print("YouTube is accessible.")
#     else:
#         print(f"Received unexpected status code {response.status_code} from YouTube.")
# except requests.ConnectionError:
#     print("Failed to connect to YouTube.")
    

# model_name = "mistralai/Ministral-8B-Instruct-2410"


# def download_model(model_name):
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         print(f"Model {model_name} downloaded successfully.")
#         return tokenizer, model
#     except Exception as e:
#         print(f"Failed to download model {model_name}: {e}")
#         return None, None

# tokenizer, model = download_model(model_name)

# # Retry logic
# if tokenizer is None or model is None:
#     print("Retrying download...")
#     tokenizer, model = download_model(model_name)


# Step 2: 
# from huggingface_hub import login
# login()
# from huggingface_hub import snapshot_download
# from pathlib import Path
# mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
# mistral_models_path.mkdir(parents=True, exist_ok=True)
# snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
#                   allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], 
#                   local_dir=mistral_models_path)


from modelscope.hub.snapshot_download import snapshot_download

        # allow_patterns=["*.model", "*.json", "*.safetensors",
        # "*.py", "*.md", "*.txt"],
        # ignore_file_pattern=["*.bin", "*.msgpack",
        # "*.h5", "*.ot",],
# step1. use snapshot_download to download the model, and avoid downloading the files with the suffixes of "bin", "msgpack", "h5", "ot" 

model_dir = snapshot_download('lukeminglkm/instagger_llama2', 
                              cache_dir='/home/zhe/models')
                            #   ignore_file_pattern=[ "*.msgpack", "*.h5", "*.ot"

# "*.model", "*.json", "*.safetensors","*.py", "*.md", "*.txt" "*.bin", "*.msgpack","*.h5", "*.ot"





