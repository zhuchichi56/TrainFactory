from modelscope.hub.snapshot_download import snapshot_download

#         # allow_patterns=["*.model", "*.json", "*.safetensors",
#         # "*.py", "*.md", "*.txt"],
#         ignore_file_pattern=["*.bin", "*.msgpack",
#         "*.h5", "*.ot",],
# step1. use snapshot_download to download the model, and avoid downloading the files with the suffixes of "bin", "msgpack", "h5", "ot" 

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', 
                              cache_dir='/home/zhe/models/trained_model')
                            #   ignore_file_pattern=[ "*.msgpack", "*.h5", "*.ot"]) 

# "*.model", "*.json", "*.safetensors","*.py", "*.md", "*.txt" "*.bin", "*.msgpack","*.h5", "*.ot"
