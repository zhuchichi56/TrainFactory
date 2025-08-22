# from modelscope.hub.api import HubApi




# # 大概用时40分钟; 还是非常不错的; 
# YOUR_ACCESS_TOKEN = '8b89653c-bf85-4a22-98be-3a9a16ae4266'
# api = HubApi()
# api.login(YOUR_ACCESS_TOKEN)

# from modelscope.hub.constants import Licenses, ModelVisibility





# # api.push_model(
# #     model_id=model_id, # 如果model_id对应的模型库不存在，将会被自动创建
# #     model_dir="/data/zhe/new/llama3-8b-WizardLM_evol_instruct_V2_196k_processed" # 指定本地模型所在目录
# # )


# model_dirs = [
#     "llama3.2-3b-alpaca_data_cleaned",
#     "llama3.2-3b-alpaca_evol_instruct_70k",
#     "llama3.2-3b-WizardLM_evol_instruct_V2_196k_processed",
#     "llama3-8b-alpaca_data_cleaned_52k",
#     "llama3-8b-alpaca_evol_instruct_70k",
#     "llama3.2-3b-tag_instruct_reward_4",
#     "llama3-8b-tag_instruct_reward_4",
#     "llama3-8b-tag_instruct_4_magpie_qwen"
# ]
# model_name = [
#     "LLAMA3.2-3b-Alpaca_data_cleaned",
#     "LLAMA3.2-3b-Alpaca_evol_instruct_70k",
#     "LLAMA3.2-3b-WizardLM_evol_instruct_V2_196k",
#     "LLAMA3-8b-Alpaca_data_cleaned_52k",
#     "LLAMA3-8b-Alpaca_evol_instruct_70k",
#     "LLAMA3.2-3b-Tag_instruct_Reward_4",
#     "LLAMA3-8b-Tag_instruct_Reward_4",
#     "LLAMA3-8b-Tag_instruct_Magpie_Qwen"
# ]


# username = 'chichi56'
# model_name = 'LLAMA3-8b-Wiz_196k'
# model_id=f"{username}/{model_name}"

# api.create_model(
#     model_id,
#     visibility=ModelVisibility.PUBLIC,
#     license=Licenses.APACHE_V2,
#     chinese_name="LLAMA3-8b-Wiz_196k",
# )

# for model_dir in model_dirs:
#     model_id = f"{username}/{model_dir}"
#     api.push_model(
#         model_id=model_id,
#         model_dir=f"/data/zhe/new/{model_dir}"
#     )

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_upload.log'),
        logging.StreamHandler()
    ]
)

# ModelScope API configuration
YOUR_ACCESS_TOKEN = '8b89653c-bf85-4a22-98be-3a9a16ae4266'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

# Model configuration
username = 'chichi56'
base_dir = "/data/zhe/new"

model_configs = [
    {
        "dir_name": "llama3.2-3b-alpaca_data_cleaned",
        "display_name": "LLAMA3.2-3b-Alpaca_data_cleaned"
    },
    {
        "dir_name": "llama3.2-3b-alpaca_evol_instruct_70k",
        "display_name": "LLAMA3.2-3b-Alpaca_evol_instruct_70k"
    },
    {
        "dir_name": "llama3.2-3b-WizardLM_evol_instruct_V2_196k_processed",
        "display_name": "LLAMA3.2-3b-WizardLM_evol_instruct_V2_196k"
    },
    {
        "dir_name": "llama3-8b-alpaca_data_cleaned_52k",
        "display_name": "LLAMA3-8b-Alpaca_data_cleaned_52k"
    },
    {
        "dir_name": "llama3-8b-alpaca_evol_instruct_70k",
        "display_name": "LLAMA3-8b-Alpaca_evol_instruct_70k"
    },
    {
        "dir_name": "llama3.2-3b-tag_instruct_reward_4",
        "display_name": "LLAMA3.2-3b-Tag_instruct_Reward_4"
    },
    {
        "dir_name": "llama3-8b-tag_instruct_reward_4",
        "display_name": "LLAMA3-8b-Tag_instruct_Reward_4"
    },
    {
        "dir_name": "llama3-8b-tag_instruct_4_magpie_qwen",
        "display_name": "LLAMA3-8b-Tag_instruct_Magpie_Qwen"
    }
]

def create_and_push_model(config):
    """
    Create and push a single model to ModelScope
    """
    try:
        model_id = f"{username}/{config['dir_name']}"
        model_path = f"{base_dir}/{config['dir_name']}"
        
        # Create model entry
        logging.info(f"Creating model: {model_id}")
        api.create_model(
            model_id,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2,
            chinese_name=config['display_name']
        )
        
        # Push model files
        logging.info(f"Pushing model files from: {model_path}")
        api.push_model(
            model_id=model_id,
            model_dir=model_path
        )
        
        logging.info(f"Successfully uploaded model: {config['display_name']}")
        return True
        
    except Exception as e:
        logging.error(f"Error uploading model {config['display_name']}: {str(e)}")
        return False

def main():
    start_time = time.time()
    successful_uploads = 0
    
    for config in model_configs:
        logging.info(f"Starting upload for: {config['display_name']}")
        if create_and_push_model(config):
            successful_uploads += 1
        time.sleep(5)  # Small delay between uploads
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60  # Convert to minutes
    
    logging.info(f"Upload process completed. Time taken: {total_time:.2f} minutes")
    logging.info(f"Successfully uploaded {successful_uploads} out of {len(model_configs)} models")

if __name__ == "__main__":
    main()