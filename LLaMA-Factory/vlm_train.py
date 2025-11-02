import subprocess
import shutil
from loguru import logger
import os
import json

# /volume/pt-train/models/Qwen2.5-VL-7B-Instruct
# 


# [
#   {
#     "image": "/volume/pt-train/users/wzhang/ghchen/zh/图片/图片/配电-图/中压配电架空线路带电线夹锈蚀/并沟线夹锈蚀56.JPG",
#     "question": "请描述这张电力设备图片中的内容",
#     "type": "图片/配电-图/中压配电架空线路带电线夹锈蚀",
#     "response": "### 1. 设备类型和名称\n从图片中可以看出，这是一组**高压输电线路的杆塔及其附件**。具体设备包括：\n- **杆塔**：用于支撑输电线路的结构，通常由钢材制成。\n- **绝缘子串**：用于隔离导线与杆塔，防止电流泄漏，由多个瓷质或复合绝缘子组成。\n- **导线**：用于传输电能的金属导线，通常为多股绞合线。\n- **横担**：杆塔上用于支撑导线和绝缘子串的水平结构，通常由钢材制成。\n- **连接金具**：用于固定导线和绝缘子串的金属部件，包括悬垂线夹、耐张线夹等。\n\n### 2. 设备的具体状态和运行情况\n- **杆塔**：杆塔整体结构较为完整，但横担部分有明显的锈蚀痕迹，表明设备已经使用了一段时间，可能需要进行防腐处理。\n- **绝缘子串**：绝缘子串安装在横担上，连接导线。绝缘子的颜色为红色，表明其可能为瓷质绝缘子。绝缘子表面没有明显的破损或污秽，但需要进一步检查其电气性能。\n- **导线**：导线悬挂在绝缘子串上，从外观上看没有明显的断股或损伤，但需要通过电气测试确认其导电性能。\n- **横担**：横担为金属结构，表面有锈蚀，表明其防腐涂层可能已经失效，需要进行维护。\n- **连接金具**：连接金具（如悬垂线夹）安装牢固，但需要检查其紧固件是否松动。\n\n### 3. 是否存在缺陷、异常或安全隐患\n- **锈蚀问题**：横担和部分金属部件存在明显的锈蚀，这可能导致结构强度下降，存在安全隐患。\n- **绝缘子状态**：虽然从外观上看绝缘子没有明显破损，但需要进一步检查其电气性能，确保其绝缘电阻符合标准。\n- **连接金具紧固性**：需要检查连接金具的紧固件是否松动，防止导线脱落或连接不良。\n- **鸟巢问题**：横担上有一个鸟巢，虽然鸟巢本身不会直接影响设备运行，但鸟粪可能污染绝缘子，降低其绝缘性能，甚至可能引发闪络事故。\n\n### 4. 设备所处的环境和位置信息\n- **环境**：设备位于一个植被茂密的区域，周围有树木和灌木丛，地面为裸露的土壤，可能是一个山区或丘陵地带。\n- **位置**：杆塔位于一条输电线路的中间位置，从导线的走向来看，线路可能跨越一定的地形起伏。\n- **气候条件**：从植被和土壤状况来看，该地区可能气候湿润，降雨较多，这会加速金属部件的锈蚀。\n\n### 5. 可能的故障原因和维护建议\n#### **可能的故障原因**\n1. **锈蚀**：由于长期暴露在潮湿环境中，金属部件（如横担和连接金具）锈蚀严重，可能导致结构强度下降。\n2. **鸟粪污染**：鸟巢的存在可能导致绝缘子被鸟粪污染，降低其绝缘性能。\n3. **连接松动**：长时间运行可能导致连接金具的紧固件松动，影响导线的稳定性。\n4. **绝缘子老化**：虽然外观无损，但长期运行可能导致绝缘子电气性能下降。\n\n#### **维护建议**\n1. **防腐处理**：\n   - 对横担和其他金属部件进行除锈处理，并重新涂刷防腐涂层，防止进一步锈蚀。\n2. **绝缘子检查**：\n   - 使用绝缘电阻测试仪对绝缘子进行电气性能测试，确保其绝缘电阻符合标准。\n   - 清理绝缘子表面的污秽，特别是鸟粪，防止闪络事故。\n3. **连接金具检查**：\n   - 检查连接金具的紧固件，确保其紧固可靠，必要时进行紧固或更换。\n4. **鸟巢处理**：\n   - 清理横担上的鸟巢，防止鸟粪污染绝缘子。\n   - 在横担上安装防鸟装置（如防鸟刺），防止鸟类再次筑巢。\n5. **定期巡检**：\n   - 定期对杆塔及其附件进行巡检，及时发现并处理锈蚀、松动等问题。\n   - 在雨季或潮湿季节增加巡检频率，防止锈蚀加剧。\n6. **环境适应性改进**：\n   - 如果该地区气候潮湿，可以考虑更换为复合绝缘子，因其具有更好的防污性能和耐腐蚀性。\n\n### 总结\n该电力设备为高压输电线路的杆塔及其附件，整体结构较为完整，但存在明显的锈蚀问题和鸟巢隐患。建议尽快进行防腐处理、绝缘子性能测试、连接金具紧固检查以及鸟巢清理，以确保设备的安全运行。同时，应加强定期巡检和维护，防止潜在故障的发生。"
#   },
#   {
#     "image": "/volume/pt-train/users/wzhang/ghchen/zh/图片/图片/变电/（9）断路器/SF6断路器/本体/瓷套表面温度/红外识别表面温度是否异常/红外识别表面温度是否异常026.jpg",
#     "question": "请描述这张电力设备图片中的内容",
#     "type": "图片/变电/（9）断路器/SF6断路器/本体/瓷套表面温度/红外识别表面温度是否异常",
#     "response": "### 1. **设备类型和名称**\n从图片中可以看出，这是一张热成像图，显示的设备为**高压电气设备**，具体为**高压隔离开关（刀闸）**或**高压断路器**的一部分。这些设备通常用于电力系统中，用于隔离电路或控制电流的通断。设备的支柱绝缘子和导电部分清晰可见，导电部分通过金属连接件与母线或其他电气设备相连。\n\n### 2. **设备的具体状态和运行情况**\n- **热成像特征**：热成像图中，设备的不同部分显示了不同的颜色，颜色越亮（如黄色、橙色），表示温度越高；颜色越暗（如紫色、蓝色），表示温度越低。\n- **导电部分**：导电部分（如连接点、触头等）显示为黄色或橙色，表明这些部位的温度较高，可能处于正常运行状态，但也可能存在过热现象。\n- **绝缘部分**：绝缘支柱（如支柱绝缘子）显示为较暗的颜色（如紫色或蓝色），表明这些部分的温度较低，符合绝缘材料的正常工作温度范围。\n\n### 3. **是否存在缺陷、异常或安全隐患**\n- **过热现象**：热成像图中，部分导电连接点（如触头、螺栓连接处）显示为黄色或橙色，表明这些部位可能存在过热现象。过热可能是由于接触不良、接触电阻增大或电流过载引起的。\n- **潜在隐患**：如果过热现象持续或加剧，可能导致设备的导电部分烧损、绝缘性能下降，甚至引发设备故障或火灾，存在安全隐患。\n\n### 4. **设备所处的环境和位置信息**\n- **环境**：设备位于户外，地面为水泥或混凝土基础，周围有其他电气设备的支柱绝缘子，表明这是一个户外高压变电站或开关站。\n- **位置**：设备安装在支柱绝缘子上，通过金属导体与母线或其他电气设备连接，属于高压电气系统的组成部分。\n\n### 5. **可能的故障原因和维护建议**\n#### **可能的故障原因**\n1. **接触不良**：导电连接点（如触头、螺栓连接处）可能存在接触不良，导致接触电阻增大，从而引起过热。\n2. **电流过载**：设备所承载的电流超过额定值，导致导电部分过热。\n3. **氧化或腐蚀**：导电连接点的金属表面可能因氧化或腐蚀而增大接触电阻，导致局部过热。\n4. **安装不当**：设备安装时螺栓未拧紧或连接不牢固，导致接触不良。\n\n#### **维护建议**\n1. **检查连接点**：对导电连接点（如触头、螺栓连接处）进行详细检查，确保接触良好，必要时进行紧固或更换。\n2. **测量接触电阻**：使用接触电阻测试仪测量导电连接点的接触电阻，确保其在正常范围内。\n3. **清理氧化层**：对导电连接点进行清洁，去除氧化层或腐蚀物，必要时涂抹导电膏以降低接触电阻。\n4. **检查电流负载**：确认设备的电流负载是否超过额定值，必要时调整负载或更换设备。\n5. **定期维护**：定期对高压电气设备进行热成像检测，及时发现过热现象并采取措施，防止故障扩大。\n6. **更换老化部件**：如果支柱绝缘子或其他部件出现老化或损坏，应及时更换，确保设备的绝缘性能和机械强度。\n\n### **总结**\n这张热成像图显示的高压电气设备（可能是隔离开关或断路器的一部分）存在局部过热现象，主要集中在导电连接点。过热可能是由于接触不良、电流过载或氧化腐蚀等原因引起的。建议立即进行详细检查和维护，以确保设备的安全运行，避免潜在的故障和安全隐患。"
#   }
# ]


gpus = ["0", "1","2", "3", "4","5","6","7"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

def load_json(data_path):
    with open(data_path, "r") as f:
        data = f.read()
    return data

def save_jsonl(data, data_path):
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
    print(f"Save to {data_path}")
    
def convert_format(input_file, output_file):
    """
    Convert from the original format:
    [
        {
            "image": "/path/to/image.jpeg",
            "question": "Question text?",
            "thinking": "",
            "summary": "Summary text"
        },
        ...
    ]
    
    To the target format:
    [
        {
            "messages": [
                {
                    "content": "<image>Question text?",
                    "role": "user"
                },
                {
                    "content": "Summary text",
                    "role": "assistant"
                }
            ],
            "images": [
                "new/path/to/image.jpeg"
            ]
        },
        ...
    ]
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in input_data:
        # Skip if question or summary is empty
        if not item["question"].strip() or not item["summary"].strip():
            continue
            
        # Extract the image path and use it directly as input
        image_path = item["image"]
        # Use the image path as is for the output
        new_image_path = image_path
        
        # Create the new format
        new_item = {
            "messages": [
                {
                    "content": f"<image>{item['question']}",
                    "role": "user"
                },
                {
                    "content": item["summary"],
                    "role": "assistant"
                }
            ],
            "images": [
                new_image_path
            ]
        }
        
        output_data.append(new_item)
    
    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. Output saved to {output_file}")

def convert_dpo_format(input_file, output_file):
    """
    Convert from the original format to DPO format which requires preferred and rejected responses:
    [
        {
            "image": "/path/to/image.jpeg",
            "question": "Question text?",
            "good_response": "Good response text",
            "bad_response": "Bad response text"
        },
        ...
    ]
    
    To the target format:
    [
        {
            "messages": [
                {
                    "content": "<image>Question text?",
                    "role": "user"
                }
            ],
            "chosen": [
                {
                    "content": "Good response text",
                    "role": "assistant"
                }
            ],
            "rejected": [
                {
                    "content": "Bad response text",
                    "role": "assistant"
                }
            ],
            "images": [
                "new/path/to/image.jpeg"
            ]
        },
        ...
    ]
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in input_data:
        # Skip if required fields are empty
        if (not item["question"].strip() or 
            not item.get("good_response", "").strip() or 
            not item.get("bad_response", "").strip()):
            continue
            
        # Extract the image path and use it directly as input
        image_path = item["image"]
        
        # Create the new format for DPO
        new_item = {
            "messages": [
                {
                    "content": f"<image>{item['question']}",
                    "role": "user"
                }
            ],
            "chosen": [
                {
                    "content": item["good_response"],
                    "role": "assistant"
                }
            ],
            "rejected": [
                {
                    "content": item["bad_response"],
                    "role": "assistant"
                }
            ],
            "images": [
                image_path
            ]
        }
        
        output_data.append(new_item)
    
    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"DPO conversion complete. Output saved to {output_file}")

def run_bash_script(model_name_or_path, data_path, output_dir, template="qwen2_vl", training_type="sft"):
    logger.info(f"{training_type.upper()} training model {model_name_or_path} with template {template}")
    
    if training_type == "sft":
        convert_format(data_path, "data/selected_data_vlm.jsonl")
        dataset_name = "SELECTED_DATA_VLM"
        script_name = "ft_vl.sh"
    else:  # DPO
        convert_dpo_format(data_path, "data/selected_data_vlm_dpo.jsonl")
        dataset_name = "SELECTED_DATA_VLM_DPO"
        script_name = "ft_vl_dpo.sh"
    
    bash_command = [
        "bash", script_name,
        "--model_name_or_path", model_name_or_path,
        "--dataset", dataset_name,
        "--output_dir", output_dir,
        "--template", template,
        "--device", ",".join(gpus)
    ]
    subprocess.run(bash_command)

def sft_vl(data_paths, base_model):
    assert base_model in ["qwen2-vl-7b", "qwen2-vl-2b"], f"Model {base_model} not found in model_base_path"
    
    model_base_path = {
        "qwen2-vl-7b": "/volume/pt-train/models/Qwen2.5-VL-7B-Instruct",
        "qwen2-vl-2b": "/volume/pt-train/models/Qwen2.5-VL-2B-Instruct"
    }
    ft_model_path = model_base_path[base_model]
    
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
        ft_output_dir = f"/volume/pt-train/users/wzhang/ghchen/zh/valid_code/plangptvl-update/LLaMA-Factory/output/{base_model}/{name}"
        logger.info(f"Finetuning model {base_model} with data {path}")
    
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=ft_output_dir,
            template="qwen2_vl",
            training_type="sft"
        )

# 最重要的参数是什么, 
def dpo_vl(data_paths, base_model, sft_model_path=None):
    assert base_model in ["qwen2-vl-7b", "qwen2-vl-2b"], f"Model {base_model} not found in model_base_path"
    
    model_base_path = {
        "qwen2-vl-7b": "/volume/pt-train/models/Qwen2.5-VL-7B-Instruct",
        "qwen2-vl-2b": "/volume/pt-train/models/Qwen2.5-VL-2B-Instruct"
    }
    if sft_model_path:
        ft_model_path = sft_model_path
        logger.info(f"Using SFT model at {sft_model_path} as starting point for DPO")
    else:
        ft_model_path = model_base_path[base_model]
        logger.info(f"Using base model {base_model} directly for DPO")
    
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
        dpo_output_dir = f"/volume/pt-train/users/wzhang/ghchen/zh/TrainFactory/LLaMA-Factory/output/{base_model}/dpo_{name}"
        logger.info(f"DPO training model {base_model} with data {path}")
    
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=dpo_output_dir,
            template="qwen2_vl",
            training_type="dpo"
        )

if __name__ == "__main__":
    # SFT training data
    sft_data_path = [
        # "/volume/pt-train/users/wzhang/ghchen/zh/valid_code/plangptvl-update/LLaMA-Factory/dev_data/mock_data.json",
        "/volume/pt-train/users/wzhang/ghchen/zh/valid_code/code4elecgpt-v/result/qwen2_32_output_revert.json"
    ]

    
    # DPO training data (contains preferred/rejected pairs)
    dpo_data_path = [
        "/volume/pt-train/users/wzhang/ghchen/zh/TrainFactory/LLaMA-Factory/data/dpo_pairs.json",
    ]
    
    # Uncomment the training type you want to run
    
    # Run SFT training
    sft_vl(sft_data_path, "qwen2-vl-7b")
    
    # Run DPO training directly on base model
    # dpo_vl(dpo_data_path, "qwen2-vl-7b")
    
    # Run DPO training on an already fine-tuned SFT model
    # sft_model = "/share/home/u24147/data/vlm_training/qwen2-vl-7b/basic_answers"
    # dpo_vl(dpo_data_path, "qwen2-vl-7b", sft_model_path=sft_model)