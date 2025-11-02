import json
import os
from loguru import logger

# 设置可见 GPU（仅在需要时）
gpus = ["4", "5"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

def load_json(data_path):
    """加载 JSON 文件"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)  # ✅ 这里必须是 json.load 而不是 f.read()

def save_json(data, data_path):
    """保存 JSON 文件"""
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 已保存到 {data_path}，共 {len(data)} 条")

def convert_to_data_format(input_file, output_file):
    """转换为目标格式"""
    try:
        data = load_json(input_file)
        if not isinstance(data, list):
            raise ValueError("输入文件不是 JSON 列表格式")

        new_data = []
        for item in data:
            if not isinstance(item, dict):
                logger.warning(f"跳过无效项：{item}")
                continue

            new_item = {
                "image": item.get("image", ""),
                "question": item.get("question", ""),
                "thinking": "",  # 空字段
                "summary": item.get("response", "")
            }
            new_data.append(new_item)

        save_json(new_data, output_file)
        logger.success(f"转换完成，共 {len(new_data)} 条数据")

    except Exception as e:
        logger.error(f"❌ 转换出错: {e}")

if __name__ == "__main__":
    convert_to_data_format(
        "/volume/pt-train/users/wzhang/ghchen/zh/valid_code/code4elecgpt-v/result/qwen2_32_output_final.json",
        "/volume/pt-train/users/wzhang/ghchen/zh/valid_code/code4elecgpt-v/result/qwen2_32_output_revert.json"
    )
