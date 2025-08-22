#!/bin/bash
# 一键导出当前激活环境信息
# 用法: source activate <your_env> && bash export_env.sh

# 输出环境基本信息
echo "==== Environment Info ====" > env_info.txt
date >> env_info.txt
echo "User: $(whoami)" >> env_info.txt
echo "Host: $(hostname)" >> env_info.txt
echo "Python version:" >> env_info.txt
python --version >> env_info.txt
echo -e "\nCUDA / Torch info:" >> env_info.txt
python -c "import torch, sys; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda if torch.cuda.is_available() else 'CPU-only'); print('Python:', sys.version)" >> env_info.txt 2>/dev/null || echo "Torch not installed" >> env_info.txt
echo "==========================" >> env_info.txt

# 导出 conda 环境
if command -v conda &>/dev/null; then
    echo "[*] Exporting conda environment..."
    conda env export > conda_env_full.yaml
    conda env export --from-history > conda_env_min.yaml
else
    echo "[!] conda not found, skipping conda export."
fi

# 导出 pip 环境
if command -v pip &>/dev/null; then
    echo "[*] Exporting pip environment..."
    pip freeze > requirements.txt
else
    echo "[!] pip not found, skipping pip export."
fi

echo "✅ 导出完成: env_info.txt, conda_env_full.yaml, conda_env_min.yaml, requirements.txt"
