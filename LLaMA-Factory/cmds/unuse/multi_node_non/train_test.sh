#!/bin/bash
# 火山引擎机器学习平台PyTorch DDP分布式训练启动脚本 (使用torchrun)
# 适用于PyTorch 1.10及以上版本

# 显示环境变量信息
echo "===== 环境变量信息 ====="
echo "MLP_WORKER_0_HOST: ${MLP_WORKER_0_HOST}"
echo "MLP_WORKER_0_PORT: ${MLP_WORKER_0_PORT}"
echo "MLP_ROLE_INDEX: ${MLP_ROLE_INDEX}"
echo "MLP_WORKER_NUM: ${MLP_WORKER_NUM}"
echo "MLP_WORKER_GPU: ${MLP_WORKER_GPU}"
echo "========================="

# 确保代码文件有执行权限
chmod +x /fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/cmds/multi_node/test_gpu.py

# 使用torchrun启动训练 (PyTorch 1.10+推荐的方式)
# torchrun是torch.distributed.launch的新版本替代品
torchrun \
    --nnodes=${MLP_WORKER_NUM} \
    --node_rank=${MLP_ROLE_INDEX} \
    --nproc_per_node=${MLP_WORKER_GPU} \
    --master_addr=${MLP_WORKER_0_HOST} \
    --master_port=${MLP_WORKER_0_PORT} \
    /fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory/cmds/multi_node/test_gpu.py \
    --epochs 10 \
    --batch_size 32 \
    --model resnet50

