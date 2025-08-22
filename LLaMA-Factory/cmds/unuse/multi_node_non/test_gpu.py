#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch DDP分布式训练示例代码
适用于火山引擎机器学习平台
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch DDP分布式训练示例')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='每个GPU的批次大小')
    parser.add_argument('--model', type=str, default='resnet18', help='模型名称')
    parser.add_argument('--local_rank', type=int, default=-1, help='本地排名 (由torch.distributed.launch设置)')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    
    return parser.parse_args()

def setup_distributed():
    """设置分布式环境"""
    # torch.distributed.launch会设置以下环境变量
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    
    # 打印分布式训练的环境信息
    print(f'本地排名: {local_rank}, 全局排名: {rank}, 世界大小: {world_size}')
    print(f'节点排名: {os.environ.get("MLP_ROLE_INDEX", "未设置")}')
    print(f'节点数量: {os.environ.get("MLP_WORKER_NUM", "未设置")}')
    
    # 初始化进程组
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
    return local_rank, rank, world_size

def get_model(model_name):
    """获取指定的模型"""
    if model_name == 'resnet18':
        return models.resnet18(pretrained=False)
    elif model_name == 'resnet50':
        return models.resnet50(pretrained=False)
    else:
        raise ValueError(f'不支持的模型: {model_name}')

def prepare_data(args, world_size, rank):
    """准备数据加载器"""
    # 定义数据转换
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据目录（如果不存在）
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 加载数据集（这里使用CIFAR-10演示，实际应用请替换为您的数据集）
    # 注意：首次运行会下载数据集
    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 使用DistributedSampler来分配数据到不同进程
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, train_sampler

def main():
    # 解析命令行参数
    args = setup_args()
    
    # 设置分布式环境
    local_rank, rank, world_size = setup_distributed()
    
    # 只在主进程中打印信息
    is_main_process = rank == 0
    
    # 创建输出目录
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'输出目录: {args.output_dir}')
    
    # 准备数据
    train_loader, train_sampler = prepare_data(args, world_size, rank)
    
    # 加载模型
    model = get_model(args.model)
    
    # 将模型移动到当前设备
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 为CIFAR-10调整模型（10个类别）
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model.fc = model.fc.to(device)
    
    # 使用DDP包装模型
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # 训练模型
    for epoch in range(args.epochs):
        # 设置训练模式
        model.train()
        
        # 必须设置sampler的epoch来保证shuffle正常工作
        train_sampler.set_epoch(epoch)
        
        # 记录损失
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练循环
        for i, (images, targets) in enumerate(train_loader):
            # 将数据移动到当前设备
            images, targets = images.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 每100个批次打印一次信息（只在主进程中打印）
            if is_main_process and (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        # 每个epoch结束后保存模型（只在主进程中保存）
        if is_main_process:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            
            # 保存最新的检查点
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_latest.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'模型已保存到 {checkpoint_path}')
    
    # 训练结束
    if is_main_process:
        print('训练完成!')
        
        # 保存最终模型
        final_model_path = os.path.join(args.output_dir, f'model_final.pth')
        torch.save(model.module.state_dict(), final_model_path)
        print(f'最终模型已保存到 {final_model_path}')
        
if __name__ == '__main__':
    main()
    