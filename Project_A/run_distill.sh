#!/bin/bash

# 确保在项目根目录
cd "$(dirname "$0")/.."

echo "========================================="
echo "Defensive Distillation 完整流程"
echo "========================================="

# 检查GPU
python -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 运行MNIST蒸馏
echo ""
echo "正在运行 MNIST 蒸馏..."
python distill.py --dataset mnist --temperature 100 --seed 42 --step all

# 运行CIFAR-10蒸馏
echo ""
echo "正在运行 CIFAR-10 蒸馏..."
python distill.py --dataset cifar10 --temperature 100 --seed 42 --step all

echo ""
echo "========================================="
echo "蒸馏完成！生成的文件："
ls -la ckpt/teacher_* ckpt/distilled_* 2>/dev/null || echo "文件未找到"
echo "========================================="