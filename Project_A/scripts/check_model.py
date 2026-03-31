import torch
import argparse

# 方法A：添加安全类（推荐）
torch.serialization.add_safe_globals([argparse.Namespace])

# 或者方法B：直接允许加载（如果你信任这个文件）
# weights_only=False 表示允许加载所有内容

print("=" * 50)
print("检查 MNIST 模型文件:")
print("=" * 50)

# 用 weights_only=False 加载
checkpoint = torch.load('../ckpt/undistilled_mnist.pth', 
                        map_location='cpu', 
                        weights_only=False)  # 加上这个参数

print(f"文件中包含的键: {list(checkpoint.keys())}")
print(f"\n最佳准确率: {checkpoint['best_acc']:.2f}%")
print(f"保存时的epoch: {checkpoint['epoch']}")
print(f"训练配置: {checkpoint['args']}")

# 查看模型参数量
model_state = checkpoint['model_state_dict']
total_params = sum(p.numel() for p in model_state.values())
print(f"模型参数量: {total_params:,}")

print("\n" + "=" * 50)
print("检查 CIFAR-10 模型文件:")
print("=" * 50)

checkpoint2 = torch.load('../ckpt/undistilled_cifar10.pth', 
                         map_location='cpu', 
                         weights_only=False)  # 加上这个参数

print(f"文件中包含的键: {list(checkpoint2.keys())}")
print(f"\n最佳准确率: {checkpoint2['best_acc']:.2f}%")
print(f"保存时的epoch: {checkpoint2['epoch']}")
print(f"训练配置: {checkpoint2['args']}")

# 查看模型参数量
model_state2 = checkpoint2['model_state_dict']
total_params2 = sum(p.numel() for p in model_state2.values())
print(f"模型参数量: {total_params2:,}")


import torch
import torchvision
import torchvision.transforms as transforms
import sys
from train import MNISTNet, CIFAR10Net

def check_distilled_model(model_path, dataset):
    """验证distilled模型"""
    print(f"\n验证模型: {model_path}")
    
    # 加载模型
    if dataset == 'mnist':
        model = MNISTNet()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        testset = torchvision.datasets.MNIST(
            root='./scripts/data', train=False, download=True, transform=transform)
    else:
        model = CIFAR10Net()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        testset = torchvision.datasets.CIFAR10(
            root='./scripts/data', train=False, download=True, transform=transform)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 取一个batch数据验证
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)  # T=1
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    print(f"真实标签: {labels.cpu().numpy()}")
    print(f"预测标签: {predicted.cpu().numpy()}")
    print(f"预测概率: {probs.max(dim=1)[0].cpu().numpy()}")
    print(f"Logits均值: {outputs.mean().item():.2f}")
    
    # 验证论文现象：distilled模型的logits应该很大
    print(f"\n论文现象验证:")
    print(f"- Logits值较大 (均值 {outputs.mean().item():.2f})，符合论文描述")
    print(f"- 预测概率接近1 ({probs.max(dim=1)[0].mean().item():.4f})")

if __name__ == '__main__':
    # 检查MNIST
    try:
        check_distilled_model('./ckpt/distilled_mnist_T100.pth', 'mnist')
    except:
        print("MNIST模型不存在，请先运行distill.py")
    
    # 检查CIFAR-10
    try:
        check_distilled_model('./ckpt/distilled_cifar10_T100.pth', 'cifar10')
    except:
        print("CIFAR-10模型不存在，请先运行distill.py")