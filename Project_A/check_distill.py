import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms

# 添加scripts目录到路径
scripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

print(f"添加的路径: {scripts_path}")

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
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    print(f"真实标签: {labels.cpu().numpy()}")
    print(f"预测标签: {predicted.cpu().numpy()}")
    print(f"预测概率: {probs.max(dim=1)[0].cpu().numpy()}")
    print(f"Logits均值: {outputs.mean().item():.2f}")
    print(f"Logits范围: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]")
    
    # 验证论文现象
    # 验证论文现象
    print(f"\n论文现象验证:")
    print(f"Logits均值: {outputs.mean().item():.2f}")

if __name__ == '__main__':
    # 检查MNIST
    try:
        check_distilled_model('./ckpt/distilled_mnist_T100.pth', 'mnist')
    except FileNotFoundError:
        print("MNIST模型不存在，请先运行distill.py")
    except Exception as e:
        print(f"MNIST验证出错: {e}")
    
    print("\n" + "="*50)
    
    # 检查CIFAR-10
    try:
        check_distilled_model('./ckpt/distilled_cifar10_T100.pth', 'cifar10')
    except FileNotFoundError:
        print("CIFAR-10模型不存在，请先运行distill.py")
    except Exception as e:
        print(f"CIFAR-10验证出错: {e}")