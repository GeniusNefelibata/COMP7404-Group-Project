import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from attacks.cw_l2 import CWL2Attack, save_attack_results
from attacks.utils import compute_metrics

# ========== CIFAR-10 Model from Task A ==========
class CIFAR10Net(nn.Module):
    """CIFAR-10 model architecture from Paper Table I/II"""
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv1(x)))
        
        # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv2(x)))
        
        # Keep 8x8
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # 8x8 -> 4x4
        x = self.pool(x)
        
        # flatten: [batch, 128, 4, 4] -> [batch, 2048]
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Return logits
        return x

import matplotlib.pyplot as plt
import numpy as np

def visualize_attack(images, adv_images, labels, targets, save_path):
    """Generate visualization of attack results"""
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    classes = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(min(5, len(images))):
        # Original
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original\n{classes[labels[i]]}')
        axes[0, i].axis('off')
        
        # Adversarial
        adv = adv_images[i].cpu().numpy().transpose(1, 2, 0)
        adv = np.clip(adv, 0, 1)
        axes[1, i].imshow(adv)
        axes[1, i].set_title(f'Adversarial\n{classes[targets[i]]}')
        axes[1, i].axis('off')
        
        # Perturbation
        pert = np.abs(adv - img)
        pert = (pert * 10).clip(0, 1)  # Amplify
        axes[2, i].imshow(pert, cmap='hot')
        axes[2, i].set_title('Perturbation\n(10x)')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def load_cifar_data(batch_size: int = 100, num_samples: int = 1000):
    """Load CIFAR-10 test data"""
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./scripts/data', train=False, download=True, transform=transform
    )
    
    indices = torch.randperm(len(testset))[:num_samples]
    test_subset = torch.utils.data.Subset(testset, indices)
    
    testloader = torch.utils.data.DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return testloader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Attack parameters
    CONFIDENCE = 0.0  # kappa = 0
    # NUM_SAMPLES = 10  # Small sample for testing
    NUM_SAMPLES = 1000  # Small sample for testing
    BATCH_SIZE = 50
    
    models_to_test = [
        {
            'name': 'undistilled',
            'path': 'ckpt/undistilled_cifar10.pth',
            'accuracy': 81.72
        },
        {
            'name': 'distilled',
            'path': 'ckpt/distilled_cifar10_T100.pth',
            'accuracy': 69.47
        }
    ]
    
    print("Loading CIFAR-10 test data...")
    testloader = load_cifar_data(batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES)
    
    for model_info in models_to_test:
        print(f"\n{'='*60}")
        print(f"Attacking {model_info['name']} CIFAR-10 model...")
        print(f"Model path: {model_info['path']}")
        
        # Create model instance
        model = CIFAR10Net().to(device)
        
        # Load checkpoint (handle PyTorch 2.6+ compatibility)
        try:
            checkpoint = torch.load(model_info['path'], map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_info['path'], map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        
        model.eval()
        print("✅ Model loaded successfully")
        
        # Initialize C&W L2 attack
        attack = CWL2Attack(
            model=model,
            device=device,
            targeted=True,
            confidence=CONFIDENCE,
            learning_rate=0.01,
            max_iterations=1000,
            binary_search_steps=9,
            initial_const=0.001,
            abort_early=True,
            clip_min=0.0,
            clip_max=1.0
        )
        
        all_results = []
        SAVE_INTERVAL = 5
        
        for batch_idx, (images, labels) in enumerate(testloader):
            print(f"Processing batch {batch_idx+1}/{len(testloader)}...")
            
            # Randomly select target classes (exclude original class)
            targets = (labels + torch.randint(1, 10, labels.shape)) % 10
            
            results, adv_images = attack.attack_batch(
                images=images,
                labels=labels,
                target_classes=targets
            )
            
            all_results.extend(results)

            # ✅ 定期保存中间结果
            if (batch_idx + 1) % SAVE_INTERVAL == 0:
                temp_path = f"outputs/logs/temp_cifar_{model_info['name']}_batch{batch_idx+1}.json"
                save_attack_results(all_results, temp_path)
                print(f" Saved intermediate results to {temp_path}")

            # 第一个batch生成可视化
            if batch_idx == 0 and adv_images is not None:
                # 创建figures目录
                vis_path = f"outputs/figures/cifar_{model_info['name']}_samples.png"
                os.makedirs('outputs/figures', exist_ok=True)
                
                # 调用可视化函数
                visualize_attack(
                    images=images,
                    adv_images=adv_images,
                    labels=labels,
                    targets=targets,
                    save_path=vis_path
                )
        
        # Compute metrics
        metrics = compute_metrics(all_results)
        print(f"\nAttack Results:")
        print(f"  Success Rate: {metrics['success_rate']:.2f}%")
        print(f"  Mean L2 Distance: {metrics['mean_l2']:.4f}")
        print(f"  Successful Samples: {metrics['successful_samples']}/{metrics['total_samples']}")
        
        # Save results
        output_path = f"outputs/logs/cw_l2_cifar_{model_info['name']}_kappa{CONFIDENCE}.json"
        save_attack_results(all_results, output_path)


if __name__ == '__main__':
    main()