import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from attacks.cw_l2 import CWL2Attack, save_attack_results
from attacks.utils import compute_metrics

# ========== MNIST Model from Task A ==========
class MNISTNet(nn.Module):
    """MNIST model architecture from Paper Table I/II"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # Return logits (no softmax) for C&W attack
        return x

def load_mnist_data(batch_size: int = 100, num_samples: int = 1000):
    """Load MNIST test data"""
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    testset = torchvision.datasets.MNIST(
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
    NUM_SAMPLES = 1000
    BATCH_SIZE = 50
    
    models_to_test = [
        {
            'name': 'undistilled',
            'path': 'ckpt/undistilled_mnist.pth',
            'accuracy': 99.35
        },
        {
            'name': 'distilled',
            'path': 'ckpt/distilled_mnist_T100.pth',
            'accuracy': 99.35
        }
    ]
    
    print("Loading MNIST test data...")
    testloader = load_mnist_data(batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES)
    
    for model_info in models_to_test:
        print(f"\n{'='*60}")
        print(f"Attacking {model_info['name']} MNIST model...")
        print(f"Model path: {model_info['path']}")
        
        # Create model instance
        model = MNISTNet().to(device)
        
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
        
        # Compute metrics
        metrics = compute_metrics(all_results)
        print(f"\nAttack Results:")
        print(f"  Success Rate: {metrics['success_rate']:.2f}%")
        print(f"  Mean L2 Distance: {metrics['mean_l2']:.4f}")
        print(f"  Successful Samples: {metrics['successful_samples']}/{metrics['total_samples']}")
        
        # Save results
        output_path = f"outputs/logs/cw_l2_mnist_{model_info['name']}_kappa{CONFIDENCE}.json"
        save_attack_results(all_results, output_path)

if __name__ == '__main__':
    main()