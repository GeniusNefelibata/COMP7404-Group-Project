import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import random
from tqdm import tqdm
import time

# ========== Training LogSet a fixed random seed (to ensure reproducibility) ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========= MNIST（Table I/II） ==========
class MNISTNet(nn.Module):
    """论文Table I/II的MNIST模型结构"""
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
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# ========== CIFAR-10（Table I/II） ==========
class CIFAR10Net(nn.Module):
    """论文Table I/II的CIFAR-10模型结构 """
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        #  128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # pool: 32x32 -> 16x16
        x = self.pool(torch.relu(self.conv1(x)))
        
        # pool: 16x16 -> 8x8
        x = self.pool(torch.relu(self.conv2(x)))
        
        # No pooling, keep 8x8
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        
        # Last pooling: 8x8 -> 4x4
        x = self.pool(x)
        
        # flatten: [batch, 128, 4, 4] -> [batch, 2048]
        x = torch.flatten(x, 1)
        
        # Fully Connected Layer
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ========== Data Loading ==========
def get_data_loaders(dataset, batch_size=128):
    """Load the MNIST or CIFAR-10 dataset"""
    
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    return trainloader, testloader

# ========== training ==========
def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch}')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss/(total/inputs.size(0)),
            'acc': 100.*correct/total
        })
    
    return running_loss/len(trainloader), 100.*correct/total

# ========== testing ==========
def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return test_loss/len(testloader), accuracy

# ========== main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='../ckpt')
    args = parser.parse_args()
    
    # fixed random seed
    set_seed(args.seed)
    
    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Use device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'Hyperparameter: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Loading data
    trainloader, testloader = get_data_loaders(args.dataset, args.batch_size)
    
    # Create model
    if args.dataset == 'mnist':
        model = MNISTNet().to(device)
    else:
        model = CIFAR10Net().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training Log
    log_file = f'../runs/train_{args.dataset}_seed{args.seed}.txt'
    os.makedirs('../runs', exist_ok=True)
    
    best_acc = 0.0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, device, epoch)
        test_loss, test_acc = test(model, testloader, criterion, device)
        scheduler.step()
        
        # Print log
        log_msg = f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%'
        print(log_msg)
        
        # Save files
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_path = f'{args.save_dir}/undistilled_{args.dataset}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }, checkpoint_path)
            print(f'✅ Save the best model {checkpoint_path}')
    
    print(f"\n🎉 Training complete! Best test accuracy: {best_acc:.2f}%")
    
    # Final test
    print("\nFinal test:")
    test_loss, test_acc = test(model, testloader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    main()