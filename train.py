import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import MNISTNet
from datetime import datetime
import torch.nn.functional as F
import os
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model initialization
    model = MNISTNet().to(device)
    
    # Print model parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        # loss = criterion(output, target)
        loss = F.nll_loss(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        # Update progress bar
        accuracy = 100 * correct / total
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'accuracy': f'{accuracy:.2f}%'
        })
    
    final_accuracy = 100 * correct / total
    print(f'\nFinal Training Accuracy: {final_accuracy:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    accuracy_str = f"{final_accuracy:.1f}"
    save_path = f'models/mnist_model_{timestamp}_acc{accuracy_str}.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

if __name__ == '__main__':
    train() 