import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import os

# model
from rope_vit import RoPEViT
from rope_performer import RopePerformerViT

def create_cifar10_dataloaders(batch_size=128, num_workers=4):
    """
    Create CIFAR-10 dataloaders with transforms.
    """

    # CIFAR-10 mean and std for normalization https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize(224, antialias=True)  # Upsample to ViT/Performer input size
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize(224, antialias=True)  # Upsample to ViT/Performer input size
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

class TrainingConfig:
    """Configuration class for training hyperparameters"""
    def __init__(self):
        # Model parameters
        self.image_size = 224
        self.patch_size = 16
        self.num_classes = 10
        self.dim = 384
        self.depth = 12
        self.heads = 6
        self.mlp_dim = 1536
        self.model_type = 'vit' # 'vit' or 'performer'
        self.rope_type = 'axial'  # 'axial' or 'mixed'
        self.rope_theta = 10.0   # 100.0 or 10.0
        
        # Training parameters
        self.epochs = 100
        self.batch_size = 128
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.warmup_epochs = 5
        
        # Hardware
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_scheduler(optimizer, config, num_training_steps):
    """Create a learning rate scheduler with warmup"""
    def lr_lambda(current_step: int):
        # Warmup steps
        warmup_steps = config.warmup_epochs * num_training_steps
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_loader, criterion, optimizer, scheduler, config, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # progress bar
        progress_bar.set_postfix({'loss': running_loss/(batch_idx+1), 'acc': 100.*correct/total})
    
    return running_loss/len(train_loader), 100.*correct/total

def evaluate(model, test_loader, criterion, config):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/len(test_loader), 100.*correct/total

def save_training_metrics(history, config):
    """
    Save training metrics to a CSV file
    """

    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create DataFrame with all metrics
    metrics_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'train_accuracy': history['train_acc'],
        'test_loss': history['test_loss'],
        'test_accuracy': history['test_acc']
    })

    model_type = config.model_type
    
    # Add configuration information as metadata
    metadata = {
        'model_type': model_type,
        'rope_type': config.rope_type,
        'rope_theta': config.rope_theta,
        'image_size': config.image_size,
        'patch_size': config.patch_size,
        'embedding_dim': config.dim,
        'depth': config.depth,
        'heads': config.heads,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'batch_size': config.batch_size,
        'warmup_epochs': config.warmup_epochs,
        'total_epochs': config.epochs
    }
    
    # Save metrics to CSV
    metrics_file = f'training_metrics_{model_type}_{config.rope_type}_{timestamp}.csv'
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save metadata to JSON
    metadata_file = f'training_config_{model_type}_{config.rope_type}_{timestamp}.json'
    pd.Series(metadata).to_json(metadata_file)
    
    return metrics_file, metadata_file

def main():
    config = TrainingConfig()
    
    train_loader, test_loader = create_cifar10_dataloaders(batch_size=config.batch_size)
    
    if config.model_type == 'performer':
        model = RopePerformerViT(
                image_size=config.image_size,
                patch_size=config.patch_size,
                num_classes=config.num_classes,
                dim=config.dim,
                depth=config.depth - 4,
                heads=config.heads - 3,
                rope_type = 'mixed'  # or 'mixed' for diagonal RoPE
            ).to(config.device)
    else:
        model = RoPEViT(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            rope_type=config.rope_type,
            rope_theta=config.rope_theta
        ).to(config.device)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    num_training_steps = len(train_loader) * config.epochs
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    # Training loop
    best_acc = 0
    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, config, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, config)
        
        # Save
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            # delete previous best model
            if f'best_model_{config.model_type}_{config.rope_type}' in locals():
                os.remove(model_file)

            best_acc = test_acc
            model_file = f'best_model_{config.model_type}_{config.rope_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': vars(config)
            }, model_file)

            
        
        print(f'\nEpoch {epoch+1}/{config.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Save all training metrics
    metrics_file, metadata_file = save_training_metrics(history, config)
    print(f'\nTraining metrics saved to {metrics_file}')
    print(f'Training metadata saved to {metadata_file}')
    
    return history

if __name__ == '__main__':
    history = main()