import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
import timm
import os
from PIL import Image
from pathlib import Path
from slot_attention import SlotAttention

#TODO implement dropout
#implement low rank
# Custom Dataset for EuroSAT
class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TransformSubset(Dataset):
    """Wrapper to apply different transforms to a subset"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.subset)

def create_dataloaders(data_dir, batch_size=32, train_split=0.7, val_split=0.15, test_split=0.15):
    """Create train/val/test dataloaders with 70/15/15 split"""
    
    # Verify splits sum to 1.0
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = EuroSATDataset(data_dir, transform=transform_train)
    val_dataset = EuroSATDataset(data_dir, transform=transform_eval)
    test_dataset = EuroSATDataset(data_dir, transform=transform_eval)
    
    # Calculate split sizes
    total_size = len(train_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Total samples: {total_size}")
    print(f"Split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Get indices for each split
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Create dataloaders - num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_subset, batch_size=batch_size, 
                            shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, 
                          shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_eurosat(
    model_name='mobilevitv2_075.cvnets_in1k',
    data_dir='EuroSATallBands',
    batch_size=32,
    epochs=50,
    lr=1e-4,
    weight_decay=0.01,
    device='cuda'
):
    """Train on EuroSAT dataset with train/val/test split"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_dir, batch_size=batch_size, 
        train_split=0.7, val_split=0.15, test_split=0.15
    )
    
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {classes}\n")
    
    # Create model
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes,drop_rate=0.1)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {total_params:.2f}M, Trainable: {trainable_params:.2f}M")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nTraining {model_name} on EuroSAT")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}\n")
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': model_name,
                'num_classes': num_classes,
                'classes': classes,
            }
            save_path = f'best_{model_name.replace("/", "_")}_eurosat.pth'
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best model to {save_path} (Val Acc: {val_acc:.2f}%)")
        
        print()
    
    print(f"Training complete! Best Val Acc: {best_val_acc:.2f}%")
    
    # Final test evaluation
    print("\n" + "="*70)
    print("Evaluating on TEST set")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save final results
    checkpoint['test_acc'] = test_acc
    torch.save(checkpoint, save_path)
    
    return model, test_acc


if __name__ == '__main__':
    # Train models
    models_to_train = [
        'mobilevitv2_075.cvnets_in1k',
    ]
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*70}")
        print(f"Training {model_name}")
        print(f"{'='*70}\n")
        
        model, test_acc = train_eurosat(
            model_name=model_name,
            data_dir=r'./EuroSAT_RGB',
            batch_size=32,
            epochs=50,
            lr=1e-4,
            weight_decay=0.01
        )
        
        results[model_name] = test_acc
    
    # Print final comparison
    print("\n" + "="*20)
    print("FINAL TEST RESULTS")
    print("="*20)
    for model_name, test_acc in results.items():
        print(f"{model_name}: {test_acc:.2f}%")