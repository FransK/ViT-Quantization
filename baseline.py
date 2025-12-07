import copy
import os
import warnings
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from torch.ao.quantization import quantize_dynamic
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from slot_attention import SlotAttention

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


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(data_dir, batch_size=32, train_split=0.7, val_split=0.15, test_split=0.15):
    """Create train/val/test dataloaders with 70/15/15 split"""
    
    # Verify splits sum to 1.0
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Data transforms
    transform_train = get_train_transform()
    transform_eval = get_eval_transform()
    
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
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
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


def get_predictions(model, dataloader, device):
    """Get all predictions and labels from a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def print_classification_report(model, dataloader, device, classes, model_name="Model"):
    """Print detailed classification report for a model."""
    preds, labels = get_predictions(model, dataloader, device)
    
    print(f"\n{'='*70}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*70}")
    print(classification_report(labels, preds, target_names=classes, digits=4))
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'predictions': preds,
        'labels': labels
    }


def compare_models(baseline_metrics, quantized_metrics, baseline_name="Baseline", quantized_name="Quantized"):
    """Print comparison between baseline and quantized model metrics."""
    print(f"\n{'='*70}")
    print(f"Comparison: {baseline_name} vs {quantized_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Baseline':>12} {'Quantized':>12} {'Diff':>12}")
    print(f"{'-'*51}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        baseline_val = baseline_metrics[metric]
        quant_val = quantized_metrics[metric]
        diff = quant_val - baseline_val
        sign = '+' if diff >= 0 else ''
        print(f"{metric.capitalize():<15} {baseline_val:>11.2f}% {quant_val:>11.2f}% {sign}{diff:>10.2f}%")
    
    print(f"{'-'*51}")

def get_model_size_mb(model_path):
    """Get model size in megabytes."""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)
    return 0.0


def get_model_memory_mb(model):
    """
    Calculate actual memory footprint of a model's parameters.
    This gives accurate size regardless of how the model is saved.
    """
    total_bytes = 0
    for name, param in model.named_parameters():
        # Get the actual dtype size
        dtype = param.dtype
        if dtype == torch.float32:
            bytes_per_elem = 4
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            bytes_per_elem = 2
        elif dtype == torch.int8 or dtype == torch.qint8 or dtype == torch.quint8:
            bytes_per_elem = 1
        else:
            bytes_per_elem = param.element_size()
        
        total_bytes += param.numel() * bytes_per_elem
    
    # Also count buffers (running_mean, running_var, etc.)
    for name, buf in model.named_buffers():
        if buf is not None:
            total_bytes += buf.numel() * buf.element_size()
    
    return total_bytes / (1024 * 1024)


def count_quantized_params(model):
    """
    Count parameters by precision level in a model.
    Returns dict with counts for FP32, INT8, and total.
    """
    fp32_params = 0
    int8_params = 0
    other_params = 0
    
    # Check for dynamically quantized modules
    quantized_linear_count = 0
    fp32_linear_count = 0
    fp32_conv_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.dynamic.Linear):
            quantized_linear_count += 1
            # Quantized linear stores weights as INT8
            if hasattr(module, 'weight'):
                int8_params += module.weight().numel()
        elif isinstance(module, nn.Linear):
            fp32_linear_count += 1
            if hasattr(module, 'weight') and module.weight is not None:
                fp32_params += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                fp32_params += module.bias.numel()
        elif isinstance(module, nn.Conv2d):
            fp32_conv_count += 1
            if hasattr(module, 'weight') and module.weight is not None:
                fp32_params += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                fp32_params += module.bias.numel()
    
    total_params = fp32_params + int8_params
    
    # Calculate theoretical sizes
    fp32_size_mb = (fp32_params * 4) / (1024 * 1024)  # 4 bytes per FP32
    int8_size_mb = (int8_params * 1) / (1024 * 1024)  # 1 byte per INT8
    total_size_mb = fp32_size_mb + int8_size_mb
    
    # What size would be if all params were FP32
    baseline_size_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        'fp32_params': fp32_params,
        'int8_params': int8_params,
        'total_params': total_params,
        'fp32_size_mb': fp32_size_mb,
        'int8_size_mb': int8_size_mb,
        'total_size_mb': total_size_mb,
        'baseline_size_mb': baseline_size_mb,
        'compression_ratio': baseline_size_mb / total_size_mb if total_size_mb > 0 else 1.0,
        'quantized_linear_layers': quantized_linear_count,
        'fp32_linear_layers': fp32_linear_count,
        'fp32_conv_layers': fp32_conv_count,
        'percent_quantized': 100 * int8_params / total_params if total_params > 0 else 0,
    }


def print_size_comparison(size_stats, checkpoint_path, ptq_path, qat_path, static_path=None):
    """
    Print size comparison from pre-computed stats.
    Memory efficient - doesn't require models to be in memory.
    """
    print(f"\n{'='*80}")
    print("Model Size Comparison (Accurate Parameter Analysis)")
    print(f"{'='*80}")
    
    # Analyze baseline model
    if 'baseline' in size_stats:
        stats = size_stats['baseline']
        print(f"\nBaseline FP32 Model:")
        print(f"  Total parameters: {stats['total_params']:,}")
        print(f"  Memory footprint: {stats['baseline_size_mb']:.2f} MB")
        print(f"  Conv2d layers: {stats['fp32_conv_layers']}")
        print(f"  Linear layers: {stats['fp32_linear_layers']}")
    
    # Analyze PTQ model  
    if 'ptq' in size_stats:
        stats = size_stats['ptq']
        print(f"\nPTQ Dynamic Quantized Model:")
        print(f"  FP32 parameters: {stats['fp32_params']:,} ({stats['fp32_size_mb']:.2f} MB)")
        print(f"  INT8 parameters: {stats['int8_params']:,} ({stats['int8_size_mb']:.2f} MB)")
        print(f"  Quantized Linear layers: {stats['quantized_linear_layers']}")
        print(f"  FP32 Conv2d layers: {stats['fp32_conv_layers']} (NOT quantized)")
        print(f"  Actual memory: {stats['total_size_mb']:.2f} MB")
        print(f"  True compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Percent of params quantized: {stats['percent_quantized']:.1f}%")
    
    # Analyze QAT model
    if 'qat' in size_stats:
        stats = size_stats['qat']
        print(f"\nQAT Dynamic Quantized Model:")
        print(f"  FP32 parameters: {stats['fp32_params']:,} ({stats['fp32_size_mb']:.2f} MB)")
        print(f"  INT8 parameters: {stats['int8_params']:,} ({stats['int8_size_mb']:.2f} MB)")
        print(f"  Quantized Linear layers: {stats['quantized_linear_layers']}")
        print(f"  FP32 Conv2d layers: {stats['fp32_conv_layers']} (NOT quantized)")
        print(f"  Actual memory: {stats['total_size_mb']:.2f} MB")
        print(f"  True compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Percent of params quantized: {stats['percent_quantized']:.1f}%")
    
    # File size comparison (for reference)
    print(f"\n{'-'*80}")
    print("File Sizes on Disk (for reference only):")
    print(f"{'-'*80}")
    
    paths = {
        "Baseline (.pth)": checkpoint_path,
        "PTQ INT8 (.pth)": ptq_path,
        "QAT INT8 (.pth)": qat_path,
    }
    if static_path and os.path.exists(static_path):
        paths["Static INT8 (.pth)"] = static_path
    
    for name, path in paths.items():
        if os.path.exists(path):
            size = get_model_size_mb(path)
            print(f"  {name}: {size:.2f} MB")
    
    print(f"{'-'*80}")


def compare_model_sizes(checkpoint_path, ptq_path, qat_path, static_path=None, ptq_model=None, qat_model=None, baseline_model=None):
    """
    Print a comparison of model sizes and compression ratios.
    Uses actual parameter analysis for accurate measurements.
    """
    print(f"\n{'='*80}")
    print("Model Size Comparison (Accurate Parameter Analysis)")
    print(f"{'='*80}")
    
    results = {}
    
    # Analyze baseline model
    if baseline_model is not None:
        stats = count_quantized_params(baseline_model)
        results['Baseline (FP32)'] = stats
        print(f"\nBaseline FP32 Model:")
        print(f"  Total parameters: {stats['total_params']:,}")
        print(f"  Memory footprint: {stats['baseline_size_mb']:.2f} MB")
        print(f"  Conv2d layers: {stats['fp32_conv_layers']}")
        print(f"  Linear layers: {stats['fp32_linear_layers']}")
    
    # Analyze PTQ model  
    if ptq_model is not None:
        stats = count_quantized_params(ptq_model)
        results['PTQ (Dynamic)'] = stats
        print(f"\nPTQ Dynamic Quantized Model:")
        print(f"  FP32 parameters: {stats['fp32_params']:,} ({stats['fp32_size_mb']:.2f} MB)")
        print(f"  INT8 parameters: {stats['int8_params']:,} ({stats['int8_size_mb']:.2f} MB)")
        print(f"  Quantized Linear layers: {stats['quantized_linear_layers']}")
        print(f"  FP32 Conv2d layers: {stats['fp32_conv_layers']} (NOT quantized)")
        print(f"  Actual memory: {stats['total_size_mb']:.2f} MB")
        print(f"  True compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Percent of params quantized: {stats['percent_quantized']:.1f}%")
    
    # Analyze QAT model
    if qat_model is not None:
        stats = count_quantized_params(qat_model)
        results['QAT (Dynamic)'] = stats
        print(f"\nQAT Dynamic Quantized Model:")
        print(f"  FP32 parameters: {stats['fp32_params']:,} ({stats['fp32_size_mb']:.2f} MB)")
        print(f"  INT8 parameters: {stats['int8_params']:,} ({stats['int8_size_mb']:.2f} MB)")
        print(f"  Quantized Linear layers: {stats['quantized_linear_layers']}")
        print(f"  FP32 Conv2d layers: {stats['fp32_conv_layers']} (NOT quantized)")
        print(f"  Actual memory: {stats['total_size_mb']:.2f} MB")
        print(f"  True compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Percent of params quantized: {stats['percent_quantized']:.1f}%")
    
    # File size comparison (for reference)
    print(f"\n{'-'*80}")
    print("File Sizes on Disk (less accurate - includes serialization overhead):")
    print(f"{'-'*80}")
    print(f"{'Model Type':<25} {'File Size':>12} {'Note':<40}")
    print(f"{'-'*80}")
    
    paths = {
        "Baseline (.pth)": checkpoint_path,
        "PTQ INT8 (.pth)": ptq_path,
        "QAT INT8 (.pth)": qat_path,
    }
    if static_path:
        paths["Static INT8 (.pth)"] = static_path
    
    for name, path in paths.items():
        if os.path.exists(path):
            size = get_model_size_mb(path)
            note = "(includes optimizer state)" if "Baseline" in name else ""
            print(f"{name:<25} {size:>10.2f} MB {note}")
        else:
            print(f"{name:<25} {'Not Found':>12}")
    
    print(f"{'-'*80}")
    
    return results

def run_full_evaluation(checkpoint_path, ptq_path, qat_path, data_dir, batch_size=32):
    """
    Run complete evaluation comparing baseline, PTQ, and QAT models.
    Prints classification reports and comparisons for all three.
    """
    _, _, test_loader, classes = create_dataloaders(data_dir, batch_size=batch_size)
    
    results = {}
    size_stats = {}  # Store size stats instead of keeping models in memory
    
    # Evaluate baseline (FP32)
    if os.path.exists(checkpoint_path):
        print("\nLoading baseline FP32 model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        baseline_model, checkpoint = load_float_model_from_checkpoint(checkpoint_path, device=device)
        
        # Get size stats immediately, then we can delete if needed
        size_stats['baseline'] = count_quantized_params(baseline_model)
        
        results['baseline'] = print_classification_report(
            baseline_model, test_loader, device, classes, 
            model_name=f"Baseline FP32 ({checkpoint['model_name']})"
        )
        del baseline_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"Baseline checkpoint not found: {checkpoint_path}")
    
    # Evaluate PTQ model (dynamic quantization - must run on CPU)
    if os.path.exists(ptq_path):
        print("\nLoading PTQ INT8 model...")
        ptq_checkpoint = torch.load(ptq_path, map_location='cpu')
        backend = ptq_checkpoint.get('backend', 'x86')
        torch.backends.quantized.engine = backend
        
        # Recreate dynamically quantized model
        float_model, _ = load_float_model_from_checkpoint(checkpoint_path, device=torch.device('cpu'))
        ptq_model = quantize_dynamic(float_model, {nn.Linear}, dtype=torch.qint8)
        del float_model  # Free memory immediately
        
        ptq_model.load_state_dict(ptq_checkpoint['model_state_dict'])
        ptq_model.eval()
        
        # Get size stats immediately
        size_stats['ptq'] = count_quantized_params(ptq_model)
        
        # Create CPU test loader
        _, _, cpu_test_loader, _ = create_dataloaders(data_dir, batch_size=batch_size)
        results['ptq'] = print_classification_report(
            ptq_model, cpu_test_loader, torch.device('cpu'), classes,
            model_name="PTQ INT8 (Dynamic)"
        )
        del ptq_model
    else:
        print(f"PTQ checkpoint not found: {ptq_path}")
    
    # Evaluate QAT model (dynamic quantization - must run on CPU)
    if os.path.exists(qat_path):
        print("\nLoading QAT INT8 model...")
        qat_checkpoint = torch.load(qat_path, map_location='cpu')
        backend = qat_checkpoint.get('backend', 'x86')
        torch.backends.quantized.engine = backend
        
        # Recreate dynamically quantized model
        float_model, _ = load_float_model_from_checkpoint(checkpoint_path, device=torch.device('cpu'))
        qat_model = quantize_dynamic(float_model, {nn.Linear}, dtype=torch.qint8)
        del float_model  # Free memory immediately
        
        qat_model.load_state_dict(qat_checkpoint['model_state_dict'])
        qat_model.eval()
        
        # Get size stats immediately
        size_stats['qat'] = count_quantized_params(qat_model)
        
        _, _, cpu_test_loader, _ = create_dataloaders(data_dir, batch_size=batch_size)
        results['qat'] = print_classification_report(
            qat_model, cpu_test_loader, torch.device('cpu'), classes,
            model_name="QAT INT8 (Simulated)"
        )
        del qat_model
    else:
        print(f"QAT checkpoint not found: {qat_path}")
    
    # Add check for Static Quantization path
    static_path = checkpoint_path.replace('.pth', '_static_int8.pth')
    
    # Evaluate Static model (Eager Mode Static Quantization - must run on CPU)
    if os.path.exists(static_path):
        print("\nLoading Static INT8 model...")
        static_checkpoint = torch.load(static_path, map_location='cpu')
        backend = static_checkpoint.get('backend', 'x86')
        torch.backends.quantized.engine = backend
        
        # To load a static quantized model, we must rebuild the quantized structure
        float_model, _ = load_float_model_from_checkpoint(checkpoint_path, device=torch.device('cpu'))
        static_model = QuantizableModelWrapper(float_model)
        static_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.ao.quantization.prepare(static_model, inplace=True)
        torch.ao.quantization.convert(static_model, inplace=True)
        
        static_model.load_state_dict(static_checkpoint['model_state_dict'])
        static_model.eval()
        
        _, _, cpu_test_loader, _ = create_dataloaders(data_dir, batch_size=batch_size)
        results['static'] = print_classification_report(
            static_model, cpu_test_loader, torch.device('cpu'), classes,
            model_name="Static PTQ (Eager)"
        )
        del static_model
    else:
        print(f"Static checkpoint not found: {static_path}")
    
    # Print comparisons
    if 'baseline' in results and 'ptq' in results:
        compare_models(results['baseline'], results['ptq'], "Baseline FP32", "PTQ INT8")
    
    if 'baseline' in results and 'qat' in results:
        compare_models(results['baseline'], results['qat'], "Baseline FP32", "QAT INT8")
    
    if 'ptq' in results and 'qat' in results:
        compare_models(results['ptq'], results['qat'], "PTQ INT8", "QAT INT8")
    
    if 'baseline' in results and 'static' in results:
        compare_models(results['baseline'], results['static'], "Baseline FP32", "Static INT8")
    
    # Print size comparison from stored stats
    print_size_comparison(size_stats, checkpoint_path, ptq_path, qat_path, static_path)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def load_float_model_from_checkpoint(checkpoint_path: str, device: torch.device, eval_mode: bool = True):
    """Recreate a timm model from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    model_kwargs = checkpoint.get('model_kwargs', {})
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if eval_mode:
        model.eval()
    return model, checkpoint


def apply_post_training_quantization(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 32,
    backend: str = 'x86'
):
    """
    Apply dynamic quantization to a stored FP32 checkpoint.
    
    Dynamic quantization quantizes weights to INT8 and computes activations
    in INT8 at runtime. Works with any model architecture including those
    with dynamic control flow (like MobileViT).
    """
    torch.backends.quantized.engine = backend
    device = torch.device('cpu')  # Quantized models run on CPU

    _, _, test_loader, _ = create_dataloaders(data_dir, batch_size=batch_size)

    float_model, checkpoint = load_float_model_from_checkpoint(checkpoint_path, device=device)
    
    # Dynamic quantization - quantizes Linear and LSTM layers
    # Works with any model architecture (no tracing required)
    quantized_model = quantize_dynamic(
        float_model,
        {nn.Linear},  # Quantize Linear layers (main compute in ViT)
        dtype=torch.qint8
    )
    
    # Verify quantization coverage
    quantized_layers = [n for n, m in quantized_model.named_modules() 
                       if isinstance(m, torch.nn.quantized.dynamic.Linear)]
    print(f"Dynamic Quantization applied to {len(quantized_layers)} Linear layers.")
    
    quantized_model.eval()

    criterion = nn.CrossEntropyLoss()
    _, test_acc = evaluate(quantized_model, test_loader, criterion, device)

    quantized_path = checkpoint_path.replace('.pth', '_ptq_int8.pth')
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'model_name': checkpoint['model_name'],
        'num_classes': checkpoint['num_classes'],
        'classes': checkpoint['classes'],
        'quantization': 'dynamic_ptq',
        'backend': backend,
        'test_acc': test_acc,
        'source_checkpoint': checkpoint_path,
        'model_kwargs': checkpoint.get('model_kwargs', {}),
    }, quantized_path)
    print(f"Dynamic PTQ complete. INT8 model accuracy: {test_acc:.2f}% saved to {quantized_path}")

    return quantized_model, quantized_path, test_acc


class QuantizationNoiseInjector(nn.Module):
    """Simulates quantization noise during training for QAT-like behavior."""
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.enabled = True
    
    def forward(self, x):
        if self.training and self.enabled:
            # Simulate quantization noise
            scale = x.abs().max() / (2 ** (self.bits - 1) - 1)
            if scale > 0:
                x_quant = torch.round(x / scale) * scale
                return x_quant
        return x


def add_quantization_noise_hooks(model):
    """Add forward hooks to simulate quantization noise during training."""
    hooks = []
    
    def create_hook():
        injector = QuantizationNoiseInjector()
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                return injector(output)
            return output
        return hook, injector
    
    injectors = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook_fn, injector = create_hook()
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
            injectors.append(injector)
    
    return hooks, injectors


def quantization_aware_finetune(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    backend: str = 'x86',
    device: str = 'cuda',
):
    """
    Fine-tune a model with simulated quantization noise, then apply dynamic quantization.
    
    This approach works with any model architecture by:
    1. Training with noise injection that simulates quantization effects
    2. Applying dynamic quantization to the fine-tuned model
    """
    torch.backends.quantized.engine = backend
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, _ = create_dataloaders(data_dir, batch_size=batch_size)

    float_model, checkpoint = load_float_model_from_checkpoint(checkpoint_path, device=device, eval_mode=False)
    
    # Add quantization noise hooks for QAT-like training
    hooks, injectors = add_quantization_noise_hooks(float_model)
    print(f"Added quantization noise to {len(hooks)} Linear layers")

    optimizer = torch.optim.AdamW(float_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = copy.deepcopy(float_model.state_dict())

    for epoch in range(epochs):
        float_model.train()
        # Enable noise during training
        for inj in injectors:
            inj.enabled = True
            
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = float_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            running_total += labels.size(0)
            running_correct += preds.eq(labels).sum().item()

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  [QAT] Epoch {epoch+1}/{epochs} Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * running_correct / running_total
        
        # Disable noise for validation
        for inj in injectors:
            inj.enabled = False
        val_loss, val_acc = evaluate(float_model, val_loader, criterion, device)

        scheduler.step()

        print(f"[QAT] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(float_model.state_dict())

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Load best state and move to CPU for quantization
    float_model.load_state_dict(best_state)
    float_model.to('cpu')
    float_model.eval()
    
    # Apply dynamic quantization to the fine-tuned model
    quantized_model = quantize_dynamic(
        float_model,
        {nn.Linear},
        dtype=torch.qint8
    )

    criterion = nn.CrossEntropyLoss()
    _, test_acc = evaluate(quantized_model, test_loader, criterion, torch.device('cpu'))

    quantized_path = checkpoint_path.replace('.pth', '_qat_int8.pth')
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'model_name': checkpoint['model_name'],
        'num_classes': checkpoint['num_classes'],
        'classes': checkpoint['classes'],
        'quantization': 'simulated_qat',
        'backend': backend,
        'test_acc': test_acc,
        'source_checkpoint': checkpoint_path,
        'model_kwargs': checkpoint.get('model_kwargs', {}),
    }, quantized_path)
    print(f"QAT fine-tuning complete. INT8 model accuracy: {test_acc:.2f}% saved to {quantized_path}")

    return quantized_model, quantized_path, test_acc


def train_eurosat(
    checkpoint_path : str,
    model_name='mobilevitv2_075.cvnets_in1k',
    data_dir='EuroSATallBands',
    batch_size=32,
    epochs=50,
    lr=1e-4,
    weight_decay=0.01,
    device='cuda',
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
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, drop_rate=0.1)
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
            model_kwargs = {}
            if hasattr(model, 'drop_rate'):
                model_kwargs['drop_rate'] = getattr(model, 'drop_rate', 0.0)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': model_name,
                'num_classes': num_classes,
                'classes': classes,
                'model_kwargs': model_kwargs,
            }
            save_path = checkpoint_path
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


def apply_onnx_quantization(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 32,
    calibration_batches: int = 10
):
    """
    Export to ONNX and apply ONNX Runtime quantization.
    
    This handles complex architectures that PyTorch native quantization cannot,
    including MobileViT with its residual connections and dynamic control flow.
    
    Returns paths to FP32 and INT8 ONNX models.
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic as onnx_quantize_dynamic
        from onnxruntime.quantization import quantize_static as onnx_quantize_static
        from onnxruntime.quantization import QuantType, CalibrationDataReader
        import onnxruntime as ort
    except ImportError:
        print("ONNX quantization requires: pip install onnx onnxruntime")
        return None, None, 0.0, 0.0
    
    print(f"\nPreparing ONNX Quantization...")
    device = torch.device('cpu')
    
    # 1. Load Float Model
    float_model, checkpoint = load_float_model_from_checkpoint(checkpoint_path, device=device)
    float_model.eval()
    
    # 2. Export to ONNX
    onnx_fp32_path = checkpoint_path.replace('.pth', '_fp32.onnx')
    onnx_int8_path = checkpoint_path.replace('.pth', '_int8.onnx')
    
    print("Exporting to ONNX format...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        float_model,
        dummy_input,
        onnx_fp32_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False  # Use legacy exporter for better compatibility
    )
    print(f"Exported FP32 ONNX model to: {onnx_fp32_path}")
    
    # Free memory - we don't need the PyTorch model anymore
    del float_model
    del dummy_input
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 3. Quantize with ONNX Runtime Static Quantization
    # Static quantization uses QLinearConv (not ConvInteger) which is well-supported
    print("Applying ONNX Runtime static quantization (all ops)...")
    
    # Create calibration data reader
    class ONNXCalibrationDataReader(CalibrationDataReader):
        def __init__(self, data_loader, num_batches=10):
            self.data_iter = iter(data_loader)
            self.num_batches = num_batches
            self.batch_count = 0
            
        def get_next(self):
            if self.batch_count >= self.num_batches:
                return None
            try:
                images, _ = next(self.data_iter)
                self.batch_count += 1
                return {'input': images.numpy()}
            except StopIteration:
                return None
    
    # Get calibration data - use smaller batch size for calibration to save memory
    cal_batch_size = min(batch_size, 16)  # Limit calibration batch size
    _, _, cal_loader, _ = create_dataloaders(data_dir, batch_size=cal_batch_size)
    calibration_reader = ONNXCalibrationDataReader(cal_loader, num_batches=calibration_batches)
    
    # Static quantization with per-channel for better accuracy
    from onnxruntime.quantization import QuantFormat, CalibrationMethod
    
    # Use MinMax calibration (uses less memory than Entropy)
    onnx_quantize_static(
        onnx_fp32_path,
        onnx_int8_path,
        calibration_reader,
        quant_format=QuantFormat.QDQ,  # Use QDQ format for better compatibility
        per_channel=False,  # Disable per-channel to reduce memory
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,  # Uses less memory than Entropy
    )
    print(f"Exported INT8 ONNX model to: {onnx_int8_path}")
    
    # 4. Get file sizes for comparison
    fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    compression_ratio = fp32_size / int8_size if int8_size > 0 else 0
    
    print(f"\nONNX Model Size Comparison:")
    print(f"  FP32: {fp32_size:.2f} MB")
    print(f"  INT8: {int8_size:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # 5. Evaluate ONNX INT8 model using ONNX Runtime
    print("\nEvaluating ONNX INT8 model...")
    _, _, test_loader, classes = create_dataloaders(data_dir, batch_size=batch_size)
    
    # Create ONNX Runtime session - prefer CUDA if available
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Try CUDA first, fall back to CPU
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("Using CUDA execution provider for ONNX Runtime")
    else:
        providers = ['CPUExecutionProvider']
        print("Using CPU execution provider for ONNX Runtime")
    
    ort_session = ort.InferenceSession(onnx_int8_path, sess_options, providers=providers)
    
    input_name = ort_session.get_inputs()[0].name
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        # Run inference
        outputs = ort_session.run(None, {input_name: images.numpy()})
        predictions = np.argmax(outputs[0], axis=1)
        
        total += labels.size(0)
        correct += (predictions == labels.numpy()).sum()
    
    test_acc = 100. * correct / total
    print(f"ONNX INT8 model accuracy: {test_acc:.2f}%")
    
    # 6. Also evaluate FP32 ONNX for comparison
    print("\nEvaluating ONNX FP32 model...")
    ort_session_fp32 = ort.InferenceSession(onnx_fp32_path, sess_options, providers=providers)
    
    correct_fp32 = 0
    total_fp32 = 0
    
    for images, labels in test_loader:
        outputs = ort_session_fp32.run(None, {input_name: images.numpy()})
        predictions = np.argmax(outputs[0], axis=1)
        
        total_fp32 += labels.size(0)
        correct_fp32 += (predictions == labels.numpy()).sum()
    
    fp32_acc = 100. * correct_fp32 / total_fp32
    print(f"ONNX FP32 model accuracy: {fp32_acc:.2f}%")
    print(f"Accuracy difference (INT8 - FP32): {test_acc - fp32_acc:+.2f}%")
    
    return onnx_int8_path, onnx_fp32_path, test_acc, fp32_acc


if __name__ == '__main__':
    # Train models
    models_to_train = [
        'resnet18.a1_in1k',  # Small, fast, quantizes well
        # 'resnet50.a1_in1k',  # Larger, more accurate
    ]
    RUN_BASELINE_TRAINING = True
    RUN_PTQ = True
    RUN_QAT = False
    RUN_ONNX = True
    
    results = {}
    
    for model_name in models_to_train:
        checkpoint_path = f'/content/drive/MyDrive/Colab Notebooks/CS 7267 Final Group Project/best_{model_name.replace("/", "_")}_eurosat.pth'
        data_dir = '/content/EuroSAT_local'

        if RUN_BASELINE_TRAINING:
            print(f"\n{'='*70}")
            print(f"Training {model_name}")
            print(f"{'='*70}\n")
            _, test_acc = train_eurosat(
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                data_dir=data_dir,
                batch_size=128,
                epochs=50,
                lr=1e-4,
                weight_decay=0.01,
            )
            results[model_name] = test_acc

        if RUN_PTQ and os.path.exists(checkpoint_path):
            print(f"\nRunning PTQ for {model_name}")
            apply_post_training_quantization(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                batch_size=128,
                backend='x86'
            )
        elif RUN_PTQ:
            print(f"Checkpoint {checkpoint_path} not found. Skipping PTQ.")

        if RUN_QAT and os.path.exists(checkpoint_path):
            print(f"\nRunning QAT fine-tuning for {model_name}")
            quantization_aware_finetune(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                batch_size=128,
                epochs=10,
                lr=1e-5,
                weight_decay=0.0,
                backend='x86',
                device='cuda'
            )
        elif RUN_QAT:
            print(f"Checkpoint {checkpoint_path} not found. Skipping QAT.")

        if RUN_ONNX and os.path.exists(checkpoint_path):
            print(f"\nRunning ONNX Quantization for {model_name}")
            apply_onnx_quantization(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                batch_size=128,
                calibration_batches=10
            )
        elif RUN_ONNX:
            print(f"Checkpoint {checkpoint_path} not found. Skipping ONNX quantization.")

    # Run full evaluation with classification reports
    ptq_path = checkpoint_path.replace('.pth', '_ptq_int8.pth')
    qat_path = checkpoint_path.replace('.pth', '_qat_int8.pth')
    
    if os.path.exists(checkpoint_path) and (os.path.exists(ptq_path) or os.path.exists(qat_path)):
        print(f"\n{'='*70}")
        print("RUNNING FULL EVALUATION WITH CLASSIFICATION REPORTS")
        print(f"{'='*70}")
        run_full_evaluation(
            checkpoint_path=checkpoint_path,
            ptq_path=ptq_path,
            qat_path=qat_path,
            data_dir=data_dir,
            batch_size=128
        )

    if results:
        print("\n" + "="*20)
        print("FINAL TEST RESULTS")
        print("="*20)
        for model_name, test_acc in results.items():
            print(f"{model_name}: {test_acc:.2f}%")