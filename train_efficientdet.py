"""
EfficientDet Object Detection Model Training
Optimized for live detection with fast inference
"""

import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
from collections import defaultdict
import shutil
from tqdm import tqdm

class WasteDataset(Dataset):
    """Dataset class for waste classification"""
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Get all class folders
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Load images
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes")
        print(f"Classes: {', '.join(self.classes)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_train_val_split(data_dir, train_ratio=0.8):
    """Split dataset into train and validation sets"""
    data_dir = Path(data_dir)
    train_dir = data_dir.parent / 'train'
    val_dir = data_dir.parent / 'val'
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Get all classes
    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    for class_name in classes:
        class_dir = data_dir / class_name
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        train_class_dir.mkdir(exist_ok=True, parents=True)
        val_class_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all images
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        
        # Shuffle
        import random
        random.shuffle(image_files)
        
        # Split
        split_idx = int(len(image_files) * train_ratio)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        # Copy files
        for img in tqdm(train_images, desc=f"Copying {class_name} train"):
            shutil.copy2(img, train_class_dir / img.name)
        
        for img in tqdm(val_images, desc=f"Copying {class_name} val"):
            shutil.copy2(img, val_class_dir / img.name)
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\nDataset split complete!")
    print(f"Train: {train_dir}")
    print(f"Val: {val_dir}")

def train_model(data_dir, epochs=50, batch_size=32, lr=0.001, device='cuda'):
    """Train EfficientNet model for classification"""
    
    # Check if train/val split exists, if not create it
    data_path = Path(data_dir)
    train_path = data_path.parent / 'train'
    val_path = data_path.parent / 'val'
    
    if not train_path.exists() or not val_path.exists():
        print("Creating train/validation split...")
        create_train_val_split(data_dir)
        train_path = data_path.parent / 'train'
        val_path = data_path.parent / 'val'
    else:
        print("Using existing train/validation split")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = WasteDataset(train_path, transform=train_transform, split='train')
    val_dataset = WasteDataset(val_path, transform=val_transform, split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    num_classes = len(train_dataset.classes)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create model - Using EfficientNet for fast inference
    print("\nLoading EfficientNet-B0 model...")
    model = torchvision.models.efficientnet_b0(pretrained=True)
    
    # Modify classifier for our number of classes
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(model.classifier[1].in_features, num_classes)
    )
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_acc = 0.0
    model_save_path = Path('models')
    model_save_path.mkdir(exist_ok=True)
    
    print(f"\nStarting training on {device}...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/(train_bar.n+1):.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/(val_bar.n+1):.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes,
                'class_to_idx': train_dataset.class_to_idx,
            }, model_save_path / 'best_model.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes,
                'class_to_idx': train_dataset.class_to_idx,
            }, model_save_path / f'checkpoint_epoch_{epoch+1}.pth')
        
        print("-" * 60)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {model_save_path / 'best_model.pth'}")
    
    # Save class mapping
    class_info = {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': train_dataset.idx_to_class
    }
    
    with open(model_save_path / 'class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"Class info saved to: {model_save_path / 'class_info.json'}")
    
    return model, train_dataset.classes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EfficientNet model for waste classification')
    parser.add_argument('--data', type=str, default='D:/garbage_detection/Try/dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EfficientNet Waste Classification Training")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset directory not found: {args.data}")
        exit(1)
    
    train_model(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )

