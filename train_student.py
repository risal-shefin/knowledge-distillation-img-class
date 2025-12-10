"""
Train a small student model using knowledge distillation from ResNet152 teacher.

This script implements pure knowledge distillation (soft targets only) to train
a lightweight MobileNetV2 student model on Mini-ImageNet dataset.
"""

import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from PIL import Image

from models import (
    create_student_model,
    create_teacher_model,
    DistillationLoss,
    extract_teacher_soft_targets,
    print_model_info
)
from utils import create_imagenet_to_mini_mapping


def load_mini_imagenet_data(batch_size=128, num_workers=4):
    """
    Load Mini-ImageNet train, validation, and test splits.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader, imagenet_to_mini_mapping
    """
    print("="*70)
    print("Loading Mini-ImageNet Dataset")
    print("="*70)
    
    # Load datasets from HuggingFace
    print("\nLoading train split...")
    train_dataset = load_dataset('timm/mini-imagenet', split='train', keep_in_memory=True)
    print(f"Train set: {len(train_dataset)} images")
    
    print("\nLoading validation split...")
    val_dataset = load_dataset('timm/mini-imagenet', split='validation', keep_in_memory=True)
    print(f"Validation set: {len(val_dataset)} images")
    
    print("\nLoading test split...")
    test_dataset = load_dataset('timm/mini-imagenet', split='test', keep_in_memory=True)
    print(f"Test set: {len(test_dataset)} images")
    
    # Create label mapping (needed for teacher logit remapping)
    imagenet_to_mini = create_imagenet_to_mini_mapping(train_dataset)
    print(f"\nCreated mapping: {len(imagenet_to_mini)} ImageNet classes -> 100 Mini-ImageNet classes")
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Custom dataset wrapper
    class MiniImageNetDataset(torch.utils.data.Dataset):
        """Wraps HuggingFace dataset with transforms."""
        
        def __init__(self, hf_dataset, transform):
            self.dataset = hf_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item['image']
            label = item['label']  # Mini-ImageNet label (0-99)
            
            # Ensure image is PIL RGB format
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    # Create wrapped datasets
    train_data = MiniImageNetDataset(train_dataset, train_transform)
    val_data = MiniImageNetDataset(val_dataset, eval_transform)
    test_data = MiniImageNetDataset(test_dataset, eval_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, imagenet_to_mini


def validate(student_model, val_loader, device):
    """
    Validate student model on validation set.
    
    Args:
        student_model: Student model to evaluate
        val_loader: Validation data loader
        device: Device for computation
    
    Returns:
        accuracy: Top-1 accuracy (%)
    """
    student_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = student_model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def train_student(
    student_model,
    teacher_model,
    train_loader,
    val_loader,
    imagenet_to_mini,
    device,
    num_epochs=50,
    learning_rate=0.001,
    temperature=4.0,
    save_dir='models'
):
    """
    Train student model using knowledge distillation.
    
    Args:
        student_model: Student model to train
        teacher_model: Teacher model (frozen)
        train_loader: Training data loader
        val_loader: Validation data loader
        imagenet_to_mini: Mapping from ImageNet to Mini-ImageNet indices
        device: Device for computation
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        temperature: Temperature for distillation
        save_dir: Directory to save checkpoints
    
    Returns:
        best_val_acc: Best validation accuracy achieved
    """
    print("\n" + "="*70)
    print("Starting Knowledge Distillation Training")
    print("="*70)
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(timestamped_save_dir, exist_ok=True)
    print(f"Model will be saved to: {timestamped_save_dir}")
    
    # Setup TensorBoard writer with same timestamp
    log_dir = os.path.join('runs', f'mobilenet_T{temperature}_lr{learning_rate}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    print(f"  View with: tensorboard --logdir=runs")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Setup loss function
    criterion = DistillationLoss(temperature=temperature)
    
    # Training state
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Temperature: {temperature}")
    print(f"  Optimizer: Adam")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Loss: Pure KD (soft targets only)")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        student_model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Extract soft targets from teacher (remapped to 100 classes)
            teacher_soft_targets = extract_teacher_soft_targets(
                teacher_model, images, imagenet_to_mini, temperature, device
            )
            
            # Forward pass through student
            student_logits = student_model(images)
            
            # Compute distillation loss
            loss = criterion(student_logits, teacher_soft_targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = train_loss / num_batches
        # Print progress
        print(f"  Epoch [{epoch+1}/{num_epochs}]", f"Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        val_acc = validate(student_model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Time tracking
        epoch_time = time.time() - epoch_start_time
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'train_loss': avg_train_loss,
                'temperature': temperature,
            }
            
            save_path = os.path.join(timestamped_save_dir, 'best_mobilenet_student.pth')
            torch.save(checkpoint, save_path)
            print(f"  ✓ Best model saved! (Accuracy: {val_acc:.2f}%)")
        
        print("-" * 70)
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nTraining Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    return best_val_acc, timestamped_save_dir


def test_student(student_model, test_loader, device, checkpoint_path):
    """
    Test student model on test set.
    
    Args:
        student_model: Student model
        test_loader: Test data loader
        device: Device for computation
        checkpoint_path: Path to best model checkpoint
    
    Returns:
        test_accuracy: Top-1 accuracy on test set (%)
    """
    print("\n" + "="*70)
    print("Evaluating Student Model on Test Set")
    print("="*70)
    
    # Load best checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_accuracy']:.2f}%)")
    
    # Evaluate on test set
    student_model.eval()
    correct = 0
    total = 0
    
    print(f"\nRunning inference on {len(test_loader.dataset)} test images...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = student_model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {total}/{len(test_loader.dataset)} images...")
    
    test_accuracy = 100 * correct / total
    inference_time = time.time() - start_time
    
    print(f"\nTest Set Results:")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")
    print(f"  Total samples: {total}")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Time per image: {inference_time / total * 1000:.2f}ms")
    
    return test_accuracy


def main():
    """Main training and evaluation pipeline."""
    print("="*70)
    print("Knowledge Distillation Training - MobileNetV2 Student")
    print("="*70)
    
    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.001
    TEMPERATURE = 4.0
    NUM_WORKERS = 4
    SAVE_DIR = 'models'
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    train_loader, val_loader, test_loader, imagenet_to_mini = load_mini_imagenet_data(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Create models
    print("\n" + "="*70)
    print("Creating Models")
    print("="*70)
    
    print("\nCreating student model (MobileNetV2)...")
    student_model = create_student_model(num_classes=100, pretrained=False)
    student_model.to(device)
    print_model_info(student_model, "Student Model (MobileNetV2)")
    
    print("\nLoading teacher model (ResNet152)...")
    teacher_model = create_teacher_model(device=device)
    print_model_info(teacher_model, "Teacher Model (ResNet152)")
    
    # Train student
    best_val_acc, model_save_dir = train_student(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        imagenet_to_mini=imagenet_to_mini,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
        save_dir=SAVE_DIR
    )
    
    # Test student
    checkpoint_path = os.path.join(model_save_dir, 'best_mobilenet_student.pth')
    test_accuracy = test_student(
        student_model=student_model,
        test_loader=test_loader,
        device=device,
        checkpoint_path=checkpoint_path
    )
    
    # Final summary
    print("\n" + "="*70)
    print("Training and Evaluation Complete!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Student Model: MobileNetV2 (~3.5M parameters)")
    print(f"  Teacher Model: ResNet152 (79.30% baseline on Mini-ImageNet)")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")
    print(f"  Model saved to: {checkpoint_path}")
    
    # Compression metrics
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    compression_ratio = teacher_params / student_params
    
    print(f"\nCompression Metrics:")
    print(f"  Teacher parameters: {teacher_params:,}")
    print(f"  Student parameters: {student_params:,}")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    print(f"  Accuracy gap: {83.26 - test_accuracy:.2f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
