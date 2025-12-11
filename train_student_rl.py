"""
Train a small student model using knowledge distillation from ResNet152 teacher.

This script implements pure knowledge distillation (soft targets only) to train
a lightweight MobileNetV2 student model on Mini-ImageNet dataset.
"""

import os
import time
from datetime import datetime
import numpy as np
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
    print_model_info,
    compute_sample_difficulty
)
from utils import create_imagenet_to_mini_mapping
from rl_curriculum_env import CurriculumEnvironment
from rl_agent import PPOAgent
from curriculum_dataloader import CurriculumDataLoader, compute_dataset_difficulty_scores


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
    
    # Create curriculum-aware train loader
    train_loader = CurriculumDataLoader(
        train_data,
        batch_size=batch_size,
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
    save_dir='models',
    use_rl_curriculum=True,
    rl_update_freq=5
):
    """
    Train student model using knowledge distillation with RL-based curriculum learning.
    
    Args:
        student_model: Student model to train
        teacher_model: Teacher model (frozen)
        train_loader: Curriculum-aware training data loader
        val_loader: Validation data loader
        imagenet_to_mini: Mapping from ImageNet to Mini-ImageNet indices
        device: Device for computation
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        temperature: Initial temperature for distillation
        save_dir: Directory to save checkpoints
        use_rl_curriculum: Whether to use RL agent for curriculum control
        rl_update_freq: Update RL agent every N epochs
    
    Returns:
        best_val_acc: Best validation accuracy achieved
    """
    print("\n" + "="*70)
    print("Starting Knowledge Distillation Training")
    print("="*70)
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_save_dir = os.path.join(save_dir, f'{timestamp}_rl')
    os.makedirs(timestamped_save_dir, exist_ok=True)
    print(f"Model will be saved to: {timestamped_save_dir}")
    
    # Setup TensorBoard writer with same timestamp
    log_dir = os.path.join('runs', f'mobilenet_T{temperature}_lr{learning_rate}_{timestamp}_rl')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    print(f"  View with: tensorboard --logdir=runs")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Setup loss function (temperature will be dynamic if using RL)
    criterion = DistillationLoss(temperature=temperature)
    
    # Training state
    best_val_acc = 0.0
    best_epoch = 0
    
    # Initialize RL curriculum learning components
    rl_env = None
    rl_agent = None
    current_temperature = temperature
    difficulty_scores = None
    
    if use_rl_curriculum:
        print("\n" + "="*70)
        print("Initializing RL Curriculum Learning")
        print("="*70)
        
        # Create RL environment
        rl_env = CurriculumEnvironment(
            num_epochs=num_epochs,
            initial_temperature=temperature,
            temp_min=2.0,
            temp_max=8.0,
            baseline_val_acc=0.0
        )
        
        # Create PPO agent
        rl_agent = PPOAgent(
            state_dim=rl_env.get_state_dim(),
            temp_actions=3,
            diff_actions=5,
            lr=3e-4,
            gamma=0.99,
            device=device
        )
        
        print(f"RL Environment:")
        print(f"  State dimension: {rl_env.get_state_dim()}")
        print(f"  Action space: {rl_env.get_action_space()}")
        print(f"  Temperature range: [{rl_env.temp_min}, {rl_env.temp_max}]")
        print(f"\nPPO Agent:")
        print(f"  Learning rate: 3e-4")
        print(f"  Discount factor (gamma): 0.99")
        print(f"  Update frequency: Every {rl_update_freq} epochs")
        
        # Get initial state
        rl_state = rl_env.reset()
        
        # Convert state to tensor on correct device
        if isinstance(rl_state, np.ndarray):
            rl_state = torch.FloatTensor(rl_state).to(device)
        elif isinstance(rl_state, torch.Tensor):
            rl_state = rl_state.to(device)
        
        # Compute initial difficulty scores
        print(f"\nComputing initial difficulty scores...")
        difficulty_scores = compute_dataset_difficulty_scores(
            teacher_model,
            student_model,
            train_loader.dataset,
            imagenet_to_mini,
            device,
            batch_size=128
        )
        train_loader.update_curriculum(
            difficulty_threshold=0.0,
            difficulty_scores=difficulty_scores
        )
    
    print(f"\n" + "="*70)
    print(f"Training Configuration")
    print(f"="*70)
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Initial Temperature: {temperature}")
    print(f"  Optimizer: Adam")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Loss: Pure KD (soft targets only)")
    print(f"  RL Curriculum: {'Enabled' if use_rl_curriculum else 'Disabled'}")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Recompute difficulty scores every 20 epochs for curriculum adaptation
        if use_rl_curriculum and epoch > 0 and epoch % 20 == 0:
            print(f"\n  Recomputing difficulty scores at epoch {epoch}...")
            difficulty_scores = compute_dataset_difficulty_scores(
                teacher_model,
                student_model,
                train_loader.dataset,
                imagenet_to_mini,
                device,
                batch_size=128
            )
        
        # RL: Get action from agent before training epoch
        if use_rl_curriculum and rl_agent is not None:
            # Ensure state is on correct device
            if isinstance(rl_state, np.ndarray):
                rl_state = torch.FloatTensor(rl_state).to(device)
            elif isinstance(rl_state, torch.Tensor):
                rl_state = rl_state.to(device)
            
            action, log_prob, value = rl_agent.get_action_with_value(rl_state)
            
            # Apply curriculum parameters
            curriculum_params = rl_env.get_current_params()
            current_temperature = curriculum_params['temperature']
            difficulty_threshold = curriculum_params['difficulty_threshold']
            
            # Update criterion with new temperature
            criterion.temperature = current_temperature
            
            # Update data loader curriculum WITH difficulty scores
            train_loader.update_curriculum(
                difficulty_threshold=difficulty_threshold,
                difficulty_scores=difficulty_scores
            )
        else:
            action, log_prob, value = None, None, None
            current_temperature = temperature
        
        # Training phase
        student_model.train()
        train_loss = 0.0
        num_batches = 0
        batch_difficulties = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Extract soft targets from teacher (remapped to 100 classes)
            teacher_soft_targets = extract_teacher_soft_targets(
                teacher_model, images, imagenet_to_mini, current_temperature, device
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
            
            # Track difficulty scores for RL reward
            if use_rl_curriculum and epoch % 10 == 0:  # Sample every 10 epochs
                with torch.no_grad():
                    batch_diff = compute_sample_difficulty(
                        teacher_model, student_model, images, labels,
                        imagenet_to_mini, device
                    )
                    batch_difficulties.extend(batch_diff)
            
        avg_train_loss = train_loss / num_batches
        avg_difficulty = np.mean(batch_difficulties) if batch_difficulties else 0.5
        
        # Validation phase
        val_acc = validate(student_model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Time tracking
        epoch_time = time.time() - epoch_start_time
        
        # RL: Step environment and store transition
        if use_rl_curriculum and rl_agent is not None:
            next_state, reward, done = rl_env.step(
                action, val_acc, avg_train_loss, avg_difficulty
            )
            
            # Convert next_state to tensor on correct device
            if isinstance(next_state, np.ndarray):
                next_state_tensor = torch.FloatTensor(next_state).to(device)
            elif isinstance(next_state, torch.Tensor):
                next_state_tensor = next_state.to(device)
            else:
                next_state_tensor = next_state
            
            # Store transition in RL buffer
            rl_agent.store_transition(
                rl_state, action, reward, log_prob, value, done
            )
            
            # Update RL agent periodically
            if (epoch + 1) % rl_update_freq == 0:
                rl_stats = rl_agent.update(next_state_tensor)
                if rl_stats:
                    print(f"\n  RL Agent Update:")
                    print(f"    Policy Loss: {rl_stats['policy_loss']:.4f}")
                    print(f"    Value Loss: {rl_stats['value_loss']:.4f}")
                    writer.add_scalar('RL/policy_loss', rl_stats['policy_loss'], epoch)
                    writer.add_scalar('RL/value_loss', rl_stats['value_loss'], epoch)
            
            rl_state = next_state_tensor
            
            # Log RL metrics
            writer.add_scalar('RL/reward', reward, epoch)
            writer.add_scalar('RL/temperature', current_temperature, epoch)
            writer.add_scalar('RL/difficulty_threshold', difficulty_threshold, epoch)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        if use_rl_curriculum:
            print(f"  Temperature: {current_temperature:.2f}")
            print(f"  Difficulty Threshold: {difficulty_threshold:.2f}")
            print(f"  RL Reward: {reward:.3f}" if 'reward' in locals() else "")
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
                'temperature': current_temperature,
                'use_rl_curriculum': use_rl_curriculum,
            }
            
            # Save RL agent state if enabled
            if use_rl_curriculum and rl_agent is not None:
                checkpoint['rl_agent_state'] = {
                    'actor_critic_state_dict': rl_agent.actor_critic.state_dict(),
                    'optimizer_state_dict': rl_agent.optimizer.state_dict(),
                }
                checkpoint['rl_env_state'] = {
                    'val_acc_history': rl_env.val_acc_history,
                    'loss_history': rl_env.loss_history,
                    'difficulty_history': rl_env.difficulty_history,
                }
            
            save_path = os.path.join(timestamped_save_dir, 'best_mobilenet_student.pth')
            torch.save(checkpoint, save_path)
            
            # Save difficulty scores separately
            if difficulty_scores is not None:
                np.save(
                    os.path.join(timestamped_save_dir, 'difficulty_scores.npy'),
                    difficulty_scores
                )
            
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
