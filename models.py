"""
Student model architectures and knowledge distillation utilities.

This module provides:
- MobileNetV2 student model with 100-class output
- Functions to extract and remap teacher logits from 1000 to 100 classes
- Knowledge distillation loss (pure soft targets, no hard labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MobileNetV2Student(nn.Module):
    """
    MobileNetV2-based student model for Mini-ImageNet (100 classes).
    
    Lightweight architecture (~3.5M parameters) suitable for efficient deployment.
    Modified classifier layer outputs 100 classes instead of 1000.
    """
    
    def __init__(self, num_classes=100, pretrained=False):
        """
        Initialize MobileNetV2 student model.
        
        Args:
            num_classes: Number of output classes (100 for Mini-ImageNet)
            pretrained: Whether to use ImageNet pretrained weights as initialization
        """
        super(MobileNetV2Student, self).__init__()
        
        # Load MobileNetV2 architecture
        if pretrained:
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        else:
            self.model = models.mobilenet_v2(weights=None)
        
        # Replace classifier for 100 classes
        # Original: Linear(1280 -> 1000)
        # Modified: Linear(1280 -> 100)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def extract_teacher_soft_targets(teacher_model, images, imagenet_to_mini_mapping, temperature=4.0, device='cuda'):
    """
    Extract soft targets from teacher model and remap from 1000 to 100 classes.
    
    This function:
    1. Gets teacher's 1000-class logits
    2. Remaps to 100 Mini-ImageNet classes using imagenet_to_mini_mapping
    3. Applies temperature scaling to create soft probability distribution
    
    Args:
        teacher_model: ResNet152 teacher model (outputs 1000 classes)
        images: Batch of input images [batch_size, 3, 224, 224]
        imagenet_to_mini_mapping: Dict mapping ImageNet-1K idx (0-999) to Mini-ImageNet idx (0-99)
        temperature: Temperature for softening probability distribution
        device: Device for computation
    
    Returns:
        Soft targets: Temperature-scaled probabilities [batch_size, 100]
    """
    teacher_model.eval()
    
    with torch.no_grad():
        # Get teacher's 1000-class logits
        teacher_logits_full = teacher_model(images)  # [batch_size, 1000]
        
        # Create mapping tensor for efficient indexing
        # For each of 1000 ImageNet classes, map to Mini-ImageNet class (or -1 if not present)
        batch_size = teacher_logits_full.size(0)
        remapped_logits = torch.zeros(batch_size, 100, device=device)
        
        # Accumulate logits for Mini-ImageNet classes
        for imagenet_idx, mini_idx in imagenet_to_mini_mapping.items():
            remapped_logits[:, mini_idx] = teacher_logits_full[:, imagenet_idx]
        
        # Apply temperature scaling and softmax to get soft targets
        soft_targets = F.softmax(remapped_logits / temperature, dim=1)
    
    return soft_targets


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss using only soft targets (no hard labels).
    
    Uses KL divergence between student and teacher probability distributions,
    with temperature scaling to soften the distributions and reveal more information.
    """
    
    def __init__(self, temperature=4.0):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature parameter for softening distributions.
                        Higher T = softer probabilities, more information transfer.
                        Typical range: [3, 6]
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_soft_targets):
        """
        Compute KL divergence loss between student and teacher.
        
        Args:
            student_logits: Raw logits from student model [batch_size, 100]
            teacher_soft_targets: Pre-computed soft targets from teacher [batch_size, 100]
                                 (already temperature-scaled and softmax-ed)
        
        Returns:
            KL divergence loss (scalar)
        """
        # Apply temperature to student logits and compute log probabilities
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Compute KL divergence: KL(teacher || student)
        # teacher_soft_targets is already softmax-ed with temperature
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_soft_targets,
            reduction='batchmean'
        )
        
        # Scale loss by T^2 to compensate for temperature scaling
        # This ensures gradient magnitude is independent of temperature
        loss = kl_loss * (self.temperature ** 2)
        
        return loss


def create_student_model(num_classes=100, pretrained=False):
    """
    Factory function to create student model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        MobileNetV2Student model
    """
    model = MobileNetV2Student(num_classes=num_classes, pretrained=pretrained)
    return model


def create_teacher_model(device='cuda'):
    """
    Load pretrained ResNet152 teacher model.
    
    Args:
        device: Device to load model on
    
    Returns:
        Frozen ResNet152 model in eval mode
    """
    print("Loading ResNet152 teacher model with ImageNet weights...")
    teacher = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    teacher.eval()
    teacher.to(device)
    
    # Freeze all parameters
    for param in teacher.parameters():
        param.requires_grad = False
    
    print(f"Teacher model loaded on {device} (frozen)")
    return teacher


def compute_sample_difficulty(
    teacher_model,
    student_model,
    images,
    labels,
    imagenet_to_mini,
    device='cuda'
):
    """
    Compute difficulty scores for each sample in a batch.
    
    Difficulty metrics:
    1. Teacher confidence (low confidence = harder)
    2. Prediction entropy (high entropy = harder)
    3. Student-teacher agreement (low agreement = harder)
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        images: Batch of images [batch_size, 3, 224, 224]
        labels: True labels [batch_size]
        imagenet_to_mini: Mapping from ImageNet to Mini-ImageNet indices
        device: Device for computation
    
    Returns:
        Difficulty scores [batch_size] in range [0, 1], where 1 = hardest
    """
    teacher_model.eval()
    student_model.eval()
    
    with torch.no_grad():
        batch_size = images.size(0)
        
        # Get teacher predictions (remapped to 100 classes)
        teacher_logits_full = teacher_model(images)  # [batch_size, 1000]
        teacher_logits = torch.zeros(batch_size, 100, device=device)
        for imagenet_idx, mini_idx in imagenet_to_mini.items():
            teacher_logits[:, mini_idx] = teacher_logits_full[:, imagenet_idx]
        teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # Get student predictions
        student_logits = student_model(images)  # [batch_size, 100]
        student_probs = F.softmax(student_logits, dim=1)
        
        # 1. Teacher confidence (1 - max_prob)
        teacher_max_prob = teacher_probs.max(dim=1)[0]
        confidence_difficulty = 1.0 - teacher_max_prob
        
        # 2. Prediction entropy (normalized)
        teacher_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-10)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(100.0))
        entropy_difficulty = teacher_entropy / max_entropy
        
        # 3. Student-teacher KL divergence
        kl_div = F.kl_div(
            torch.log(student_probs + 1e-10),
            teacher_probs,
            reduction='none'
        ).sum(dim=1)
        kl_difficulty = torch.clamp(kl_div / 5.0, 0, 1)  # Normalize to [0, 1]
        
        # Combine metrics (weighted average)
        difficulty = (
            0.3 * confidence_difficulty +
            0.3 * entropy_difficulty +
            0.4 * kl_difficulty
        )
        
        return difficulty.cpu().numpy()


def print_model_info(model, model_name="Model"):
    """Print model architecture information."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_name} Information:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    print(f"  Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (fp32)")
