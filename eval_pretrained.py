"""
Evaluate a pretrained ImageNet model on ImageNet validation dataset.

This script loads a pretrained image classification model (ResNet50) with ImageNet weights
and evaluates its accuracy on the ImageNet validation dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from PIL import Image
import time

from utils import (
    load_imagenet_labels,
    create_mini_to_imagenet_mapping,
    get_label_name
)


def load_pretrained_model():
    """Load pretrained ResNet50 model with ImageNet weights."""
    print("Loading pretrained ResNet50 model with ImageNet weights...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()  # Set to evaluation mode
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded successfully on {device}!")
    
    return model, device


def load_imagenet():
    """Load Mini-ImageNet dataset from Hugging Face and align labels."""
    print("\nLoading Mini-ImageNet test dataset from Hugging Face...")
    
    # Load dataset from Hugging Face with keep_in_memory=True
    dataset = load_dataset('timm/mini-imagenet', split='test', keep_in_memory=True)
    
    print(f"Test set: {len(dataset)} images")
    print("Dataset loaded in memory (not saved to disk)")

    # Create label mapping using utility function
    label_mapping = create_mini_to_imagenet_mapping(dataset)
    print(f"Aligned {len(label_mapping)} dataset classes to ImageNet-1K indices")
    
    # Define transforms for preprocessing (standard ImageNet preprocessing)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Custom dataset wrapper to apply transforms and label mapping
    class MiniImageNetDataset(torch.utils.data.Dataset):
        """Wraps Mini-ImageNet dataset with transforms and label mapping."""
        
        def __init__(self, hf_dataset, transform, label_mapping):
            self.dataset = hf_dataset
            self.transform = transform
            self.label_mapping = label_mapping
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item['image']
            label = item['label']
            
            # Map Mini-ImageNet label to ImageNet-1K index
            if label not in self.label_mapping:
                raise KeyError(f"Label {label} missing from ImageNet mapping.")
            imagenet_label = self.label_mapping[label]
            
            # Ensure image is PIL RGB format
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image, imagenet_label
    
    val_dataset = MiniImageNetDataset(dataset, transform, label_mapping)
    
    return val_dataset


def evaluate_model(model, test_dataset, device, imagenet_labels, batch_size=32):
    """
    Evaluate the model on the ImageNet validation set.
    
    Args:
        model: The pretrained model to evaluate
        test_dataset: ImageNet test dataset
        device: Device to run inference on
        imagenet_labels: List of human-readable ImageNet class names
        batch_size: Batch size for evaluation
    """
    num_samples = len(test_dataset)
    print(f"\nEvaluating model on full test set ({num_samples} images)...")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    all_predictions = []
    all_labels = []
    top5_correct = 0
    total_samples = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get top-1 predictions
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get top-5 predictions
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct = top5_pred.eq(labels.view(1, -1).expand_as(top5_pred))
            top5_correct += correct[:5].reshape(-1).float().sum(0).item()
            total_samples += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {min((batch_idx + 1) * batch_size, num_samples)}/{num_samples} images...")
    
    end_time = time.time()
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"\nInference completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per image: {(end_time - start_time) / len(all_labels) * 1000:.2f} ms")
    
    # Calculate accuracies
    top1_accuracy = accuracy_score(all_labels, all_predictions)
    top5_accuracy = top5_correct / total_samples
    
    print(f"\nResults:")
    print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
    
    # Show sample predictions
    print("\nSample predictions (first 5 images):")
    for i in range(min(5, len(all_labels))):
        true_label = get_label_name(all_labels[i], imagenet_labels)
        pred_label = get_label_name(all_predictions[i], imagenet_labels)
        correct = "✓" if all_labels[i] == all_predictions[i] else "✗"
        print(f"  Image {i+1}: True: {true_label} | Predicted: {pred_label} {correct}")
    
    return top1_accuracy, top5_accuracy


def main():
    """Main execution function."""
    print("="*70)
    print("Pretrained ImageNet Model Evaluation")
    print("="*70)

    # Load human-readable ImageNet labels
    imagenet_labels = load_imagenet_labels()
    
    # Load pretrained model
    model, device = load_pretrained_model()
    
    # Load Mini-ImageNet dataset with label mapping
    val_dataset = load_imagenet()
    
    top1_acc, top5_acc = evaluate_model(
        model, 
        val_dataset,
        device,
        imagenet_labels=imagenet_labels,
        batch_size=64,
    )
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
