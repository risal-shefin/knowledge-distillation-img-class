"""
Curriculum-aware DataLoader with dynamic sample ordering.

This module provides a data loader that can reorder training samples
based on difficulty scores controlled by the RL agent.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class CurriculumSampler(Sampler):
    """
    Sampler that orders samples based on difficulty scores and curriculum strategy.
    
    Supports multiple curriculum strategies:
    - 'easy_to_hard': Start with easy samples, progressively include harder ones (maintains full dataset size)
    - 'hard_mining': Weighted sampling favoring harder samples above threshold
    - 'mixed': Weighted sampling across full difficulty spectrum (threshold controls distribution)
    - 'adaptive': Focus on specific difficulty range (sliding window)
    - 'random': Random sampling (baseline)
    """
    
    def __init__(
        self,
        difficulty_scores,
        difficulty_threshold=0.0,
        batch_size=128,
        strategy='mixed',
        epoch_size_ratio=0.7
    ):
        """
        Initialize curriculum sampler.
        
        Args:
            difficulty_scores: Array of difficulty scores for all samples [num_samples]
            difficulty_threshold: Threshold for sample selection [0, 1]
                - easy_to_hard: 0=all easy, 1=include all difficulties progressively
                - hard_mining: 0=uniform, 1=strongly favor hard samples
                - mixed: 0=favor easy, 1=favor hard (continuous weighting)
                - adaptive: controls center of difficulty window
            batch_size: Batch size for training
            strategy: Curriculum strategy ('easy_to_hard', 'hard_mining', 'mixed', 'adaptive', 'random')
            epoch_size_ratio: Fraction of dataset to use per epoch [0.5, 1.0]
                              Lower values = faster epochs, stronger curriculum focus
        """
        self.difficulty_scores = np.array(difficulty_scores)
        self.difficulty_threshold = difficulty_threshold
        self.batch_size = batch_size
        self.strategy = strategy
        self.num_samples = len(difficulty_scores)
        self.epoch_size_ratio = np.clip(epoch_size_ratio, 0.5, 1.0)
        
        # Create sample indices based on strategy
        self.indices = self._create_indices()
    
    def _create_indices(self):
        """
        Create sample indices based on curriculum strategy.
        
        Returns:
            Array of sample indices
        """
        if self.strategy == 'random':
            # Random baseline
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)
            return indices
        
        # Sort samples by difficulty
        sorted_indices = np.argsort(self.difficulty_scores)
        sorted_scores = self.difficulty_scores[sorted_indices]
        
        if self.strategy == 'easy_to_hard':
            # Progressive curriculum: start with easy, gradually include harder samples
            # threshold=0.0: focus on easiest samples only
            # threshold=1.0: uniform coverage of all difficulties
            
            # Calculate target epoch size
            target_size = max(int(self.num_samples * self.epoch_size_ratio), self.batch_size)
            
            # Calculate coverage range (increases with threshold)
            coverage_ratio = 0.5 + 0.5 * self.difficulty_threshold  # Range: 0.5 to 1.0
            num_covered = max(int(self.num_samples * coverage_ratio), self.batch_size)
            
            # Divide covered samples into 5 difficulty buckets
            bucket_indices = np.array_split(sorted_indices[:num_covered], 5)
            
            # Calculate sampling weights for each bucket (favor easier buckets)
            # Exponentially decay from easiest to hardest
            base_multipliers = np.array([5, 3, 2, 1, 1])  # Favor easy buckets
            
            # Adjust multipliers based on threshold
            # At threshold=0: heavily favor easiest bucket
            # At threshold=1: more uniform distribution
            scaling = 1.0 + (1.0 - self.difficulty_threshold) * 2.0
            multipliers = base_multipliers ** scaling
            multipliers = multipliers / multipliers.sum()
            
            # Calculate samples per bucket to reach target_size
            samples_per_bucket = (multipliers * target_size).astype(int)
            
            # Adjust to ensure exact total
            diff = target_size - samples_per_bucket.sum()
            samples_per_bucket[0] += diff  # Add remainder to easiest bucket
            
            # Sample from each bucket using fast shuffle-and-slice (no np.random.choice)
            selected_indices = []
            for bucket_idx, bucket_samples in zip(bucket_indices, samples_per_bucket):
                if len(bucket_idx) > 0 and bucket_samples > 0:
                    if len(bucket_idx) >= bucket_samples:
                        # Shuffle bucket and take first N samples (fast)
                        shuffled = bucket_idx.copy()
                        np.random.shuffle(shuffled)
                        selected_indices.append(shuffled[:bucket_samples])
                    else:
                        # Use all samples from bucket if it's smaller than allocation
                        selected_indices.append(bucket_idx)
            
            selected_indices = np.concatenate(selected_indices)
            np.random.shuffle(selected_indices)
            return selected_indices
        
        elif self.strategy == 'hard_mining':
            # Weighted sampling favoring harder samples
            # threshold=0.0: uniform sampling
            # threshold=1.0: strongly favor hardest samples
            
            # Calculate target epoch size
            target_size = max(int(self.num_samples * self.epoch_size_ratio), self.batch_size)
            
            # Divide all samples into 5 difficulty buckets (easy to hard)
            bucket_indices = np.array_split(sorted_indices, 5)
            
            if self.difficulty_threshold > 0.0:
                # Create exponential multipliers favoring harder buckets
                # Scale strength based on threshold
                strength = self.difficulty_threshold * 2.0
                base_multipliers = np.array([1, 1, 2, 3, 5])  # Favor hard buckets
                multipliers = base_multipliers ** (1.0 + strength)
            else:
                # Uniform distribution when threshold is 0
                multipliers = np.ones(5)
            
            multipliers = multipliers / multipliers.sum()
            
            # Calculate samples per bucket to reach target_size
            samples_per_bucket = (multipliers * target_size).astype(int)
            
            # Adjust to ensure exact total
            diff = target_size - samples_per_bucket.sum()
            samples_per_bucket[-1] += diff  # Add remainder to hardest bucket
            
            # Sample from each bucket using fast shuffle-and-slice (no np.random.choice)
            selected_indices = []
            for bucket_idx, bucket_samples in zip(bucket_indices, samples_per_bucket):
                if len(bucket_idx) > 0 and bucket_samples > 0:
                    if len(bucket_idx) >= bucket_samples:
                        # Shuffle bucket and take first N samples (fast)
                        shuffled = bucket_idx.copy()
                        np.random.shuffle(shuffled)
                        selected_indices.append(shuffled[:bucket_samples])
                    else:
                        # Use all samples from bucket if it's smaller than allocation
                        selected_indices.append(bucket_idx)
            
            selected_indices = np.concatenate(selected_indices)
            np.random.shuffle(selected_indices)
            return selected_indices
        
        elif self.strategy == 'mixed':
            # Weighted sampling across full difficulty spectrum
            # threshold=0.0: favor easy samples (exponential decay)
            # threshold=0.5: uniform sampling
            # threshold=1.0: favor hard samples (exponential growth)
            
            # Calculate target epoch size
            target_size = max(int(self.num_samples * self.epoch_size_ratio), self.batch_size)
            
            # Divide all samples into 5 difficulty buckets (easy to hard)
            bucket_indices = np.array_split(sorted_indices, 5)
            
            if self.difficulty_threshold < 0.5:
                # Favor easy samples - exponentially decay from easy to hard
                strength = 2.0 * (0.5 - self.difficulty_threshold)  # Range: [0, 1]
                base_multipliers = np.array([5, 3, 2, 1, 1])  # Easy to hard
                multipliers = base_multipliers ** (1.0 + strength)
            elif self.difficulty_threshold > 0.5:
                # Favor hard samples - exponentially grow from easy to hard
                strength = 2.0 * (self.difficulty_threshold - 0.5)  # Range: [0, 1]
                base_multipliers = np.array([1, 1, 2, 3, 5])  # Easy to hard
                multipliers = base_multipliers ** (1.0 + strength)
            else:
                # Uniform sampling at threshold=0.5
                multipliers = np.ones(5)
            
            multipliers = multipliers / multipliers.sum()
            
            # Calculate samples per bucket to reach target_size
            samples_per_bucket = (multipliers * target_size).astype(int)
            
            # Adjust to ensure exact total
            diff = target_size - samples_per_bucket.sum()
            if self.difficulty_threshold < 0.5:
                samples_per_bucket[0] += diff  # Add remainder to easiest bucket
            else:
                samples_per_bucket[-1] += diff  # Add remainder to hardest bucket
            
            # Sample from each bucket using fast shuffle-and-slice (no np.random.choice)
            selected_indices = []
            for bucket_idx, bucket_samples in zip(bucket_indices, samples_per_bucket):
                if len(bucket_idx) > 0 and bucket_samples > 0:
                    if len(bucket_idx) >= bucket_samples:
                        # Shuffle bucket and take first N samples (fast)
                        shuffled = bucket_idx.copy()
                        np.random.shuffle(shuffled)
                        selected_indices.append(shuffled[:bucket_samples])
                    else:
                        # Use all samples from bucket if it's smaller than allocation
                        selected_indices.append(bucket_idx)
            
            selected_indices = np.concatenate(selected_indices)
            np.random.shuffle(selected_indices)
            return selected_indices
        
        elif self.strategy == 'adaptive':
            # Focus on specific difficulty range (sliding window)
            # threshold controls the center of the window
            # Window size controlled by epoch_size_ratio
            
            window_size = self.epoch_size_ratio  # Use epoch_size_ratio as window size
            half_window = window_size / 2
            
            # Calculate window center based on threshold
            center = self.difficulty_threshold
            
            # Calculate window boundaries
            start_ratio = max(0.0, center - half_window)
            end_ratio = min(1.0, center + half_window)
            
            # Convert to indices
            start_idx = int(start_ratio * self.num_samples)
            end_idx = int(end_ratio * self.num_samples)
            
            # Ensure minimum window size
            if end_idx - start_idx < self.batch_size:
                # Expand window if too small
                mid_idx = (start_idx + end_idx) // 2
                start_idx = max(0, mid_idx - self.batch_size // 2)
                end_idx = min(self.num_samples, start_idx + self.batch_size)
            
            # Select samples from window
            available_indices = sorted_indices[start_idx:end_idx]
            
            selected_indices = available_indices
            
            np.random.shuffle(selected_indices)
            return selected_indices
        
        else:
            # Default to random
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)
            return indices
    
    def __iter__(self):
        """Iterate through indices."""
        return iter(self.indices)
    
    def __len__(self):
        """Return number of samples."""
        return len(self.indices)


class CurriculumDataLoader:
    """
    Wrapper for DataLoader with curriculum learning support.
    
    Dynamically updates sample ordering based on RL agent's curriculum parameters.
    """
    
    def __init__(
        self,
        dataset,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        epoch_size_ratio=0.7
    ):
        """
        Initialize curriculum data loader.
        
        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
            epoch_size_ratio: Fraction of dataset to use per epoch [0.5, 1.0]
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.epoch_size_ratio = epoch_size_ratio
        
        # Initialize with uniform difficulty scores (updated later)
        self.difficulty_scores = np.ones(len(dataset)) * 0.5
        
        # Current curriculum parameters
        self.difficulty_threshold = 0.0
        self.strategy = 'mixed'
        # self.strategy = 'adaptive'
        
        # Create initial data loader
        self._create_loader()
    
    def _create_loader(self):
        """Create PyTorch DataLoader with current curriculum settings."""
        sampler = CurriculumSampler(
            difficulty_scores=self.difficulty_scores,
            difficulty_threshold=self.difficulty_threshold,
            batch_size=self.batch_size,
            strategy=self.strategy,
            epoch_size_ratio=self.epoch_size_ratio
        )
        
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
    
    def update_curriculum(self, difficulty_threshold, difficulty_scores=None):
        """
        Update curriculum parameters and recreate data loader.
        
        Args:
            difficulty_threshold: New difficulty threshold [0, 1]
            difficulty_scores: Optional updated difficulty scores for all samples
        """
        self.difficulty_threshold = difficulty_threshold
        
        if difficulty_scores is not None:
            self.difficulty_scores = np.array(difficulty_scores)
        
        # Recreate loader with new parameters
        self._create_loader()
    
    def set_strategy(self, strategy):
        """
        Change curriculum strategy.
        
        Args:
            strategy: One of ['easy_to_hard', 'hard_mining', 'mixed', 'adaptive', 'random']
        """
        self.strategy = strategy
        self._create_loader()
    
    def get_loader(self):
        """Get current PyTorch DataLoader."""
        return self.loader
    
    def __len__(self):
        """Return number of batches."""
        return len(self.loader)
    
    def __iter__(self):
        """Iterate through batches."""
        return iter(self.loader)


def compute_dataset_difficulty_scores(
    teacher_model,
    student_model,
    dataset,
    imagenet_to_mini,
    device='cuda',
    batch_size=128
):
    """
    Compute difficulty scores for entire dataset.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        dataset: PyTorch Dataset
        imagenet_to_mini: Label mapping
        device: Device for computation
        batch_size: Batch size for processing
    
    Returns:
        Array of difficulty scores [num_samples]
    """
    from models import compute_sample_difficulty
    
    # Create temporary data loader
    temp_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    all_difficulties = []
    
    teacher_model.eval()
    student_model.eval()
    
    print(f"Computing difficulty scores for {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(temp_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            difficulties = compute_sample_difficulty(
                teacher_model,
                student_model,
                images,
                labels,
                imagenet_to_mini,
                device
            )
            
            all_difficulties.extend(difficulties)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} samples...")
    
    difficulty_array = np.array(all_difficulties)
    
    print(f"Difficulty scores computed:")
    print(f"  Mean: {difficulty_array.mean():.3f}")
    print(f"  Std: {difficulty_array.std():.3f}")
    print(f"  Min: {difficulty_array.min():.3f}")
    print(f"  Max: {difficulty_array.max():.3f}")
    
    return difficulty_array
