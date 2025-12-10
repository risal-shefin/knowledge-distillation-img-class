"""
RL Environment for Curriculum Learning in Knowledge Distillation.

This module defines the environment that the RL agent interacts with to
control curriculum parameters (temperature, difficulty thresholds) during training.
"""

import numpy as np
import torch


class CurriculumEnvironment:
    """
    Environment for curriculum learning decisions in knowledge distillation.
    
    State Space:
    - Current epoch progress (normalized)
    - Recent validation accuracy trend (last 3 epochs)
    - Current training loss
    - Loss trend (increasing/decreasing)
    - Current temperature
    - Average sample difficulty score
    
    Action Space:
    - Temperature adjustment: [-0.5, 0, +0.5]
    - Difficulty threshold: [0.0, 0.25, 0.5, 0.75, 1.0]
    
    Reward:
    - Primary: Validation accuracy improvement
    - Penalty: Loss explosion (encourage stability)
    - Bonus: Convergence speed
    """
    
    def __init__(
        self,
        num_epochs,
        initial_temperature=4.0,
        temp_min=2.0,
        temp_max=8.0,
        baseline_val_acc=0.0
    ):
        """
        Initialize curriculum environment.
        
        Args:
            num_epochs: Total number of training epochs
            initial_temperature: Starting temperature value
            temp_min: Minimum allowed temperature
            temp_max: Maximum allowed temperature
            baseline_val_acc: Baseline validation accuracy for reward normalization
        """
        self.num_epochs = num_epochs
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.baseline_val_acc = baseline_val_acc
        
        # State tracking
        self.current_epoch = 0
        self.current_temperature = initial_temperature
        self.current_difficulty_threshold = 0.0  # 0=all samples, 1=hardest only
        
        # History for state computation
        self.val_acc_history = []
        self.loss_history = []
        self.difficulty_history = []
        
        # Previous state for reward calculation
        self.prev_val_acc = 0.0
        self.prev_loss = float('inf')
        
        # State and action dimensions
        self.state_dim = 8
        self.action_dim = 2  # temperature_delta, difficulty_threshold
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_epoch = 0
        self.current_temperature = 4.0
        self.current_difficulty_threshold = 0.0
        self.val_acc_history = []
        self.loss_history = []
        self.difficulty_history = []
        self.prev_val_acc = 0.0
        self.prev_loss = float('inf')
        
        return self._get_state()
    
    def _get_state(self):
        """
        Compute current state representation.
        
        Returns:
            State vector of shape (state_dim,)
        """
        # Epoch progress (0 to 1)
        epoch_progress = self.current_epoch / self.num_epochs
        
        # Validation accuracy trend (last 3 epochs)
        if len(self.val_acc_history) >= 3:
            val_acc_trend = np.mean(np.diff(self.val_acc_history[-3:]))
        else:
            val_acc_trend = 0.0
        
        # Current validation accuracy (normalized to [0, 1])
        current_val_acc = self.val_acc_history[-1] / 100.0 if self.val_acc_history else 0.0
        
        # Loss trend (last 3 epochs)
        if len(self.loss_history) >= 3:
            loss_trend = np.mean(np.diff(self.loss_history[-3:]))
        else:
            loss_trend = 0.0
        
        # Current loss (normalized)
        current_loss = min(self.loss_history[-1], 10.0) / 10.0 if self.loss_history else 1.0
        
        # Temperature (normalized to [0, 1])
        temp_normalized = (self.current_temperature - self.temp_min) / (self.temp_max - self.temp_min)
        
        # Average difficulty score
        avg_difficulty = np.mean(self.difficulty_history) if self.difficulty_history else 0.5
        
        # Difficulty threshold
        difficulty_threshold = self.current_difficulty_threshold
        
        state = np.array([
            epoch_progress,
            val_acc_trend,
            current_val_acc,
            loss_trend,
            current_loss,
            temp_normalized,
            avg_difficulty,
            difficulty_threshold
        ], dtype=np.float32)
        
        return state
    
    def step(self, action, val_acc, train_loss, avg_difficulty_score):
        """
        Execute action and compute reward.
        
        Args:
            action: Tuple of (temperature_delta, difficulty_threshold)
                    temperature_delta in [-1, 0, 1] -> actual delta [-0.5, 0, 0.5]
                    difficulty_threshold in [0, 1, 2, 3, 4] -> [0.0, 0.25, 0.5, 0.75, 1.0]
            val_acc: Current validation accuracy (%)
            train_loss: Current training loss
            avg_difficulty_score: Average difficulty score of samples
        
        Returns:
            next_state, reward, done
        """
        # Parse action
        temp_action_idx, diff_action_idx = action
        
        # Apply temperature adjustment
        temp_delta_map = {0: -0.5, 1: 0.0, 2: 0.5}
        temp_delta = temp_delta_map.get(temp_action_idx, 0.0)
        self.current_temperature = np.clip(
            self.current_temperature + temp_delta,
            self.temp_min,
            self.temp_max
        )
        
        # Apply difficulty threshold adjustment
        diff_threshold_map = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
        self.current_difficulty_threshold = diff_threshold_map.get(diff_action_idx, 0.0)
        
        # Update history
        self.val_acc_history.append(val_acc)
        self.loss_history.append(train_loss)
        self.difficulty_history.append(avg_difficulty_score)
        self.current_epoch += 1
        
        # Compute reward
        reward = self._compute_reward(val_acc, train_loss)
        
        # Update previous state
        self.prev_val_acc = val_acc
        self.prev_loss = train_loss
        
        # Check if done
        done = self.current_epoch >= self.num_epochs
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, done
    
    def _compute_reward(self, val_acc, train_loss):
        """
        Compute reward based on performance improvements.
        
        Reward components:
        1. Validation accuracy improvement (primary)
        2. Loss stability penalty
        3. Convergence speed bonus
        
        Args:
            val_acc: Current validation accuracy (%)
            train_loss: Current training loss
        
        Returns:
            Scalar reward value
        """
        reward = 0.0
        
        # 1. Validation accuracy improvement (range: -10 to +10)
        if self.prev_val_acc > 0:
            acc_improvement = val_acc - self.prev_val_acc
            reward += acc_improvement * 2.0  # Scale up importance
        
        # 2. Loss stability penalty (penalize sudden increases)
        if self.prev_loss < float('inf'):
            loss_change = train_loss - self.prev_loss
            if loss_change > 0:  # Loss increased
                reward -= min(loss_change * 5.0, 5.0)  # Cap penalty at -5
        
        # 3. Early high accuracy bonus (encourage fast convergence)
        if val_acc > 60.0 and self.current_epoch < self.num_epochs * 0.3:
            reward += 2.0
        
        # 4. Absolute performance bonus (encourage high accuracy)
        if val_acc > 70.0:
            reward += (val_acc - 70.0) * 0.1
        
        return reward
    
    def get_current_params(self):
        """
        Get current curriculum parameters.
        
        Returns:
            Dictionary with temperature and difficulty_threshold
        """
        return {
            'temperature': self.current_temperature,
            'difficulty_threshold': self.current_difficulty_threshold
        }
    
    def get_state_dim(self):
        """Return state dimension."""
        return self.state_dim
    
    def get_action_space(self):
        """
        Return action space specification.
        
        Returns:
            Tuple of (temp_actions, diff_actions)
        """
        return (3, 5)  # 3 temperature actions, 5 difficulty actions
