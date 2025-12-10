"""
PPO Agent for Curriculum Learning Control.

This module implements a Proximal Policy Optimization (PPO) agent that learns
to dynamically adjust curriculum parameters during knowledge distillation training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs probability distributions over actions
    Critic: Estimates state value function V(s)
    """
    
    def __init__(self, state_dim, temp_actions=3, diff_actions=5, hidden_dim=128):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            temp_actions: Number of temperature adjustment actions
            diff_actions: Number of difficulty threshold actions
            hidden_dim: Hidden layer dimension
        """
        super(ActorCritic, self).__init__()
        
        self.temp_actions = temp_actions
        self.diff_actions = diff_actions
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor heads (separate for each action dimension)
        self.actor_temp = nn.Linear(hidden_dim, temp_actions)
        self.actor_diff = nn.Linear(hidden_dim, diff_actions)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """
        Forward pass through network.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            temp_logits, diff_logits, value
        """
        features = self.shared(state)
        
        temp_logits = self.actor_temp(features)
        diff_logits = self.actor_diff(features)
        value = self.critic(features)
        
        return temp_logits, diff_logits, value
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state: State tensor [state_dim]
            deterministic: If True, select argmax instead of sampling
        
        Returns:
            (temp_action, diff_action), (temp_log_prob, diff_log_prob)
        """
        # Handle state tensor - preserve device
        if isinstance(state, torch.Tensor):
            state = state.unsqueeze(0) if state.dim() == 1 else state
            # Move to model's device if needed
            state = state.to(next(self.parameters()).device)
        else:
            # Convert numpy/list to tensor on model's device
            device = next(self.parameters()).device
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        temp_logits, diff_logits, _ = self.forward(state)
        
        # Get probability distributions
        temp_probs = F.softmax(temp_logits, dim=-1)
        diff_probs = F.softmax(diff_logits, dim=-1)
        
        if deterministic:
            temp_action = torch.argmax(temp_probs, dim=-1).item()
            diff_action = torch.argmax(diff_probs, dim=-1).item()
            temp_log_prob = torch.log(temp_probs[0, temp_action])
            diff_log_prob = torch.log(diff_probs[0, diff_action])
        else:
            # Sample from distributions
            temp_dist = torch.distributions.Categorical(temp_probs)
            diff_dist = torch.distributions.Categorical(diff_probs)
            
            temp_action = temp_dist.sample().item()
            diff_action = diff_dist.sample().item()
            
            # Ensure tensors are on same device as distribution
            device = temp_probs.device
            temp_log_prob = temp_dist.log_prob(torch.tensor(temp_action, device=device))
            diff_log_prob = diff_dist.log_prob(torch.tensor(diff_action, device=device))
        
        return (temp_action, diff_action), (temp_log_prob.item(), diff_log_prob.item())
    
    def evaluate_actions(self, states, temp_actions, diff_actions):
        """
        Evaluate actions for PPO update.
        
        Args:
            states: State tensor [batch_size, state_dim]
            temp_actions: Temperature actions [batch_size]
            diff_actions: Difficulty actions [batch_size]
        
        Returns:
            values, temp_log_probs, diff_log_probs, entropy
        """
        temp_logits, diff_logits, values = self.forward(states)
        
        # Get probability distributions
        temp_probs = F.softmax(temp_logits, dim=-1)
        diff_probs = F.softmax(diff_logits, dim=-1)
        
        temp_dist = torch.distributions.Categorical(temp_probs)
        diff_dist = torch.distributions.Categorical(diff_probs)
        
        # Get log probabilities of taken actions
        temp_log_probs = temp_dist.log_prob(temp_actions)
        diff_log_probs = diff_dist.log_prob(diff_actions)
        
        # Compute entropy for exploration bonus
        temp_entropy = temp_dist.entropy()
        diff_entropy = diff_dist.entropy()
        entropy = temp_entropy + diff_entropy
        
        return values, temp_log_probs, diff_log_probs, entropy


class PPOAgent:
    """
    PPO Agent for curriculum learning control.
    
    Implements Proximal Policy Optimization to learn optimal curriculum strategies.
    """
    
    def __init__(
        self,
        state_dim,
        temp_actions=3,
        diff_actions=5,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=1.0,  # Value loss coefficient
        c2=0.01,  # Entropy coefficient
        max_grad_norm=0.5,
        device='cuda'
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: State space dimension
            temp_actions: Number of temperature actions
            diff_actions: Number of difficulty actions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device for computation
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        
        # Create actor-critic network
        self.actor_critic = ActorCritic(
            state_dim,
            temp_actions,
            diff_actions,
            hidden_dim=128
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Experience buffer
        self.states = []
        self.temp_actions = []
        self.diff_actions = []
        self.rewards = []
        self.temp_log_probs = []
        self.diff_log_probs = []
        self.values = []
        self.dones = []
        
        # Training stats
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
    
    def select_action(self, state, deterministic=False):
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: If True, select argmax action
        
        Returns:
            action: (temp_action, diff_action)
        """
        with torch.no_grad():
            actions, log_probs = self.actor_critic.get_action(state, deterministic)
        
        return actions
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """
        Store transition in buffer.
        
        Args:
            state: Current state
            action: Taken action (temp_action, diff_action)
            reward: Received reward
            log_prob: Log probability (temp_log_prob, diff_log_prob)
            value: Estimated value
            done: Episode done flag
        """
        self.states.append(state)
        self.temp_actions.append(action[0])
        self.diff_actions.append(action[1])
        self.rewards.append(reward)
        self.temp_log_probs.append(log_prob[0])
        self.diff_log_probs.append(log_prob[1])
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value estimate for next state
        
        Returns:
            advantages, returns
        """
        values = self.values + [next_value]
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_state, num_epochs=4, batch_size=64):
        """
        Update policy using PPO.
        
        Args:
            next_state: Final state for value bootstrapping
            num_epochs: Number of optimization epochs
            batch_size: Mini-batch size
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.states) == 0:
            return {}
        
        # Compute advantages and returns
        with torch.no_grad():
            # Handle next_state tensor on correct device
            if isinstance(next_state, torch.Tensor):
                next_state_tensor = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state
                next_state_tensor = next_state_tensor.to(self.device)
            else:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, _, next_value = self.actor_critic(next_state_tensor)
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        # Handle states that might already be GPU tensors
        if isinstance(self.states[0], torch.Tensor):
            states = torch.stack([s.squeeze() if s.dim() > 1 else s for s in self.states]).to(self.device)
        else:
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
        temp_actions = torch.LongTensor(self.temp_actions).to(self.device)
        diff_actions = torch.LongTensor(self.diff_actions).to(self.device)
        old_temp_log_probs = torch.FloatTensor(self.temp_log_probs).to(self.device)
        old_diff_log_probs = torch.FloatTensor(self.diff_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        dataset_size = len(self.states)
        indices = np.arange(dataset_size)
        
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_total_losses = []
        
        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_temp_actions = temp_actions[batch_indices]
                batch_diff_actions = diff_actions[batch_indices]
                batch_old_temp_log_probs = old_temp_log_probs[batch_indices]
                batch_old_diff_log_probs = old_diff_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                values, temp_log_probs, diff_log_probs, entropy = \
                    self.actor_critic.evaluate_actions(
                        batch_states,
                        batch_temp_actions,
                        batch_diff_actions
                    )
                
                values = values.squeeze()
                
                # Compute ratios
                temp_ratio = torch.exp(temp_log_probs - batch_old_temp_log_probs)
                diff_ratio = torch.exp(diff_log_probs - batch_old_diff_log_probs)
                ratio = temp_ratio * diff_ratio
                
                # Surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy.mean()
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_total_losses.append(loss.item())
        
        # Clear buffer
        self.clear_buffer()
        
        # Store stats
        stats = {
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses),
            'total_loss': np.mean(epoch_total_losses)
        }
        
        return stats
    
    def clear_buffer(self):
        """Clear experience buffer."""
        self.states = []
        self.temp_actions = []
        self.diff_actions = []
        self.rewards = []
        self.temp_log_probs = []
        self.diff_log_probs = []
        self.values = []
        self.dones = []
    
    def save(self, path):
        """Save agent checkpoint."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_action_with_value(self, state, deterministic=False):
        """
        Get action, log probability, and value estimate.
        
        Args:
            state: Current state
            deterministic: If True, select argmax action
        
        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            action, log_prob = self.actor_critic.get_action(state, deterministic)
            
            # Handle state tensor on correct device
            if isinstance(state, torch.Tensor):
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                state_tensor = state_tensor.to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, value = self.actor_critic(state_tensor)
            value = value.item()
        
        return action, log_prob, value
