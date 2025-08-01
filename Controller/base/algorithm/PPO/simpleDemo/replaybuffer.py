import torch
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any
import random

class PPOBuffer:
    """PPO经验回放缓冲区"""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
        # 存储的数据结构
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def store(self, 
              state: Dict[str, torch.Tensor],
              action: Tuple[int, int],
              reward: float,
              value: float,
              log_prob: Tuple[float, float],
              done: bool):
        """存储一个经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """计算优势函数和回报"""
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            
            # 计算回报
            if i == len(self.rewards) - 1:
                returns.insert(0, self.rewards[i])
            else:
                returns.insert(0, self.rewards[i] + gamma * returns[0] * (1 - self.dones[i]))
        
        self.advantages = advantages
        self.returns = returns
        
    def get_batch(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """获取一个批次的数据"""
        if batch_size is None:
            batch_size = len(self.states)
        
        # 随机采样
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))
        
        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_advantages = torch.tensor([self.advantages[i] for i in indices], dtype=torch.float32)
        batch_returns = torch.tensor([self.returns[i] for i in indices], dtype=torch.float32)
        # 处理log_probs，确保维度一致
        log_probs_list = []
        for i in indices:
            log_prob = self.log_probs[i]
            if isinstance(log_prob, tuple):
                # 如果是元组，将两个log概率相加
                total_log_prob = log_prob[0] + log_prob[1]
            else:
                total_log_prob = log_prob
            log_probs_list.append(total_log_prob)
        
        batch_old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32)
        
        return {
            'states': batch_states,
            'actions': batch_actions,
            'advantages': batch_advantages,
            'returns': batch_returns,
            'old_log_probs': batch_old_log_probs
        }
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def __len__(self):
        return len(self.states)

class EpisodeBuffer:
    """单次episode的缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def store(self, 
              state: Dict[str, torch.Tensor],
              action: Tuple[int, int],
              reward: float,
              value: float,
              log_prob: Tuple[float, float],
              done: bool):
        """存储一个经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get_episode_data(self) -> Dict[str, Any]:
        """获取整个episode的数据"""
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones
        }
    
    def clear(self):
        """清空episode缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)

class PrioritizedBuffer:
    """带优先级的经验回放缓冲区"""
    
    def __init__(self, buffer_size: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.buffer_size = buffer_size
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def store(self, experience: Dict[str, Any], priority: float = None):
        """存储经验"""
        if priority is None:
            priority = np.max(self.priorities[:self.size]) if self.size > 0 else 1.0
        
        if self.size < self.buffer_size:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """采样经验"""
        if self.size < batch_size:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            # 基于优先级采样
            priorities = self.priorities[:self.size]
            probabilities = priorities / np.sum(priorities)
            indices = np.random.choice(self.size, size=batch_size, p=probabilities)
        
        # 计算重要性采样权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # 归一化
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
    
    def __len__(self):
        return self.size
