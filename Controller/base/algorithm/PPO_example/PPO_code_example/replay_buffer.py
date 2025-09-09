#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Any
import heapq

class ReplayBuffer:
    """改进的经验回放缓冲区"""
    
    def __init__(self, 
                 capacity: int = 10000,
                 use_priority: bool = True,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            use_priority: 是否使用优先级经验回放
            alpha: 优先级指数 (0=均匀采样, 1=纯优先级)
            beta: 重要性采样指数 (0=无修正, 1=完全修正)
            beta_increment: beta的增量
            epsilon: 避免优先级为0的小常数
        """
        self.capacity = capacity
        self.use_priority = use_priority
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 经验存储
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity) if use_priority else None
        
        # 统计信息
        self.position = 0
        self.size = 0
        self.count = 0  # 添加计数器，用于跟踪转移数量
        
    def push(self, 
             state: Dict[str, torch.Tensor],
             mapping_action: List[int],
             bandwidth_action: List[int],
             reward: float,
             value: float,
             mapping_log_prob: List[float],
             bandwidth_log_prob: List[float],
             done: bool,
             priority: Optional[float] = None):
        """
        存储经验
        
        Args:
            state: 环境状态
            mapping_action: 映射动作
            bandwidth_action: 带宽动作
            reward: 奖励
            value: 价值估计
            mapping_log_prob: 映射动作的log概率
            bandwidth_log_prob: 带宽动作的log概率
            done: 终止标志
            priority: 优先级（如果为None，则使用最大优先级）
        """
        experience = {
            'state': state,
            'mapping_action': mapping_action,
            'bandwidth_action': bandwidth_action,
            'reward': reward,
            'value': value,
            'mapping_log_prob': mapping_log_prob,
            'bandwidth_log_prob': bandwidth_log_prob,
            'done': done
        }
        
        if self.use_priority:
            # 使用优先级经验回放
            if priority is None:
                # 新经验的优先级设为当前最大优先级
                priority = max(self.priorities) if self.priorities else 1.0
            
            if self.size < self.capacity:
                self.buffer.append(experience)
                self.priorities.append(priority)
                self.size += 1
            else:
                # 替换最旧的经验
                self.buffer[self.position] = experience
                self.priorities[self.position] = priority
                self.position = (self.position + 1) % self.capacity
            
            self.count += 1  # 增加计数器
        else:
            # 普通经验回放
            self.buffer.append(experience)
            self.size = len(self.buffer)
            self.count += 1  # 增加计数器
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], List[int]]:
        """
        采样经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            experiences: 经验列表
            indices: 采样索引
        """
        if self.size < batch_size:
            # 如果缓冲区中的经验不足，返回所有经验
            experiences = list(self.buffer)
            indices = list(range(len(experiences)))
            return experiences, indices
        
        if self.use_priority:
            # 优先级采样
            return self._priority_sample(batch_size)
        else:
            # 均匀采样
            indices = random.sample(range(self.size), batch_size)
            experiences = [self.buffer[i] for i in indices]
            return experiences, indices
    
    def _priority_sample(self, batch_size: int) -> Tuple[List[Dict], List[int]]:
        """优先级采样"""
        # 计算采样概率
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        # 计算重要性采样权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """更新优先级"""
        if not self.use_priority:
            return
        
        for idx, priority in zip(indices, priorities):
            if idx < self.size:
                self.priorities[idx] = priority + self.epsilon
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        if self.priorities:
            self.priorities.clear()
        self.position = 0
        self.size = 0
        self.count = 0  # 重置计数器
    
    def reset_count(self):
        """重置计数器"""
        self.count = 0
    
    def __len__(self):
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    """优先级经验回放缓冲区的简化版本"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity=capacity, use_priority=True, alpha=alpha, beta=beta)
    
    def push(self, 
             state: Dict[str, torch.Tensor],
             mapping_action: List[int],
             bandwidth_action: List[int],
             reward: float,
             value: float,
             mapping_log_prob: List[float],
             bandwidth_log_prob: List[float],
             done: bool,
             td_error: Optional[float] = None):
        """
        存储经验，使用TD误差作为优先级
        
        Args:
            td_error: TD误差，用于计算优先级
        """
        if td_error is not None:
            priority = abs(td_error) + self.epsilon
        else:
            priority = max(self.priorities) if self.priorities else 1.0
        
        super().push(state, mapping_action, bandwidth_action, reward, value, 
                    mapping_log_prob, bandwidth_log_prob, done, priority)

class MultiStepReplayBuffer(ReplayBuffer):
    """多步经验回放缓冲区"""
    
    def __init__(self, 
                 capacity: int = 10000,
                 n_steps: int = 3,
                 gamma: float = 0.99,
                 use_priority: bool = True):
        """
        初始化多步经验回放缓冲区
        
        Args:
            n_steps: n步学习的步数
            gamma: 折扣因子
        """
        super().__init__(capacity=capacity, use_priority=use_priority)
        self.n_steps = n_steps
        self.gamma = gamma
        
        # 临时存储n步经验
        self.temp_buffer = deque(maxlen=n_steps)
    
    def push(self, 
             state: Dict[str, torch.Tensor],
             mapping_action: List[int],
             bandwidth_action: List[int],
             reward: float,
             value: float,
             mapping_log_prob: List[float],
             bandwidth_log_prob: List[float],
             done: bool,
             priority: Optional[float] = None):
        """存储经验，支持n步学习"""
        
        # 添加到临时缓冲区
        self.temp_buffer.append({
            'state': state,
            'mapping_action': mapping_action,
            'bandwidth_action': bandwidth_action,
            'reward': reward,
            'value': value,
            'mapping_log_prob': mapping_log_prob,
            'bandwidth_log_prob': bandwidth_log_prob,
            'done': done
        })
        
        # 如果临时缓冲区满了或者episode结束，计算n步回报
        if len(self.temp_buffer) == self.n_steps or done:
            n_step_experience = self._compute_n_step_return()
            super().push(**n_step_experience, priority=priority)
            
            # 如果不是episode结束，保留最后n-1步的经验
            if not done:
                self.temp_buffer = deque(list(self.temp_buffer)[-self.n_steps+1:], maxlen=self.n_steps)
    
    def _compute_n_step_return(self) -> Dict[str, Any]:
        """计算n步回报"""
        # 计算累积奖励
        total_reward = 0
        for i, exp in enumerate(self.temp_buffer):
            total_reward += (self.gamma ** i) * exp['reward']
        
        # 获取第一个经验的其他信息
        first_exp = self.temp_buffer[0]
        
        return {
            'state': first_exp['state'],
            'mapping_action': first_exp['mapping_action'],
            'bandwidth_action': first_exp['bandwidth_action'],
            'reward': total_reward,
            'value': first_exp['value'],
            'mapping_log_prob': first_exp['mapping_log_prob'],
            'bandwidth_log_prob': first_exp['bandwidth_log_prob'],
            'done': any(exp['done'] for exp in self.temp_buffer)
        }
    
    def clear(self):
        """清空缓冲区"""
        super().clear()
        self.temp_buffer.clear()

class ExperienceBuffer:
    """简单的经验缓冲区（保持向后兼容）"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Dict[str, Any]):
        """存储经验"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """采样经验"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
