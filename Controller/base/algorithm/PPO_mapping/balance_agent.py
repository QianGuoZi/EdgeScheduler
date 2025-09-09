#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class BalanceAgent(nn.Module):
    """
    PPO_balance Agent
    结合PPO_mapping的PPO算法实现和A2C的简化状态表示
    
    状态表示（来自A2C）：
    - 物理节点可用资源: [num_physical_nodes, 2] (CPU, Memory)
    - 当前任务需求: [2] (CPU, Memory)
    
    动作空间：
    - 选择将当前任务映射到哪个物理节点
    """
    
    def __init__(self, 
                 num_physical_nodes: int = 10,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 device: str = None):
        super(BalanceAgent, self).__init__()
        
        self.num_physical_nodes = num_physical_nodes
        self.hidden_dim = hidden_dim
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 状态维度：物理节点资源 + 任务需求
        # 每个物理节点的可用CPU和内存资源量 (2 * num_physical_nodes)
        # 额外加上当前任务的资源需求 (2)
        self.state_dim = 2 * num_physical_nodes + 2
        
        # 状态编码器（将状态编码为向量）
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        # 策略网络（输出物理节点概率分布）
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_physical_nodes)
        )
        
        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # 移动到设备
        self.to(self.device)
        
        # 初始化权重
        self._init_weights()
        
        print(f"✅ BalanceAgent初始化完成")
        print(f"   设备: {self.device}")
        print(f"   状态维度: {self.state_dim}")
        print(f"   隐藏维度: {hidden_dim}")
        print(f"   物理节点数: {num_physical_nodes}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def _encode_state(self, physical_resources: torch.Tensor, 
                     task_requirements: torch.Tensor) -> torch.Tensor:
        """
        编码状态为固定长度的向量（A2C的状态表示）
        
        Args:
            physical_resources: [num_physical_nodes, 2] - 每个物理节点的可用CPU和内存
            task_requirements: [2] - 当前任务的CPU和内存需求
            
        Returns:
            state_vector: [state_dim] - 编码后的状态向量
        """
        # 确保输入张量在正确的设备上
        physical_resources = physical_resources.to(self.device)
        task_requirements = task_requirements.to(self.device)
        
        # 检查输入是否包含无效值
        if torch.isnan(physical_resources).any() or torch.isinf(physical_resources).any():
            print("警告: physical_resources包含无效值，用零填充")
            physical_resources = torch.zeros_like(physical_resources)
        
        if torch.isnan(task_requirements).any() or torch.isinf(task_requirements).any():
            print("警告: task_requirements包含无效值，用零填充")
            task_requirements = torch.zeros_like(task_requirements)
        
        # 填充或截断物理资源到固定大小
        num_nodes = physical_resources.size(0)
        if num_nodes < self.num_physical_nodes:
            # 如果物理节点数不足，用零填充
            padding = torch.zeros(self.num_physical_nodes - num_nodes, 2, 
                                device=self.device)
            physical_padded = torch.cat([physical_resources, padding], dim=0)
        else:
            # 如果物理节点数超出，截断
            physical_padded = physical_resources[:self.num_physical_nodes]
        
        # 简单的归一化：除以100（假设资源值通常在0-100范围内）
        physical_normalized = physical_padded / 100.0
        task_normalized = task_requirements / 100.0
        
        # 拼接状态：[物理节点可用资源, 任务需求]
        state_vector = torch.cat([
            physical_normalized.flatten(),  # [num_physical_nodes * 2]
            task_normalized                 # [2]
        ], dim=0)
        
        # 确保状态向量没有无效值
        if torch.isnan(state_vector).any() or torch.isinf(state_vector).any():
            print("警告: state_vector包含无效值，用零向量替换")
            state_vector = torch.zeros_like(state_vector)
        
        return state_vector
    
    def forward(self, physical_resources: torch.Tensor, 
               task_requirements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            physical_resources: [num_physical_nodes, 2] - 物理节点可用资源
            task_requirements: [2] - 任务资源需求
            
        Returns:
            action_logits: [num_physical_nodes] - 动作logits
            value: [1] - 状态价值
            state_encoding: [hidden_dim] - 状态编码
        """
        # 编码状态
        state_vector = self._encode_state(physical_resources, task_requirements)
        state_encoding = self.state_encoder(state_vector)
        
        # 策略输出
        action_logits = self.policy(state_encoding)
        
        # 价值估计
        value = self.critic(state_encoding)
        
        return action_logits, value, state_encoding
    
    def select_action(self, physical_resources: torch.Tensor, 
                     task_requirements: torch.Tensor,
                     valid_actions: Optional[torch.Tensor] = None,
                     temperature: float = 1.0) -> Tuple[int, float, float]:
        """
        选择动作
        
        Args:
            physical_resources: [num_physical_nodes, 2] - 物理节点可用资源
            task_requirements: [2] - 任务资源需求
            valid_actions: [num_physical_nodes] - 有效动作掩码（可选）
            temperature: 温度参数，用于控制探索
            
        Returns:
            action: 选择的动作（物理节点索引）
            log_prob: 动作的对数概率
            value: 状态价值
        """
        self.eval()
        with torch.no_grad():
            # 前向传播
            action_logits, value, _ = self.forward(physical_resources, task_requirements)
            
            # 检查logits是否包含NaN或Inf
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                print(f"警告: action_logits包含无效值: {action_logits}")
                # 用零向量替换无效的logits
                action_logits = torch.zeros_like(action_logits)
            
            # 应用温度缩放
            scaled_logits = action_logits / temperature
            
            # 应用动作掩码（限制有效动作）
            if valid_actions is not None:
                valid_actions = valid_actions.to(self.device)
                # 检查是否有有效动作
                if not valid_actions.any():
                    print("警告: 没有有效动作，选择第一个动作")
                    return 0, 0.0, value.item()
                
                scaled_logits = scaled_logits.masked_fill(~valid_actions, float('-inf'))
            
            # 添加数值稳定性检查
            max_logit = scaled_logits.max()
            if torch.isinf(max_logit) or torch.isnan(max_logit):
                print("警告: scaled_logits包含无效值，使用均匀分布")
                if valid_actions is not None:
                    action_probs = valid_actions.float() / valid_actions.sum().float()
                else:
                    action_probs = torch.ones(self.num_physical_nodes, device=self.device) / self.num_physical_nodes
            else:
                # 数值稳定的softmax计算
                scaled_logits = scaled_logits - max_logit  # 防止溢出
                action_probs = F.softmax(scaled_logits, dim=-1)
            
            # 检查概率是否有效
            if torch.isnan(action_probs).any() or (action_probs <= 0).all():
                print("警告: action_probs包含无效值，使用均匀分布")
                if valid_actions is not None:
                    action_probs = valid_actions.float() / valid_actions.sum().float()
                else:
                    action_probs = torch.ones(self.num_physical_nodes, device=self.device) / self.num_physical_nodes
            
            # 确保概率和为1（数值稳定性）
            action_probs = action_probs / action_probs.sum()
            
            # 采样动作
            try:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            except ValueError as e:
                print(f"警告: 概率分布创建失败: {e}")
                print(f"action_probs: {action_probs}")
                # 备用方案：随机选择有效动作
                if valid_actions is not None:
                    valid_indices = torch.nonzero(valid_actions).squeeze(-1)
                    if len(valid_indices) > 0:
                        action = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
                    else:
                        action = 0
                else:
                    action = torch.randint(0, self.num_physical_nodes, (1,)).item()
                log_prob = 0.0
            
            return action, log_prob, value.item()
    
    def calculate_loss(self, states: List[Dict], actions: List[int], rewards: List[float], 
                      dones: List[bool], gamma: float = 0.99, 
                      ppo_clip: float = 0.2, value_coef: float = 0.5, 
                      entropy_coef: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        计算PPO损失
        
        Args:
            states: 状态列表，每个状态包含physical_resources和task_requirements
            actions: 动作列表
            rewards: 奖励列表
            dones: 结束标志列表
            gamma: 折扣因子
            ppo_clip: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵损失系数
            
        Returns:
            损失字典
        """
        self.train()
        
        # 准备数据
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # 获取当前策略的输出
        values = []
        action_logits_list = []
        old_log_probs = []
        
        for i, state in enumerate(states):
            physical_resources = state['physical_resources']
            task_requirements = state['task_requirements']
            valid_actions = state.get('valid_actions', None)
            
            logits, value, _ = self.forward(physical_resources, task_requirements)
            values.append(value)
            action_logits_list.append(logits)
            
            # 计算旧策略的log概率
            if valid_actions is not None:
                valid_actions = valid_actions.to(self.device)
                logits_masked = logits.masked_fill(~valid_actions, float('-inf'))
            else:
                logits_masked = logits
            
            # 数值稳定的softmax
            max_logit = logits_masked.max()
            if not torch.isinf(max_logit) and not torch.isnan(max_logit):
                logits_masked = logits_masked - max_logit
            
            old_probs = F.softmax(logits_masked, dim=-1)
            old_log_prob = torch.log(old_probs[actions[i]] + 1e-8)
            old_log_probs.append(old_log_prob)
        
        values = torch.cat(values)
        old_log_probs = torch.stack(old_log_probs)
        
        # 计算GAE优势
        advantages = torch.zeros_like(reward_batch)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1] if t + 1 < len(rewards) else 0
            
            delta = reward_batch[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * 0.95 * last_advantage * (not dones[t])
            last_advantage = advantages[t]
        
        # 计算目标值
        targets = advantages + values.detach()
        
        # 重新前向传播获得当前策略的输出
        action_logits_current = []
        values_current = []
        
        for i, state in enumerate(states):
            physical_resources = state['physical_resources']
            task_requirements = state['task_requirements']
            
            logits, value, _ = self.forward(physical_resources, task_requirements)
            action_logits_current.append(logits)
            values_current.append(value)
        
        values_current = torch.cat(values_current)
        
        # 计算策略损失（PPO clip）
        policy_loss = 0
        entropy_loss = 0
        
        for i in range(len(actions)):
            state = states[i]
            valid_actions = state.get('valid_actions', None)
            
            # 应用动作掩码
            if valid_actions is not None:
                valid_actions = valid_actions.to(self.device)
                logits_masked = action_logits_current[i].masked_fill(~valid_actions, float('-inf'))
            else:
                logits_masked = action_logits_current[i]
            
            # 数值稳定的softmax
            max_logit = logits_masked.max()
            if not torch.isinf(max_logit) and not torch.isnan(max_logit):
                logits_masked = logits_masked - max_logit
            
            # 当前策略概率
            current_probs = F.softmax(logits_masked, dim=-1)
            current_log_prob = torch.log(current_probs[actions[i]] + 1e-8)
            
            # 重要性比率
            ratio = torch.exp(current_log_prob - old_log_probs[i])
            
            # PPO clip
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages[i]
            policy_loss -= torch.min(surr1, surr2)
            
            # 熵损失
            entropy_loss -= entropy_coef * torch.sum(current_probs * torch.log(current_probs + 1e-8))
        
        policy_loss = policy_loss / len(actions)
        entropy_loss = entropy_loss / len(actions)
        
        # 计算价值损失
        value_loss = value_coef * F.mse_loss(values_current, targets)
        
        # 总损失
        total_loss = policy_loss + value_loss + entropy_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'mean_advantage': advantages.mean(),
            'mean_value': values.mean()
        }
    
    def update(self, loss_dict: Dict[str, torch.Tensor]):
        """更新网络参数"""
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
    
    def save_checkpoint(self, filepath: str):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'num_physical_nodes': self.num_physical_nodes,
                'hidden_dim': self.hidden_dim,
                'state_dim': self.state_dim,
            }
        }
        torch.save(checkpoint, filepath)
        print(f"✅ Balance模型检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Balance模型检查点已从 {filepath} 加载")
        return checkpoint.get('config', {})
