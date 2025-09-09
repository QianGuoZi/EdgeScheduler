#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class A2CAgent(nn.Module):
    """
    A2C (Advantage Actor-Critic) Agent
    专门负责任务节点到物理节点的映射决策
    
    状态表示：
    - 每个物理节点的可用CPU和内存资源量
    
    动作空间：
    - 选择将当前任务节点映射到哪个物理节点
    """
    
    def __init__(self, 
                 num_physical_nodes: int = 10,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 device: str = None):
        super(A2CAgent, self).__init__()
        
        self.num_physical_nodes = num_physical_nodes
        self.hidden_dim = hidden_dim
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 状态维度：每个物理节点的可用CPU和内存资源量 (2 * num_physical_nodes)
        # 额外加上当前任务的资源需求 (2)
        self.state_dim = 2 * num_physical_nodes + 2
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor网络（策略网络）- 输出物理节点概率分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_physical_nodes)
        )
        
        # Critic网络（价值网络）- 输出状态价值
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
        
        print(f"✅ A2CAgent初始化完成")
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
        编码状态为固定长度的向量
        
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
               task_requirements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            physical_resources: [num_physical_nodes, 2] - 物理节点可用资源
            task_requirements: [2] - 任务资源需求
            
        Returns:
            action_logits: [num_physical_nodes] - 动作logits
            value: [1] - 状态价值
        """
        # 编码状态
        state_vector = self._encode_state(physical_resources, task_requirements)
        
        # 特征提取
        features = self.feature_extractor(state_vector)
        
        # Actor输出
        action_logits = self.actor(features)
        
        # Critic输出
        value = self.critic(features)
        
        return action_logits, value
    
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
            action_logits, value = self.forward(physical_resources, task_requirements)
            
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
    
    def evaluate_actions(self, physical_resources_batch: List[torch.Tensor],
                        task_requirements_batch: List[torch.Tensor],
                        actions_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于训练）
        
        Args:
            physical_resources_batch: 批次物理资源状态
            task_requirements_batch: 批次任务需求
            actions_batch: 批次动作
            
        Returns:
            log_probs: 动作对数概率
            values: 状态价值
            entropies: 策略熵
        """
        self.train()
        
        batch_size = len(physical_resources_batch)
        log_probs = []
        values = []
        entropies = []
        
        for i in range(batch_size):
            # 前向传播
            action_logits, value = self.forward(
                physical_resources_batch[i], 
                task_requirements_batch[i]
            )
            
            # 检查logits是否包含NaN或Inf
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                print(f"警告: 训练时action_logits包含无效值: {action_logits}")
                # 用零向量替换无效的logits
                action_logits = torch.zeros_like(action_logits)
            
            # 数值稳定的softmax计算
            max_logit = action_logits.max()
            if not torch.isinf(max_logit) and not torch.isnan(max_logit):
                action_logits = action_logits - max_logit  # 防止溢出
            
            # 计算概率分布
            action_probs = F.softmax(action_logits, dim=-1)
            
            # 检查概率是否有效
            if torch.isnan(action_probs).any() or (action_probs <= 0).all():
                print("警告: 训练时action_probs包含无效值，使用均匀分布")
                action_probs = torch.ones_like(action_probs) / action_probs.size(0)
            
            # 确保概率和为1
            action_probs = action_probs / action_probs.sum()
            
            try:
                action_dist = torch.distributions.Categorical(action_probs)
                # 计算对数概率和熵
                log_prob = action_dist.log_prob(actions_batch[i])
                entropy = action_dist.entropy()
            except ValueError as e:
                print(f"警告: 训练时概率分布创建失败: {e}")
                # 使用默认值
                log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)
                entropy = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            log_probs.append(log_prob)
            values.append(value.squeeze())
            entropies.append(entropy)
        
        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)
    
    def calculate_loss(self, 
                      physical_resources_batch: List[torch.Tensor],
                      task_requirements_batch: List[torch.Tensor],
                      actions_batch: torch.Tensor,
                      rewards_batch: torch.Tensor,
                      dones_batch: torch.Tensor,
                      gamma: float = 0.99,
                      value_coef: float = 0.5,
                      entropy_coef: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        计算A2C损失
        
        Args:
            physical_resources_batch: 批次物理资源状态
            task_requirements_batch: 批次任务需求
            actions_batch: 批次动作
            rewards_batch: 批次奖励
            dones_batch: 批次结束标志
            gamma: 折扣因子
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            
        Returns:
            损失字典
        """
        # 确保输入在正确设备上
        actions_batch = actions_batch.to(self.device)
        rewards_batch = rewards_batch.to(self.device)
        dones_batch = dones_batch.to(self.device)
        
        # 评估动作
        log_probs, values, entropies = self.evaluate_actions(
            physical_resources_batch, task_requirements_batch, actions_batch
        )
        
        # 计算折扣奖励（简化版本，假设每个step都是独立的）
        returns = rewards_batch.clone()
        
        # 计算优势
        advantages = returns - values.detach()
        
        # Actor损失（策略梯度）
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic损失（价值函数）
        critic_loss = F.mse_loss(values, returns)
        
        # 熵损失（鼓励探索）
        entropy_loss = -entropies.mean()
        
        # 总损失
        total_loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
        
        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss,
            'mean_advantage': advantages.mean(),
            'mean_value': values.mean(),
            'mean_return': returns.mean()
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
        print(f"✅ A2C模型检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ A2C模型检查点已从 {filepath} 加载")
        return checkpoint.get('config', {})
