#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class SimpleSequentialAgent(nn.Module):
    """
    简化的Sequential PPO Agent
    专注于解决收敛问题，使用简单的MLP架构
    每步只输出一个动作：映射阶段输出物理节点索引，带宽阶段输出带宽等级
    
    状态表示：
    - 物理节点：每个节点包含3个特征 [CPU可用量, 内存可用量, 链路可用带宽均值]
    - 虚拟节点：每个节点包含3个特征 [CPU需求, 内存需求, 链路带宽需求均值]
    - 决策状态：10个特征 [当前步骤、阶段、进度等]
    
    注意：物理节点特征从原始格式 [CPU总量, 内存总量, CPU使用率, 内存使用率, 链路可用带宽均值] 
    转换为 [CPU可用量, 内存可用量, 链路可用带宽均值]，其中可用量 = 总量 * (1 - 使用率)
    虚拟节点的链路带宽需求均值 = 连接的所有链路带宽区间(min+max)/2的均值
    """
    
    def __init__(self, 
                 max_physical_nodes: int = 10,
                 max_virtual_nodes: int = 8,
                 bandwidth_levels: int = 5,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 device: str = None):
        super(SimpleSequentialAgent, self).__init__()
        
        self.max_physical_nodes = max_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
        self.bandwidth_levels = bandwidth_levels
        self.hidden_dim = hidden_dim
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 状态编码器（将复杂的图状态编码为向量）
        self.state_encoder = nn.Sequential(
            nn.Linear(self._calculate_state_dim(), hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        # 映射策略头（输出物理节点概率分布）
        self.mapping_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_physical_nodes)
        )
        
        # 带宽策略头（输出带宽等级概率分布）
        self.bandwidth_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bandwidth_levels)
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
        
        # print(f"✅ SimpleSequentialAgent初始化完成")
        # print(f"   设备: {self.device}")
        # print(f"   状态维度: {self._calculate_state_dim()}")
        # print(f"   隐藏维度: {hidden_dim}")
        # print(f"   最大物理节点数: {max_physical_nodes}")
        # print(f"   带宽等级数: {bandwidth_levels}")
    
    def _calculate_state_dim(self):
        """计算状态向量的维度"""
        # 简化的状态表示：
        # 物理网络摘要 + 虚拟网络摘要 + 当前决策状态
        # 原始物理特征：[CPU可用量，内存可用量，链路可用带宽均值]
        physical_summary_dim = self.max_physical_nodes * 3  # 每个物理节点：CPU可用量，内存可用量，链路可用带宽均值
        # 虚拟特征：[CPU需求，内存需求，链路带宽需求均值]
        virtual_summary_dim = self.max_virtual_nodes * 3    # 每个虚拟节点：CPU需求，内存需求，链路带宽需求均值
        decision_state_dim = 10  # 当前步骤、阶段等
        
        return physical_summary_dim + virtual_summary_dim + decision_state_dim
    
    def _encode_state(self, state: Dict) -> torch.Tensor:
        """将复杂的字典状态编码为固定长度的向量"""
        features = []
        
        # 1. 物理网络特征编码
        physical_features = state['physical_features']  # [num_physical_nodes, 5] - 原始格式：CPU总量，内存总量，CPU使用率，内存使用率，链路可用带宽均值
        num_physical_nodes = physical_features.size(0)
        
        # 转换为可用资源：总量 * (1 - 使用率)，保留链路可用带宽均值
        # 输入格式：[CPU总量, 内存总量, CPU使用率, 内存使用率, 链路可用带宽均值]
        # 输出格式：[CPU可用量, 内存可用量, 链路可用带宽均值]
        cpu_available = physical_features[:, 0] * (1 - physical_features[:, 2])  # CPU可用量
        memory_available = physical_features[:, 1] * (1 - physical_features[:, 3])  # 内存可用量
        avg_available_bandwidth = physical_features[:, 4]  # 连接的链路可用带宽均值
        
        # 组合为3个特征：[CPU可用量, 内存可用量, 链路可用带宽均值]
        available_features = torch.stack([cpu_available, memory_available, avg_available_bandwidth], dim=1)
        
        # 填充到最大节点数
        if num_physical_nodes < self.max_physical_nodes:
            padding = torch.zeros(self.max_physical_nodes - num_physical_nodes, 3, 
                                device=physical_features.device)
            physical_padded = torch.cat([available_features, padding], dim=0)
        else:
            physical_padded = available_features[:self.max_physical_nodes]
        
        features.append(physical_padded.flatten())  # [max_physical_nodes * 3]
        
        # 2. 虚拟网络特征编码
        virtual_features = state['virtual_features']  # [num_virtual_nodes, 3] - 格式：CPU需求，内存需求，链路带宽需求均值
        num_virtual_nodes = virtual_features.size(0)
        
        # 填充到最大节点数
        if num_virtual_nodes < self.max_virtual_nodes:
            padding = torch.zeros(self.max_virtual_nodes - num_virtual_nodes, 3, 
                                device=virtual_features.device)
            virtual_padded = torch.cat([virtual_features, padding], dim=0)
        else:
            virtual_padded = virtual_features[:self.max_virtual_nodes]
        
        features.append(virtual_padded.flatten())  # [max_virtual_nodes * 3]
        
        # 3. 决策状态特征
        decision_features = torch.zeros(10, device=physical_features.device)  # 10个特征
        decision_features[0] = state.get('current_step', 0) / 10.0  # 归一化步数
        decision_features[1] = 1.0 if state.get('mapping_phase', True) else 0.0  # 当前阶段
        decision_features[2] = state.get('current_virtual_node', 0) / self.max_virtual_nodes  # 当前虚拟节点
        decision_features[3] = state.get('current_link_index', 0) / 10.0  # 当前链路索引
        decision_features[4] = state.get('num_virtual_nodes', 0) / self.max_virtual_nodes  # 虚拟节点总数
        decision_features[5] = state.get('num_virtual_links', 0) / 20.0  # 虚拟链路总数
        
        # 映射进度
        partial_mapping = state.get('partial_mapping', [])
        if partial_mapping:
            mapped_count = sum(1 for x in partial_mapping if x != -1)
            decision_features[6] = mapped_count / len(partial_mapping)  # 映射完成度
        
        # 多任务信息（在多作业环境中由 state 提供；单作业环境下默认为0）
        # 这里等价于在原单任务决策状态基础上额外增加3个标量信息：
        # - multi_queue_len_norm: 队列长度（归一化）
        # - multi_completed_norm: 已完成任务数（归一化）
        # - multi_has_current_task: 是否有任务在处理（0/1）
        decision_features[7] = state.get('multi_queue_len_norm', 0.0)
        decision_features[8] = state.get('multi_completed_norm', 0.0)
        decision_features[9] = state.get('multi_has_current_task', 0.0)
        
        features.append(decision_features)
        
        # 拼接所有特征
        state_vector = torch.cat(features, dim=0)
        return state_vector
    
    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            action_logits: 动作logits
            value: 状态价值
            state_encoding: 状态编码（用于其他计算）
        """
        # 编码状态
        state_vector = self._encode_state(state)
        state_encoding = self.state_encoder(state_vector)
        
        # 根据当前阶段选择对应的策略头
        if state.get('mapping_phase', True):
            action_logits = self.mapping_policy(state_encoding)
        else:
            action_logits = self.bandwidth_policy(state_encoding)
        
        # 价值估计
        value = self.critic(state_encoding)
        
        return action_logits, value, state_encoding
    
    def select_action(self, state: Dict, temperature: float = 1.0) -> Tuple[int, float, float]:
        """
        选择动作
        
        Returns:
            action: 选择的动作（整数）
            log_prob: 动作的对数概率
            value: 状态价值
        """
        self.eval()
        with torch.no_grad():
            # 将状态移动到正确的设备
            state = self._move_state_to_device(state)
            
            # 前向传播
            action_logits, value, _ = self.forward(state)
            
            # 应用温度缩放
            scaled_logits = action_logits / temperature
            
            # 动作掩码（如果需要的话）
            action_mask = self._get_action_mask(state)
            if action_mask is not None:
                scaled_logits = scaled_logits.masked_fill(~action_mask, float('-inf'))
            
            # 计算动作概率
            action_probs = F.softmax(scaled_logits, dim=-1)
            
            # 采样动作
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def _move_state_to_device(self, state: Dict) -> Dict:
        """将状态张量移动到正确的设备"""
        new_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                new_state[key] = value.to(self.device)
            else:
                new_state[key] = value
        return new_state
    
    def _get_action_mask(self, state: Dict) -> Optional[torch.Tensor]:
        """
        获取动作掩码（可选实现约束）
        
        Returns:
            mask: 布尔张量，True表示有效动作，False表示无效动作
        """
        if state.get('mapping_phase', True):
            # 映射阶段：所有物理节点都是有效的（简化处理）
            num_physical_nodes = state.get('num_physical_nodes', self.max_physical_nodes)
            mask = torch.zeros(self.max_physical_nodes, dtype=torch.bool, device=self.device)
            mask[:num_physical_nodes] = True
            return mask
        else:
            # 带宽阶段：所有带宽等级都是有效的
            mask = torch.ones(self.bandwidth_levels, dtype=torch.bool, device=self.device)
            return mask
    
    def calculate_loss(self, states: List[Dict], actions: List[int], rewards: List[float], 
                      dones: List[bool], gamma: float = 0.99) -> Dict[str, torch.Tensor]:
        """
        计算PPO损失（简化版本）
        """
        self.train()
        
        # 准备数据
        state_batch = []
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # 编码所有状态
        for state in states:
            state = self._move_state_to_device(state)
            state_batch.append(self._encode_state(state))
        state_batch = torch.stack(state_batch)
        
        # 计算优势和目标值
        values = []
        action_logits_list = []
        
        for i, state in enumerate(states):
            state = self._move_state_to_device(state)
            logits, value, _ = self.forward(state)
            values.append(value)
            action_logits_list.append(logits)
        
        values = torch.cat(values)
        
        # 计算GAE优势（简化版本）
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
        
        # 重新前向传播获得当前策略的动作概率
        action_logits_current = []
        values_current = []
        
        for i, state in enumerate(states):
            state = self._move_state_to_device(state)
            logits, value, _ = self.forward(state)
            action_logits_current.append(logits)
            values_current.append(value)
        
        values_current = torch.cat(values_current)
        
        # 计算策略损失
        policy_loss = 0
        for i in range(len(actions)):
            action_probs = F.softmax(action_logits_current[i], dim=-1)
            action_log_prob = torch.log(action_probs[actions[i]] + 1e-8)
            policy_loss -= action_log_prob * advantages[i].detach()
        
        policy_loss = policy_loss / len(actions)
        
        # 计算价值损失
        value_loss = F.mse_loss(values_current, targets)
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
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