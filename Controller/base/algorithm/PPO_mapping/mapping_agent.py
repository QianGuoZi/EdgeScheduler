#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class MappingAgent(nn.Module):
    """
    PPO_mapping Agent
    专门负责节点映射决策，带宽分配由贪心策略处理
    
    状态表示：
    - 物理节点：每个节点包含3个特征 [CPU可用量, 内存可用量, 链路可用带宽均值]
    - 虚拟节点：每个节点包含3个特征 [CPU需求, 内存需求, 链路带宽需求均值]
    - 决策状态：包含当前步骤、当前虚拟节点、已映射信息等
    """
    
    def __init__(self, 
                 max_physical_nodes: int = 10,
                 max_virtual_nodes: int = 8,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 device: str = None):
        super(MappingAgent, self).__init__()
        
        self.max_physical_nodes = max_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
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
        
        print(f"✅ MappingAgent初始化完成")
        print(f"   设备: {self.device}")
        print(f"   状态维度: {self._calculate_state_dim()}")
        print(f"   隐藏维度: {hidden_dim}")
        print(f"   最大物理节点数: {max_physical_nodes}")
    
    def _calculate_state_dim(self):
        """计算状态向量的维度"""
        # 简化的状态表示：
        # 物理网络摘要 + 虚拟网络摘要 + 当前决策状态
        physical_summary_dim = self.max_physical_nodes * 3  # 每个物理节点：CPU可用量，内存可用量，链路可用带宽均值
        virtual_summary_dim = self.max_virtual_nodes * 3    # 每个虚拟节点：CPU需求，内存需求，链路带宽需求均值
        decision_state_dim = 10  # 当前步骤、当前虚拟节点、映射进度等
        
        return physical_summary_dim + virtual_summary_dim + decision_state_dim
    
    def _encode_state(self, state: Dict) -> torch.Tensor:
        """将复杂的字典状态编码为固定长度的向量"""
        features = []
        
        # 1. 物理网络特征编码
        physical_features = state['physical_features']  # [num_physical_nodes, 5]
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
        virtual_features = state['virtual_features']  # [num_virtual_nodes, 3]
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
        decision_features = torch.zeros(10, device=physical_features.device)
        decision_features[0] = state.get('current_step', 0) / 10.0  # 归一化步数
        decision_features[1] = state.get('current_virtual_node', 0) / self.max_virtual_nodes  # 当前虚拟节点
        decision_features[2] = state.get('num_virtual_nodes', 0) / self.max_virtual_nodes  # 虚拟节点总数
        decision_features[3] = state.get('num_virtual_links', 0) / 20.0  # 虚拟链路总数
        
        # 映射进度
        node_mapping = state.get('node_mapping', [])
        if node_mapping:
            mapped_count = sum(1 for x in node_mapping if x != -1)
            decision_features[4] = mapped_count / len(node_mapping)  # 映射完成度
            
            # 当前虚拟节点的连接信息
            current_vnode = state.get('current_virtual_node', 0)
            if current_vnode < len(node_mapping):
                virtual_edges = state.get('virtual_edges', torch.tensor([]))
                if virtual_edges.numel() > 0:
                    # 计算当前虚拟节点的连接数
                    connections = 0
                    for i in range(virtual_edges.size(1)):
                        if virtual_edges[0, i] == current_vnode or virtual_edges[1, i] == current_vnode:
                            connections += 1
                    decision_features[5] = connections / 10.0  # 归一化连接数
                    
                    # 计算已连接的映射节点数（用于同节点映射奖励估计）
                    connected_mapped = 0
                    for i in range(virtual_edges.size(1)):
                        if virtual_edges[0, i] == current_vnode:
                            other_vnode = virtual_edges[1, i].item()
                            if other_vnode < len(node_mapping) and node_mapping[other_vnode] != -1:
                                connected_mapped += 1
                        elif virtual_edges[1, i] == current_vnode:
                            other_vnode = virtual_edges[0, i].item()
                            if other_vnode < len(node_mapping) and node_mapping[other_vnode] != -1:
                                connected_mapped += 1
                    decision_features[6] = connected_mapped / 10.0 if connections > 0 else 0.0
        
        features.append(decision_features)
        
        # 拼接所有特征
        state_vector = torch.cat(features, dim=0)
        return state_vector
    
    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            action_logits: 动作logits（物理节点选择）
            value: 状态价值
            state_encoding: 状态编码（用于其他计算）
        """
        # 编码状态
        state_vector = self._encode_state(state)
        state_encoding = self.state_encoder(state_vector)
        
        # 映射策略
        action_logits = self.mapping_policy(state_encoding)
        
        # 价值估计
        value = self.critic(state_encoding)
        
        return action_logits, value, state_encoding
    
    def select_action(self, state: Dict, temperature: float = 1.0) -> Tuple[int, float, float]:
        """
        选择动作
        
        Returns:
            action: 选择的动作（物理节点索引）
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
            
            # 动作掩码（只考虑有效的物理节点）
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
        获取动作掩码（限制有效的物理节点选择）
        
        Returns:
            mask: 布尔张量，True表示有效动作，False表示无效动作
        """
        num_physical_nodes = state.get('num_physical_nodes', self.max_physical_nodes)
        mask = torch.zeros(self.max_physical_nodes, dtype=torch.bool, device=self.device)
        mask[:num_physical_nodes] = True
        return mask
    
    def calculate_loss(self, states: List[Dict], actions: List[int], rewards: List[float], 
                      dones: List[bool], gamma: float = 0.99, 
                      ppo_clip: float = 0.2, value_coef: float = 0.5, 
                      entropy_coef: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        计算PPO损失
        """
        self.train()
        
        # 准备数据
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # 编码所有状态
        state_batch = []
        for state in states:
            state = self._move_state_to_device(state)
            state_batch.append(self._encode_state(state))
        state_batch = torch.stack(state_batch)
        
        # 获取当前策略的输出
        values = []
        action_logits_list = []
        old_log_probs = []
        
        for i, state in enumerate(states):
            state = self._move_state_to_device(state)
            logits, value, _ = self.forward(state)
            values.append(value)
            action_logits_list.append(logits)
            
            # 计算旧策略的log概率
            action_mask = self._get_action_mask(state)
            if action_mask is not None:
                logits_masked = logits.masked_fill(~action_mask, float('-inf'))
            else:
                logits_masked = logits
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
            state = self._move_state_to_device(state)
            logits, value, _ = self.forward(state)
            action_logits_current.append(logits)
            values_current.append(value)
        
        values_current = torch.cat(values_current)
        
        # 计算策略损失（PPO clip）
        policy_loss = 0
        entropy_loss = 0
        
        for i in range(len(actions)):
            # 应用动作掩码
            state = self._move_state_to_device(states[i])
            action_mask = self._get_action_mask(state)
            if action_mask is not None:
                logits_masked = action_logits_current[i].masked_fill(~action_mask, float('-inf'))
            else:
                logits_masked = action_logits_current[i]
            
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
                'max_physical_nodes': self.max_physical_nodes,
                'max_virtual_nodes': self.max_virtual_nodes,
                'hidden_dim': self.hidden_dim,
            }
        }
        torch.save(checkpoint, filepath)
        print(f"✅ 模型检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 模型检查点已从 {filepath} 加载")
        return checkpoint.get('config', {})
