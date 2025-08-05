#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import deque
import random
import math
from constraint_manager import ConstraintManager

class GraphEncoder(nn.Module):
    """图神经网络编码器，用于编码物理节点和虚拟工作节点"""
    
    def __init__(self, node_features: int, hidden_dim: int = 128, num_layers: int = 3):
        super(GraphEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 图卷积层
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(node_features, hidden_dim, heads=8, concat=False))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=8, concat=False))
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: 节点特征 [num_nodes, node_features]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_features]
        """
        h = x
        
        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        
        return self.output_proj(h)

class MappingActor(nn.Module):
    """映射Actor，负责输出所有虚拟任务节点的映射结果"""
    
    def __init__(self, 
                 physical_node_dim: int,
                 virtual_node_dim: int,
                 hidden_dim: int = 128,
                 num_physical_nodes: int = 10,
                 max_virtual_nodes: int = 8):
        super(MappingActor, self).__init__()
        
        self.num_physical_nodes = num_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
        self.hidden_dim = hidden_dim
        
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 注意力机制，用于计算物理节点和虚拟节点的匹配度
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 全局映射策略网络 - 输出所有虚拟节点的映射
        self.global_mapping_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_physical_nodes)  # 每个虚拟节点输出一个物理节点选择
        )
        
        # 映射约束检查层
        self.constraint_checker = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr):
        """
        Args:
            physical_features: 物理节点特征 [num_physical_nodes, physical_node_dim]
            physical_edge_index: 物理网络边索引 [2, num_physical_edges]
            physical_edge_attr: 物理网络边特征 [num_physical_edges, edge_features]
            virtual_features: 虚拟节点特征 [num_virtual_nodes, virtual_node_dim]
            virtual_edge_index: 虚拟网络边索引 [2, num_virtual_edges]
            virtual_edge_attr: 虚拟网络边特征 [num_virtual_edges, edge_features]
        
        Returns:
            mapping_logits: 所有虚拟节点的映射logits [num_virtual_nodes, num_physical_nodes]
            constraint_scores: 约束满足度分数 [num_virtual_nodes, num_physical_nodes]
        """
        # 编码物理网络和虚拟网络
        physical_encoded = self.physical_encoder(physical_features, physical_edge_index, physical_edge_attr)
        virtual_encoded = self.virtual_encoder(virtual_features, virtual_edge_index, virtual_edge_attr)
        
        # 注意力机制：计算虚拟节点对物理节点的注意力
        # [num_virtual_nodes, hidden_dim] -> [1, num_virtual_nodes, hidden_dim]
        virtual_encoded_expanded = virtual_encoded.unsqueeze(0)
        physical_encoded_expanded = physical_encoded.unsqueeze(0)
        
        # 计算注意力
        attended_virtual, attention_weights = self.attention(
            virtual_encoded_expanded, 
            physical_encoded_expanded, 
            physical_encoded_expanded
        )
        attended_virtual = attended_virtual.squeeze(0)  # [num_virtual_nodes, hidden_dim]
        
        # 合并编码特征
        combined_features = torch.cat([attended_virtual, virtual_encoded], dim=1)
        
        # 获取实际的虚拟节点数量
        actual_virtual_nodes = virtual_features.size(0)
        
        # 全局映射策略：为每个虚拟节点输出映射logits
        mapping_logits = self.global_mapping_head(combined_features)  # [actual_virtual_nodes, num_physical_nodes]
        
        # 约束检查：计算每个虚拟节点的约束满足度
        constraint_scores = self.constraint_checker(combined_features)
        # 为每个虚拟节点对每个物理节点计算约束分数
        constraint_scores = constraint_scores.squeeze(-1).unsqueeze(1).expand(-1, self.num_physical_nodes)
        
        return mapping_logits, constraint_scores, attention_weights

class BandwidthActor(nn.Module):
    """带宽Actor，负责输出所有虚拟链路的带宽分配结果"""
    
    def __init__(self, 
                 physical_node_dim: int,
                 virtual_node_dim: int,
                 hidden_dim: int = 128,
                 bandwidth_levels: int = 10,
                 max_virtual_nodes: int = 8):
        super(BandwidthActor, self).__init__()
        
        self.bandwidth_levels = bandwidth_levels
        self.max_virtual_nodes = max_virtual_nodes
        self.hidden_dim = hidden_dim
        
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 链路编码器：专门处理虚拟链路信息
        self.link_encoder = nn.Sequential(
            nn.Linear(virtual_node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 全局带宽分配策略网络
        self.global_bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 物理 + 虚拟 + 链路特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, bandwidth_levels)  # 每个链路输出一个带宽等级选择
        )
        
        # 带宽约束检查层
        self.bandwidth_constraint_checker = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                mapping_result):
        """
        Args:
            physical_features: 物理节点特征
            physical_edge_index: 物理网络边索引
            physical_edge_attr: 物理网络边特征
            virtual_features: 虚拟节点特征
            virtual_edge_index: 虚拟网络边索引
            virtual_edge_attr: 虚拟网络边特征
            mapping_result: 映射结果 [num_virtual_nodes] (物理节点索引)
        
        Returns:
            bandwidth_logits: 所有虚拟链路的带宽logits [num_links, bandwidth_levels]
            constraint_scores: 带宽约束满足度分数 [num_links, bandwidth_levels]
        """
        # 编码物理网络和虚拟网络
        physical_encoded = self.physical_encoder(physical_features, physical_edge_index, physical_edge_attr)
        virtual_encoded = self.virtual_encoder(virtual_features, virtual_edge_index, virtual_edge_attr)
        
        # 构建虚拟链路特征
        num_virtual_nodes = virtual_features.size(0)
        link_features = []
        link_indices = []
        
        for i in range(num_virtual_nodes):
            for j in range(i + 1, num_virtual_nodes):
                # 合并两个虚拟节点的特征
                link_feature = torch.cat([virtual_features[i], virtual_features[j]], dim=0)
                link_features.append(link_feature)
                link_indices.append([i, j])
        
        if link_features:
            link_features = torch.stack(link_features)  # [num_links, virtual_node_dim * 2]
            link_encoded = self.link_encoder(link_features)  # [num_links, hidden_dim]
        else:
            link_encoded = torch.empty(0, self.hidden_dim, device=virtual_features.device)
        
        # 注意力机制：考虑映射结果的影响
        if link_encoded.size(0) > 0:
            # 根据映射结果调整注意力
            mapped_physical_features = physical_encoded[mapping_result]  # [num_virtual_nodes, hidden_dim]
            
            # 计算链路对物理路径的注意力
            link_encoded_expanded = link_encoded.unsqueeze(0)
            mapped_physical_expanded = mapped_physical_features.unsqueeze(0)
            
            attended_links, link_attention_weights = self.attention(
                link_encoded_expanded,
                mapped_physical_expanded,
                mapped_physical_expanded
            )
            attended_links = attended_links.squeeze(0)  # [num_links, hidden_dim]
            
            # 合并所有特征
            combined_features = torch.cat([
                attended_links,  # 链路特征
                virtual_encoded.mean(dim=0).expand(link_encoded.size(0), -1),  # 全局虚拟特征
                physical_encoded.mean(dim=0).expand(link_encoded.size(0), -1)   # 全局物理特征
            ], dim=1)
            
            # 全局带宽分配策略
            bandwidth_logits = self.global_bandwidth_head(combined_features)  # [num_links, bandwidth_levels]
            
            # 带宽约束检查
            constraint_scores = self.bandwidth_constraint_checker(combined_features)
            constraint_scores = constraint_scores.expand(-1, self.bandwidth_levels)
            
        else:
            bandwidth_logits = torch.empty(0, self.bandwidth_levels, device=virtual_features.device)
            constraint_scores = torch.empty(0, self.bandwidth_levels, device=virtual_features.device)
            link_attention_weights = None
            link_indices = []
        
        return bandwidth_logits, constraint_scores, link_attention_weights, link_indices

class Critic(nn.Module):
    """Critic网络，评估状态价值"""
    
    def __init__(self, 
                 physical_node_dim: int,
                 virtual_node_dim: int,
                 hidden_dim: int = 128):
        super(Critic, self).__init__()
        
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 全局价值评估网络
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, 
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr):
        """
        Args:
            physical_features: 物理节点特征
            physical_edge_index: 物理网络边索引
            physical_edge_attr: 物理网络边特征
            virtual_features: 虚拟节点特征
            virtual_edge_index: 虚拟网络边索引
            virtual_edge_attr: 虚拟网络边特征
        """
        # 编码物理网络和虚拟网络
        physical_encoded = self.physical_encoder(physical_features, physical_edge_index, physical_edge_attr)
        virtual_encoded = self.virtual_encoder(virtual_features, virtual_edge_index, virtual_edge_attr)
        
        # 全局特征聚合
        global_physical = torch.mean(physical_encoded, dim=0)  # [hidden_dim]
        global_virtual = torch.mean(virtual_encoded, dim=0)    # [hidden_dim]
        
        # 合并特征
        combined_features = torch.cat([global_physical, global_virtual], dim=0)
        
        # 价值评估
        value = self.value_head(combined_features)
        
        return value

class TwoStagePPOAgent:
    """两阶段PPO智能体，使用两个独立的Actor"""
    
    def __init__(self, 
                 physical_node_dim: int,
                 virtual_node_dim: int,
                 num_physical_nodes: int,
                 max_virtual_nodes: int,
                 bandwidth_levels: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络参数
        self.num_physical_nodes = num_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
        self.bandwidth_levels = bandwidth_levels
        
        # 创建网络
        self.mapping_actor = MappingActor(
            physical_node_dim, virtual_node_dim, 
            num_physical_nodes=num_physical_nodes,
            max_virtual_nodes=max_virtual_nodes
        ).to(self.device)
        
        self.bandwidth_actor = BandwidthActor(
            physical_node_dim, virtual_node_dim,
            bandwidth_levels=bandwidth_levels,
            max_virtual_nodes=max_virtual_nodes
        ).to(self.device)
        
        self.critic = Critic(physical_node_dim, virtual_node_dim).to(self.device)
        
        # 优化器
        self.mapping_optimizer = torch.optim.Adam(self.mapping_actor.parameters(), lr=lr)
        self.bandwidth_optimizer = torch.optim.Adam(self.bandwidth_actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # PPO参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # 约束管理器
        self.constraint_manager = ConstraintManager(bandwidth_levels=bandwidth_levels)
        
        # 经验缓冲区
        self.states = []
        self.mapping_actions = []
        self.bandwidth_actions = []
        self.rewards = []
        self.values = []
        self.mapping_log_probs = []
        self.bandwidth_log_probs = []
        self.dones = []
    
    def select_actions(self, state):
        """
        选择映射和带宽分配动作
        
        Args:
            state: 环境状态字典
        
        Returns:
            mapping_action: 映射动作 [num_virtual_nodes]
            bandwidth_action: 带宽动作 [num_links]
            mapping_log_prob: 映射动作的log概率
            bandwidth_log_prob: 带宽动作的log概率
            value: 状态价值
        """
        # 提取状态信息
        physical_features = state['physical_features'].to(self.device)
        physical_edge_index = state['physical_edge_index'].to(self.device)
        physical_edge_attr = state['physical_edge_attr'].to(self.device)
        virtual_features = state['virtual_features'].to(self.device)
        virtual_edge_index = state['virtual_edge_index'].to(self.device)
        virtual_edge_attr = state['virtual_edge_attr'].to(self.device)
        
        # 生成节点映射约束
        node_constraints = self.constraint_manager.generate_node_mapping_constraints(
            physical_features, virtual_features, physical_edge_index, virtual_edge_index
        )
        
        # 第一阶段：映射Actor
        mapping_logits, constraint_scores, _ = self.mapping_actor(
            physical_features, physical_edge_index, physical_edge_attr,
            virtual_features, virtual_edge_index, virtual_edge_attr
        )
        
        # 应用约束管理器生成的约束
        mapping_logits = self.constraint_manager.apply_node_mapping_constraints(
            mapping_logits, node_constraints, temperature=1.0
        )
        
        # 应用原有的约束分数（如果有的话）
        mapping_logits = mapping_logits + torch.log(constraint_scores + 1e-8)
        
        # 检查数值稳定性
        if torch.isnan(mapping_logits).any() or torch.isinf(mapping_logits).any():
            print("警告：mapping_logits 包含 NaN 或 Inf 值，使用均匀分布")
            # 保持梯度信息，只替换无效值
            mapping_logits = torch.where(torch.isnan(mapping_logits) | torch.isinf(mapping_logits), 
                                       torch.zeros_like(mapping_logits), mapping_logits)
        
        mapping_probs = F.softmax(mapping_logits, dim=-1)
        
        # 采样映射动作
        mapping_dist = torch.distributions.Categorical(mapping_probs)
        mapping_action = mapping_dist.sample()
        mapping_log_prob = mapping_dist.log_prob(mapping_action)
        
        # 第二阶段：带宽Actor
        bandwidth_logits, bandwidth_constraint_scores, _, link_indices = self.bandwidth_actor(
            physical_features, physical_edge_index, physical_edge_attr,
            virtual_features, virtual_edge_index, virtual_edge_attr,
            mapping_action
        )
        
        if bandwidth_logits.size(0) > 0:
            # 生成带宽约束
            # 从环境状态中获取链路特定的带宽映射
            link_bandwidth_mappings = state.get('bandwidth_mapping', {})
            # 传递期望的链路数量以匹配bandwidth_logits的形状
            expected_num_links = bandwidth_logits.size(0)
            bandwidth_constraints = self.constraint_manager.generate_bandwidth_constraints(
                virtual_edge_attr, link_bandwidth_mappings, virtual_edge_index, expected_num_links
            )
            
            # 应用约束管理器生成的带宽约束
            bandwidth_logits = self.constraint_manager.apply_bandwidth_constraints(
                bandwidth_logits, bandwidth_constraints, temperature=1.0
            )
            
            # 应用原有的带宽约束分数（如果有的话）
            bandwidth_logits = bandwidth_logits + torch.log(bandwidth_constraint_scores + 1e-8)
            
            # 检查数值稳定性
            if torch.isnan(bandwidth_logits).any() or torch.isinf(bandwidth_logits).any():
                print("警告：bandwidth_logits 包含 NaN 或 Inf 值，使用均匀分布")
                # 保持梯度信息，只替换无效值
                bandwidth_logits = torch.where(torch.isnan(bandwidth_logits) | torch.isinf(bandwidth_logits), 
                                             torch.zeros_like(bandwidth_logits), bandwidth_logits)
            
            bandwidth_probs = F.softmax(bandwidth_logits, dim=-1)
            
            # 采样带宽动作
            bandwidth_dist = torch.distributions.Categorical(bandwidth_probs)
            bandwidth_action = bandwidth_dist.sample()
            bandwidth_log_prob = bandwidth_dist.log_prob(bandwidth_action)
        else:
            bandwidth_action = torch.empty(0, dtype=torch.long, device=self.device)
            bandwidth_log_prob = torch.empty(0, dtype=torch.float, device=self.device)
        print(f"TwoStagePPOAgent select_actions mapping_action: {mapping_action}")
        print(f"TwoStagePPOAgent select_actions bandwidth_action: {bandwidth_action}")
        # Critic评估
        value = self.critic(
            physical_features, physical_edge_index, physical_edge_attr,
            virtual_features, virtual_edge_index, virtual_edge_attr
        )
        
        return (mapping_action.cpu().detach().numpy(), bandwidth_action.cpu().detach().numpy(),
                mapping_log_prob.cpu().detach().numpy(), bandwidth_log_prob.cpu().detach().numpy(),
                value.cpu().item(), link_indices)
    
    def store_transition(self, state, mapping_action, bandwidth_action, 
                        reward, value, mapping_log_prob, bandwidth_log_prob, done):
        """存储经验"""
        self.states.append(state)
        self.mapping_actions.append(mapping_action)
        self.bandwidth_actions.append(bandwidth_action)
        self.rewards.append(reward)
        self.values.append(value)
        self.mapping_log_probs.append(mapping_log_prob)
        self.bandwidth_log_probs.append(bandwidth_log_prob)
        self.dones.append(done)
    
    def update(self):
        """更新网络"""
        if len(self.states) < 2:
            return
        
        # 计算优势函数
        advantages = self._compute_advantages()
        
        # 转换为tensor
        states = self.states
        # 处理不同长度的mapping_actions
        mapping_actions = []
        for actions in self.mapping_actions:
            if isinstance(actions, np.ndarray):
                mapping_actions.append(torch.tensor(actions, dtype=torch.long, device=self.device))
            else:
                mapping_actions.append(torch.tensor(np.array(actions), dtype=torch.long, device=self.device))
        
        bandwidth_actions = [torch.tensor(actions, dtype=torch.long, device=self.device) for actions in self.bandwidth_actions]
        old_mapping_log_probs = [torch.tensor(probs, dtype=torch.float32, device=self.device) for probs in self.mapping_log_probs]
        old_bandwidth_log_probs = [torch.tensor(probs, dtype=torch.float32, device=self.device) for probs in self.bandwidth_log_probs]
        returns = advantages + torch.tensor(self.values, dtype=torch.float32, device=self.device)
        
        # 更新映射Actor
        self._update_mapping_actor(states, mapping_actions, old_mapping_log_probs, advantages, returns)
        
        # 更新带宽Actor - 传递映射动作
        self._update_bandwidth_actor(states, bandwidth_actions, old_bandwidth_log_probs, advantages, returns, mapping_actions)
        
        # 更新Critic
        self._update_critic(states, returns)
        
        # 清空缓冲区
        self.states.clear()
        self.mapping_actions.clear()
        self.bandwidth_actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.mapping_log_probs.clear()
        self.bandwidth_log_probs.clear()
        self.dones.clear()
    
    def _update_mapping_actor(self, states, mapping_actions, old_log_probs, advantages, returns):
        """更新映射Actor"""
        # 重新计算当前策略的概率
        current_log_probs = []
        entropies = []
        
        for i, state in enumerate(states):
            physical_features = state['physical_features'].to(self.device)
            physical_edge_index = state['physical_edge_index'].to(self.device)
            physical_edge_attr = state['physical_edge_attr'].to(self.device)
            virtual_features = state['virtual_features'].to(self.device)
            virtual_edge_index = state['virtual_edge_index'].to(self.device)
            virtual_edge_attr = state['virtual_edge_attr'].to(self.device)
            
            # 生成节点映射约束
            node_constraints = self.constraint_manager.generate_node_mapping_constraints(
                physical_features, virtual_features, physical_edge_index, virtual_edge_index
            )
            
            mapping_logits, constraint_scores, _ = self.mapping_actor(
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr
            )
            
            # 应用约束管理器生成的约束
            mapping_logits = self.constraint_manager.apply_node_mapping_constraints(
                mapping_logits, node_constraints, temperature=1.0
            )
            
            # 应用原有的约束分数（如果有的话）
            mapping_logits = mapping_logits + torch.log(constraint_scores + 1e-8)
            
            # 检查数值稳定性
            if torch.isnan(mapping_logits).any() or torch.isinf(mapping_logits).any():
                print("警告：更新阶段 mapping_logits 包含 NaN 或 Inf 值，使用均匀分布")
                # 保持梯度信息，只替换无效值
                mapping_logits = torch.where(torch.isnan(mapping_logits) | torch.isinf(mapping_logits), 
                                           torch.zeros_like(mapping_logits), mapping_logits)
            
            mapping_probs = F.softmax(mapping_logits, dim=-1)
            
            mapping_dist = torch.distributions.Categorical(mapping_probs)
            current_log_prob = mapping_dist.log_prob(mapping_actions[i])
            entropy = mapping_dist.entropy().mean()
            
            current_log_probs.append(current_log_prob)
            entropies.append(entropy)
        
        if current_log_probs:
            # 处理不同长度的log概率
            all_current_log_probs = []
            all_old_log_probs = []
            all_advantages = []
            all_entropies = []
            
            for i, (current_log_prob, old_log_prob, entropy) in enumerate(zip(current_log_probs, old_log_probs, entropies)):
                all_current_log_probs.append(current_log_prob)
                all_old_log_probs.append(old_log_prob)
                all_entropies.append(entropy)
                # 为每个虚拟节点分配相同的优势值
                all_advantages.extend([advantages[i]] * len(current_log_prob))
            
            # 连接所有log概率
            current_log_probs = torch.cat(all_current_log_probs)
            old_log_probs = torch.cat(all_old_log_probs)
            all_advantages = torch.tensor(all_advantages, device=self.device)
            entropies = torch.stack(all_entropies)
            
            # 计算比率
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * all_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * all_advantages
            mapping_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失
            entropy_loss = -entropies.mean()
            
            # 总损失
            total_loss = mapping_loss + self.entropy_coef * entropy_loss
            
            # 更新
            self.mapping_optimizer.zero_grad()
            total_loss.backward()
            self.mapping_optimizer.step()
    
    def _update_bandwidth_actor(self, states, bandwidth_actions, old_log_probs, advantages, returns, mapping_actions=None):
        """更新带宽Actor"""
        # 重新计算当前策略的概率
        current_log_probs = []
        entropies = []
        
        for i, state in enumerate(states):
            physical_features = state['physical_features'].to(self.device)
            physical_edge_index = state['physical_edge_index'].to(self.device)
            physical_edge_attr = state['physical_edge_attr'].to(self.device)
            virtual_features = state['virtual_features'].to(self.device)
            virtual_edge_index = state['virtual_edge_index'].to(self.device)
            virtual_edge_attr = state['virtual_edge_attr'].to(self.device)
            
            # 使用实际的映射动作，如果没有提供则使用随机动作
            if mapping_actions is not None and i < len(mapping_actions):
                mapping_action = torch.tensor(mapping_actions[i], dtype=torch.long, device=self.device)
            # else:
            #     # 如果没有映射动作，使用随机动作作为后备
            #     mapping_action = torch.randint(0, self.num_physical_nodes, (self.max_virtual_nodes,), device=self.device)
            
            print(f"update bandwidth_actor mapping_action: {mapping_action}")
            bandwidth_logits, bandwidth_constraint_scores, _, _ = self.bandwidth_actor(
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                mapping_action
            )
            
            if bandwidth_logits.size(0) > 0:
                # 生成带宽约束
                link_bandwidth_mappings = state.get('bandwidth_mapping', {})
                expected_num_links = bandwidth_logits.size(0)
                bandwidth_constraints = self.constraint_manager.generate_bandwidth_constraints(
                    virtual_edge_attr, link_bandwidth_mappings, virtual_edge_index, expected_num_links
                )
                
                # 应用约束管理器生成的带宽约束
                bandwidth_logits = self.constraint_manager.apply_bandwidth_constraints(
                    bandwidth_logits, bandwidth_constraints, temperature=1.0
                )
                
                # 应用原有的带宽约束分数（如果有的话）
                bandwidth_logits = bandwidth_logits + torch.log(bandwidth_constraint_scores + 1e-8)
                
                # 检查数值稳定性
                if torch.isnan(bandwidth_logits).any() or torch.isinf(bandwidth_logits).any():
                    print("警告：更新阶段 bandwidth_logits 包含 NaN 或 Inf 值，使用均匀分布")
                    # 保持梯度信息，只替换无效值
                    bandwidth_logits = torch.where(torch.isnan(bandwidth_logits) | torch.isinf(bandwidth_logits), 
                                                 torch.zeros_like(bandwidth_logits), bandwidth_logits)
                
                bandwidth_probs = F.softmax(bandwidth_logits, dim=-1)
                
                bandwidth_dist = torch.distributions.Categorical(bandwidth_probs)
                current_log_prob = bandwidth_dist.log_prob(bandwidth_actions[i])
                entropy = bandwidth_dist.entropy().mean()
                
                current_log_probs.append(current_log_prob)
                entropies.append(entropy)
        
        if current_log_probs:
            # 处理不同长度的log概率
            all_current_log_probs = []
            all_old_log_probs = []
            all_advantages = []
            all_entropies = []
            
            for i, (current_log_prob, old_log_prob, entropy) in enumerate(zip(current_log_probs, old_log_probs, entropies)):
                all_current_log_probs.append(current_log_prob)
                all_old_log_probs.append(old_log_prob)
                all_entropies.append(entropy)
                # 为每个链路分配相同的优势值
                all_advantages.extend([advantages[i]] * len(current_log_prob))
            
            # 连接所有log概率
            current_log_probs = torch.cat(all_current_log_probs)
            old_log_probs = torch.cat(all_old_log_probs)
            all_advantages = torch.tensor(all_advantages, device=self.device)
            entropies = torch.stack(all_entropies)
            
            # 计算比率
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * all_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * all_advantages
            bandwidth_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失
            entropy_loss = -entropies.mean()
            
            # 总损失
            total_loss = bandwidth_loss + self.entropy_coef * entropy_loss
            
            # 更新
            self.bandwidth_optimizer.zero_grad()
            total_loss.backward()
            self.bandwidth_optimizer.step()
    
    def _update_critic(self, states, returns):
        """更新Critic"""
        values = []
        
        for state in states:
            physical_features = state['physical_features'].to(self.device)
            physical_edge_index = state['physical_edge_index'].to(self.device)
            physical_edge_attr = state['physical_edge_attr'].to(self.device)
            virtual_features = state['virtual_features'].to(self.device)
            virtual_edge_index = state['virtual_edge_index'].to(self.device)
            virtual_edge_attr = state['virtual_edge_attr'].to(self.device)
            
            value = self.critic(
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr
            )
            values.append(value)
        
        values = torch.cat(values)
        returns = returns[:len(values)]
        
        # 价值损失
        value_loss = F.mse_loss(values, returns)
        
        # 更新
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
    
    def _compute_advantages(self):
        """计算优势函数"""
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages