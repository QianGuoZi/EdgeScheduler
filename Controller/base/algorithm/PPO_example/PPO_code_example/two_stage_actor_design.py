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
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, MultiStepReplayBuffer

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
                 max_physical_nodes: int = 20,  # 改为最大物理节点数
                 max_virtual_nodes: int = 8):
        super(MappingActor, self).__init__()
        
        self.max_physical_nodes = max_physical_nodes  # 最大物理节点数
        self.max_virtual_nodes = max_virtual_nodes
        self.hidden_dim = hidden_dim
        
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 注意力机制，用于计算物理节点和虚拟节点的匹配度
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 全局映射策略网络 - 输出所有虚拟节点的映射
        # 使用最大物理节点数作为输出维度，运行时动态调整
        self.global_mapping_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_physical_nodes)  # 使用最大物理节点数
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
        
        # 新增：GCN特征投影层
        self.physical_gcn_proj = None  # 将在第一次使用时动态创建
        self.virtual_gcn_proj = None   # 将在第一次使用时动态创建

        self.device_synced = False
        
    def forward(self, 
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                physical_gcn_features=None, virtual_gcn_features=None):
        """
        Args:
            physical_features: 物理节点特征 [num_physical_nodes, physical_node_dim]
            physical_edge_index: 物理网络边索引 [2, num_physical_edges]
            physical_edge_attr: 物理网络边特征 [num_physical_edges, edge_features]
            virtual_features: 虚拟节点特征 [num_virtual_nodes, virtual_node_dim]
            virtual_edge_index: 虚拟网络边索引 [2, num_virtual_edges]
            virtual_edge_attr: 虚拟网络边特征 [num_virtual_edges, edge_features]
            physical_gcn_features: 物理网络GCN特征 [num_physical_nodes, gcn_dim] (可选)
            virtual_gcn_features: 虚拟网络GCN特征 [num_virtual_nodes, gcn_dim] (可选)
        
        Returns:
            mapping_logits: 所有虚拟节点的映射logits [num_virtual_nodes, num_physical_nodes]
            constraint_scores: 约束满足度分数 [num_virtual_nodes, num_physical_nodes]
        """
        # 确保所有模块都在正确的设备上
        target_device = physical_features.device
        self.to(target_device)
        
        # 编码物理网络和虚拟网络
        physical_encoded = self.physical_encoder(physical_features, physical_edge_index, physical_edge_attr)
        virtual_encoded = self.virtual_encoder(virtual_features, virtual_edge_index, virtual_edge_attr)
        
        # 新增：如果提供了GCN特征，将其与编码特征融合
        if physical_gcn_features is not None:
            # 确保GCN特征在正确的设备上
            if physical_gcn_features.device != physical_encoded.device:
                if not self.device_synced:
                    print(f"🔧 同步物理GCN特征到设备: {physical_encoded.device}")
                physical_gcn_features = physical_gcn_features.to(physical_encoded.device)
            
            # 确保GCN特征维度与编码特征维度匹配
            if physical_gcn_features.size(1) != physical_encoded.size(1):
                # 动态创建投影层
                if self.physical_gcn_proj is None:
                    self.physical_gcn_proj = nn.Linear(physical_gcn_features.size(1), physical_encoded.size(1)).to(physical_encoded.device)
                    # 将投影层注册为模块参数
                    self.add_module('physical_gcn_proj', self.physical_gcn_proj)
                else:
                    # 确保现有投影层在正确的设备上
                    self.physical_gcn_proj = self.physical_gcn_proj.to(physical_encoded.device)
                
                # 确保投影层权重在正确的设备上
                if self.physical_gcn_proj.weight.device != physical_gcn_features.device:
                    print(f"🔧 同步物理GCN投影层权重到设备: {physical_gcn_features.device}")
                    self.physical_gcn_proj = self.physical_gcn_proj.to(physical_gcn_features.device)
                
                physical_gcn_features = self.physical_gcn_proj(physical_gcn_features)
            # 融合特征：原始编码 + GCN特征
            physical_encoded = physical_encoded + 0.5 * physical_gcn_features
        
        if virtual_gcn_features is not None:
            # 确保GCN特征在正确的设备上
            if virtual_gcn_features.device != virtual_encoded.device:
                # print(f"🔧 同步虚拟GCN特征到设备: {virtual_encoded.device}")
                virtual_gcn_features = virtual_gcn_features.to(virtual_encoded.device)
            
            # 确保GCN特征维度与编码特征维度匹配
            if virtual_gcn_features.size(1) != virtual_encoded.size(1):
                # 动态创建投影层
                if self.virtual_gcn_proj is None:
                    self.virtual_gcn_proj = nn.Linear(virtual_gcn_features.size(1), virtual_encoded.size(1)).to(virtual_encoded.device)
                    # 将投影层注册为模块参数
                    self.add_module('virtual_gcn_proj', self.virtual_gcn_proj)
                else:
                    # 确保现有投影层在正确的设备上
                    self.virtual_gcn_proj = self.virtual_gcn_proj.to(virtual_encoded.device)
                
                # 确保投影层权重在正确的设备上
                if self.virtual_gcn_proj.weight.device != virtual_gcn_features.device:
                    if not self.device_synced:
                        print(f"🔧 同步虚拟GCN投影层权重到设备: {virtual_gcn_features.device}")
                    self.virtual_gcn_proj = self.virtual_gcn_proj.to(virtual_gcn_features.device)
                
                virtual_gcn_features = self.virtual_gcn_proj(virtual_gcn_features)
            # 融合特征：原始编码 + GCN特征
            virtual_encoded = virtual_encoded + 0.5 * virtual_gcn_features
        
        self.device_synced = True
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
        
        # 获取实际的虚拟节点数量和物理节点数量
        actual_virtual_nodes = virtual_features.size(0)
        actual_num_physical_nodes = physical_features.size(0)
        
        # 全局映射策略：为每个虚拟节点输出映射logits
        mapping_logits = self.global_mapping_head(combined_features)  # [actual_virtual_nodes, max_physical_nodes]
        
        # 🚀 动态调整：只保留实际物理节点数量的输出
        if mapping_logits.size(1) > actual_num_physical_nodes:
            mapping_logits = mapping_logits[:, :actual_num_physical_nodes]
        elif mapping_logits.size(1) < actual_num_physical_nodes:
            # 如果输出维度小于实际物理节点数，需要扩展
            # 这种情况通常不会发生，因为max_physical_nodes应该足够大
            print(f"警告：实际物理节点数({actual_num_physical_nodes})超过了最大物理节点数({self.max_physical_nodes})")
            # 使用零填充扩展
            padding = torch.zeros(actual_virtual_nodes, actual_num_physical_nodes - mapping_logits.size(1), 
                                device=mapping_logits.device)
            mapping_logits = torch.cat([mapping_logits, padding], dim=1)
        
        # 🚀 添加负载均衡约束：防止过度集中到同一物理节点
        # mapping_logits = self._apply_load_balancing_constraints(mapping_logits, physical_features, virtual_features)
        
        # 约束检查：计算每个虚拟节点的约束满足度
        constraint_scores = self.constraint_checker(combined_features)
        # 为每个虚拟节点对每个物理节点计算约束分数
        constraint_scores = constraint_scores.squeeze(-1).unsqueeze(1).expand(-1, actual_num_physical_nodes)
        
        return mapping_logits, constraint_scores, attention_weights
    
    def _apply_load_balancing_constraints(self, mapping_logits, physical_features, virtual_features):
        """
        应用负载均衡约束，防止过度集中到同一物理节点
        
        Args:
            mapping_logits: 原始映射logits [num_virtual_nodes, num_physical_nodes]
            physical_features: 物理节点特征
            virtual_features: 虚拟节点特征
            
        Returns:
            constrained_logits: 应用约束后的logits
        """
        constrained_logits = mapping_logits.clone()
        
        # 计算每个物理节点的当前负载
        actual_num_physical_nodes = physical_features.size(0)
        current_loads = torch.zeros(actual_num_physical_nodes, device=mapping_logits.device)
        
        # 从物理特征中提取当前使用率
        for i in range(actual_num_physical_nodes):
            cpu_usage = physical_features[i, 2]  # CPU使用率
            memory_usage = physical_features[i, 3]  # 内存使用率
            current_loads[i] = (cpu_usage + memory_usage) / 2
        
        # 计算虚拟节点的资源需求
        virtual_cpu_demands = virtual_features[:, 0]  # CPU需求
        virtual_memory_demands = virtual_features[:, 1]  # 内存需求
        
        # 为每个虚拟节点应用负载均衡约束
        for virtual_node in range(mapping_logits.size(0)):
            # 计算如果映射到每个物理节点后的负载
            for physical_node in range(actual_num_physical_nodes):
                # 估算映射后的负载
                estimated_cpu_load = current_loads[physical_node] + virtual_cpu_demands[virtual_node] / physical_features[physical_node, 0]
                estimated_memory_load = current_loads[physical_node] + virtual_memory_demands[virtual_node] / physical_features[physical_node, 1]
                estimated_total_load = (estimated_cpu_load + estimated_memory_load) / 2
                
                # 如果负载过高，降低该物理节点的logits
                if estimated_total_load > 0.7:  # 70%阈值
                    penalty = (estimated_total_load - 0.7) * 5.0  # 惩罚强度
                    constrained_logits[virtual_node, physical_node] -= penalty
                
                # 如果负载过高（超过80%），给予更强惩罚
                if estimated_total_load > 0.8:
                    penalty = (estimated_total_load - 0.8) * 10.0  # 强惩罚
                    constrained_logits[virtual_node, physical_node] -= penalty
        
        return constrained_logits

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
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 物理 + 虚拟 + 链路 + 同一节点映射特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, bandwidth_levels)  # 每个链路输出一个带宽等级选择
        )
        
        # 带宽约束检查层
        self.bandwidth_constraint_checker = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # 更新输入维度
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 新增：GCN特征投影层
        self.physical_gcn_proj = None  # 将在第一次使用时动态创建
        self.virtual_gcn_proj = None   # 将在第一次使用时动态创建

        self.device_synced = False
        
    def forward(self, 
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                mapping_result,
                physical_gcn_features=None, virtual_gcn_features=None):
        """
        Args:
            physical_features: 物理节点特征
            physical_edge_index: 物理网络边索引
            physical_edge_attr: 物理网络边特征
            virtual_features: 虚拟节点特征
            virtual_edge_index: 虚拟网络边索引 [2, num_virtual_edges]
            virtual_edge_attr: 虚拟网络边特征 [num_virtual_edges, edge_features]
            mapping_result: 映射结果 [num_virtual_nodes] (物理节点索引)
            physical_gcn_features: 物理网络GCN特征 [num_physical_nodes, gcn_dim] (可选)
            virtual_gcn_features: 虚拟网络GCN特征 [num_virtual_nodes, gcn_dim] (可选)
        
        Returns:
            bandwidth_logits: 虚拟链路的带宽logits [num_virtual_edges, bandwidth_levels]
            constraint_scores: 带宽约束满足度分数 [num_virtual_edges, bandwidth_levels]
            link_attention_weights: 链路注意力权重
            link_indices: 链路索引列表 [[src, dst], ...] (基于实际virtual_edge_index)
        """
        # 确保所有模块都在正确的设备上
        target_device = physical_features.device
        self.to(target_device)
        
        # 编码物理网络和虚拟网络
        physical_encoded = self.physical_encoder(physical_features, physical_edge_index, physical_edge_attr)
        virtual_encoded = self.virtual_encoder(virtual_features, virtual_edge_index, virtual_edge_attr)
        
        # 新增：如果提供了GCN特征，将其与编码特征融合
        if physical_gcn_features is not None:
            # 确保GCN特征在正确的设备上
            if physical_gcn_features.device != physical_encoded.device:
                if not self.device_synced:
                    print(f"🔧 同步物理GCN特征到设备: {physical_encoded.device}")
                physical_gcn_features = physical_gcn_features.to(physical_encoded.device)
            
            # 确保GCN特征维度与编码特征维度匹配
            if physical_gcn_features.size(1) != physical_encoded.size(1):
                # 动态创建投影层
                if self.physical_gcn_proj is None:
                    self.physical_gcn_proj = nn.Linear(physical_gcn_features.size(1), physical_encoded.size(1)).to(physical_encoded.device)
                    # 将投影层注册为模块参数
                    self.add_module('physical_gcn_proj', self.physical_gcn_proj)
                else:
                    # 确保现有投影层在正确的设备上
                    self.physical_gcn_proj = self.physical_gcn_proj.to(physical_encoded.device)
                
                # 确保投影层权重在正确的设备上
                if self.physical_gcn_proj.weight.device != physical_gcn_features.device:
                    print(f"🔧 同步物理GCN投影层权重到设备: {physical_gcn_features.device}")
                    self.physical_gcn_proj = self.physical_gcn_proj.to(physical_gcn_features.device)
                
                physical_gcn_features = self.physical_gcn_proj(physical_gcn_features)
            # 融合特征：原始编码 + GCN特征
            physical_encoded = physical_encoded + 0.5 * physical_gcn_features
        
        if virtual_gcn_features is not None:
            # 确保GCN特征在正确的设备上
            if virtual_gcn_features.device != virtual_encoded.device:
                if not self.device_synced:
                    print(f"🔧 同步虚拟GCN特征到设备: {virtual_encoded.device}")
                virtual_gcn_features = virtual_gcn_features.to(virtual_encoded.device)
            
            # 确保GCN特征维度与编码特征维度匹配
            if virtual_gcn_features.size(1) != virtual_encoded.size(1):
                # 动态创建投影层
                if self.virtual_gcn_proj is None:
                    self.virtual_gcn_proj = nn.Linear(virtual_gcn_features.size(1), virtual_encoded.size(1)).to(virtual_encoded.device)
                    # 将投影层注册为模块参数
                    self.add_module('virtual_gcn_proj', self.virtual_gcn_proj)
                else:
                    # 确保现有投影层在正确的设备上
                    self.virtual_gcn_proj = self.virtual_gcn_proj.to(virtual_encoded.device)
                
                # 确保投影层权重在正确的设备上
                if self.virtual_gcn_proj.weight.device != virtual_gcn_features.device:
                    if not self.device_synced:
                        print(f"🔧 同步虚拟GCN投影层权重到设备: {virtual_gcn_features.device}")
                    self.virtual_gcn_proj = self.virtual_gcn_proj.to(virtual_gcn_features.device)
                
                virtual_gcn_features = self.virtual_gcn_proj(virtual_gcn_features)
            # 融合特征：原始编码 + GCN特征
            virtual_encoded = virtual_encoded + 0.5 * virtual_gcn_features
        
        self.device_synced = True
        
        # 基于实际的虚拟边构建链路特征
        num_virtual_edges = virtual_edge_index.size(1)
        link_features = []
        link_indices = []
        
        if num_virtual_edges > 0:
            # 从virtual_edge_index中提取实际的链路
            for i in range(num_virtual_edges):
                src = virtual_edge_index[0, i].item()
                dst = virtual_edge_index[1, i].item()
                
                # 合并两个虚拟节点的特征
                link_feature = torch.cat([virtual_features[src], virtual_features[dst]], dim=0)
                link_features.append(link_feature)
                link_indices.append([src, dst])
            
            link_features = torch.stack(link_features)  # [num_virtual_edges, virtual_node_dim * 2]
            link_encoded = self.link_encoder(link_features)  # [num_virtual_edges, hidden_dim]
            
            # 注意力机制：考虑映射结果的影响
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
            attended_links = attended_links.squeeze(0)  # [num_virtual_edges, hidden_dim]
            
            # 🚀 优化：为同一物理节点映射的链路添加特殊特征
            same_node_features = self._compute_same_node_mapping_features(
                link_indices, mapping_result, virtual_encoded, physical_encoded
            )
            
            # 合并所有特征
            combined_features = torch.cat([
                attended_links,  # 链路特征
                virtual_encoded.mean(dim=0).expand(link_encoded.size(0), -1),  # 全局虚拟特征
                physical_encoded.mean(dim=0).expand(link_encoded.size(0), -1),  # 全局物理特征
                same_node_features  # 同一节点映射特征
            ], dim=1)
            
            # 全局带宽分配策略
            bandwidth_logits = self.global_bandwidth_head(combined_features)  # [num_virtual_edges, bandwidth_levels]
            
            # 🚀 优化：为同一物理节点映射的链路增强最大带宽级别的logits
            bandwidth_logits = self._enhance_same_node_bandwidth_logits(
                bandwidth_logits, link_indices, mapping_result
            )
            
            # 带宽约束检查
            constraint_scores = self.bandwidth_constraint_checker(combined_features)
            # 🔧 修复：使用实际的带宽等级数量而不是固定的self.bandwidth_levels
            actual_bandwidth_levels = bandwidth_logits.size(1)
            constraint_scores = constraint_scores.expand(-1, actual_bandwidth_levels)
            
        else:
            # 🔧 修复：使用实际的带宽等级数量而不是固定的self.bandwidth_levels
            actual_bandwidth_levels = self.bandwidth_levels  # 这里可以使用固定值，因为没有实际的logits
            bandwidth_logits = torch.empty(0, actual_bandwidth_levels, device=virtual_features.device)
            constraint_scores = torch.empty(0, actual_bandwidth_levels, device=virtual_features.device)
            link_attention_weights = None
            link_indices = []
        
        return bandwidth_logits, constraint_scores, link_attention_weights, link_indices
    
    def _compute_same_node_mapping_features(self, link_indices, mapping_result, virtual_encoded, physical_encoded):
        """
        计算同一物理节点映射的特征
        
        Args:
            link_indices: 链路索引列表 [[src, dst], ...]
            mapping_result: 映射结果 [num_virtual_nodes]
            virtual_encoded: 虚拟节点编码 [num_virtual_nodes, hidden_dim]
            physical_encoded: 物理节点编码 [num_physical_nodes, hidden_dim]
            
        Returns:
            same_node_features: 同一节点映射特征 [num_links, hidden_dim]
        """
        same_node_features = torch.zeros(len(link_indices), self.hidden_dim, device=virtual_encoded.device)
        
        for i, (src, dst) in enumerate(link_indices):
            src_physical = mapping_result[src]
            dst_physical = mapping_result[dst]
            
            if src_physical == dst_physical:
                # 同一物理节点映射：使用物理节点的编码特征
                same_node_features[i] = physical_encoded[src_physical]
            else:
                # 不同物理节点映射：使用两个物理节点的平均编码特征
                same_node_features[i] = (physical_encoded[src_physical] + physical_encoded[dst_physical]) / 2
        
        return same_node_features
    
    def _enhance_same_node_bandwidth_logits(self, bandwidth_logits, link_indices, mapping_result):
        """
        为同一物理节点映射的链路增强最大带宽级别的logits
        
        Args:
            bandwidth_logits: 带宽logits [num_links, bandwidth_levels]
            link_indices: 链路索引列表 [[src, dst], ...]
            mapping_result: 映射结果 [num_virtual_nodes]
            
        Returns:
            enhanced_logits: 增强后的带宽logits
        """
        enhanced_logits = bandwidth_logits.clone()
        
        for i, (src, dst) in enumerate(link_indices):
            src_physical = mapping_result[src]
            dst_physical = mapping_result[dst]
            
            if src_physical == dst_physical:
                # 同一物理节点映射：增强最大带宽级别的logits
                max_bandwidth_level = self.bandwidth_levels - 1
                enhancement_factor = 2.0  # 增强因子
                enhanced_logits[i, max_bandwidth_level] += enhancement_factor
                
                # 可选：稍微降低其他级别的logits
                other_levels = torch.arange(self.bandwidth_levels, device=bandwidth_logits.device) != max_bandwidth_level
                enhanced_logits[i, other_levels] -= 0.5
        
        return enhanced_logits

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
        
        # 新增：GCN特征投影层
        self.physical_gcn_proj = None  # 将在第一次使用时动态创建
        self.virtual_gcn_proj = None   # 将在第一次使用时动态创建
        
        # 新增：设备同步标志
        self.device_synced = False
    
    def forward(self, 
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                physical_gcn_features=None, virtual_gcn_features=None):
        """
        Args:
            physical_features: 物理节点特征
            physical_edge_index: 物理网络边索引
            physical_edge_attr: 物理网络边特征
            virtual_features: 虚拟节点特征
            virtual_edge_index: 虚拟网络边索引
            virtual_edge_attr: 虚拟网络边特征
            physical_gcn_features: 物理网络GCN特征 [num_physical_nodes, gcn_dim] (可选)
            virtual_gcn_features: 虚拟网络GCN特征 [num_virtual_nodes, gcn_dim] (可选)
        """
        # 确保所有模块都在正确的设备上
        target_device = physical_features.device
        self.to(target_device)
        
        # 编码物理网络和虚拟网络
        physical_encoded = self.physical_encoder(physical_features, physical_edge_index, physical_edge_attr)
        virtual_encoded = self.virtual_encoder(virtual_features, virtual_edge_index, virtual_edge_attr)
        
        # 新增：如果提供了GCN特征，将其与编码特征融合
        if physical_gcn_features is not None:
            # 确保GCN特征在正确的设备上
            if physical_gcn_features.device != physical_encoded.device:
                if not self.device_synced:
                    print(f"🔧 同步物理GCN特征到设备: {physical_encoded.device}")
                physical_gcn_features = physical_gcn_features.to(physical_encoded.device)
            
            # 确保GCN特征维度与编码特征维度匹配
            if physical_gcn_features.size(1) != physical_encoded.size(1):
                # 动态创建投影层
                if self.physical_gcn_proj is None:
                    self.physical_gcn_proj = nn.Linear(physical_gcn_features.size(1), physical_encoded.size(1)).to(physical_encoded.device)
                    # 将投影层注册为模块参数
                    self.add_module('physical_gcn_proj', self.physical_gcn_proj)
                else:
                    # 确保现有投影层在正确的设备上
                    self.physical_gcn_proj = self.physical_gcn_proj.to(physical_encoded.device)
                
                # 确保投影层权重在正确的设备上
                if self.physical_gcn_proj.weight.device != physical_gcn_features.device:
                    if not self.device_synced:
                        print(f"🔧 同步物理GCN投影层权重到设备: {physical_gcn_features.device}")
                    self.physical_gcn_proj = self.physical_gcn_proj.to(physical_gcn_features.device)
                
                physical_gcn_features = self.physical_gcn_proj(physical_gcn_features)
                
                # 确保投影后的特征也在正确的设备上
                if physical_gcn_features.device != physical_encoded.device:
                    if not self.device_synced:
                        print(f"🔧 同步投影后的物理GCN特征到设备: {physical_encoded.device}")
                    physical_gcn_features = physical_gcn_features.to(physical_encoded.device)
            
            # 融合特征：原始编码 + GCN特征
            # 最终设备检查，确保所有特征都在同一设备上
            if physical_gcn_features.device != physical_encoded.device:
                print(f"⚠️ 最终检查：物理GCN特征设备 {physical_gcn_features.device} 与编码特征设备 {physical_encoded.device} 不匹配")
                physical_gcn_features = physical_gcn_features.to(physical_encoded.device)
            physical_encoded = physical_encoded + 0.5 * physical_gcn_features
        
        if virtual_gcn_features is not None:
            # 确保GCN特征维度与编码特征维度匹配
            if virtual_gcn_features.size(1) != virtual_encoded.size(1):
                # 动态创建投影层
                if self.virtual_gcn_proj is None:
                    self.virtual_gcn_proj = nn.Linear(virtual_gcn_features.size(1), virtual_encoded.size(1)).to(virtual_encoded.device)
                    # 将投影层注册为模块参数
                    self.add_module('virtual_gcn_proj', self.virtual_gcn_proj)
                else:
                    # 确保现有投影层在正确的设备上
                    self.virtual_gcn_proj = self.virtual_gcn_proj.to(virtual_encoded.device)
                
                # 确保投影层权重在正确的设备上
                if self.virtual_gcn_proj.weight.device != virtual_gcn_features.device:
                    if not self.device_synced:
                        print(f"🔧 同步虚拟GCN投影层权重到设备: {virtual_gcn_features.device}")
                    self.virtual_gcn_proj = self.virtual_gcn_proj.to(virtual_gcn_features.device)
                
                virtual_gcn_features = self.virtual_gcn_proj(virtual_gcn_features)
                
                # 确保投影后的特征也在正确的设备上
                if virtual_gcn_features.device != virtual_encoded.device:
                    if not self.device_synced:
                        print(f"🔧 同步投影后的虚拟GCN特征到设备: {virtual_encoded.device}")
                    virtual_gcn_features = virtual_gcn_features.to(virtual_encoded.device)
            
            # 融合特征：原始编码 + GCN特征
            # 最终设备检查，确保所有特征都在同一设备上
            if virtual_gcn_features.device != virtual_encoded.device:
                print(f"⚠️ 最终检查：虚拟GCN特征设备 {virtual_gcn_features.device} 与编码特征设备 {virtual_encoded.device} 不匹配")
                virtual_gcn_features = virtual_gcn_features.to(virtual_encoded.device)
            virtual_encoded = virtual_encoded + 0.5 * virtual_gcn_features
        
        # 标记设备已同步
        self.device_synced = True
        
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
                 max_physical_nodes: int,  # 改为最大物理节点数
                 max_virtual_nodes: int,
                 bandwidth_levels: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 # 经验回放参数
                 replay_buffer_type: str = "prioritized",  # "simple", "prioritized", "multistep"
                 replay_buffer_size: int = 10000,
                 batch_size: int = 64,
                 update_frequency: int = 4,  # 每多少个episode更新一次
                 priority_alpha: float = 0.6,
                 priority_beta: float = 0.4,
                 n_steps: int = 3,
                 n_epochs: int = 4,  # 添加n_epochs参数
                 # 新增：梯度裁剪参数
                 max_grad_norm: float = 2.0,  # 最大梯度范数
                 task_gradient_clip: float = 1.0,  # 任务梯度裁剪阈值
                 head_gradient_clip: float = 0.8):  # 注意力头梯度裁剪阈值
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络参数
        self.max_physical_nodes = max_physical_nodes  # 最大物理节点数
        self.max_virtual_nodes = max_virtual_nodes
        self.bandwidth_levels = bandwidth_levels
        
        # 新增：梯度裁剪参数
        self.max_grad_norm = max_grad_norm
        self.task_gradient_clip = task_gradient_clip
        self.head_gradient_clip = head_gradient_clip
        
        # 新增：梯度统计
        self.gradient_stats = {
            'mapping_actor': {'total_norms': [], 'clipped_count': 0, 'head_clipped_count': 0, 'task_clipped_count': 0},
            'bandwidth_actor': {'total_norms': [], 'clipped_count': 0, 'head_clipped_count': 0, 'task_clipped_count': 0},
            'critic': {'total_norms': [], 'clipped_count': 0, 'head_clipped_count': 0, 'task_clipped_count': 0}
        }
        
        # 创建网络
        self.mapping_actor = MappingActor(
            physical_node_dim, virtual_node_dim, 
            max_physical_nodes=max_physical_nodes,  # 使用最大物理节点数
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
        
        # 经验回放参数
        self.replay_buffer_type = replay_buffer_type
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.episode_count = 0
        
        # 添加n_epochs参数
        self.n_epochs = n_epochs
        
        # 创建经验回放缓冲区
        if replay_buffer_type == "prioritized":
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=replay_buffer_size,
                alpha=priority_alpha,
                beta=priority_beta
            )
        elif replay_buffer_type == "multistep":
            self.replay_buffer = MultiStepReplayBuffer(
                capacity=replay_buffer_size,
                n_steps=n_steps,
                gamma=gamma
            )
        else:  # simple
            self.replay_buffer = ReplayBuffer(
                capacity=replay_buffer_size,
                use_priority=False
            )
        
        # 保持向后兼容的旧缓冲区（用于在线学习）
        self.states = []
        self.mapping_actions = []
        self.bandwidth_actions = []
        self.rewards = []
        self.values = []
        self.mapping_log_probs = []
        self.bandwidth_log_probs = []
        self.dones = []
    
    def _clip_gradients(self, model, max_norm=None):
        """
        应用梯度裁剪
        
        Args:
            model: 要裁剪梯度的模型
            max_norm: 最大梯度范数，如果为None则使用self.max_grad_norm
        """
        if max_norm is None:
            max_norm = self.max_grad_norm
        
        # 计算总梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # 应用梯度裁剪
        clipped = False
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
            clipped = True
        
        return total_norm, clipped
    
    def _clip_attention_head_gradients(self, model):
        """
        专门裁剪注意力头的梯度，防止某一头主导更新
        """
        clipped_count = 0

        def clip_param(module_name: str, param_name: str, param: torch.Tensor):
            nonlocal clipped_count
            if param is None or param.grad is None:
                return
            param_norm = param.grad.data.norm(2)
            if param_norm > self.head_gradient_clip:
                clip_coef = self.head_gradient_clip / (param_norm + 1e-6)
                param.grad.data.mul_(clip_coef)
                clipped_count += 1
                print(f"🔧 裁剪注意力头梯度: {module_name}.{param_name}, 范数: {param_norm:.4f} -> {(param_norm * clip_coef):.4f}")

        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # in_proj_* 在 MultiheadAttention 自身上
                if hasattr(module, "in_proj_weight"):
                    clip_param(name, "in_proj_weight", module.in_proj_weight)
                if hasattr(module, "in_proj_bias"):
                    clip_param(name, "in_proj_bias", module.in_proj_bias)

                # out_proj 是一个 Linear 子模块
                if hasattr(module, "out_proj") and module.out_proj is not None:
                    if hasattr(module.out_proj, "weight"):
                        clip_param(name, "out_proj.weight", module.out_proj.weight)
                    if hasattr(module.out_proj, "bias"):
                        clip_param(name, "out_proj.bias", module.out_proj.bias)

        return clipped_count
    
    def _clip_task_specific_gradients(self, model, task_type):
        """
        裁剪任务特定层的梯度，防止某一任务主导更新
        
        Args:
            model: 要裁剪梯度的模型
            task_type: 任务类型 ("mapping" 或 "bandwidth")
        """
        task_layers = []
        clipped_count = 0
        
        if task_type == "mapping":
            # 映射任务的特定层
            if hasattr(model, 'global_mapping_head'):
                task_layers.append(model.global_mapping_head)
            if hasattr(model, 'constraint_checker'):
                task_layers.append(model.constraint_checker)
        elif task_type == "bandwidth":
            # 带宽任务的特定层
            if hasattr(model, 'global_bandwidth_head'):
                task_layers.append(model.global_bandwidth_head)
            if hasattr(model, 'bandwidth_constraint_checker'):
                task_layers.append(model.bandwidth_constraint_checker)
            if hasattr(model, 'link_encoder'):
                task_layers.append(model.link_encoder)
        
        # 对任务特定层应用梯度裁剪
        for layer in task_layers:
            if layer is not None:
                layer_norm = 0.0
                for p in layer.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        layer_norm += param_norm.item() ** 2
                layer_norm = layer_norm ** (1. / 2)
                
                if layer_norm > self.task_gradient_clip:
                    clip_coef = self.task_gradient_clip / (layer_norm + 1e-6)
                    for p in layer.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(clip_coef)
                    clipped_count += 1
                    print(f"🔧 裁剪{task_type}任务梯度: 范数 {layer_norm:.4f} -> {layer_norm * clip_coef:.4f}")
        
        return clipped_count
    
    def _apply_comprehensive_gradient_clipping(self, model, task_type):
        """
        应用全面的梯度裁剪策略
        
        Args:
            model: 要裁剪梯度的模型
            task_type: 任务类型 ("mapping", "bandwidth" 或 "critic")
        """
        # 1. 裁剪注意力头梯度
        head_clipped_count = self._clip_attention_head_gradients(model)
        
        # 2. 裁剪任务特定层梯度
        task_clipped_count = 0
        if task_type in ["mapping", "bandwidth"]:
            task_clipped_count = self._clip_task_specific_gradients(model, task_type)
        
        # 3. 裁剪整体梯度
        total_norm, overall_clipped = self._clip_gradients(model)
        
        # 4. 更新统计信息
        if task_type in self.gradient_stats:
            self.gradient_stats[task_type]['total_norms'].append(total_norm)
            if overall_clipped:
                self.gradient_stats[task_type]['clipped_count'] += 1
            self.gradient_stats[task_type]['head_clipped_count'] += head_clipped_count
            self.gradient_stats[task_type]['task_clipped_count'] += task_clipped_count
        
        return total_norm, overall_clipped, head_clipped_count, task_clipped_count
    
    def get_gradient_stats(self, task_type=None):
        """
        获取梯度统计信息
        
        Args:
            task_type: 任务类型，如果为None则返回所有任务的统计信息
            
        Returns:
            gradient_stats: 梯度统计信息
        """
        if task_type is None:
            return self.gradient_stats
        
        if task_type in self.gradient_stats:
            stats = self.gradient_stats[task_type]
            if stats['total_norms']:
                stats['avg_norm'] = np.mean(stats['total_norms'])
                stats['max_norm'] = np.max(stats['total_norms'])
                stats['min_norm'] = np.min(stats['total_norms'])
            else:
                stats['avg_norm'] = 0.0
                stats['max_norm'] = 0.0
                stats['min_norm'] = 0.0
            return stats
        
        return None
    
    def reset_gradient_stats(self, task_type=None):
        """
        重置梯度统计信息
        
        Args:
            task_type: 任务类型，如果为None则重置所有任务的统计信息
        """
        if task_type is None:
            for key in self.gradient_stats:
                self.gradient_stats[key] = {
                    'total_norms': [], 
                    'clipped_count': 0, 
                    'head_clipped_count': 0, 
                    'task_clipped_count': 0
                }
        elif task_type in self.gradient_stats:
            self.gradient_stats[task_type] = {
                'total_norms': [], 
                'clipped_count': 0, 
                'head_clipped_count': 0, 
                'task_clipped_count': 0
            }
    
    def print_gradient_summary(self):
        """打印梯度统计摘要"""
        print("\n" + "="*60)
        print("🔧 梯度裁剪统计摘要")
        print("="*60)
        
        for task_name, stats in self.gradient_stats.items():
            if stats['total_norms']:
                avg_norm = np.mean(stats['total_norms'])
                max_norm = np.max(stats['total_norms'])
                min_norm = np.min(stats['total_norms'])
                print(f"{task_name:>15}: 平均范数: {avg_norm:.4f}, 最大范数: {max_norm:.4f}, 最小范数: {min_norm:.4f}")
                print(f"{'':>15}  整体裁剪: {stats['clipped_count']}, 注意力头裁剪: {stats['head_clipped_count']}, 任务层裁剪: {stats['task_clipped_count']}")
            else:
                print(f"{task_name:>15}: 暂无数据")
        print("="*60 + "\n")
    
    def select_actions(self, state, temperature=1.0):
        """
        选择映射和带宽分配动作
        
        Args:
            state: 环境状态字典
            temperature: 温度参数，控制探索程度（1.0为正常，>1.0增加探索，<1.0减少探索）
        
        Returns:
            mapping_action: 映射动作 [num_virtual_nodes]
            bandwidth_action: 带宽动作 [num_links]
            mapping_log_prob: 映射动作的log概率
            bandwidth_log_prob: 带宽动作的log概率
            value: 状态价值
        """
        # 提取状态信息
        physical_features = state['physical_features'].to(self.device)
        physical_edge_index = state['physical_edges'].to(self.device)
        physical_edge_attr = state['physical_edge_features'].to(self.device)
        # physical_num_nodes = state['physical_num_nodes']
        virtual_features = state['virtual_features'].to(self.device)
        virtual_edge_index = state['virtual_edges'].to(self.device)
        virtual_edge_attr = state['virtual_edge_features'].to(self.device)
        # virtual_num_nodes = state['virtual_num_nodes']
        
        # 新增：提取GCN特征
        physical_gcn_features = state.get('physical_gcn_features', None)
        virtual_gcn_features = state.get('virtual_gcn_features', None)
        
        if physical_gcn_features is not None:
            physical_gcn_features = physical_gcn_features.to(self.device)
        if virtual_gcn_features is not None:
            virtual_gcn_features = virtual_gcn_features.to(self.device)
        
        # 生成节点映射约束
        node_constraints = self.constraint_manager.generate_node_mapping_constraints(
            physical_features, virtual_features, physical_edge_index, virtual_edge_index
        )
        
        # 第一阶段：映射Actor
        mapping_logits, constraint_scores, _ = self.mapping_actor(
            physical_features, physical_edge_index, physical_edge_attr,
            virtual_features, virtual_edge_index, virtual_edge_attr,
            physical_gcn_features, virtual_gcn_features
        )
        
        # 应用约束管理器生成的约束
        mapping_logits = self.constraint_manager.apply_node_mapping_constraints(
            mapping_logits, node_constraints, temperature
        )
        
        # 应用原有的约束分数（如果有的话）
        mapping_logits = mapping_logits + 0.5 * torch.log(constraint_scores + 1e-8)
        
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
        mapping_entropy = mapping_dist.entropy().mean().item()  # 计算映射策略熵
        
        # 第二阶段：带宽Actor
        bandwidth_logits, bandwidth_constraint_scores, _, link_indices = self.bandwidth_actor(
            physical_features, physical_edge_index, physical_edge_attr,
            virtual_features, virtual_edge_index, virtual_edge_attr,
            mapping_action,
            physical_gcn_features, virtual_gcn_features
        )
        
        if bandwidth_logits.size(0) > 0:
            # 生成带宽约束
            # 从环境状态中获取链路特定的带宽映射
            link_bandwidth_mappings = state.get('bandwidth_mapping', {})
            # 使用实际的虚拟边信息生成约束
            bandwidth_constraints = self.constraint_manager.generate_bandwidth_constraints(
                virtual_edge_attr, link_bandwidth_mappings, virtual_edge_index
            )
            
            # 应用约束管理器生成的带宽约束
            bandwidth_logits = self.constraint_manager.apply_bandwidth_constraints(
                bandwidth_logits, bandwidth_constraints, temperature
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
            
            # 🚀 优化：当两个虚拟节点映射到同一物理节点时，强制分配最大带宽级别
            bandwidth_action = self._optimize_bandwidth_for_same_node_mapping(
                bandwidth_probs, mapping_action, link_indices
            )
            
            # 计算带宽动作的log概率（考虑强制分配的影响）
            bandwidth_log_prob = self._compute_optimized_bandwidth_log_prob(
                bandwidth_probs, bandwidth_action, mapping_action, link_indices
            )
            
            # 计算带宽策略熵
            bandwidth_dist = torch.distributions.Categorical(bandwidth_probs)
            bandwidth_entropy = bandwidth_dist.entropy().mean().item()
        else:
            bandwidth_action = torch.empty(0, dtype=torch.long, device=self.device)
            bandwidth_log_prob = torch.empty(0, dtype=torch.float32, device=self.device)
            bandwidth_entropy = 0.0  # 没有链路时熵为0
        
        print(f"TwoStagePPOAgent select_actions mapping_action: {mapping_action}")
        print(f"TwoStagePPOAgent select_actions bandwidth_action: {bandwidth_action}")
        
        # Critic评估
        value = self.critic(
            physical_features, physical_edge_index, physical_edge_attr,
            virtual_features, virtual_edge_index, virtual_edge_attr,
            physical_gcn_features, virtual_gcn_features
        )
        
        return (mapping_action.cpu().detach().numpy(), bandwidth_action.cpu().detach().numpy(),
                mapping_log_prob.cpu().detach().numpy(), bandwidth_log_prob.cpu().detach().numpy(),
                value.cpu().item(), mapping_entropy, bandwidth_entropy, link_indices)
    
    def _optimize_bandwidth_for_same_node_mapping(self, bandwidth_probs, mapping_action, link_indices):
        """
        优化带宽分配：当两个虚拟节点映射到同一物理节点时，有概率分配最大带宽级别
        
        Args:
            bandwidth_probs: 带宽概率分布 [num_links, bandwidth_levels]
            mapping_action: 映射动作 [num_virtual_nodes]
            link_indices: 链路索引列表 [[src, dst], ...]
            
        Returns:
            optimized_bandwidth_action: 优化后的带宽动作
        """
        # 创建优化后的带宽动作
        optimized_bandwidth_action = torch.zeros(bandwidth_probs.size(0), dtype=torch.long, device=self.device)
        
        for i, (src, dst) in enumerate(link_indices):
            src_physical = mapping_action[src]
            dst_physical = mapping_action[dst]
            
            # 检查是否映射到同一物理节点
            if src_physical == dst_physical:
                # 有70%的概率分配最大带宽级别，30%的概率正常采样
                if torch.rand(1, device=self.device) < 1:
                    max_bandwidth_level = self.bandwidth_levels - 1
                    optimized_bandwidth_action[i] = max_bandwidth_level
                    # print(f"🔗 优化：虚拟节点{src}和{dst}映射到同一物理节点{src_physical}，分配最大带宽级别{max_bandwidth_level}")
                else:
                    # 正常采样，但偏向高带宽级别
                    bandwidth_dist = torch.distributions.Categorical(bandwidth_probs[i])
                    optimized_bandwidth_action[i] = bandwidth_dist.sample()
                    # print(f"🔗 优化：虚拟节点{src}和{dst}映射到同一物理节点{src_physical}，正常采样带宽级别{optimized_bandwidth_action[i]}")
            else:
                # 正常采样带宽级别
                bandwidth_dist = torch.distributions.Categorical(bandwidth_probs[i])
                optimized_bandwidth_action[i] = bandwidth_dist.sample()
        
        return optimized_bandwidth_action
    
    def _compute_optimized_bandwidth_log_prob(self, bandwidth_probs, bandwidth_action, mapping_action, link_indices):
        """
        计算优化后带宽动作的log概率
        
        Args:
            bandwidth_probs: 带宽概率分布 [num_links, bandwidth_levels]
            bandwidth_action: 优化后的带宽动作 [num_links]
            mapping_action: 映射动作 [num_virtual_nodes]
            link_indices: 链路索引列表 [[src, dst], ...]
            
        Returns:
            log_probs: 带宽动作的log概率 [num_links]
        """
        log_probs = torch.zeros(bandwidth_probs.size(0), dtype=torch.float, device=self.device)
        
        for i, (src, dst) in enumerate(link_indices):
            src_physical = mapping_action[src]
            dst_physical = mapping_action[dst]
            
            if src_physical == dst_physical:
                # 对于同一物理节点的映射，使用最大带宽级别的概率
                max_bandwidth_level = self.bandwidth_levels - 1
                log_probs[i] = torch.log(bandwidth_probs[i, max_bandwidth_level] + 1e-8)
            else:
                # 对于不同物理节点的映射，使用正常采样的概率
                log_probs[i] = torch.log(bandwidth_probs[i, bandwidth_action[i]] + 1e-8)
        
        return log_probs
    
    def store_transition(self, state, mapping_action, bandwidth_action, 
                        reward, value, mapping_log_prob, bandwidth_log_prob, done,
                        td_error: Optional[float] = None):
        """存储经验到回放缓冲区"""
        
        # 存储到新的ReplayBuffer
        if self.replay_buffer_type == "prioritized":
            self.replay_buffer.push(
                state=state,
                mapping_action=mapping_action,
                bandwidth_action=bandwidth_action,
                reward=reward,
                value=value,
                mapping_log_prob=mapping_log_prob,
                bandwidth_log_prob=bandwidth_log_prob,
                done=done,
                td_error=td_error
            )
        else:
            self.replay_buffer.push(
                state=state,
                mapping_action=mapping_action,
                bandwidth_action=bandwidth_action,
                reward=reward,
                value=value,
                mapping_log_prob=mapping_log_prob,
                bandwidth_log_prob=bandwidth_log_prob,
                done=done,
                priority=(abs(td_error) if td_error is not None else None)
            )
        
        # 同时存储到旧缓冲区（保持向后兼容）
        self.states.append(state)
        self.mapping_actions.append(mapping_action)
        self.bandwidth_actions.append(bandwidth_action)
        self.rewards.append(reward)
        self.values.append(value)
        self.mapping_log_probs.append(mapping_log_prob)
        self.bandwidth_log_probs.append(bandwidth_log_prob)
        self.dones.append(done)
        
        # 如果episode结束，增加计数
        if done:
            self.episode_count += 1
        
        # 当经验回放缓冲区中的转移数量达到batch_size时，自动更新智能体
        if hasattr(self.replay_buffer, 'count') and self.replay_buffer.count >= self.batch_size:
            print(f"🔄 经验回放缓冲区达到batch_size({self.batch_size})，自动更新智能体...")
            self._update_from_replay_buffer()
            self.replay_buffer.reset_count()  # 重置计数器
            print(f"✅ 智能体更新完成，缓冲区大小: {len(self.replay_buffer)}")
    
    def update(self, use_replay_buffer: bool = True):
        """更新网络"""
        
        if use_replay_buffer and len(self.replay_buffer) >= self.batch_size:
            # 使用经验回放缓冲区进行批量更新
            self._update_from_replay_buffer()
        else:
            # 使用原有的在线更新方式
            self._update_online()
    
    def _update_from_replay_buffer(self):
        """从经验回放缓冲区更新网络"""
        # 采样经验
        if self.replay_buffer_type == "prioritized":
            experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        else:
            experiences, indices = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(len(experiences), device=self.device)
        
        # 提取数据
        states = [exp['state'] for exp in experiences]
        mapping_actions = [exp['mapping_action'] for exp in experiences]
        bandwidth_actions = [exp['bandwidth_action'] for exp in experiences]
        rewards = [exp['reward'] for exp in experiences]
        values = [exp['value'] for exp in experiences]
        mapping_log_probs = [exp['mapping_log_prob'] for exp in experiences]
        bandwidth_log_probs = [exp['bandwidth_log_prob'] for exp in experiences]
        dones = [exp['done'] for exp in experiences]
        
        # 计算优势函数
        advantages = self._compute_advantages_from_batch(rewards, values, dones)
        # 对优势做标准化（每个更新批次）
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False) + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        
        # 转换为tensor
        mapping_actions = [torch.tensor(actions, dtype=torch.long, device=self.device) for actions in mapping_actions]
        bandwidth_actions = [torch.tensor(actions, dtype=torch.long, device=self.device) for actions in bandwidth_actions]
        old_mapping_log_probs = [torch.tensor(probs, dtype=torch.float32, device=self.device) for probs in mapping_log_probs]
        old_bandwidth_log_probs = [torch.tensor(probs, dtype=torch.float32, device=self.device) for probs in bandwidth_log_probs]
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        
        # 多轮更新 - 使用n_epochs参数
        for epoch in range(self.n_epochs):
            # 更新网络
            self._update_mapping_actor(states, mapping_actions, old_mapping_log_probs, advantages, returns, weights)
            self._update_bandwidth_actor(states, bandwidth_actions, old_bandwidth_log_probs, advantages, returns, mapping_actions, weights)
            self._update_critic(states, returns, weights)
        
        # 如果是优先级经验回放，更新优先级
        if self.replay_buffer_type == "prioritized":
            # 计算新的TD误差作为优先级
            new_priorities = self._compute_td_errors(states, returns)
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        # 每10个episode打印一次梯度统计摘要
        if self.episode_count % 10 == 0:
            self.print_gradient_summary()
    
    def _update_online(self):
        """原有的在线更新方式"""
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
        
        # 多轮更新 - 使用n_epochs参数
        for epoch in range(self.n_epochs):
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
    
    def _update_mapping_actor(self, states, mapping_actions, old_log_probs, advantages, returns, weights=None):
        """更新映射Actor"""
        # 重新计算当前策略的概率
        current_log_probs = []
        entropies = []
        logprob_lengths = []  # 每个样本的log_prob数量（=虚拟节点数）
        used_indices = []     # 对应样本索引（用于权重对齐）
        
        for i, state in enumerate(states):
            physical_features = state['physical_features'].to(self.device)
            physical_edge_index = state['physical_edges'].to(self.device)
            physical_edge_attr = state['physical_edge_features'].to(self.device)
            # physical_num_nodes = state['physical_num_nodes']
            virtual_features = state['virtual_features'].to(self.device)
            virtual_edge_index = state['virtual_edges'].to(self.device)
            virtual_edge_attr = state['virtual_edge_features'].to(self.device)
            # virtual_num_nodes = state['virtual_num_nodes']
            
            # 生成节点映射约束
            node_constraints = self.constraint_manager.generate_node_mapping_constraints(
                physical_features, virtual_features, physical_edge_index, virtual_edge_index
            )
            
            mapping_logits, constraint_scores, _ = self.mapping_actor(
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                physical_gcn_features=state.get('physical_gcn_features', None),
                virtual_gcn_features=state.get('virtual_gcn_features', None)
            )
            
            # 应用约束管理器生成的约束
            mapping_logits = self.constraint_manager.apply_node_mapping_constraints(
                mapping_logits, node_constraints, temperature=1.0
            )
            
            # 应用原有的约束分数（如果有的话）
            mapping_logits = mapping_logits + 0.5 * torch.log(constraint_scores + 1e-8)
            
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
            logprob_lengths.append(current_log_prob.numel())
            used_indices.append(i)
        
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
            
            # 展开样本权重以与逐决策log_prob对齐
            expanded_weights = None
            sample_weights = None
            if weights is not None:
                sample_weights = weights[used_indices]
                repeats = torch.tensor(logprob_lengths, device=self.device)
                expanded_weights = torch.repeat_interleave(sample_weights, repeats)
            
            # 计算比率
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * all_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * all_advantages
            if expanded_weights is not None:
                min_surr = torch.min(surr1, surr2)
                mapping_loss = -(min_surr * expanded_weights).sum() / (expanded_weights.sum() + 1e-8)
            else:
                mapping_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失
            if sample_weights is not None:
                entropy_loss = -(entropies * sample_weights).sum() / (sample_weights.sum() + 1e-8)
            else:
                entropy_loss = -entropies.mean()
            
            # 总损失
            total_loss = mapping_loss + self.entropy_coef * entropy_loss
            
            # 更新
            self.mapping_optimizer.zero_grad()
            total_loss.backward()
            
            # 应用梯度裁剪
            total_norm, overall_clipped, head_clipped, task_clipped = self._apply_comprehensive_gradient_clipping(self.mapping_actor, "mapping")
            print(f"🔧 映射Actor梯度裁剪完成，总范数: {total_norm:.4f}, 整体裁剪: {overall_clipped}, 注意力头裁剪: {head_clipped}, 任务层裁剪: {task_clipped}")
            
            self.mapping_optimizer.step()
    
    def _update_bandwidth_actor(self, states, bandwidth_actions, old_log_probs, advantages, returns, mapping_actions=None, weights=None):
        """更新带宽Actor"""
        # 重新计算当前策略的概率
        current_log_probs = []
        entropies = []
        logprob_lengths = []  # 每个样本的log_prob数量（=该样本链路数）
        used_indices = []     # 有有效链路的样本索引
        
        for i, state in enumerate(states):
            physical_features = state['physical_features'].to(self.device)
            physical_edge_index = state['physical_edges'].to(self.device)
            physical_edge_attr = state['physical_edge_features'].to(self.device)
            # physical_num_nodes = state['physical_num_nodes']
            virtual_features = state['virtual_features'].to(self.device)
            virtual_edge_index = state['virtual_edges'].to(self.device)
            virtual_edge_attr = state['virtual_edge_features'].to(self.device)
            # virtual_num_nodes = state['virtual_num_nodes']
            
            # 使用实际的映射动作，如果没有提供则使用随机动作
            if mapping_actions is not None and i < len(mapping_actions):
                # 修复UserWarning：使用clone().detach()而不是torch.tensor()
                if isinstance(mapping_actions[i], torch.Tensor):
                    mapping_action = mapping_actions[i].clone().detach().to(self.device)
                else:
                    mapping_action = torch.tensor(mapping_actions[i], dtype=torch.long, device=self.device)
            else:
                raise ValueError("mapping_actions is None")

            bandwidth_logits, bandwidth_constraint_scores, _, _ = self.bandwidth_actor(
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                mapping_action,
                physical_gcn_features=state.get('physical_gcn_features', None),
                virtual_gcn_features=state.get('virtual_gcn_features', None)
            )
            
            if bandwidth_logits.size(0) > 0:
                # 生成带宽约束
                link_bandwidth_mappings = state.get('bandwidth_mapping', {})
                bandwidth_constraints = self.constraint_manager.generate_bandwidth_constraints(
                    virtual_edge_attr, link_bandwidth_mappings, virtual_edge_index
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
                logprob_lengths.append(current_log_prob.numel())
                used_indices.append(i)
        
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
            
            # 展开样本权重以与逐决策log_prob对齐
            expanded_weights = None
            sample_weights = None
            if weights is not None and len(used_indices) > 0:
                sample_weights = weights[used_indices]
                repeats = torch.tensor(logprob_lengths, device=self.device)
                expanded_weights = torch.repeat_interleave(sample_weights, repeats)
            
            # 计算比率
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * all_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * all_advantages
            if expanded_weights is not None:
                min_surr = torch.min(surr1, surr2)
                bandwidth_loss = -(min_surr * expanded_weights).sum() / (expanded_weights.sum() + 1e-8)
            else:
                bandwidth_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失
            if sample_weights is not None:
                entropy_loss = -(entropies * sample_weights).sum() / (sample_weights.sum() + 1e-8)
            else:
                entropy_loss = -entropies.mean()
            
            # 总损失
            total_loss = bandwidth_loss + self.entropy_coef * entropy_loss
            
            # 更新
            self.bandwidth_optimizer.zero_grad()
            total_loss.backward()
            
            # 应用梯度裁剪
            total_norm, overall_clipped, head_clipped, task_clipped = self._apply_comprehensive_gradient_clipping(self.bandwidth_actor, "bandwidth")
            print(f"🔧 带宽Actor梯度裁剪完成，总范数: {total_norm:.4f}, 整体裁剪: {overall_clipped}, 注意力头裁剪: {head_clipped}, 任务层裁剪: {task_clipped}")
            
            self.bandwidth_optimizer.step()
    
    def _update_critic(self, states, returns, weights=None):
        """更新Critic"""
        values = []
        
        for i, state in enumerate(states):
            physical_features = state['physical_features'].to(self.device)
            physical_edge_index = state['physical_edges'].to(self.device)
            physical_edge_attr = state['physical_edge_features'].to(self.device)
            # physical_num_nodes = state['physical_num_nodes']
            virtual_features = state['virtual_features'].to(self.device)
            virtual_edge_index = state['virtual_edges'].to(self.device)
            virtual_edge_attr = state['virtual_edge_features'].to(self.device)
            # virtual_num_nodes = state['virtual_num_nodes']
            
            value = self.critic(
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                physical_gcn_features=state.get('physical_gcn_features', None),
                virtual_gcn_features=state.get('virtual_gcn_features', None)
            )
            values.append(value)
        
        values = torch.cat(values)
        returns = returns[:len(values)]
        
        # 价值损失
        value_loss = F.mse_loss(values, returns, reduction='none')
        
        # 如果有权重，应用权重
        if weights is not None:
            value_loss = (value_loss * weights).mean()
        else:
            value_loss = value_loss.mean()
        
        # 更新
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        
        # 应用梯度裁剪
        total_norm, overall_clipped, head_clipped, task_clipped = self._apply_comprehensive_gradient_clipping(self.critic, "critic")
        print(f"🔧 Critic梯度裁剪完成，总范数: {total_norm:.4f}, 整体裁剪: {overall_clipped}, 注意力头裁剪: {head_clipped}, 任务层裁剪: {task_clipped}")
        
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

    def _compute_advantages_from_batch(self, rewards, values, dones):
        """从批量经验计算优势函数"""
        # 转换为张量
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
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

    def _compute_td_errors(self, states, returns):
        """计算TD误差作为优先级"""
        values = []
        for state in states:
            physical_features = state['physical_features'].to(self.device)
            physical_edge_index = state['physical_edges'].to(self.device)
            physical_edge_attr = state['physical_edge_features'].to(self.device)
            virtual_features = state['virtual_features'].to(self.device)
            virtual_edge_index = state['virtual_edges'].to(self.device)
            virtual_edge_attr = state['virtual_edge_features'].to(self.device)
            
            value = self.critic(
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                physical_gcn_features=state.get('physical_gcn_features', None),
                virtual_gcn_features=state.get('virtual_gcn_features', None)
            )
            values.append(value)
        
        values = torch.cat(values)
        returns = returns[:len(values)]
        
        td_errors = torch.abs(values - returns)
        return td_errors.detach().cpu().numpy().tolist()