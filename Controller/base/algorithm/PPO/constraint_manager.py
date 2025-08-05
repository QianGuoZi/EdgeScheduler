#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx

class ConstraintManager:
    """约束管理器，用于在动作选择之前生成和应用约束"""
    
    def __init__(self, bandwidth_levels: int = 10):
        self.bandwidth_levels = bandwidth_levels
        
    def generate_node_mapping_constraints(self, 
                                        physical_features: torch.Tensor,
                                        virtual_features: torch.Tensor,
                                        physical_edge_index: torch.Tensor,
                                        virtual_edge_index: torch.Tensor) -> torch.Tensor:
        """
        生成节点映射约束矩阵
        
        Args:
            physical_features: 物理节点特征 [num_physical_nodes, features]
            virtual_features: 虚拟节点特征 [num_virtual_nodes, features]
            physical_edge_index: 物理网络边索引
            virtual_edge_index: 虚拟网络边索引
            
        Returns:
            constraint_matrix: 约束矩阵 [num_virtual_nodes, num_physical_nodes]
                             1.0表示可行，0.0表示不可行
        """
        num_virtual_nodes = virtual_features.size(0)
        num_physical_nodes = physical_features.size(0)
        
        # 初始化约束矩阵，默认所有映射都可行
        constraint_matrix = torch.ones(num_virtual_nodes, num_physical_nodes, 
                                     device=virtual_features.device)
        
        # 检查资源约束
        for virtual_idx in range(num_virtual_nodes):
            virtual_cpu = virtual_features[virtual_idx, 0]    # CPU需求
            virtual_memory = virtual_features[virtual_idx, 1] # 内存需求
            
            for physical_idx in range(num_physical_nodes):
                physical_cpu = physical_features[physical_idx, 0]      # 总CPU
                physical_memory = physical_features[physical_idx, 1]   # 总内存
                cpu_usage = physical_features[physical_idx, 2]         # CPU使用率
                memory_usage = physical_features[physical_idx, 3]      # 内存使用率
                
                # 计算可用资源
                available_cpu = physical_cpu * (1 - cpu_usage)
                available_memory = physical_memory * (1 - memory_usage)
                
                # 检查资源是否足够
                if virtual_cpu > available_cpu or virtual_memory > available_memory:
                    constraint_matrix[virtual_idx, physical_idx] = 0.0
        
        return constraint_matrix
    
    def generate_bandwidth_constraints(self,
                                     virtual_edge_features: torch.Tensor,
                                     link_bandwidth_mappings: Dict[str, Dict[int, int]],
                                     virtual_edges: torch.Tensor,
                                     expected_num_links: int = None) -> torch.Tensor:
        """
        生成带宽分配约束矩阵（支持链路特定的带宽映射）
        
        Args:
            virtual_edge_features: 虚拟链路特征 [num_links, 2] (min_bandwidth, max_bandwidth)
            link_bandwidth_mappings: 每个链路的带宽映射字典 {link_key: {level: bandwidth}}
            virtual_edges: 虚拟边索引 [2, num_links]
            expected_num_links: 期望的链路数量，如果提供则扩展约束矩阵
            
        Returns:
            constraint_matrix: 约束矩阵 [num_links, bandwidth_levels]
                             1.0表示可行，0.0表示不可行
        """
        num_links = virtual_edge_features.size(0)
        
        # 初始化约束矩阵
        constraint_matrix = torch.ones(num_links, self.bandwidth_levels, 
                                     device=virtual_edge_features.device)
        
        # 检查每个链路的带宽约束
        for link_idx in range(num_links):
            min_bandwidth = virtual_edge_features[link_idx, 0]
            max_bandwidth = virtual_edge_features[link_idx, 1]
            
            # 获取链路信息
            src, dst = virtual_edges[:, link_idx]
            link_key = f"{src.item()}_{dst.item()}"
            
            if link_key in link_bandwidth_mappings:
                link_mapping = link_bandwidth_mappings[link_key]
                
                for level in range(self.bandwidth_levels):
                    if level in link_mapping:
                        allocated_bandwidth = link_mapping[level]
                        
                        # 检查带宽是否在允许范围内
                        if allocated_bandwidth < min_bandwidth or allocated_bandwidth > max_bandwidth:
                            constraint_matrix[link_idx, level] = 0.0
                    else:
                        # 如果该等级不存在，标记为不可行
                        constraint_matrix[link_idx, level] = 0.0
            else:
                # 如果找不到链路的映射，所有等级都标记为不可行
                constraint_matrix[link_idx, :] = 0.0
        
        # 如果指定了期望的链路数量且不匹配，则调整约束矩阵
        if expected_num_links is not None and num_links != expected_num_links:
            print(f"调整约束矩阵：从 {num_links} 到 {expected_num_links} 个链路")
            
            if expected_num_links > num_links:
                # 扩展约束矩阵
                new_constraint_matrix = torch.ones(expected_num_links, self.bandwidth_levels, 
                                                 device=virtual_edge_features.device)
                new_constraint_matrix[:num_links, :] = constraint_matrix
                constraint_matrix = new_constraint_matrix
            else:
                # 截取约束矩阵
                constraint_matrix = constraint_matrix[:expected_num_links, :]
        
        return constraint_matrix
    
    def apply_node_mapping_constraints(self, 
                                     logits: torch.Tensor, 
                                     constraints: torch.Tensor,
                                     temperature: float = 1.0) -> torch.Tensor:
        """
        应用节点映射约束到logits
        
        Args:
            logits: 原始logits [num_virtual_nodes, num_physical_nodes]
            constraints: 约束矩阵 [num_virtual_nodes, num_physical_nodes]
            temperature: 温度参数，控制探索程度
            
        Returns:
            constrained_logits: 应用约束后的logits
        """
        # 将不可行的选择设置为负无穷
        masked_logits = logits.clone()
        masked_logits[constraints == 0.0] = float('-inf')
        
        # 应用温度缩放
        constrained_logits = masked_logits / temperature
        
        return constrained_logits
    
    def apply_bandwidth_constraints(self,
                                  logits: torch.Tensor,
                                  constraints: torch.Tensor,
                                  temperature: float = 1.0) -> torch.Tensor:
        """
        应用带宽约束到logits
        
        Args:
            logits: 原始logits [num_links, bandwidth_levels]
            constraints: 约束矩阵 [num_links, bandwidth_levels]
            temperature: 温度参数，控制探索程度
            
        Returns:
            constrained_logits: 应用约束后的logits
        """
        # 检查形状是否匹配
        if logits.shape != constraints.shape:
            print(f"警告：logits形状 {logits.shape} 与约束形状 {constraints.shape} 不匹配")
            print(f"调整约束矩阵形状以匹配logits")
            
            # 如果约束矩阵的行数少于logits，需要扩展约束矩阵
            if constraints.shape[0] < logits.shape[0]:
                # 创建新的约束矩阵，默认所有带宽等级都可行
                new_constraints = torch.ones(logits.shape, device=constraints.device)
                # 复制原有的约束
                new_constraints[:constraints.shape[0], :] = constraints
                constraints = new_constraints
            elif constraints.shape[0] > logits.shape[0]:
                # 截取约束矩阵以匹配logits
                constraints = constraints[:logits.shape[0], :]
        
        # 将不可行的选择设置为负无穷
        masked_logits = logits.clone()
        masked_logits[constraints == 0.0] = float('-inf')
        
        # 应用温度缩放
        constrained_logits = masked_logits / temperature
        
        return constrained_logits
    
    def check_mapping_feasibility(self, 
                                mapping_action: torch.Tensor,
                                constraints: torch.Tensor) -> bool:
        """
        检查映射动作是否满足约束
        
        Args:
            mapping_action: 映射动作 [num_virtual_nodes]
            constraints: 约束矩阵 [num_virtual_nodes, num_physical_nodes]
            
        Returns:
            bool: 是否满足约束
        """
        for virtual_idx, physical_idx in enumerate(mapping_action):
            if constraints[virtual_idx, physical_idx] == 0.0:
                return False
        return True
    
    def check_bandwidth_feasibility(self,
                                  bandwidth_action: torch.Tensor,
                                  constraints: torch.Tensor) -> bool:
        """
        检查带宽动作是否满足约束
        
        Args:
            bandwidth_action: 带宽动作 [num_links]
            constraints: 约束矩阵 [num_links, bandwidth_levels]
            
        Returns:
            bool: 是否满足约束
        """
        for link_idx, level in enumerate(bandwidth_action):
            if constraints[link_idx, level] == 0.0:
                return False
        return True
    
    def get_feasible_actions_count(self, constraints: torch.Tensor) -> Dict[str, int]:
        """
        获取可行动作的数量统计
        
        Args:
            constraints: 约束矩阵
            
        Returns:
            Dict: 包含可行动作数量的统计信息
        """
        if len(constraints.shape) == 2:
            # 检查是否是带宽约束（通过形状判断）
            if constraints.shape[1] >= 10:  # 带宽等级通常>=10
                # 带宽约束: [num_links, bandwidth_levels]
                feasible_per_link = torch.sum(constraints, dim=1)
                total_feasible = torch.sum(constraints)
                total_possible = constraints.numel()
                
                return {
                    'total_feasible': total_feasible.item(),
                    'total_possible': total_possible,
                    'feasible_per_link': feasible_per_link.tolist(),
                    'min_feasible_per_link': torch.min(feasible_per_link).item(),
                    'max_feasible_per_link': torch.max(feasible_per_link).item()
                }
            else:
                # 节点映射约束: [num_virtual_nodes, num_physical_nodes]
                feasible_per_node = torch.sum(constraints, dim=1)
                total_feasible = torch.sum(constraints)
                total_possible = constraints.numel()
                
                return {
                    'total_feasible': total_feasible.item(),
                    'total_possible': total_possible,
                    'feasible_per_node': feasible_per_node.tolist(),
                    'min_feasible_per_node': torch.min(feasible_per_node).item(),
                    'max_feasible_per_node': torch.max(feasible_per_node).item()
                }
        else:
            # 其他维度的约束
            total_feasible = torch.sum(constraints)
            total_possible = constraints.numel()
            
            return {
                'total_feasible': total_feasible.item(),
                'total_possible': total_possible
            }
