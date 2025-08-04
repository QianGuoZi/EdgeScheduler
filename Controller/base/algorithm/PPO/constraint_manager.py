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
                                     bandwidth_mapping: Dict[int, int],
                                     expected_num_links: int = None) -> torch.Tensor:
        """
        生成带宽分配约束矩阵
        
        Args:
            virtual_edge_features: 虚拟链路特征 [num_links, 2] (min_bandwidth, max_bandwidth)
            bandwidth_mapping: 带宽等级到实际带宽的映射
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
            
            for level in range(self.bandwidth_levels):
                allocated_bandwidth = bandwidth_mapping[level]
                
                # 检查带宽是否在允许范围内
                if allocated_bandwidth < min_bandwidth or allocated_bandwidth > max_bandwidth:
                    constraint_matrix[link_idx, level] = 0.0
        
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

def test_constraint_manager():
    """测试约束管理器"""
    print("🧪 测试约束管理器")
    print("=" * 60)
    
    # 创建约束管理器
    constraint_manager = ConstraintManager(bandwidth_levels=10)
    
    # 测试数据
    physical_features = torch.tensor([
        [100, 200, 0.3, 0.4],  # 节点0: 100CPU, 200内存, 30%CPU使用, 40%内存使用
        [150, 300, 0.5, 0.6],  # 节点1: 150CPU, 300内存, 50%CPU使用, 60%内存使用
        [80, 150, 0.2, 0.3],   # 节点2: 80CPU, 150内存, 20%CPU使用, 30%内存使用
    ])
    
    virtual_features = torch.tensor([
        [30, 50],   # 虚拟节点0: 需要30CPU, 50内存
        [40, 80],   # 虚拟节点1: 需要40CPU, 80内存
        [25, 40],   # 虚拟节点2: 需要25CPU, 40内存
    ])
    
    print("物理节点特征:")
    for i, features in enumerate(physical_features):
        cpu, memory, cpu_usage, memory_usage = features
        available_cpu = cpu * (1 - cpu_usage)
        available_memory = memory * (1 - memory_usage)
        print(f"   节点{i}: 总CPU={cpu}, 总内存={memory}, 可用CPU={available_cpu:.1f}, 可用内存={available_memory:.1f}")
    
    print("\n虚拟节点需求:")
    for i, features in enumerate(virtual_features):
        cpu, memory = features
        print(f"   虚拟节点{i}: 需要CPU={cpu}, 需要内存={memory}")
    
    # 生成节点映射约束
    physical_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    virtual_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    
    node_constraints = constraint_manager.generate_node_mapping_constraints(
        physical_features, virtual_features, physical_edge_index, virtual_edge_index
    )
    
    print(f"\n节点映射约束矩阵:")
    print(node_constraints)
    
    # 统计可行动作
    node_stats = constraint_manager.get_feasible_actions_count(node_constraints)
    print(f"\n节点映射约束统计:")
    print(f"   总可行映射: {node_stats['total_feasible']}")
    print(f"   总可能映射: {node_stats['total_possible']}")
    print(f"   每个虚拟节点的可行物理节点数: {node_stats['feasible_per_node']}")
    
    # 测试带宽约束
    virtual_edge_features = torch.tensor([
        [20, 60],   # 链路0: 最小20, 最大60
        [30, 80],   # 链路1: 最小30, 最大80
    ])
    
    bandwidth_mapping = {i: 10 + i * 10 for i in range(10)}  # 10, 20, 30, ..., 100
    
    print(f"\n带宽映射: {bandwidth_mapping}")
    
    bandwidth_constraints = constraint_manager.generate_bandwidth_constraints(
        virtual_edge_features, bandwidth_mapping
    )
    
    print(f"\n带宽约束矩阵:")
    print(bandwidth_constraints)
    print(f"带宽约束矩阵形状: {bandwidth_constraints.shape}")
    
    # 统计带宽约束
    bandwidth_stats = constraint_manager.get_feasible_actions_count(bandwidth_constraints)
    print(f"\n带宽约束统计:")
    print(f"   总可行带宽等级: {bandwidth_stats['total_feasible']}")
    print(f"   总可能带宽等级: {bandwidth_stats['total_possible']}")
    print(f"   统计信息键: {list(bandwidth_stats.keys())}")
    if 'feasible_per_link' in bandwidth_stats:
        print(f"   每个链路的可行带宽等级数: {bandwidth_stats['feasible_per_link']}")
    
    # 测试约束应用
    print(f"\n测试约束应用:")
    
    # 模拟原始logits
    original_node_logits = torch.randn(3, 3)
    print(f"原始节点映射logits:")
    print(original_node_logits)
    
    # 应用约束
    constrained_node_logits = constraint_manager.apply_node_mapping_constraints(
        original_node_logits, node_constraints, temperature=1.0
    )
    
    print(f"\n应用约束后的节点映射logits:")
    print(constrained_node_logits)
    
    # 测试可行性检查
    print(f"\n测试可行性检查:")
    
    # 测试一个可行的映射
    feasible_mapping = torch.tensor([0, 0, 2])  # 虚拟节点0->物理节点0, 1->0, 2->2
    is_feasible = constraint_manager.check_mapping_feasibility(feasible_mapping, node_constraints)
    print(f"   映射 {feasible_mapping.tolist()}: {'可行' if is_feasible else '不可行'}")
    
    # 测试一个不可行的映射
    infeasible_mapping = torch.tensor([1, 1, 1])  # 虚拟节点2的资源需求超过物理节点1的可用资源
    is_feasible = constraint_manager.check_mapping_feasibility(infeasible_mapping, node_constraints)
    print(f"   映射 {infeasible_mapping.tolist()}: {'可行' if is_feasible else '不可行'}")
    
    print(f"\n✅ 约束管理器测试完成！")

if __name__ == "__main__":
    test_constraint_manager() 