#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from two_stage_actor_design import TwoStagePPOAgent
from two_stage_environment import TwoStageNetworkSchedulerEnvironment
from constraint_manager import ConstraintManager

def test_constraint_integration():
    """测试约束管理器与智能体的集成"""
    print("🧪 测试约束管理器与智能体的集成")
    print("=" * 60)
    
    # 创建环境
    env = TwoStageNetworkSchedulerEnvironment(
        num_physical_nodes=4,
        max_virtual_nodes=3,
        bandwidth_levels=10,
        physical_cpu_range=(100, 200),
        physical_memory_range=(200, 400),
        virtual_cpu_range=(20, 60),
        virtual_memory_range=(40, 120),
        use_network_scheduler=True
    )
    
    print("✅ 环境创建成功")
    
    # 重置环境
    state = env.reset()
    print(f"✅ 环境重置成功")
    print(f"   物理节点数: {env.num_physical_nodes}")
    print(f"   虚拟节点数: {env.virtual_work['num_nodes']}")
    print(f"   带宽映射: {state['bandwidth_mapping']}")
    
    # 创建智能体
    agent = TwoStagePPOAgent(
        physical_node_dim=4,  # CPU, 内存, CPU使用率, 内存使用率
        virtual_node_dim=2,   # CPU需求, 内存需求
        num_physical_nodes=env.num_physical_nodes,
        max_virtual_nodes=env.max_virtual_nodes,
        bandwidth_levels=env.bandwidth_levels
    )
    
    print("✅ 智能体创建成功")
    
    # 测试约束生成
    print(f"\n🎯 测试约束生成:")
    
    # 生成节点映射约束
    node_constraints = agent.constraint_manager.generate_node_mapping_constraints(
        state['physical_features'], state['virtual_features'],
        state['physical_edge_index'], state['virtual_edge_index']
    )
    
    print(f"节点映射约束矩阵形状: {node_constraints.shape}")
    print(f"节点映射约束矩阵:")
    print(node_constraints)
    
    # 统计节点映射约束
    node_stats = agent.constraint_manager.get_feasible_actions_count(node_constraints)
    print(f"\n节点映射约束统计:")
    print(f"   总可行映射: {node_stats['total_feasible']}")
    print(f"   总可能映射: {node_stats['total_possible']}")
    print(f"   每个虚拟节点的可行物理节点数: {node_stats['feasible_per_node']}")
    
    # 生成带宽约束
    bandwidth_constraints = agent.constraint_manager.generate_bandwidth_constraints(
        state['virtual_edge_attr'], state['bandwidth_mapping']
    )
    
    print(f"\n带宽约束矩阵形状: {bandwidth_constraints.shape}")
    print(f"带宽约束矩阵:")
    print(bandwidth_constraints)
    
    # 统计带宽约束
    bandwidth_stats = agent.constraint_manager.get_feasible_actions_count(bandwidth_constraints)
    print(f"\n带宽约束统计:")
    print(f"   总可行带宽等级: {bandwidth_stats['total_feasible']}")
    print(f"   总可能带宽等级: {bandwidth_stats['total_possible']}")
    print(f"   每个链路的可行带宽等级数: {bandwidth_stats['feasible_per_link']}")
    
    # 测试动作选择
    print(f"\n🎯 测试约束动作选择:")
    
    # 选择动作（包含约束）
    mapping_action, bandwidth_action, mapping_log_prob, bandwidth_log_prob, value, link_indices = agent.select_actions(state)
    
    print(f"映射动作: {mapping_action}")
    print(f"带宽动作: {bandwidth_action}")
    print(f"映射log概率: {mapping_log_prob}")
    print(f"带宽log概率: {bandwidth_log_prob}")
    print(f"状态价值: {value:.4f}")
    
    # 验证动作可行性
    print(f"\n🔍 验证动作可行性:")
    
    # 检查映射动作可行性
    mapping_feasible = agent.constraint_manager.check_mapping_feasibility(
        torch.tensor(mapping_action), node_constraints
    )
    print(f"   映射动作可行性: {'✅ 可行' if mapping_feasible else '❌ 不可行'}")
    
    # 检查带宽动作可行性
    if len(bandwidth_action) > 0:
        bandwidth_feasible = agent.constraint_manager.check_bandwidth_feasibility(
            torch.tensor(bandwidth_action), bandwidth_constraints
        )
        print(f"   带宽动作可行性: {'✅ 可行' if bandwidth_feasible else '❌ 不可行'}")
    else:
        print(f"   带宽动作可行性: ✅ 无链路，无需检查")
    
    # 执行动作
    print(f"\n🚀 执行动作:")
    next_state, reward, done, info = env.step(mapping_action, bandwidth_action)
    
    print(f"   奖励: {reward:.4f}")
    print(f"   是否结束: {done}")
    print(f"   是否有效: {info['is_valid']}")
    print(f"   约束违反: {info['constraint_violations']}")
    
    # 测试多次动作选择，观察约束效果
    print(f"\n🔄 测试多次动作选择:")
    
    feasible_count = 0
    total_tests = 10
    
    for i in range(total_tests):
        # 重置环境
        state = env.reset()
        
        # 选择动作
        mapping_action, bandwidth_action, _, _, _, _ = agent.select_actions(state)
        
        # 执行动作
        _, reward, _, info = env.step(mapping_action, bandwidth_action)
        
        if info['is_valid']:
            feasible_count += 1
        
        print(f"   测试{i+1}: 奖励={reward:.3f}, 有效={info['is_valid']}")
    
    print(f"\n📊 约束效果统计:")
    print(f"   总测试次数: {total_tests}")
    print(f"   有效动作次数: {feasible_count}")
    print(f"   有效动作率: {feasible_count/total_tests:.2%}")
    
    print(f"\n✅ 约束管理器与智能体集成测试完成！")

def test_constraint_effectiveness():
    """测试约束的有效性"""
    print("\n🧪 测试约束的有效性")
    print("=" * 60)
    
    # 创建约束管理器
    constraint_manager = ConstraintManager(bandwidth_levels=10)
    
    # 创建测试数据：资源不足的情况
    physical_features = torch.tensor([
        [50, 100, 0.8, 0.9],   # 节点0: 资源很少，使用率很高
        [100, 200, 0.5, 0.6],  # 节点1: 资源中等
        [200, 400, 0.2, 0.3],  # 节点2: 资源充足
    ])
    
    virtual_features = torch.tensor([
        [30, 50],   # 虚拟节点0: 需要30CPU, 50内存
        [40, 80],   # 虚拟节点1: 需要40CPU, 80内存
        [60, 120],  # 虚拟节点2: 需要60CPU, 120内存
    ])
    
    print("物理节点可用资源:")
    for i, features in enumerate(physical_features):
        cpu, memory, cpu_usage, memory_usage = features
        available_cpu = cpu * (1 - cpu_usage)
        available_memory = memory * (1 - memory_usage)
        print(f"   节点{i}: 可用CPU={available_cpu:.1f}, 可用内存={available_memory:.1f}")
    
    print("\n虚拟节点需求:")
    for i, features in enumerate(virtual_features):
        cpu, memory = features
        print(f"   虚拟节点{i}: 需要CPU={cpu}, 需要内存={memory}")
    
    # 生成约束
    physical_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    virtual_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    
    node_constraints = constraint_manager.generate_node_mapping_constraints(
        physical_features, virtual_features, physical_edge_index, virtual_edge_index
    )
    
    print(f"\n节点映射约束矩阵:")
    print(node_constraints)
    
    # 分析约束效果
    print(f"\n约束分析:")
    for virtual_idx in range(virtual_features.size(0)):
        feasible_nodes = torch.where(node_constraints[virtual_idx] == 1.0)[0]
        print(f"   虚拟节点{virtual_idx}可映射到物理节点: {feasible_nodes.tolist()}")
    
    # 测试带宽约束
    virtual_edge_features = torch.tensor([
        [20, 40],   # 链路0: 最小20, 最大40
        [50, 100],  # 链路1: 最小50, 最大100
    ])
    
    bandwidth_mapping = {i: 10 + i * 10 for i in range(10)}  # 10, 20, 30, ..., 100
    
    bandwidth_constraints = constraint_manager.generate_bandwidth_constraints(
        virtual_edge_features, bandwidth_mapping
    )
    
    print(f"\n带宽约束矩阵:")
    print(bandwidth_constraints)
    
    # 分析带宽约束效果
    print(f"\n带宽约束分析:")
    for link_idx in range(virtual_edge_features.size(0)):
        min_bw, max_bw = virtual_edge_features[link_idx]
        feasible_levels = torch.where(bandwidth_constraints[link_idx] == 1.0)[0]
        feasible_bandwidths = [bandwidth_mapping[level.item()] for level in feasible_levels]
        print(f"   链路{link_idx} (需求{min_bw}-{max_bw}): 可行等级{feasible_levels.tolist()}, 对应带宽{feasible_bandwidths}")
    
    print(f"\n✅ 约束有效性测试完成！")

if __name__ == "__main__":
    test_constraint_integration()
    test_constraint_effectiveness() 