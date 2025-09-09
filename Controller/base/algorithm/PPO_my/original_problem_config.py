#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始问题定义的标准配置
基于define_problem.md的数学模型
"""

# 原始问题配置
ORIGINAL_CONFIG = {
    # 物理拓扑配置 (Physical Topology)
    'physical_topology': {
        'num_nodes': 10,  # |P| = 10个物理节点
        'cpu_range': (50, 100),  # CPU_P(p) 资源范围
        'memory_range': (50, 100),  # RAM_P(p) 资源范围  
        'bandwidth_range': (100, 1000),  # BW_R(r) 链路带宽范围
        'connectivity_prob': 0.3,  # 物理网络连接概率
        'initial_usage': (0.1, 0.5),  # 初始资源使用率范围
    },
    
    # 任务拓扑配置 (Task Topology)  
    # 'task_topology': {
    #     'num_nodes_range': (3, 6),  # |N| 任务节点数范围
    #     'cpu_demand_range': (10, 50),  # CPU_N(n) 需求范围
    #     'memory_demand_range': (20, 100),  # RAM_N(n) 需求范围
    #     'bandwidth_min_range': (10, 100),  # BW_min(v) 最小带宽需求
    #     'bandwidth_max_range': (100, 200),  # BW_max(v) 最大带宽需求
    #     'connectivity_prob': 0.4,  # 任务P2P网络连接概率
    # },
    'task_topology': {
        'num_nodes_range': (3, 8),  # |N| 任务节点数范围
        'cpu_demand_range': (10, 50),  # CPU_N(n) 需求范围
        'memory_demand_range': (10, 50),  # RAM_N(n) 需求范围
        'bandwidth_min_range': (10, 50),  # BW_min(v) 最小带宽需求
        'bandwidth_max_range': (50, 100),  # BW_max(v) 最大带宽需求
        'connectivity_prob': 0.4,  # 任务P2P网络连接概率
    },

    # 优化目标权重 (Optimization Weights)
    'optimization_weights': {
        # 负载均衡度 L 的权重
        'w1_cpu': 0.4,  # CPU负载权重
        'w2_memory': 0.4,  # 内存负载权重  
        'w3_bandwidth': 0.2,  # 带宽负载权重
        
        # 全局目标函数权重
        'gamma1_load_balance': 0.6,  # 负载均衡权重
        'gamma2_bandwidth_satisfaction': 0.4,  # 带宽满足度权重
    },
    
    # 动作空间配置
    'action_space': {
        'bandwidth_levels': 10,  # 带宽分配的离散等级数
    },
    
    # 环境配置
    'environment': {
        'max_steps': 1,  # 两阶段动作在一步内完成
        'seed': 42,  # 随机种子
    }
}