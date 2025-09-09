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
        'num_nodes': 10,  # |P| = 10个物理节点（参考SB3配置）
        'cpu_range': (50, 200),  # CPU_P(p) 资源范围
        'memory_range': (100, 400),  # RAM_P(p) 资源范围  
        'bandwidth_range': (100, 1000),  # BW_R(r) 链路带宽范围
        'connectivity_prob': 0.3,  # 物理网络连接概率
        'initial_usage': (0.1, 0.5),  # 初始资源使用率范围
    },
    
    # 任务拓扑配置 (Task Topology)  
    'task_topology': {
        'num_nodes_range': (3, 8),  # |N| 任务节点数范围
        'cpu_demand_range': (10, 50),  # CPU_N(n) 需求范围
        'memory_demand_range': (20, 100),  # RAM_N(n) 需求范围
        'bandwidth_min_range': (10, 100),  # BW_min(v) 最小带宽需求
        'bandwidth_max_range': (100, 200),  # BW_max(v) 最大带宽需求
        'connectivity_prob': 0.4,  # 任务P2P网络连接概率
    },
    
    # 优化目标权重 (Optimization Weights)
    'optimization_weights': {
        # 负载均衡度 L 的权重
        'w1_cpu': 0.4,  # CPU负载权重
        'w2_memory': 0.4,  # 内存负载权重  
        'w3_bandwidth': 0.2,  # 带宽负载权重
        
        # 全局目标函数权重
        'gamma1_load_balance': 0.7,  # 负载均衡权重
        'gamma2_bandwidth_satisfaction': 0.3,  # 带宽满足度权重
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

def get_two_stage_env_config():
    """
    转换为TwoStageEnvironment的配置格式
    """
    config = ORIGINAL_CONFIG
    return {
        'num_physical_nodes': config['physical_topology']['num_nodes'],
        'max_virtual_nodes': config['task_topology']['num_nodes_range'][1],
        'bandwidth_levels': config['action_space']['bandwidth_levels'],
        
        # 物理资源范围
        'physical_cpu_range': config['physical_topology']['cpu_range'],
        'physical_memory_range': config['physical_topology']['memory_range'],
        'physical_bandwidth_range': config['physical_topology']['bandwidth_range'],
        
        # 虚拟需求范围
        'virtual_cpu_range': config['task_topology']['cpu_demand_range'],
        'virtual_memory_range': config['task_topology']['memory_demand_range'],
        'virtual_bandwidth_range': (
            config['task_topology']['bandwidth_min_range'][0],
            config['task_topology']['bandwidth_max_range'][1]
        ),
        
        # 连接概率
        'physical_connectivity_prob': config['physical_topology']['connectivity_prob'],
        'virtual_connectivity_prob': config['task_topology']['connectivity_prob'],
        
        # 虚拟节点数范围
        'virtual_nodes_range': config['task_topology']['num_nodes_range'],
        
        # 其他
        'use_network_scheduler': True,
        'seed': config['environment']['seed'],
    }

def get_reward_weights():
    """
    获取奖励函数权重配置
    """
    weights = ORIGINAL_CONFIG['optimization_weights']
    return {
        'load_balance_weights': {
            'cpu': weights['w1_cpu'],
            'memory': weights['w2_memory'],
            'bandwidth': weights['w3_bandwidth'],
        },
        'global_weights': {
            'load_balance': weights['gamma1_load_balance'],
            'bandwidth_satisfaction': weights['gamma2_bandwidth_satisfaction'],
        }
    }

def get_sequential_env_config():
    """
    转换为SequentialEnvironment的配置格式
    保持与原始问题一致但增加难度
    """
    config = ORIGINAL_CONFIG
    return {
        'num_physical_nodes': config['physical_topology']['num_nodes'],
        'max_virtual_nodes': config['task_topology']['num_nodes_range'][1],
        'bandwidth_levels': config['action_space']['bandwidth_levels'],
        
        # 物理资源范围 - 适当减少以增加难度
        'physical_cpu_range': (40, 150),  # 减少物理资源
        'physical_memory_range': (80, 300),
        'physical_bandwidth_range': (80, 800),
        
        # 虚拟需求范围 - 适当增加以增加难度
        'virtual_cpu_range': (15, 60),  # 增加需求
        'virtual_memory_range': (30, 120),
        'virtual_bandwidth_range': (15, 250),
        
        # 连接概率
        'physical_connectivity_prob': config['physical_topology']['connectivity_prob'],
        'virtual_connectivity_prob': config['task_topology']['connectivity_prob'],
        
        # 虚拟节点数范围
        'virtual_nodes_range': config['task_topology']['num_nodes_range'],
        
        # 其他
        'seed': config['environment']['seed'],
    }

if __name__ == "__main__":
    print("原始问题配置:")
    print("-" * 50)
    
    env_config = get_two_stage_env_config()
    print("\nTwoStageEnvironment配置:")
    for key, value in env_config.items():
        print(f"  {key}: {value}")
    
    weights = get_reward_weights()
    print("\n奖励权重配置:")
    print(f"  负载均衡权重: {weights['load_balance_weights']}")
    print(f"  全局权重: {weights['global_weights']}")