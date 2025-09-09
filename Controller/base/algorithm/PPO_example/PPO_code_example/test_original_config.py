#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试原始问题配置的脚本
验证环境参数和奖励函数是否正确对齐原始问题定义
"""

import numpy as np
import torch
from two_stage_environment import TwoStageNetworkSchedulerEnvironment
from original_problem_config import get_two_stage_env_config, get_reward_weights
from original_reward import integrate_with_network_scheduler

def test_original_configuration():
    """测试原始配置的环境"""
    print("=" * 60)
    print("测试原始问题配置")
    print("=" * 60)
    
    # 获取原始配置
    env_config = get_two_stage_env_config()
    weights = get_reward_weights()
    
    print("\n📋 环境配置:")
    print(f"  物理节点数: {env_config['num_physical_nodes']}")
    print(f"  虚拟节点数范围: {env_config['virtual_nodes_range']}")
    print(f"  物理资源范围:")
    print(f"    CPU: {env_config['physical_cpu_range']}")
    print(f"    内存: {env_config['physical_memory_range']}")
    print(f"    带宽: {env_config['physical_bandwidth_range']}")
    print(f"  虚拟需求范围:")
    print(f"    CPU: {env_config['virtual_cpu_range']}")
    print(f"    内存: {env_config['virtual_memory_range']}")
    print(f"    带宽: {env_config['virtual_bandwidth_range']}")
    
    print(f"\n⚖️ 奖励权重:")
    print(f"  负载均衡权重: {weights['load_balance_weights']}")
    print(f"  全局权重: {weights['global_weights']}")
    
    # 创建环境
    env = TwoStageNetworkSchedulerEnvironment(**env_config)
    
    # 重置环境以初始化network_scheduler
    _ = env.reset()
    
    # 集成原始奖励函数
    if env.use_network_scheduler and env.network_scheduler is not None:
        integrate_with_network_scheduler(env.network_scheduler)
    
    print("\n" + "=" * 60)
    print("开始测试不同策略")
    print("=" * 60)
    
    # 测试多个episode
    strategies = {
        'random': lambda state, env: random_policy(state, env),
        'greedy': lambda state, env: greedy_policy(state, env),
        'balanced': lambda state, env: balanced_policy(state, env),
    }
    
    results = {name: [] for name in strategies}
    
    num_episodes = 10
    for episode in range(num_episodes):
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        for strategy_name, strategy_func in strategies.items():
            # 重置环境
            state = env.reset()
            
            # 获取动作
            mapping_action, bandwidth_action = strategy_func(state, env)
            
            # 执行动作
            next_state, reward, done, info = env.step(mapping_action, bandwidth_action)
            
            # 使用原始奖励函数
            if env.use_network_scheduler and hasattr(env.network_scheduler, 'calculate_original_reward'):
                original_reward = env.network_scheduler.calculate_original_reward(env.virtual_work_obj)
                components = env.network_scheduler.get_original_reward_components(env.virtual_work_obj)
                
                print(f"\n  {strategy_name.upper()} 策略:")
                print(f"    映射成功: {components['mapping_success']}")
                if components['mapping_success']:
                    print(f"    L (负载均衡度): {components['L']:.4f}")
                    print(f"    D_BW (带宽满足度): {components['D_BW']:.4f}")
                    print(f"    原始奖励: {original_reward:.4f}")
                else:
                    print(f"    映射失败率: {1 - components.get('mapped_ratio', 0):.2%}")
                
                results[strategy_name].append(original_reward)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    
    for strategy_name, rewards in results.items():
        if rewards:
            successful = [r for r in rewards if r > -0.5]
            success_rate = len(successful) / len(rewards)
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            print(f"\n{strategy_name.upper()} 策略:")
            print(f"  成功率: {success_rate:.1%}")
            print(f"  平均奖励: {avg_reward:.4f} ± {std_reward:.4f}")
            if successful:
                print(f"  成功时平均奖励: {np.mean(successful):.4f}")

def random_policy(state, env):
    """随机策略"""
    num_virtual_nodes = state['virtual_num_nodes']
    num_virtual_edges = state['virtual_edges'].shape[1]
    
    mapping_action = np.random.randint(0, env.num_physical_nodes, num_virtual_nodes)
    bandwidth_action = np.random.randint(0, env.bandwidth_levels, num_virtual_edges)
    
    return mapping_action, bandwidth_action

def greedy_policy(state, env):
    """贪心策略：选择资源最多的物理节点"""
    num_virtual_nodes = state['virtual_num_nodes']
    num_virtual_edges = state['virtual_edges'].shape[1]
    
    # 获取物理节点资源
    physical_features = state['physical_features'].numpy()
    
    # 计算每个物理节点的可用资源得分
    scores = []
    for i in range(env.num_physical_nodes):
        cpu = physical_features[i][0]
        memory = physical_features[i][1]
        cpu_usage = physical_features[i][2]
        memory_usage = physical_features[i][3]
        
        available_cpu = cpu * (1 - cpu_usage)
        available_memory = memory * (1 - memory_usage)
        
        score = available_cpu + available_memory
        scores.append(score)
    
    # 贪心分配：优先分配到资源最多的节点
    sorted_nodes = np.argsort(scores)[::-1]
    mapping_action = []
    
    for i in range(num_virtual_nodes):
        # 循环使用排序后的节点
        mapping_action.append(sorted_nodes[i % len(sorted_nodes)])
    
    # 带宽使用中等级别
    bandwidth_action = np.full(num_virtual_edges, env.bandwidth_levels // 2)
    
    return np.array(mapping_action), bandwidth_action

def balanced_policy(state, env):
    """均衡策略：尽量平均分配到不同物理节点"""
    num_virtual_nodes = state['virtual_num_nodes']
    num_virtual_edges = state['virtual_edges'].shape[1]
    
    # 均匀分配到不同物理节点
    mapping_action = []
    for i in range(num_virtual_nodes):
        mapping_action.append(i % env.num_physical_nodes)
    
    # 带宽使用较高级别
    bandwidth_action = np.full(num_virtual_edges, int(env.bandwidth_levels * 0.7))
    
    return np.array(mapping_action), bandwidth_action

if __name__ == "__main__":
    test_original_configuration()