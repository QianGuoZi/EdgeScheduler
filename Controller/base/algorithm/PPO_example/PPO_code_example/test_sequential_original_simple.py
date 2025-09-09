#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试原始配置的Sequential环境
不依赖matplotlib，避免numpy版本问题
"""

import torch
import numpy as np
import time
from sequential_environment import SequentialNetworkSchedulerEnvironment
from sequential_agent import SimpleSequentialAgent
from original_problem_config import ORIGINAL_CONFIG

def test_original_config():
    """测试原始配置的训练"""
    print("=" * 60)
    print("测试原始配置Sequential PPO")
    print("=" * 60)
    
    # 使用原始问题配置
    original_config = ORIGINAL_CONFIG
    
    # 环境配置
    env_config = {
        'num_physical_nodes': original_config['physical_topology']['num_nodes'],
        'max_virtual_nodes': original_config['task_topology']['num_nodes_range'][1],
        'bandwidth_levels': original_config['action_space']['bandwidth_levels'],
        'physical_cpu_range': original_config['physical_topology']['cpu_range'],
        'physical_memory_range': original_config['physical_topology']['memory_range'],
        'physical_bandwidth_range': original_config['physical_topology']['bandwidth_range'],
        'virtual_cpu_range': original_config['task_topology']['cpu_demand_range'],
        'virtual_memory_range': original_config['task_topology']['memory_demand_range'],
        'virtual_bandwidth_range': (
            original_config['task_topology']['bandwidth_min_range'][0],
            original_config['task_topology']['bandwidth_max_range'][1]
        ),
        'physical_connectivity_prob': original_config['physical_topology']['connectivity_prob'],
        'virtual_connectivity_prob': original_config['task_topology']['connectivity_prob'],
        'virtual_nodes_range': original_config['task_topology']['num_nodes_range'],
        'seed': 42
    }
    
    # Agent配置
    agent_config = {
        'max_physical_nodes': original_config['physical_topology']['num_nodes'],
        'max_virtual_nodes': original_config['task_topology']['num_nodes_range'][1],
        'bandwidth_levels': original_config['action_space']['bandwidth_levels'],
        'hidden_dim': 128,
        'lr': 5e-4
    }
    
    # 创建环境和agent
    env = SequentialNetworkSchedulerEnvironment(**env_config)
    agent = SimpleSequentialAgent(**agent_config)
    
    print(f"\n环境配置:")
    print(f"  物理节点数: {env_config['num_physical_nodes']}")
    print(f"  虚拟节点范围: {env_config['virtual_nodes_range']}")
    print(f"  物理资源: CPU{env_config['physical_cpu_range']}, Mem{env_config['physical_memory_range']}")
    print(f"  虚拟需求: CPU{env_config['virtual_cpu_range']}, Mem{env_config['virtual_memory_range']}")
    
    # 测试参数
    num_episodes = 500  # 短期测试
    batch_size = 32
    update_frequency = 32
    
    # 训练统计
    episode_rewards = []
    success_rates = []
    episode_lengths = []
    
    # 经验缓冲
    experience_buffer = {
        'states': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    
    print(f"\n开始训练 {num_episodes} episodes...")
    print("-" * 60)
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        
        # 温度衰减
        temperature = max(0.1, 1.0 - episode / num_episodes)
        
        # 收集一个episode
        step_count = 0
        max_steps = 30
        
        while not done and step_count < max_steps:
            # 选择动作
            action, log_prob, value = agent.select_action(state, temperature)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_length += 1
            step_count += 1
        
        # 判断成功
        all_nodes_mapped = all(node != -1 for node in env.partial_mapping) if env.partial_mapping is not None else False
        episode_success = all_nodes_mapped and episode_reward > 0
        
        # 记录统计
        episode_rewards.append(episode_reward)
        success_rates.append(1.0 if episode_success else 0.0)
        episode_lengths.append(episode_length)
        
        # 添加到缓冲区
        experience_buffer['states'].extend(episode_states)
        experience_buffer['actions'].extend(episode_actions)
        experience_buffer['rewards'].extend(episode_rewards)
        experience_buffer['dones'].extend(episode_dones)
        
        # 更新agent
        if (episode + 1) % update_frequency == 0 and len(experience_buffer['states']) >= batch_size:
            loss_dict = agent.calculate_loss(
                experience_buffer['states'],
                experience_buffer['actions'],
                experience_buffer['rewards'],
                experience_buffer['dones']
            )
            agent.update(loss_dict)
            
            # 清空缓冲区
            for key in experience_buffer:
                experience_buffer[key] = []
            
            loss_value = loss_dict['total_loss'].item() if hasattr(loss_dict['total_loss'], 'item') else loss_dict['total_loss']
        else:
            loss_value = 0
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            recent_success = success_rates[-50:]
            recent_lengths = episode_lengths[-50:]
            
            avg_reward = np.mean(recent_rewards)
            avg_success = np.mean(recent_success)
            avg_length = np.mean(recent_lengths)
            
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {avg_reward:6.3f} | "
                  f"Success: {avg_success:5.1%} | "
                  f"Length: {avg_length:4.1f} | "
                  f"Loss: {loss_value:.4f}")
    
    # 训练完成
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"\n✅ 训练完成！总用时: {total_time:.2f}秒")
    
    # 最终统计
    print("\n📊 最终统计 (最后100轮):")
    final_window = min(100, len(episode_rewards))
    if final_window > 0:
        final_rewards = episode_rewards[-final_window:]
        final_success = success_rates[-final_window:]
        final_lengths = episode_lengths[-final_window:]
        
        print(f"  平均奖励: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
        print(f"  成功率: {np.mean(final_success):.1%}")
        print(f"  平均长度: {np.mean(final_lengths):.1f}")
        
        # 成功episode分析
        successful_indices = [i for i, s in enumerate(final_success) if s > 0]
        if successful_indices:
            successful_rewards = [final_rewards[i] for i in successful_indices]
            print(f"\n  成功episode:")
            print(f"    数量: {len(successful_indices)}/{final_window}")
            print(f"    平均奖励: {np.mean(successful_rewards):.3f}")
        else:
            print(f"\n  ⚠️ 最后{final_window}轮没有成功的episode")
    
    # 对比不同阶段
    if len(episode_rewards) >= 300:
        print("\n📈 学习进展:")
        stages = [
            ("前100轮", 0, 100),
            ("中间轮次", 200, 300),
            ("最后100轮", -100, None)
        ]
        
        for name, start, end in stages:
            if end is None:
                stage_rewards = episode_rewards[start:]
                stage_success = success_rates[start:]
            else:
                stage_rewards = episode_rewards[start:end]
                stage_success = success_rates[start:end]
            
            print(f"  {name}: 奖励={np.mean(stage_rewards):.3f}, 成功率={np.mean(stage_success):.1%}")
    
    return {
        'episode_rewards': episode_rewards,
        'success_rates': success_rates,
        'episode_lengths': episode_lengths,
        'final_success_rate': np.mean(success_rates[-final_window:]) if final_window > 0 else 0
    }

if __name__ == "__main__":
    results = test_original_config()
    
    # 判断训练效果
    if results['final_success_rate'] > 0.3:
        print("\n✨ 训练效果良好，成功率达到30%以上")
    elif results['final_success_rate'] > 0.1:
        print("\n⚡ 训练有一定效果，但还需要更多训练")
    else:
        print("\n⚠️ 训练效果不理想，可能需要调整参数或增加训练轮数")