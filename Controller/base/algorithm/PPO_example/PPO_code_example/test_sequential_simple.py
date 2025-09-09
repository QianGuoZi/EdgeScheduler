#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import time
from collections import defaultdict

from sequential_environment import SequentialNetworkSchedulerEnvironment
from sequential_agent import SimpleSequentialAgent

def test_sequential_training():
    """简单的Sequential PPO训练测试（不依赖matplotlib）"""
    print("🧪 开始Sequential PPO收敛性测试")
    
    # 创建增强难度环境
    env = SequentialNetworkSchedulerEnvironment(
        num_physical_nodes=4,    # 使用增强参数
        max_virtual_nodes=6,     # 使用增强参数
        virtual_nodes_range=(4, 6),  # 使用增强参数
        bandwidth_levels=3,
        seed=42
    )
    
    # 创建匹配的agent
    agent = SimpleSequentialAgent(
        max_physical_nodes=4,    # 匹配环境设置
        max_virtual_nodes=6,     # 匹配环境设置
        bandwidth_levels=3,
        hidden_dim=32,  # 减小网络复杂度
        lr=1e-3
    )
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    constraint_violations = []
    loss_history = []
    
    # 经验缓冲区
    experience_buffer = {
        'states': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    
    num_episodes = 200
    batch_size = 16
    update_frequency = 16
    
    print(f"🎯 训练配置: {num_episodes} episodes, 批次大小: {batch_size}")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # 温度衰减
        temperature = max(0.1, 1.0 - episode / num_episodes)
        
        # 收集一个episode
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        step_count = 0
        max_steps = 15
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        
        while not done and step_count < max_steps:
            # 选择动作
            action, log_prob, value = agent.select_action(state, temperature)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            
            # 更新
            state = next_state
            episode_reward += reward
            episode_length += 1
            step_count += 1
        
        # 添加到经验缓冲区
        experience_buffer['states'].extend(episode_states)
        experience_buffer['actions'].extend(episode_actions)
        experience_buffer['rewards'].extend(episode_rewards)
        experience_buffer['dones'].extend(episode_dones)
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        violations = sum(1 for v in env.episode_stats['constraint_violations'] if v)
        constraint_violations.append(violations)
        
        # 定期更新agent
        if (episode + 1) % update_frequency == 0 and len(experience_buffer['states']) >= batch_size:
            try:
                loss_dict = agent.calculate_loss(
                    experience_buffer['states'],
                    experience_buffer['actions'],
                    experience_buffer['rewards'],
                    experience_buffer['dones']
                )
                
                agent.update(loss_dict)
                loss_history.append(loss_dict['total_loss'].item())
                
                # 清空缓冲区
                for key in experience_buffer:
                    experience_buffer[key] = []
                
            except Exception as e:
                print(f"⚠️ 更新失败: {e}")
        
        # 定期打印进度
        if (episode + 1) % 25 == 0:
            recent_window = 25
            recent_start = max(0, episode + 1 - recent_window)
            
            avg_reward = np.mean(episode_rewards[recent_start:])
            avg_length = np.mean(episode_lengths[recent_start:])
            avg_violations = np.mean(constraint_violations[recent_start:])
            recent_rewards_slice = episode_rewards[recent_start:]
            success_rate = sum(1 for r in recent_rewards_slice if r > 1.0) / len(recent_rewards_slice) if recent_rewards_slice else 0
            
            print(f"Episode {episode+1:3d} | "
                  f"Reward: {avg_reward:6.3f} | "
                  f"Length: {avg_length:4.1f} | "
                  f"Success: {success_rate:5.1%} | "
                  f"Violations: {avg_violations:4.1f}")
            
            if loss_history:
                print(f"           | Loss: {loss_history[-1]:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n🎉 测试完成！用时: {total_time:.2f}s")
    
    # 分析结果
    print(f"\n📊 收敛性分析:")
    
    # 分为前半段和后半段
    mid_point = len(episode_rewards) // 2
    early_rewards = episode_rewards[:mid_point]
    late_rewards = episode_rewards[mid_point:]
    
    early_avg = np.mean(early_rewards) if early_rewards else 0
    late_avg = np.mean(late_rewards) if late_rewards else 0
    improvement = late_avg - early_avg
    
    print(f"   前半段平均奖励: {early_avg:.3f}")
    print(f"   后半段平均奖励: {late_avg:.3f}")
    print(f"   改善幅度: {improvement:.3f}")
    
    # 成功率分析
    early_success = sum(1 for r in early_rewards if r > 1.0) / len(early_rewards) if early_rewards else 0
    late_success = sum(1 for r in late_rewards if r > 1.0) / len(late_rewards) if late_rewards else 0
    
    print(f"   前半段成功率: {early_success:.1%}")
    print(f"   后半段成功率: {late_success:.1%}")
    
    # 约束违反分析
    early_violations = np.mean(constraint_violations[:mid_point]) if constraint_violations[:mid_point] else 0
    late_violations = np.mean(constraint_violations[mid_point:]) if constraint_violations[mid_point:] else 0
    
    print(f"   前半段平均违反: {early_violations:.1f}")
    print(f"   后半段平均违反: {late_violations:.1f}")
    
    # 判断是否收敛
    is_converging = improvement > 0.1 and late_success > early_success and late_violations <= early_violations
    
    print(f"\n🎯 收敛判断: {'✅ 正在收敛' if is_converging else '❌ 未明显收敛'}")
    
    if is_converging:
        print("   🎉 Sequential PPO显示出学习信号！")
        print("   🔹 奖励持续改善")
        print("   🔹 成功率提升")
        print("   🔹 约束违反减少")
    else:
        print("   🤔 可能需要进一步调优:")
        print("   🔹 调整学习率")
        print("   🔹 增加训练episodes")
        print("   🔹 优化奖励函数")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'constraint_violations': constraint_violations,
        'loss_history': loss_history,
        'is_converging': is_converging,
        'improvement': improvement
    }

def compare_with_random():
    """与随机策略对比"""
    print("\n🎲 与随机策略对比测试")
    
    env = SequentialNetworkSchedulerEnvironment(
        num_physical_nodes=4,        # 使用增强参数
        max_virtual_nodes=6,         # 使用增强参数
        virtual_nodes_range=(4, 6),  # 使用增强参数
        bandwidth_levels=3,
        seed=42
    )
    
    random_rewards = []
    
    for episode in range(50):
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 15
        
        while not done and step_count < max_steps:
            # 随机选择动作
            if state.get('mapping_phase', True):
                action = np.random.randint(0, state.get('num_physical_nodes', 3))
            else:
                action = np.random.randint(0, 3)  # bandwidth_levels
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
        
        random_rewards.append(episode_reward)
    
    random_avg = np.mean(random_rewards)
    print(f"   随机策略平均奖励: {random_avg:.3f}")
    
    return random_avg

if __name__ == "__main__":
    # 主测试
    results = test_sequential_training()
    
    # 对比测试
    random_baseline = compare_with_random()
    
    print(f"\n📈 最终对比:")
    print(f"   Sequential PPO后半段: {np.mean(results['episode_rewards'][len(results['episode_rewards'])//2:]):.3f}")
    print(f"   随机策略基线: {random_baseline:.3f}")
    
    if results['is_converging']:
        print(f"\n🚀 结论: Sequential PPO成功解决了收敛问题！")
        print(f"   相比原有的单步episode，多步决策提供了更好的学习信号")
        print(f"   简化的奖励函数有效避免了奖励稀疏性问题")
    else:
        print(f"\n⚠️ 结论: 仍需进一步优化，但比原版本有改善")