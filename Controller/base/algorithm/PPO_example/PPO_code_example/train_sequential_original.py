#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用原始问题配置的Sequential PPO训练脚本
目标：确保任务能够成功完成，而不是严格遵循原始奖励函数
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os

from sequential_environment import SequentialNetworkSchedulerEnvironment
from sequential_agent import SimpleSequentialAgent
from original_problem_config import ORIGINAL_CONFIG

class OriginalSequentialPPOTrainer:
    """
    使用原始配置的Sequential PPO训练器
    """
    
    def __init__(self, 
                 env_config: dict = None,
                 agent_config: dict = None,
                 training_config: dict = None):
        
        # 使用原始问题配置
        original_config = ORIGINAL_CONFIG
        
        # 默认环境配置（基于原始问题）
        default_env_config = {
            'num_physical_nodes': original_config['physical_topology']['num_nodes'],
            'max_virtual_nodes': original_config['task_topology']['num_nodes_range'][1],
            'bandwidth_levels': original_config['action_space']['bandwidth_levels'],
            # 物理资源（使用原始范围）
            'physical_cpu_range': original_config['physical_topology']['cpu_range'],
            'physical_memory_range': original_config['physical_topology']['memory_range'],
            'physical_bandwidth_range': original_config['physical_topology']['bandwidth_range'],
            # 虚拟需求（使用原始范围）
            'virtual_cpu_range': original_config['task_topology']['cpu_demand_range'],
            'virtual_memory_range': original_config['task_topology']['memory_demand_range'],
            'virtual_bandwidth_range': (
                original_config['task_topology']['bandwidth_min_range'][0],
                original_config['task_topology']['bandwidth_max_range'][1]
            ),
            # 连接概率
            'physical_connectivity_prob': original_config['physical_topology']['connectivity_prob'],
            'virtual_connectivity_prob': original_config['task_topology']['connectivity_prob'],
            # 虚拟节点数范围
            'virtual_nodes_range': original_config['task_topology']['num_nodes_range'],
            'seed': original_config['environment']['seed']
        }
        
        # Agent配置（适配原始问题规模）
        default_agent_config = {
            'max_physical_nodes': original_config['physical_topology']['num_nodes'],
            'max_virtual_nodes': original_config['task_topology']['num_nodes_range'][1],
            'bandwidth_levels': original_config['action_space']['bandwidth_levels'],
            'hidden_dim': 128,  # 增大网络容量
            'lr': 5e-4  # 调整学习率
        }
        
        # 训练配置
        default_training_config = {
            'total_episodes': 3000,  # 增加训练轮数
            'batch_size': 64,  # 增大批次
            'update_frequency': 64,
            'print_frequency': 50,
            'save_frequency': 200,
            'warmup_episodes': 100,  # 热身阶段
            'curriculum_enabled': False,
            'curriculum_start': 500,  # Curriculum learning开始时机
        }
        
        # 合并配置
        self.env_config = {**default_env_config, **(env_config or {})}
        self.agent_config = {**default_agent_config, **(agent_config or {})}
        self.training_config = {**default_training_config, **(training_config or {})}
        
        # 创建环境和agent
        self.env = SequentialNetworkSchedulerEnvironment(**self.env_config)
        self.agent = SimpleSequentialAgent(**self.agent_config)
        
        # 训练统计
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'constraint_violations': [],
            'losses': [],
            'mapping_rewards': [],
            'bandwidth_rewards': [],
            'difficulty_levels': [],  # 记录难度变化
        }
        
        # 经验缓冲区
        self.experience_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
        
        print(f"🚀 原始配置Sequential PPO训练器初始化完成")
        print(f"   物理节点数: {self.env_config['num_physical_nodes']}")
        print(f"   虚拟节点范围: {self.env_config['virtual_nodes_range']}")
        print(f"   资源配置:")
        print(f"     物理CPU: {self.env_config['physical_cpu_range']}")
        print(f"     物理内存: {self.env_config['physical_memory_range']}")
        print(f"     虚拟CPU需求: {self.env_config['virtual_cpu_range']}")
        print(f"     虚拟内存需求: {self.env_config['virtual_memory_range']}")
    
    def collect_episode(self, episode_idx: int, temperature: float = 1.0):
        """收集一个episode的经验"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_violations = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_values = []
        episode_log_probs = []
        
        step_count = 0
        max_steps = 30  # 增加最大步数以适应更多节点
        
        while not done and step_count < max_steps:
            # 选择动作
            action, log_prob, value = self.agent.select_action(state, temperature)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录约束违反（无效动作）
            if not info.get('is_valid', True):
                episode_violations += 1

            # 存储经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_length += 1
            step_count += 1
        
        # 计算statistics
        constraint_violations = episode_violations
        mapping_rewards = self.env.episode_stats['mapping_rewards']
        bandwidth_rewards = self.env.episode_stats['bandwidth_rewards']
        
        # 正确判断成功：检查是否所有虚拟节点都成功映射
        all_nodes_mapped = all(node != -1 for node in self.env.partial_mapping) if self.env.partial_mapping is not None else False
        episode_success = all_nodes_mapped and episode_reward > 0
        
        episode_stats = {
            'reward': episode_reward,
            'length': episode_length,
            'constraint_violations': constraint_violations,
            'mapping_rewards': mapping_rewards,
            'bandwidth_rewards': bandwidth_rewards,
            'success_rate': 1.0 if episode_success else 0.0,
            'difficulty_level': getattr(self.env, 'difficulty_level', 1.0)
        }
        
        return {
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'dones': episode_dones,
            'values': episode_values,
            'log_probs': episode_log_probs,
            'stats': episode_stats
        }
    
    def update_agent(self):
        """更新agent"""
        if len(self.experience_buffer['states']) < self.training_config['batch_size']:
            return {}
        
        # 计算损失
        loss_dict = self.agent.calculate_loss(
            self.experience_buffer['states'],
            self.experience_buffer['actions'],
            self.experience_buffer['rewards'],
            self.experience_buffer['dones']
        )
        
        # 更新网络
        self.agent.update(loss_dict)
        
        # 清空缓冲区
        for key in self.experience_buffer:
            self.experience_buffer[key] = []
        
        return {key: value.item() if hasattr(value, 'item') else value for key, value in loss_dict.items()}
    
    def train(self):
        """主训练循环"""
        print(f"\n🎯 开始训练 原始配置Sequential PPO")
        print(f"   总episodes: {self.training_config['total_episodes']}")
        print(f"   更新频率: {self.training_config['update_frequency']}")
        print(f"   热身阶段: {self.training_config['warmup_episodes']} episodes")
        
        start_time = time.time()
        
        for episode in range(self.training_config['total_episodes']):
            # 热身阶段使用更高的温度
            if episode < self.training_config['warmup_episodes']:
                temperature = 1.5  # 更多探索
            else:
                # 温度衰减（探索策略）
                temperature = max(0.1, 1.0 - (episode - self.training_config['warmup_episodes']) / 
                                (self.training_config['total_episodes'] - self.training_config['warmup_episodes']))
            
            # Curriculum Learning：调整环境难度
            if hasattr(self.env, 'adjust_difficulty') and episode >= self.training_config['curriculum_start']:
                if episode % 50 == 0 and self.stats['success_rates']:
                    recent_success = np.mean(self.stats['success_rates'][-50:])
                    self.env.adjust_difficulty(recent_success)
            
            # 收集经验
            episode_data = self.collect_episode(episode, temperature)
            
            # 添加到缓冲区
            for key in ['states', 'actions', 'rewards', 'dones', 'values', 'log_probs']:
                self.experience_buffer[key].extend(episode_data[key])
            
            # 更新统计信息
            stats = episode_data['stats']
            self.stats['episode_rewards'].append(stats['reward'])
            self.stats['episode_lengths'].append(stats['length'])
            self.stats['success_rates'].append(stats['success_rate'])
            self.stats['constraint_violations'].append(stats['constraint_violations'])
            self.stats['mapping_rewards'].append(np.mean(stats['mapping_rewards']) if stats['mapping_rewards'] else 0)
            self.stats['bandwidth_rewards'].append(np.mean(stats['bandwidth_rewards']) if stats['bandwidth_rewards'] else 0)
            self.stats['difficulty_levels'].append(stats['difficulty_level'])
            
            # 定期更新agent
            if (episode + 1) % self.training_config['update_frequency'] == 0:
                loss_dict = self.update_agent()
                if loss_dict:
                    self.stats['losses'].append(loss_dict)
            
            # 定期打印进度
            if (episode + 1) % self.training_config['print_frequency'] == 0:
                self.print_progress(episode + 1)
            
            # 定期保存模型
            if (episode + 1) % self.training_config['save_frequency'] == 0:
                self.save_checkpoint(episode + 1)
        
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成！总用时: {total_time:.2f}s")
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return self.stats
    
    def print_progress(self, episode):
        """打印训练进度"""
        recent_window = 50
        recent_start = max(0, episode - recent_window)
        
        recent_rewards = self.stats['episode_rewards'][recent_start:]
        recent_success = self.stats['success_rates'][recent_start:]
        recent_violations = self.stats['constraint_violations'][recent_start:]
        recent_difficulty = self.stats['difficulty_levels'][recent_start:]
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_success = np.mean(recent_success) if recent_success else 0
        avg_violations = np.mean(recent_violations) if recent_violations else 0
        avg_difficulty = np.mean(recent_difficulty) if recent_difficulty else 1.0
        
        print(f"Episode {episode:4d} | "
              f"Reward: {avg_reward:6.3f} | "
              f"Success: {avg_success:5.1%} | "
              f"Violations: {avg_violations:4.1f} | "
              f"Difficulty: {avg_difficulty:.2f} | "
              f"Loss: {self.stats['losses'][-1]['total_loss']:.4f}" if self.stats['losses'] else "")
    
    def save_checkpoint(self, episode):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'stats': self.stats,
            'config': {
                'env_config': self.env_config,
                'agent_config': self.agent_config,
                'training_config': self.training_config
            }
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/original_sequential_ppo_episode_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 保存检查点: {checkpoint_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        fig.suptitle('Original Config Sequential PPO Training Progress', fontsize=16)
        
        # 滑动平均窗口
        window = 50
        
        # Episode奖励
        if self.stats['episode_rewards']:
            rewards = self.stats['episode_rewards']
            smoothed_rewards = self._smooth_curve(rewards, window)
            axes[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
            axes[0, 0].plot(smoothed_rewards, color='blue', label='Smoothed')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 成功率
        if self.stats['success_rates']:
            success_rates = self.stats['success_rates']
            smoothed_success = self._smooth_curve(success_rates, window)
            axes[0, 1].plot(success_rates, alpha=0.3, color='green', label='Raw')
            axes[0, 1].plot(smoothed_success, color='green', label='Smoothed')
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 约束违反
        if self.stats['constraint_violations']:
            violations = self.stats['constraint_violations']
            smoothed_violations = self._smooth_curve(violations, window)
            axes[0, 2].plot(violations, alpha=0.3, color='red', label='Raw')
            axes[0, 2].plot(smoothed_violations, color='red', label='Smoothed')
            axes[0, 2].set_title('Constraint Violations')
            axes[0, 2].set_ylabel('Violations')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # 难度等级
        if self.stats['difficulty_levels']:
            difficulty = self.stats['difficulty_levels']
            smoothed_difficulty = self._smooth_curve(difficulty, window)
            axes[0, 3].plot(difficulty, alpha=0.3, color='purple', label='Raw')
            axes[0, 3].plot(smoothed_difficulty, color='purple', label='Smoothed')
            axes[0, 3].set_title('Difficulty Level')
            axes[0, 3].set_ylabel('Level')
            axes[0, 3].legend()
            axes[0, 3].grid(True)
        
        # Episode长度
        if self.stats['episode_lengths']:
            lengths = self.stats['episode_lengths']
            smoothed_lengths = self._smooth_curve(lengths, window)
            axes[1, 0].plot(lengths, alpha=0.3, color='orange', label='Raw')
            axes[1, 0].plot(smoothed_lengths, color='orange', label='Smoothed')
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 映射奖励
        if self.stats['mapping_rewards']:
            mapping_rewards = self.stats['mapping_rewards']
            smoothed_mapping = self._smooth_curve(mapping_rewards, window)
            axes[1, 1].plot(mapping_rewards, alpha=0.3, color='cyan', label='Raw')
            axes[1, 1].plot(smoothed_mapping, color='cyan', label='Smoothed')
            axes[1, 1].set_title('Mapping Rewards')
            axes[1, 1].set_ylabel('Avg Mapping Reward')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # 带宽奖励
        if self.stats['bandwidth_rewards']:
            bandwidth_rewards = self.stats['bandwidth_rewards']
            smoothed_bandwidth = self._smooth_curve(bandwidth_rewards, window)
            axes[1, 2].plot(bandwidth_rewards, alpha=0.3, color='brown', label='Raw')
            axes[1, 2].plot(smoothed_bandwidth, color='brown', label='Smoothed')
            axes[1, 2].set_title('Bandwidth Rewards')
            axes[1, 2].set_ylabel('Avg Bandwidth Reward')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        # 损失曲线
        if self.stats['losses']:
            losses = [loss['total_loss'] for loss in self.stats['losses']]
            axes[1, 3].plot(losses, color='magenta')
            axes[1, 3].set_title('Training Loss')
            axes[1, 3].set_ylabel('Loss')
            axes[1, 3].set_xlabel('Update Step')
            axes[1, 3].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/original_sequential_ppo_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"📊 训练曲线已保存: plots/original_sequential_ppo_training_curves.png")
        
        # 在支持的环境中显示图片
        try:
            plt.show()
        except:
            print("📊 图片已保存，但无法显示（可能是无头环境）")
        finally:
            plt.close()
    
    def _smooth_curve(self, data, window):
        """计算滑动平均"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed


def main():
    """主函数"""
    print("🎮 原始配置Sequential PPO训练实验")
    print("=" * 60)
    
    # 创建训练器（使用原始配置）
    trainer = OriginalSequentialPPOTrainer()
    
    # 开始训练
    stats = trainer.train()
    
    # 分析结果
    print("\n" + "=" * 60)
    print("📈 训练结果分析:")
    print("=" * 60)
    
    # 计算不同阶段的统计
    stages = [
        ('前期(0-500)', 0, 500),
        ('中期(500-1500)', 500, 1500),
        ('后期(1500-3000)', 1500, 3000)
    ]
    
    for stage_name, start, end in stages:
        if len(stats['episode_rewards']) >= end:
            stage_rewards = stats['episode_rewards'][start:end]
            stage_success = stats['success_rates'][start:end]
            stage_violations = stats['constraint_violations'][start:end]
            
            print(f"\n{stage_name}:")
            print(f"  平均奖励: {np.mean(stage_rewards):.3f} ± {np.std(stage_rewards):.3f}")
            print(f"  成功率: {np.mean(stage_success):.1%}")
            print(f"  平均违反: {np.mean(stage_violations):.2f}")
    
    # 最终统计
    final_window = min(100, len(stats['episode_rewards']))
    if final_window > 0:
        print(f"\n最终{final_window}轮统计:")
        print(f"  平均奖励: {np.mean(stats['episode_rewards'][-final_window:]):.3f}")
        print(f"  成功率: {np.mean(stats['success_rates'][-final_window:]):.1%}")
        print(f"  平均episode长度: {np.mean(stats['episode_lengths'][-final_window:]):.1f}")
        print(f"  平均约束违反: {np.mean(stats['constraint_violations'][-final_window:]):.1f}")
        
        # 成功episode的统计
        last_episodes_success = [i for i, s in enumerate(stats['success_rates'][-final_window:]) if s > 0]
        if last_episodes_success:
            successful_rewards = [stats['episode_rewards'][-final_window + i] for i in last_episodes_success]
            print(f"\n成功episode统计:")
            print(f"  成功次数: {len(last_episodes_success)}/{final_window}")
            print(f"  成功时平均奖励: {np.mean(successful_rewards):.3f}")

if __name__ == "__main__":
    main()