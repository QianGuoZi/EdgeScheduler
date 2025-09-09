#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2C (Advantage Actor-Critic) 训练脚本
用于网络调度任务的强化学习训练
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import os
from datetime import datetime
import uuid
import json

from a2c_environment import A2CNetworkEnvironment
from a2c_agent import A2CAgent
from original_problem_config import ORIGINAL_CONFIG

class A2CTrainer:
    """
    A2C训练器
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
            # 物理资源（使用原始范围）
            'physical_cpu_range': original_config['physical_topology']['cpu_range'],
            'physical_memory_range': original_config['physical_topology']['memory_range'],
            'physical_bandwidth_range': original_config['physical_topology']['bandwidth_range'],
            # 任务需求（使用原始范围）
            'task_cpu_range': original_config['task_topology']['cpu_demand_range'],
            'task_memory_range': original_config['task_topology']['memory_demand_range'],
            'task_bandwidth_range': (
                original_config['task_topology']['bandwidth_min_range'][0],
                original_config['task_topology']['bandwidth_max_range'][1]
            ),
            # 连接概率
            'physical_connectivity_prob': original_config['physical_topology']['connectivity_prob'],
            'task_connectivity_prob': original_config['task_topology']['connectivity_prob'],
            # 任务节点数范围
            'task_nodes_range': original_config['task_topology']['num_nodes_range'],
            # 物理资源初始使用率
            'initial_usage_range': original_config['physical_topology']['initial_usage'],
            'seed': original_config['environment']['seed']
        }
        
        # Agent配置
        default_agent_config = {
            'num_physical_nodes': original_config['physical_topology']['num_nodes'],
            'hidden_dim': 128,  # 减小网络容量
            'lr': 1e-5         # 降低学习率
        }
        
        # 训练配置
        default_training_config = {
            'total_episodes': 2000,  
            'batch_size': 32,        
            'update_frequency': 10,   
            'print_frequency': 50,   
            'save_frequency': 200,
            'gamma': 0.99,           
            'value_coef': 0.5,
            'entropy_coef': 0.01,    
            'max_grad_norm': 1.0,    
        }
        
        # 合并配置
        self.env_config = {**default_env_config, **(env_config or {})}
        self.agent_config = {**default_agent_config, **(agent_config or {})}
        self.training_config = {**default_training_config, **(training_config or {})}
        
        # 创建环境和智能体
        self.env = A2CNetworkEnvironment(**self.env_config)
        self.agent = A2CAgent(**self.agent_config)
        
        # 训练统计
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'acceptance_rates': [],
            'load_balance_rewards': [],
            'rejection_penalties': [],
            'bandwidth_success_rates': [],
            'losses': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropy_losses': []
        }
        
        # 经验缓冲区（用于批量更新）
        self.experience_buffer = []
        
        # 创建输出目录
        self.setup_output_directories()
        
        print(f"🚀 A2CTrainer初始化完成")
        print(f"   环境: {self.env_config['num_physical_nodes']}个物理节点")
        print(f"   任务范围: {self.env_config['task_nodes_range']}")
        print(f"   智能体: 隐藏维度{self.agent_config['hidden_dim']}, 学习率{self.agent_config['lr']}")
        print(f"   训练: {self.training_config['total_episodes']}轮, 批次大小{self.training_config['batch_size']}")
    
    def setup_output_directories(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"a2c_{timestamp}_{str(uuid.uuid4())[:8]}"
        
        self.output_dir = f"a2c_results/{self.run_id}"
        self.checkpoint_dir = f"{self.output_dir}/checkpoints"
        self.plots_dir = f"{self.output_dir}/plots"
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f"📁 输出目录: {self.output_dir}")
    
    def collect_episode(self):
        """收集一个episode的经验"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_experiences = []
        
        done = False
        while not done:
            # 获取当前状态
            physical_resources = state['physical_resources']
            task_requirements = state['task_requirements']
            valid_actions = state['valid_actions']
            
            # 选择动作
            action, log_prob, value = self.agent.select_action(
                physical_resources, task_requirements, valid_actions
            )
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            experience = {
                'physical_resources': physical_resources,
                'task_requirements': task_requirements,
                'action': action,
                'reward': reward,
                'value': value,
                'log_prob': log_prob,
                'done': done,
                'info': info
            }
            episode_experiences.append(experience)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        return {
            'experiences': episode_experiences,
            'total_reward': episode_reward,
            'length': episode_length,
            'final_info': info
        }
    
    def update_agent(self):
        """更新智能体"""
        if len(self.experience_buffer) < self.training_config['batch_size']:
            return None
        
        # 准备批次数据
        physical_resources_batch = []
        task_requirements_batch = []
        actions_batch = []
        rewards_batch = []
        dones_batch = []
        
        for exp in self.experience_buffer:
            physical_resources_batch.append(exp['physical_resources'])
            task_requirements_batch.append(exp['task_requirements'])
            actions_batch.append(exp['action'])
            rewards_batch.append(exp['reward'])
            dones_batch.append(exp['done'])
        
        # 转换为张量
        actions_batch = torch.tensor(actions_batch, dtype=torch.long)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
        dones_batch = torch.tensor(dones_batch, dtype=torch.bool)
        
        # 计算损失
        loss_dict = self.agent.calculate_loss(
            physical_resources_batch=physical_resources_batch,
            task_requirements_batch=task_requirements_batch,
            actions_batch=actions_batch,
            rewards_batch=rewards_batch,
            dones_batch=dones_batch,
            gamma=self.training_config['gamma'],
            value_coef=self.training_config['value_coef'],
            entropy_coef=self.training_config['entropy_coef']
        )
        
        # 更新网络
        self.agent.update(loss_dict)
        
        # 清空缓冲区
        self.experience_buffer = []
        
        return loss_dict
    
    def train(self):
        """主训练循环"""
        print(f"🎯 开始训练A2C算法")
        print(f"总训练轮数: {self.training_config['total_episodes']}")
        print("-" * 80)
        
        best_acceptance_rate = 0.0
        recent_rewards = deque(maxlen=100)
        recent_acceptance_rates = deque(maxlen=100)
        
        for episode in range(self.training_config['total_episodes']):
            episode_start_time = time.time()
            
            # 收集episode经验
            episode_data = self.collect_episode()
            
            # 添加到经验缓冲区
            for exp in episode_data['experiences']:
                self.experience_buffer.append(exp)
            
            # 更新统计信息
            episode_reward = episode_data['total_reward']
            episode_length = episode_data['length']
            final_info = episode_data['final_info']
            
            # 计算接受率
            env_stats = self.env.episode_stats
            acceptance_rate = env_stats['accepted_tasks'] / max(1, env_stats['total_tasks'])
            bandwidth_success = env_stats['final_bandwidth_success']
            
            # 记录统计信息
            self.stats['episode_rewards'].append(episode_reward)
            self.stats['episode_lengths'].append(episode_length)
            self.stats['acceptance_rates'].append(acceptance_rate)
            self.stats['bandwidth_success_rates'].append(float(bandwidth_success))
            
            recent_rewards.append(episode_reward)
            recent_acceptance_rates.append(acceptance_rate)
            
            # 记录详细奖励信息
            if env_stats['load_balance_rewards']:
                avg_load_balance = np.mean(env_stats['load_balance_rewards'])
                self.stats['load_balance_rewards'].append(avg_load_balance)
            
            if env_stats['rejection_penalties']:
                avg_rejection_penalty = np.mean(env_stats['rejection_penalties'])
                self.stats['rejection_penalties'].append(avg_rejection_penalty)
            
            # 更新智能体
            loss_dict = None
            if (episode + 1) % self.training_config['update_frequency'] == 0:
                loss_dict = self.update_agent()
                if loss_dict:
                    self.stats['losses'].append(loss_dict['total_loss'].item())
                    self.stats['actor_losses'].append(loss_dict['actor_loss'].item())
                    self.stats['critic_losses'].append(loss_dict['critic_loss'].item())
                    self.stats['entropy_losses'].append(loss_dict['entropy_loss'].item())
            
            # 打印进度
            if (episode + 1) % self.training_config['print_frequency'] == 0:
                episode_time = time.time() - episode_start_time
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                avg_acceptance = np.mean(recent_acceptance_rates) if recent_acceptance_rates else 0.0
                
                print(f"Episode {episode + 1:4d}/{self.training_config['total_episodes']}")
                print(f"  奖励: {episode_reward:8.3f} | 平均: {avg_reward:8.3f}")
                print(f"  接受率: {acceptance_rate:6.1%} | 平均: {avg_acceptance:6.1%}")
                print(f"  长度: {episode_length:3d} | 带宽成功: {'✓' if bandwidth_success else '✗'}")
                print(f"  时间: {episode_time:6.2f}s", end="")
                
                if loss_dict:
                    print(f" | 损失: {loss_dict['total_loss'].item():.4f}", end="")
                
                print()
                
                # 检查是否需要保存最佳模型
                if avg_acceptance > best_acceptance_rate:
                    best_acceptance_rate = avg_acceptance
                    self.save_best_model(episode, avg_acceptance, avg_reward)
            
            # 定期保存检查点
            if (episode + 1) % self.training_config['save_frequency'] == 0:
                self.save_checkpoint(episode)
                self.plot_training_curves()
        
        print(f"🎉 训练完成！最佳接受率: {best_acceptance_rate:.1%}")
        
        # 最终保存
        self.save_checkpoint(self.training_config['total_episodes'] - 1, final=True)
        self.plot_training_curves()
        self.save_training_summary()
    
    def save_best_model(self, episode, acceptance_rate, avg_reward):
        """保存最佳模型"""
        model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.agent.save_checkpoint(model_path)
        
        # 保存最佳模型信息
        best_info = {
            'episode': episode,
            'acceptance_rate': acceptance_rate,
            'avg_reward': avg_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = os.path.join(self.checkpoint_dir, "best_model_info.json")
        with open(info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        print(f"💾 新的最佳模型已保存 (接受率: {acceptance_rate:.1%})")
    
    def save_checkpoint(self, episode, final=False):
        """保存训练检查点"""
        suffix = "final" if final else f"ep_{episode + 1}"
        model_path = os.path.join(self.checkpoint_dir, f"model_{suffix}.pth")
        self.agent.save_checkpoint(model_path)
        
        # 保存训练统计
        stats_path = os.path.join(self.checkpoint_dir, f"stats_{suffix}.npz")
        # 将统计数据转换为numpy数组
        stats_to_save = {}
        for key, value in self.stats.items():
            if value:  # 只保存非空的统计数据
                stats_to_save[key] = np.array(value)
        
        np.savez(stats_path, **stats_to_save)
        
        if final:
            print(f"💾 最终检查点已保存")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.stats['episode_rewards']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'A2C训练曲线 - {self.run_id}', fontsize=14)
        
        episodes = range(len(self.stats['episode_rewards']))
        
        # 奖励曲线
        axes[0, 0].plot(episodes, self.stats['episode_rewards'], alpha=0.3, color='blue')
        if len(self.stats['episode_rewards']) > 10:
            window = min(50, len(self.stats['episode_rewards']) // 10)
            smoothed = np.convolve(self.stats['episode_rewards'], np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.stats['episode_rewards'])), smoothed, color='red', linewidth=2)
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # 接受率曲线
        if self.stats['acceptance_rates']:
            axes[0, 1].plot(episodes, self.stats['acceptance_rates'], color='green', linewidth=2)
            axes[0, 1].set_title('Task Acceptance Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Acceptance Rate')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True)
        
        # Episode长度
        axes[0, 2].plot(episodes, self.stats['episode_lengths'], alpha=0.5, color='purple')
        if len(self.stats['episode_lengths']) > 10:
            window = min(50, len(self.stats['episode_lengths']) // 10)
            smoothed = np.convolve(self.stats['episode_lengths'], np.ones(window)/window, mode='valid')
            axes[0, 2].plot(range(window-1, len(self.stats['episode_lengths'])), smoothed, color='orange', linewidth=2)
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].grid(True)
        
        # 损失曲线
        if self.stats['losses']:
            loss_episodes = range(0, len(self.stats['losses']) * self.training_config['update_frequency'], 
                                self.training_config['update_frequency'])
            axes[1, 0].plot(loss_episodes, self.stats['losses'], color='red', linewidth=2)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Total Loss')
            axes[1, 0].grid(True)
        
        # 带宽成功率
        if self.stats['bandwidth_success_rates']:
            axes[1, 1].plot(episodes, self.stats['bandwidth_success_rates'], color='cyan', linewidth=2)
            axes[1, 1].set_title('Bandwidth Success Rate')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True)
        
        # 负载均衡奖励
        if self.stats['load_balance_rewards']:
            lb_episodes = range(len(self.stats['load_balance_rewards']))
            axes[1, 2].plot(lb_episodes, self.stats['load_balance_rewards'], color='brown', linewidth=2)
            axes[1, 2].set_title('Load Balance Reward')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('LB Reward')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_summary(self):
        """保存训练总结"""
        summary = {
            'config': {
                'env_config': self.env_config,
                'agent_config': self.agent_config,
                'training_config': self.training_config
            },
            'results': {
                'total_episodes': len(self.stats['episode_rewards']),
                'final_acceptance_rate': self.stats['acceptance_rates'][-1] if self.stats['acceptance_rates'] else 0,
                'best_acceptance_rate': max(self.stats['acceptance_rates']) if self.stats['acceptance_rates'] else 0,
                'average_reward': np.mean(self.stats['episode_rewards'][-100:]) if self.stats['episode_rewards'] else 0,
                'average_length': np.mean(self.stats['episode_lengths'][-100:]) if self.stats['episode_lengths'] else 0,
                'final_bandwidth_success_rate': np.mean(self.stats['bandwidth_success_rates'][-100:]) if self.stats['bandwidth_success_rates'] else 0,
            },
            'run_info': {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'output_dir': self.output_dir
            }
        }
        
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 训练总结已保存到: {summary_path}")


def main():
    """主函数"""
    print("🚀 启动A2C训练")
    
    # 可以在这里调整配置
    env_config = {
        # 'num_physical_nodes': 8,  # 可以调整物理节点数
        # 'task_nodes_range': (4, 7),  # 可以调整任务节点数范围
    }
    
    agent_config = {
        # 'hidden_dim': 256,        # 可以调整网络容量
        # 'lr': 5e-4,              # 可以调整学习率
    }
    
    training_config = {
        # 'total_episodes': 3000,   # 可以调整训练轮数
        # 'batch_size': 64,        # 可以调整批次大小
        # 'update_frequency': 5,   # 可以调整更新频率
    }
    
    # 创建训练器
    trainer = A2CTrainer(
        env_config=env_config,
        agent_config=agent_config,
        training_config=training_config
    )
    
    try:
        # 开始训练
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        print("正在保存当前进度...")
        trainer.save_checkpoint(len(trainer.stats['episode_rewards']) - 1, final=True)
        trainer.plot_training_curves()
        trainer.save_training_summary()
        print("✅ 进度已保存")
    
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试保存当前进度
        try:
            trainer.save_checkpoint(len(trainer.stats['episode_rewards']) - 1, final=True)
            trainer.plot_training_curves()
            trainer.save_training_summary()
            print("✅ 错误前的进度已保存")
        except:
            print("❌ 无法保存进度")


if __name__ == "__main__":
    main()
