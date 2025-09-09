#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO_mapping训练脚本
Actor只负责节点调度，带宽分配使用贪心策略
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
from datetime import datetime
import uuid

from mapping_environment import MappingNetworkSchedulerEnvironment
from mapping_agent import MappingAgent
from original_problem_config import ORIGINAL_CONFIG

class MappingPPOTrainer:
    """
    PPO_mapping训练器
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
            'hidden_dim': 128,
            'lr': 3e-4
        }
        
        # 训练配置
        default_training_config = {
            'total_episodes': 2000,
            'batch_size': 32,
            'update_frequency': 32,
            'print_frequency': 50,
            'save_frequency': 200,
            'warmup_episodes': 100,
            'curriculum_enabled': True,
            'curriculum_start': 300,
            'ppo_clip': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'gamma': 0.99,
        }
        
        # 合并配置
        self.env_config = {**default_env_config, **(env_config or {})}
        self.agent_config = {**default_agent_config, **(agent_config or {})}
        self.training_config = {**default_training_config, **(training_config or {})}
        
        # 创建环境和agent
        self.env = MappingNetworkSchedulerEnvironment(**self.env_config, curriculum_enabled=True)
        self.agent = MappingAgent(**self.agent_config)
        
        # 训练统计
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'constraint_violations': [],
            'losses': [],
            'mapping_rewards': [],
            'difficulty_levels': [],
            'bandwidth_success_rates': [],
            'final_rewards': [],
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
        
        # 创建输出目录
        self.setup_output_directories()
        
        print(f"🚀 MappingPPOTrainer初始化完成")
        print(f"   环境: {self.env_config['num_physical_nodes']}个物理节点, {self.env_config['max_virtual_nodes']}个最大虚拟节点")
        print(f"   智能体: 隐藏维度{self.agent_config['hidden_dim']}, 学习率{self.agent_config['lr']}")
        print(f"   训练: {self.training_config['total_episodes']}轮, 批次大小{self.training_config['batch_size']}")
    
    def setup_output_directories(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"mapping_ppo_{timestamp}_{str(uuid.uuid4())[:8]}"
        
        self.output_dir = f"mapping_results/{self.run_id}"
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
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_values = []
        episode_log_probs = []
        
        done = False
        while not done:
            # 选择动作
            action, log_prob, value = self.agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        return {
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'dones': episode_dones,
            'values': episode_values,
            'log_probs': episode_log_probs,
            'total_reward': episode_reward,
            'length': episode_length,
            'info': info
        }
    
    def update_agent(self):
        """更新agent"""
        if len(self.experience_buffer['states']) < self.training_config['batch_size']:
            return None
        
        # 计算损失
        loss_dict = self.agent.calculate_loss(
            states=self.experience_buffer['states'],
            actions=self.experience_buffer['actions'],
            rewards=self.experience_buffer['rewards'],
            dones=self.experience_buffer['dones'],
            gamma=self.training_config['gamma'],
            ppo_clip=self.training_config['ppo_clip'],
            value_coef=self.training_config['value_coef'],
            entropy_coef=self.training_config['entropy_coef']
        )
        
        # 更新网络
        self.agent.update(loss_dict)
        
        # 清空缓冲区
        self.experience_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
        
        return loss_dict
    
    def train(self):
        """主训练循环"""
        print(f"🎯 开始训练PPO_mapping算法")
        print(f"总训练轮数: {self.training_config['total_episodes']}")
        print("-" * 80)
        
        best_success_rate = 0.0
        recent_rewards = []
        recent_success = []
        
        for episode in range(self.training_config['total_episodes']):
            episode_start_time = time.time()
            
            # 收集episode经验
            episode_data = self.collect_episode()
            
            # 添加到经验缓冲区
            self.experience_buffer['states'].extend(episode_data['states'])
            self.experience_buffer['actions'].extend(episode_data['actions'])
            self.experience_buffer['rewards'].extend(episode_data['rewards'])
            self.experience_buffer['dones'].extend(episode_data['dones'])
            
            # 更新统计信息
            episode_reward = episode_data['total_reward']
            episode_length = episode_data['length']
            episode_success = episode_data.get('info', {}).get('final_reward', 0) > 0
            
            self.stats['episode_rewards'].append(episode_reward)
            self.stats['episode_lengths'].append(episode_length)
            
            recent_rewards.append(episode_reward)
            recent_success.append(episode_success)
            
            # 保持最近100个episode的记录
            if len(recent_rewards) > 100:
                recent_rewards = recent_rewards[-100:]
                recent_success = recent_success[-100:]
            
            # 记录其他统计信息
            if hasattr(self.env, 'difficulty_level'):
                self.stats['difficulty_levels'].append(self.env.difficulty_level)
            
            # 计算成功率
            success_rate = np.mean(recent_success) if recent_success else 0.0
            self.stats['success_rates'].append(success_rate)
            
            # 更新agent（每隔一定频率）
            loss_dict = None
            if (episode + 1) % self.training_config['update_frequency'] == 0:
                if episode >= self.training_config['warmup_episodes']:
                    loss_dict = self.update_agent()
                    if loss_dict:
                        self.stats['losses'].append({
                            'episode': episode,
                            'total_loss': loss_dict['total_loss'].item(),
                            'policy_loss': loss_dict['policy_loss'].item(),
                            'value_loss': loss_dict['value_loss'].item(),
                            'entropy_loss': loss_dict['entropy_loss'].item(),
                        })
            
            # 打印进度
            if (episode + 1) % self.training_config['print_frequency'] == 0:
                episode_time = time.time() - episode_start_time
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                
                print(f"Episode {episode + 1:4d}/{self.training_config['total_episodes']}")
                print(f"  奖励: {episode_reward:8.3f} | 平均: {avg_reward:8.3f}")
                print(f"  成功率: {success_rate:6.1%} | 长度: {episode_length:3d}")
                print(f"  时间: {episode_time:6.2f}s", end="")
                
                if hasattr(self.env, 'difficulty_level'):
                    print(f" | 难度: {self.env.difficulty_level:.2f}", end="")
                
                if loss_dict:
                    print(f" | 损失: {loss_dict['total_loss'].item():.4f}", end="")
                
                print()
                
                # 检查是否需要保存最佳模型
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    self.save_best_model(episode, success_rate, avg_reward)
            
            # 定期保存检查点
            if (episode + 1) % self.training_config['save_frequency'] == 0:
                self.save_checkpoint(episode)
                self.plot_training_curves()
        
        print(f"🎉 训练完成！最佳成功率: {best_success_rate:.1%}")
        
        # 最终保存
        self.save_checkpoint(self.training_config['total_episodes'] - 1, final=True)
        self.plot_training_curves()
        self.save_training_summary()
    
    def save_best_model(self, episode, success_rate, avg_reward):
        """保存最佳模型"""
        model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.agent.save_checkpoint(model_path)
        
        # 保存最佳模型信息
        best_info = {
            'episode': episode,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        info_path = os.path.join(self.checkpoint_dir, "best_model_info.json")
        with open(info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        print(f"💾 新的最佳模型已保存 (成功率: {success_rate:.1%})")
    
    def save_checkpoint(self, episode, final=False):
        """保存训练检查点"""
        suffix = "final" if final else f"ep_{episode + 1}"
        model_path = os.path.join(self.checkpoint_dir, f"model_{suffix}.pth")
        self.agent.save_checkpoint(model_path)
        
        # 保存训练统计
        stats_path = os.path.join(self.checkpoint_dir, f"stats_{suffix}.npz")
        np.savez(stats_path, **self.stats)
        
        if final:
            print(f"💾 最终检查点已保存")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.stats['episode_rewards']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'PPO_mapping训练曲线 - {self.run_id}', fontsize=14)
        
        # 奖励曲线
        episodes = range(len(self.stats['episode_rewards']))
        axes[0, 0].plot(episodes, self.stats['episode_rewards'], alpha=0.3, color='blue')
        if len(self.stats['episode_rewards']) > 10:
            window = min(50, len(self.stats['episode_rewards']) // 10)
            smoothed = np.convolve(self.stats['episode_rewards'], np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.stats['episode_rewards'])), smoothed, color='red', linewidth=2)
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # 成功率曲线
        if self.stats['success_rates']:
            axes[0, 1].plot(episodes, self.stats['success_rates'], color='green', linewidth=2)
            axes[0, 1].set_title('Success Rates')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rates')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True)
        
        # Episode长度
        axes[1, 0].plot(episodes, self.stats['episode_lengths'], alpha=0.5, color='purple')
        if len(self.stats['episode_lengths']) > 10:
            window = min(50, len(self.stats['episode_lengths']) // 10)
            smoothed = np.convolve(self.stats['episode_lengths'], np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(self.stats['episode_lengths'])), smoothed, color='orange', linewidth=2)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # 损失曲线
        if self.stats['losses']:
            loss_episodes = [loss['episode'] for loss in self.stats['losses']]
            total_losses = [loss['total_loss'] for loss in self.stats['losses']]
            axes[1, 1].plot(loss_episodes, total_losses, color='red', linewidth=2)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Total Loss')
            axes[1, 1].grid(True)
        
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
                'final_success_rate': self.stats['success_rates'][-1] if self.stats['success_rates'] else 0,
                'best_success_rate': max(self.stats['success_rates']) if self.stats['success_rates'] else 0,
                'average_reward': np.mean(self.stats['episode_rewards'][-100:]) if self.stats['episode_rewards'] else 0,
                'average_length': np.mean(self.stats['episode_lengths'][-100:]) if self.stats['episode_lengths'] else 0,
            },
            'run_info': {
                'run_id': self.run_id,
                'start_time': datetime.now().isoformat(),
                'output_dir': self.output_dir
            }
        }
        
        import json
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 训练总结已保存到: {summary_path}")

def main():
    """主函数"""
    print("🚀 启动PPO_mapping训练")
    
    # 可以在这里调整配置
    env_config = {
        # 'num_physical_nodes': 6,  # 可以调整物理节点数
        # 'max_virtual_nodes': 5,   # 可以调整最大虚拟节点数
    }
    
    agent_config = {
        # 'hidden_dim': 256,        # 可以调整网络容量
        # 'lr': 5e-4,              # 可以调整学习率
    }
    
    training_config = {
        # 'total_episodes': 3000,   # 可以调整训练轮数
        # 'batch_size': 64,        # 可以调整批次大小
    }
    
    # 创建训练器
    trainer = MappingPPOTrainer(
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
