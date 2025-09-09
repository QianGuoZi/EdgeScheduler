#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO_balance训练脚本
结合PPO_mapping的PPO算法实现和A2C的状态/奖励设计
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
from datetime import datetime
import uuid

from balance_environment import PPOBalanceNetworkEnvironment
from balance_agent import BalanceAgent
from original_problem_config import ORIGINAL_CONFIG

class BalancePPOTrainer:
    """
    PPO_balance训练器
    结合PPO算法和负载均衡奖励设计
    """
    
    def __init__(self, 
                 env_config: dict = None,
                 agent_config: dict = None,
                 training_config: dict = None):
        
        # 使用原始问题配置作为基础
        original_config = ORIGINAL_CONFIG
        
        # 默认环境配置（基于A2C的设计）
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
            'initial_usage_range': (0.1, 0.5),
            'seed': original_config['environment']['seed']
        }
        
        # Agent配置（适配原始问题规模）
        default_agent_config = {
            'num_physical_nodes': original_config['physical_topology']['num_nodes'],
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
        self.env = PPOBalanceNetworkEnvironment(**self.env_config)
        self.agent = BalanceAgent(**self.agent_config)
        
        # 训练统计
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'acceptance_rates': [],
            'load_balance_rewards': [],
            'losses': [],
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
        
        print(f"🚀 BalancePPOTrainer初始化完成")
        print(f"   环境: {self.env_config['num_physical_nodes']}个物理节点, 任务范围{self.env_config['task_nodes_range']}")
        print(f"   智能体: 隐藏维度{self.agent_config['hidden_dim']}, 学习率{self.agent_config['lr']}")
        print(f"   训练: {self.training_config['total_episodes']}轮, 批次大小{self.training_config['batch_size']}")
    
    def setup_output_directories(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"balance_ppo_{timestamp}_{str(uuid.uuid4())[:8]}"
        
        self.output_dir = f"balance_results/{self.run_id}"
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
            # 从状态中提取物理资源和任务需求
            physical_resources = state['physical_resources']
            task_requirements = state['task_requirements']
            valid_actions = state.get('valid_actions', None)
            
            # 选择动作
            action, log_prob, value = self.agent.select_action(
                physical_resources, task_requirements, valid_actions
            )
            
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
        print(f"🎯 开始PPO_balance训练")
        print(f"   目标episodes: {self.training_config['total_episodes']}")
        print(f"   更新频率: 每{self.training_config['update_frequency']}个episodes")
        
        start_time = time.time()
        best_acceptance_rate = 0.0
        
        for episode in range(self.training_config['total_episodes']):
            # 收集episode经验
            episode_data = self.collect_episode()
            
            # 添加到经验缓冲区
            self.experience_buffer['states'].extend(episode_data['states'])
            self.experience_buffer['actions'].extend(episode_data['actions'])
            self.experience_buffer['rewards'].extend(episode_data['rewards'])
            self.experience_buffer['dones'].extend(episode_data['dones'])
            self.experience_buffer['values'].extend(episode_data['values'])
            self.experience_buffer['log_probs'].extend(episode_data['log_probs'])
            
            # 记录统计信息
            self.stats['episode_rewards'].append(episode_data['total_reward'])
            self.stats['episode_lengths'].append(episode_data['length'])
            
            # 从环境统计中提取信息
            env_stats = self.env.episode_stats
            acceptance_rate = env_stats['accepted_tasks'] / max(1, env_stats['total_tasks'])
            self.stats['acceptance_rates'].append(acceptance_rate)
            
            # 计算平均负载均衡奖励
            if env_stats['load_balance_rewards']:
                avg_lb_reward = np.mean(env_stats['load_balance_rewards'])
                self.stats['load_balance_rewards'].append(avg_lb_reward)
            else:
                self.stats['load_balance_rewards'].append(0.0)
            
            # 带宽分配成功率
            self.stats['bandwidth_success_rates'].append(
                1.0 if env_stats['final_bandwidth_success'] else 0.0
            )
            
            # 最终奖励
            final_reward = episode_data['info'].get('final_reward', 0.0)
            self.stats['final_rewards'].append(final_reward)
            
            # 定期更新agent
            loss_dict = None
            if (episode + 1) % self.training_config['update_frequency'] == 0:
                loss_dict = self.update_agent()
                if loss_dict:
                    self.stats['losses'].append({
                        'episode': episode + 1,
                        'total_loss': loss_dict['total_loss'].item(),
                        'policy_loss': loss_dict['policy_loss'].item(),
                        'value_loss': loss_dict['value_loss'].item(),
                        'entropy_loss': loss_dict['entropy_loss'].item(),
                    })
            
            # 定期输出统计信息和检查最佳模型
            if (episode + 1) % self.training_config['print_frequency'] == 0:
                # 计算最近的平均指标
                recent_rewards = self.stats['episode_rewards'][-self.training_config['print_frequency']:]
                recent_acceptance = self.stats['acceptance_rates'][-self.training_config['print_frequency']:]
                recent_bandwidth = self.stats['bandwidth_success_rates'][-self.training_config['print_frequency']:]
                recent_lb_rewards = self.stats['load_balance_rewards'][-self.training_config['print_frequency']:]
                
                avg_reward = np.mean(recent_rewards)
                avg_acceptance = np.mean(recent_acceptance)
                avg_bandwidth = np.mean(recent_bandwidth)
                avg_lb_reward = np.mean(recent_lb_rewards)
                
                print(f"📊 Episode {episode + 1:4d} | "
                      f"平均奖励: {avg_reward:7.3f} | "
                      f"接受率: {avg_acceptance:5.2%} | "
                      f"带宽成功率: {avg_bandwidth:5.2%} | "
                      f"负载均衡奖励: {avg_lb_reward:6.3f}", end="")
                
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
        import json
        import os
        
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
        import os
        
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
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Episode奖励
        if self.stats['episode_rewards']:
            axes[0].plot(self.stats['episode_rewards'], alpha=0.6, label='Episode奖励')
            # 计算移动平均
            window_size = 50
            if len(self.stats['episode_rewards']) >= window_size:
                moving_avg = np.convolve(self.stats['episode_rewards'], 
                                       np.ones(window_size)/window_size, mode='valid')
                axes[0].plot(range(window_size-1, len(self.stats['episode_rewards'])), 
                           moving_avg, 'r-', label=f'{window_size}-episode移动平均')
            axes[0].set_title('Episode奖励')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('奖励')
            axes[0].legend()
            axes[0].grid(True)
        
        # 2. 任务接受率
        if self.stats['acceptance_rates']:
            axes[1].plot(self.stats['acceptance_rates'], alpha=0.6, label='接受率')
            # 移动平均
            window_size = 50
            if len(self.stats['acceptance_rates']) >= window_size:
                moving_avg = np.convolve(self.stats['acceptance_rates'], 
                                       np.ones(window_size)/window_size, mode='valid')
                axes[1].plot(range(window_size-1, len(self.stats['acceptance_rates'])), 
                           moving_avg, 'r-', label=f'{window_size}-episode移动平均')
            axes[1].set_title('任务接受率')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('接受率')
            axes[1].set_ylim(0, 1.1)
            axes[1].legend()
            axes[1].grid(True)
        
        # 3. 带宽分配成功率
        if self.stats['bandwidth_success_rates']:
            axes[2].plot(self.stats['bandwidth_success_rates'], alpha=0.6, label='带宽成功率')
            # 移动平均
            window_size = 50
            if len(self.stats['bandwidth_success_rates']) >= window_size:
                moving_avg = np.convolve(self.stats['bandwidth_success_rates'], 
                                       np.ones(window_size)/window_size, mode='valid')
                axes[2].plot(range(window_size-1, len(self.stats['bandwidth_success_rates'])), 
                           moving_avg, 'r-', label=f'{window_size}-episode移动平均')
            axes[2].set_title('带宽分配成功率')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('成功率')
            axes[2].set_ylim(0, 1.1)
            axes[2].legend()
            axes[2].grid(True)
        
        # 4. 负载均衡奖励
        if self.stats['load_balance_rewards']:
            axes[3].plot(self.stats['load_balance_rewards'], alpha=0.6, label='负载均衡奖励')
            # 移动平均
            window_size = 50
            if len(self.stats['load_balance_rewards']) >= window_size:
                moving_avg = np.convolve(self.stats['load_balance_rewards'], 
                                       np.ones(window_size)/window_size, mode='valid')
                axes[3].plot(range(window_size-1, len(self.stats['load_balance_rewards'])), 
                           moving_avg, 'r-', label=f'{window_size}-episode移动平均')
            axes[3].set_title('负载均衡奖励')
            axes[3].set_xlabel('Episode')
            axes[3].set_ylabel('奖励')
            axes[3].legend()
            axes[3].grid(True)
        
        # 5. 训练损失
        if self.stats['losses']:
            episodes = [loss['episode'] for loss in self.stats['losses']]
            total_losses = [loss['total_loss'] for loss in self.stats['losses']]
            policy_losses = [loss['policy_loss'] for loss in self.stats['losses']]
            value_losses = [loss['value_loss'] for loss in self.stats['losses']]
            
            axes[4].plot(episodes, total_losses, label='总损失')
            axes[4].plot(episodes, policy_losses, label='策略损失')
            axes[4].plot(episodes, value_losses, label='价值损失')
            axes[4].set_title('训练损失')
            axes[4].set_xlabel('Episode')
            axes[4].set_ylabel('损失')
            axes[4].legend()
            axes[4].grid(True)
        
        # 6. Episode长度
        if self.stats['episode_lengths']:
            axes[5].plot(self.stats['episode_lengths'], alpha=0.6, label='Episode长度')
            # 移动平均
            window_size = 50
            if len(self.stats['episode_lengths']) >= window_size:
                moving_avg = np.convolve(self.stats['episode_lengths'], 
                                       np.ones(window_size)/window_size, mode='valid')
                axes[5].plot(range(window_size-1, len(self.stats['episode_lengths'])), 
                           moving_avg, 'r-', label=f'{window_size}-episode移动平均')
            axes[5].set_title('Episode长度')
            axes[5].set_xlabel('Episode')
            axes[5].set_ylabel('步数')
            axes[5].legend()
            axes[5].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 训练曲线已保存到: {self.plots_dir}/training_curves.png")
    
    def save_training_summary(self):
        """保存训练总结"""
        import json
        import os
        
        # 计算最终统计
        final_stats = {}
        if self.stats['episode_rewards']:
            final_stats['total_episodes'] = len(self.stats['episode_rewards'])
            final_stats['final_avg_reward'] = float(np.mean(self.stats['episode_rewards'][-100:]) if len(self.stats['episode_rewards']) >= 100 else np.mean(self.stats['episode_rewards']))
            final_stats['best_reward'] = float(max(self.stats['episode_rewards']))
            final_stats['final_acceptance_rate'] = float(np.mean(self.stats['acceptance_rates'][-100:]) if len(self.stats['acceptance_rates']) >= 100 else np.mean(self.stats['acceptance_rates']))
            final_stats['best_acceptance_rate'] = float(max(self.stats['acceptance_rates']))
            final_stats['final_bandwidth_success'] = float(np.mean(self.stats['bandwidth_success_rates'][-100:]) if len(self.stats['bandwidth_success_rates']) >= 100 else np.mean(self.stats['bandwidth_success_rates']))
            final_stats['final_lb_reward'] = float(np.mean(self.stats['load_balance_rewards'][-100:]) if len(self.stats['load_balance_rewards']) >= 100 else np.mean(self.stats['load_balance_rewards']))
        
        summary = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'env_config': self.env_config,
                'agent_config': self.agent_config,
                'training_config': self.training_config
            },
            'final_stats': final_stats,
            'training_history': {
                'episode_rewards': self.stats['episode_rewards'],
                'acceptance_rates': self.stats['acceptance_rates'],
                'bandwidth_success_rates': self.stats['bandwidth_success_rates'],
                'load_balance_rewards': self.stats['load_balance_rewards']
            }
        }
        
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 训练总结已保存到: {summary_path}")


def main():
    """主函数"""
    import signal
    import sys
    
    print("🚀 启动PPO_balance训练")
    
    # 创建训练器
    trainer = BalancePPOTrainer()
    
    # 设置信号处理
    def signal_handler(signum, frame):
        print(f"\n⚠️ 接收到信号 {signum}，正在保存进度...")
        try:
            print("正在保存当前进度...")
            trainer.save_checkpoint(len(trainer.stats['episode_rewards']) - 1, final=True)
            trainer.plot_training_curves()
            trainer.save_training_summary()
            print("✅ 进度已保存")
        except Exception as e:
            print(f"❌ 保存进度时出错: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    try:
        # 开始训练
        trainer.train()
        print("✅ PPO_balance训练完成!")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        # 尝试保存当前进度
        try:
            trainer.save_checkpoint(len(trainer.stats['episode_rewards']) - 1, final=True)
            trainer.plot_training_curves()
            trainer.save_training_summary()
            print("✅ 错误前的进度已保存")
        except:
            print("❌ 无法保存进度")
        raise


if __name__ == "__main__":
    main()
