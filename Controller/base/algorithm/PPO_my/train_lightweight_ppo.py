#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级启发式PPO训练脚本
使用原始PPO架构 + 增强的奖励函数和少量状态特征
"""

from typing import Dict, List
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
from datetime import datetime

from lightweight_heuristic_integration_environment import LightweightHeuristicEnvironment
from sequential_agent import SimpleSequentialAgent  # 使用原始PPO Agent
from original_problem_config import ORIGINAL_CONFIG


class LightweightHeuristicPPOTrainer:
    """
    轻量级启发式PPO训练器
    保持原始PPO架构，只修改环境的奖励函数和状态表示
    """
    
    def __init__(self, 
                 env_config: dict = None,
                 agent_config: dict = None,
                 training_config: dict = None):
        
        # 使用原始问题配置
        original_config = ORIGINAL_CONFIG
        
        # 默认环境配置
        default_env_config = {
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
            'seed': original_config['environment']['seed'],
            'curriculum_enabled': True,
            # 轻量级启发式参数
            'heuristic_reward_weight': 0.4,  # 启发式奖励权重
            'enable_load_balance_reward': True,
            'enable_resource_efficiency_reward': True,
            'enable_progress_reward': True
        }
        
        # Agent配置 - 使用原始PPO，但状态维度需要适配新特征
        default_agent_config = {
            'max_physical_nodes': original_config['physical_topology']['num_nodes'],
            'max_virtual_nodes': original_config['task_topology']['num_nodes_range'][1],
            'bandwidth_levels': original_config['action_space']['bandwidth_levels'],
            'hidden_dim': 128,  # 保持原始大小
            'lr': 3e-4
        }
        
        # 训练配置
        default_training_config = {
            'total_episodes': 3000,
            'batch_size': 64,
            'update_frequency': 64,
            'print_frequency': 50,
            'save_frequency': 300,
            'evaluation_frequency': 200,
            # 启发式权重调度
            'enable_reward_weight_schedule': True,
            'initial_heuristic_weight': 0.6,
            'final_heuristic_weight': 0.2,
            'weight_decay_episodes': 2000,
        }
        
        # 合并配置
        self.env_config = {**default_env_config, **(env_config or {})}
        self.agent_config = {**default_agent_config, **(agent_config or {})}
        self.training_config = {**default_training_config, **(training_config or {})}
        
        # 创建环境和agent
        self.env = LightweightHeuristicEnvironment(**self.env_config)
        self.agent = SimpleSequentialAgent(**self.agent_config)
        
        # 训练统计
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'heuristic_metrics': {
                'load_balance_scores': [],
                'resource_efficiency_scores': [],
                'progress_rewards': []
            },
            'reward_components': {
                'base_rewards': [],
                'heuristic_rewards': [],
                'progress_rewards': []
            },
            'losses': {
                'total': [],
                'policy': [],
                'value': []
            },
            'heuristic_weights': []  # 记录启发式权重变化
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
        
        # 实验管理 - 与sequential版本保持一致的命名方式
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"lightweight_{self.experiment_timestamp}"
        
        # 创建实验专属的保存目录
        self.checkpoint_dir = f"checkpoints/{self.experiment_name}"
        self.plot_dir = f"plots/{self.experiment_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        print(f"🚀 轻量级启发式PPO训练器初始化完成")
        print(f"   实验时间: {self.experiment_timestamp}")
        print(f"   实验名称: {self.experiment_name}")
        print(f"   检查点目录: {self.checkpoint_dir}")
        print(f"   图表目录: {self.plot_dir}")
        print(f"   使用原始PPO架构: ✅")
        print(f"   启发式奖励权重: {self.env_config['heuristic_reward_weight']}")
        print(f"   新增状态特征: 5个（负载均衡、资源效率、进度等）")
    
    def calculate_current_heuristic_weight(self, episode: int) -> float:
        """计算当前的启发式奖励权重（可选的衰减策略）"""
        if not self.training_config.get('enable_reward_weight_schedule', False):
            return self.env_config['heuristic_reward_weight']
        
        initial_weight = self.training_config['initial_heuristic_weight']
        final_weight = self.training_config['final_heuristic_weight']
        decay_episodes = self.training_config['weight_decay_episodes']
        
        if episode >= decay_episodes:
            return final_weight
        
        # 线性衰减
        progress = episode / decay_episodes
        current_weight = initial_weight - progress * (initial_weight - final_weight)
        return current_weight
    
    def collect_episode(self, episode_idx: int):
        """收集一个episode的经验"""
        # 调整启发式权重
        current_heuristic_weight = self.calculate_current_heuristic_weight(episode_idx)
        self.env.heuristic_reward_weight = current_heuristic_weight
        
        # 重置环境
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Episode数据
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_values = []
        episode_log_probs = []
        
        # 奖励分解记录
        base_reward_sum = 0
        heuristic_reward_sum = 0
        progress_reward_sum = 0
        
        step_count = 0
        max_steps = 30
        
        while not done and step_count < max_steps:
            # 使用原始PPO选择动作
            action, log_prob, value = self.agent.select_action(state, temperature=1.0)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录奖励分解（如果可用）
            progress_reward = info.get('progress_reward', 0)
            progress_reward_sum += progress_reward
            
            # 估算基础奖励和启发式奖励的分解
            estimated_base_reward = reward / (1 + current_heuristic_weight)
            estimated_heuristic_reward = reward - estimated_base_reward
            base_reward_sum += estimated_base_reward
            heuristic_reward_sum += estimated_heuristic_reward
            
            # 记录经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            
            episode_reward += reward
            episode_length += 1
            step_count += 1
            state = next_state
        
        # 计算episode统计
        success = info.get('success', False) if 'info' in locals() else False
        
        # 获取启发式指标
        heuristic_metrics = self.env.get_heuristic_metrics_summary()
        
        episode_data = {
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'dones': episode_dones,
            'values': episode_values,
            'log_probs': episode_log_probs,
            'total_reward': episode_reward,
            'length': episode_length,
            'success': success,
            'heuristic_weight': current_heuristic_weight,
            'reward_components': {
                'base_reward': base_reward_sum,
                'heuristic_reward': heuristic_reward_sum,
                'progress_reward': progress_reward_sum
            },
            'heuristic_metrics': heuristic_metrics
        }
        
        return episode_data
    
    def update_agent(self, batch_data: List[Dict]) -> Dict:
        """更新agent参数（使用原始PPO更新）"""
        # 提取批次数据
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        
        for episode_data in batch_data:
            all_states.extend(episode_data['states'])
            all_actions.extend(episode_data['actions'])
            all_rewards.extend(episode_data['rewards'])
            all_dones.extend(episode_data['dones'])
        
        # 使用原始PPO的损失计算
        loss_dict = self.agent.calculate_loss(all_states, all_actions, all_rewards, all_dones)
        
        # 更新网络
        self.agent.update(loss_dict)
        
        return loss_dict
    
    def evaluate_agent(self, num_episodes: int = 10) -> Dict:
        """评估agent性能"""
        print(f"🔍 开始评估 ({num_episodes} episodes)...")
        
        # 保存当前权重设置
        original_weight = self.env.heuristic_reward_weight
        
        # 评估时使用固定的权重
        self.env.heuristic_reward_weight = 0.3
        
        eval_rewards = []
        eval_successes = []
        eval_lengths = []
        eval_metrics = []
        
        for _ in range(num_episodes):
            episode_data = self.collect_episode(episode_idx=999999)  # 使用大数字避免权重调度影响
            
            eval_rewards.append(episode_data['total_reward'])
            eval_successes.append(episode_data['success'])
            eval_lengths.append(episode_data['length'])
            eval_metrics.append(episode_data['heuristic_metrics'])
        
        # 恢复原权重
        self.env.heuristic_reward_weight = original_weight
        
        # 计算平均指标
        avg_metrics = {}
        if eval_metrics and len(eval_metrics[0]) > 0:
            for key in eval_metrics[0].keys():
                values = [m.get(key, 0) for m in eval_metrics if key in m]
                if values:
                    avg_metrics[f'avg_{key}'] = np.mean(values)
        
        evaluation_results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'success_rate': np.mean(eval_successes),
            'mean_length': np.mean(eval_lengths),
            **avg_metrics
        }
        
        print(f"   评估结果: 奖励={evaluation_results['mean_reward']:.3f}±{evaluation_results['std_reward']:.3f}, "
              f"成功率={evaluation_results['success_rate']:.1%}")
        
        return evaluation_results
    
    def train(self):
        """主训练循环"""
        print(f"🚀 开始轻量级启发式PPO训练")
        print(f"   总episodes: {self.training_config['total_episodes']}")
        print(f"   批次大小: {self.training_config['batch_size']}")
        
        start_time = time.time()
        batch_data = []
        
        for episode in range(self.training_config['total_episodes']):
            # 收集episode数据
            episode_data = self.collect_episode(episode)
            batch_data.append(episode_data)
            
            # 记录统计信息
            self.stats['episode_rewards'].append(episode_data['total_reward'])
            self.stats['episode_lengths'].append(episode_data['length'])
            self.stats['success_rates'].append(1.0 if episode_data['success'] else 0.0)
            self.stats['heuristic_weights'].append(episode_data['heuristic_weight'])
            
            # 记录启发式指标
            for key, value in episode_data['heuristic_metrics'].items():
                if key not in self.stats['heuristic_metrics']:
                    self.stats['heuristic_metrics'][key] = []
                self.stats['heuristic_metrics'][key].append(value)
            
            # 记录奖励分解
            for key, value in episode_data['reward_components'].items():
                if key not in self.stats['reward_components']:
                    self.stats['reward_components'][key] = []
                self.stats['reward_components'][key].append(value)
            
            # 更新agent
            if len(batch_data) >= self.training_config['batch_size']:
                loss_dict = self.update_agent(batch_data)
                
                # 记录损失
                for key, value in loss_dict.items():
                    if key in self.stats['losses']:
                        self.stats['losses'][key].append(value.item() if hasattr(value, 'item') else value)
                
                batch_data.clear()
            
            # 打印进度
            if (episode + 1) % self.training_config['print_frequency'] == 0:
                recent_rewards = self.stats['episode_rewards'][-100:]
                recent_success = self.stats['success_rates'][-100:]
                recent_weight = self.stats['heuristic_weights'][-1]
                
                # 计算平均启发式指标
                avg_load_balance = np.mean(self.stats['heuristic_metrics'].get('load_balance_mean', [0])[-10:]) if 'load_balance_mean' in self.stats['heuristic_metrics'] else 0
                avg_resource_efficiency = np.mean(self.stats['heuristic_metrics'].get('resource_efficiency_mean', [0])[-10:]) if 'resource_efficiency_mean' in self.stats['heuristic_metrics'] else 0
                
                print(f"Episode {episode + 1:4d}: "
                      f"奖励={np.mean(recent_rewards):6.3f}, "
                      f"成功率={np.mean(recent_success):5.1%}, "
                      f"启发式权重={recent_weight:5.3f}, "
                      f"负载均衡={avg_load_balance:.3f}, "
                      f"资源效率={avg_resource_efficiency:.3f}")
            
            # 定期评估
            if (episode + 1) % self.training_config['evaluation_frequency'] == 0:
                eval_metrics = self.evaluate_agent()
            
            # 保存检查点
            if (episode + 1) % self.training_config['save_frequency'] == 0:
                self.save_checkpoint(episode + 1)
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"✅ 训练完成! 用时: {training_time:.1f}秒")
        
        # 最终评估
        print(f"🏆 最终评估:")
        final_eval = self.evaluate_agent(num_episodes=50)
        
        # 保存最终结果
        self.save_final_results(final_eval)
        self.plot_training_curves()
    
    def save_checkpoint(self, episode: int):
        """保存训练检查点"""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'stats': self.stats,
            'config': {
                'env_config': self.env_config,
                'agent_config': self.agent_config,
                'training_config': self.training_config
            },
            'experiment_info': {
                'experiment_timestamp': self.experiment_timestamp,
                'experiment_name': self.experiment_name
            }
        }
        
        # 使用实验专属的检查点目录
        checkpoint_path = f'{self.checkpoint_dir}/episode_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 保存检查点: {checkpoint_path}")
        
        # 同时保存最新的检查点
        latest_checkpoint_path = f'{self.checkpoint_dir}/latest.pt'
        torch.save(checkpoint, latest_checkpoint_path)
        print(f"💾 保存最新检查点: {latest_checkpoint_path}")
        
        # 保存实验配置信息
        self._save_experiment_config()
    
    def save_final_results(self, final_eval: Dict):
        """保存最终结果"""
        # 最终检查点
        final_checkpoint = {
            'episode': self.training_config['total_episodes'],
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'stats': self.stats,
            'final_evaluation': final_eval,
            'config': {
                'env_config': self.env_config,
                'agent_config': self.agent_config,
                'training_config': self.training_config
            },
            'experiment_info': {
                'experiment_timestamp': self.experiment_timestamp,
                'experiment_name': self.experiment_name
            }
        }
        
        # 保存最终检查点
        final_checkpoint_path = f'{self.checkpoint_dir}/final.pt'
        torch.save(final_checkpoint, final_checkpoint_path)
        print(f"📊 最终检查点已保存: {final_checkpoint_path}")
        
        # 保存到根目录便于查看
        root_final_path = f'checkpoints/{self.experiment_name}_final.pt'
        torch.save(final_checkpoint, root_final_path)
        print(f"📊 最终检查点已保存到根目录: {root_final_path}")
    
    def _save_experiment_config(self):
        """保存实验配置信息 - 与sequential版本保持一致"""
        import json
        
        experiment_info = {
            'experiment_timestamp': self.experiment_timestamp,
            'experiment_name': self.experiment_name,
            'creation_time': datetime.now().isoformat(),
            'algorithm_type': 'lightweight_heuristic_ppo',  # 标识算法类型
            'env_config': self.env_config,
            'agent_config': self.agent_config,
            'training_config': self.training_config
        }
        
        # 保存到实验目录
        config_path = f'{self.checkpoint_dir}/experiment_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, ensure_ascii=False, indent=2)
        
        # 保存到根目录
        root_config_path = f'checkpoints/{self.experiment_name}_config.json'
        with open(root_config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, ensure_ascii=False, indent=2)
        
        print(f"📝 实验配置已保存: {config_path}")
        print(f"📝 实验配置已保存到根目录: {root_config_path}")
    
    def get_experiment_info(self):
        """获取实验信息 - 与sequential版本保持一致"""
        return {
            'experiment_timestamp': self.experiment_timestamp,
            'experiment_name': self.experiment_name,
            'checkpoint_dir': self.checkpoint_dir,
            'plot_dir': self.plot_dir
        }
    
    def _smooth_curve(self, data, window):
        """计算滑动平均 - 与sequential版本保持一致"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Lightweight Heuristic PPO Training - {self.experiment_name}', fontsize=16)
        
        # 1. Episode Rewards
        axes[0, 0].plot(self.stats['episode_rewards'], alpha=0.3, color='blue')
        if len(self.stats['episode_rewards']) > 100:
            rewards_smooth = np.convolve(self.stats['episode_rewards'], np.ones(100)/100, mode='valid')
            axes[0, 0].plot(range(99, len(self.stats['episode_rewards'])), rewards_smooth, color='blue', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # 2. Success Rate
        if len(self.stats['success_rates']) > 100:
            success_smooth = np.convolve(self.stats['success_rates'], np.ones(100)/100, mode='valid')
            axes[0, 1].plot(range(99, len(self.stats['success_rates'])), success_smooth, color='green', linewidth=2)
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True)
        
        # 3. Heuristic Weight Schedule
        axes[0, 2].plot(self.stats['heuristic_weights'], color='red', linewidth=2)
        axes[0, 2].set_title('Heuristic Reward Weight')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Weight')
        axes[0, 2].grid(True)
        
        # 4. Load Balance Score
        if 'load_balance_mean' in self.stats['heuristic_metrics']:
            load_balance_data = self.stats['heuristic_metrics']['load_balance_mean']
            if len(load_balance_data) > 50:
                smooth_data = np.convolve(load_balance_data, np.ones(50)/50, mode='valid')
                axes[1, 0].plot(range(49, len(load_balance_data)), smooth_data, color='orange', linewidth=2)
        axes[1, 0].set_title('Load Balance Score')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True)
        
        # 5. Resource Efficiency
        if 'resource_efficiency_mean' in self.stats['heuristic_metrics']:
            efficiency_data = self.stats['heuristic_metrics']['resource_efficiency_mean']
            if len(efficiency_data) > 50:
                smooth_data = np.convolve(efficiency_data, np.ones(50)/50, mode='valid')
                axes[1, 1].plot(range(49, len(efficiency_data)), smooth_data, color='purple', linewidth=2)
        axes[1, 1].set_title('Resource Efficiency')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].grid(True)
        
        # 6. Reward Components
        if self.stats['reward_components']['base_rewards']:
            episodes = range(len(self.stats['reward_components']['base_rewards']))
            axes[1, 2].plot(episodes, self.stats['reward_components']['base_rewards'], label='Base Reward', alpha=0.7)
            axes[1, 2].plot(episodes, self.stats['reward_components']['heuristic_rewards'], label='Heuristic Reward', alpha=0.7)
        axes[1, 2].set_title('Reward Components')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # 保存图片到实验专属目录
        plot_filename = f'{self.plot_dir}/training_curves.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 训练曲线已保存: {plot_filename}")
        
        # 同时保存到实验根目录，方便查看
        root_plot_filename = f'plots/{self.experiment_name}_training_curves.png'
        plt.savefig(root_plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 训练曲线已保存到根目录: {root_plot_filename}")
        
        # 在支持的环境中显示图片
        try:
            plt.show()
        except:
            print("📊 图片已保存，但无法显示（可能是无头环境）")
        finally:
            plt.close()


def main():
    """主函数 - 与sequential版本保持一致的风格"""
    print("🎮 轻量级启发式PPO训练实验")
    print("=" * 60)
    
    # 自定义训练配置
    custom_training_config = {
        'total_episodes': 3000,
        'batch_size': 64,
        'print_frequency': 50,
        'evaluation_frequency': 200,
        'save_frequency': 300,
        # 启发式权重调度
        'enable_reward_weight_schedule': True,
        'initial_heuristic_weight': 0.6,
        'final_heuristic_weight': 0.2,
        'weight_decay_episodes': 2000,
    }
    
    # 自定义环境配置
    custom_env_config = {
        'heuristic_reward_weight': 0.4,
        'enable_load_balance_reward': True,
        'enable_resource_efficiency_reward': True,
        'enable_progress_reward': True
    }
    
    # 创建训练器（使用轻量级启发式配置）
    trainer = LightweightHeuristicPPOTrainer(
        env_config=custom_env_config,
        training_config=custom_training_config
    )
    
    # 开始训练
    trainer.train()
    
    # 显示实验信息
    experiment_info = trainer.get_experiment_info()
    print("\n" + "=" * 60)
    print("🔬 实验信息:")
    print("=" * 60)
    print(f"实验时间: {experiment_info['experiment_timestamp']}")
    print(f"实验名称: {experiment_info['experiment_name']}")
    print(f"检查点目录: {experiment_info['checkpoint_dir']}")
    print(f"图表目录: {experiment_info['plot_dir']}")
    
    # 分析结果
    print("\n" + "=" * 60)
    print("📈 训练结果分析:")
    print("=" * 60)
    
    # 计算不同阶段的统计
    stages = [
        ('前期(0-500)', 0, 500),
        ('中期(500-1000)', 500, 1000),
        ('后期(1000-2000)', 1000, 2000)
    ]
    
    for stage_name, start, end in stages:
        if len(trainer.stats['episode_rewards']) >= end:
            stage_rewards = trainer.stats['episode_rewards'][start:end]
            stage_success = trainer.stats['success_rates'][start:end]
            stage_weights = trainer.stats['heuristic_weights'][start:end]
            
            print(f"\n{stage_name}:")
            print(f"  平均奖励: {np.mean(stage_rewards):.3f} ± {np.std(stage_rewards):.3f}")
            print(f"  成功率: {np.mean(stage_success):.1%}")
            print(f"  平均启发式权重: {np.mean(stage_weights):.3f}")
    
    # 最终统计
    final_window = min(100, len(trainer.stats['episode_rewards']))
    if final_window > 0:
        print(f"\n最终{final_window}轮统计:")
        print(f"  平均奖励: {np.mean(trainer.stats['episode_rewards'][-final_window:]):.3f}")
        print(f"  成功率: {np.mean(trainer.stats['success_rates'][-final_window:]):.1%}")
        print(f"  平均episode长度: {np.mean(trainer.stats['episode_lengths'][-final_window:]):.1f}")
        
        # 启发式指标统计
        if 'load_balance_mean' in trainer.stats['heuristic_metrics']:
            load_balance_data = trainer.stats['heuristic_metrics']['load_balance_mean'][-final_window:]
            if load_balance_data:
                print(f"  平均负载均衡分数: {np.mean(load_balance_data):.3f}")
        
        if 'resource_efficiency_mean' in trainer.stats['heuristic_metrics']:
            efficiency_data = trainer.stats['heuristic_metrics']['resource_efficiency_mean'][-final_window:]
            if efficiency_data:
                print(f"  平均资源效率: {np.mean(efficiency_data):.3f}")
        
        # 成功episode的统计
        last_episodes_success = [i for i, s in enumerate(trainer.stats['success_rates'][-final_window:]) if s > 0]
        if last_episodes_success:
            successful_rewards = [trainer.stats['episode_rewards'][-final_window + i] for i in last_episodes_success]
            print(f"\n成功episode统计:")
            print(f"  成功次数: {len(last_episodes_success)}/{final_window}")
            print(f"  成功时平均奖励: {np.mean(successful_rewards):.3f}")
    
    print("\n🎉 轻量级启发式PPO训练完成!")


if __name__ == "__main__":
    main()
