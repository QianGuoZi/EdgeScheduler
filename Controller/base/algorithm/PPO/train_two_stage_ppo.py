#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from two_stage_actor_design import TwoStagePPOAgent
from two_stage_environment import TwoStageNetworkSchedulerEnvironment

class TwoStagePPOTrainer:
    """两阶段PPO训练器"""
    
    def __init__(self, 
                 num_physical_nodes_range: Tuple[int, int] = (5, 10),
                 max_virtual_nodes_range: Tuple[int, int] = (3, 8),
                 bandwidth_levels: int = 10,
                 # 物理节点资源范围
                 physical_cpu_range: Tuple[int, int] = (50, 200),
                 physical_memory_range: Tuple[int, int] = (100, 400),
                 physical_bandwidth_range: Tuple[int, int] = (100, 1000),
                 # 虚拟节点资源范围
                 virtual_cpu_range: Tuple[int, int] = (10, 50),
                 virtual_memory_range: Tuple[int, int] = (20, 100),
                 virtual_bandwidth_range: Tuple[int, int] = (10, 200),
                 # 网络连接概率
                 physical_connectivity_prob: float = 0.3,
                 virtual_connectivity_prob: float = 0.4,
                 # 训练参数
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 # 文件管理
                 model_dir: str = "models",
                 stats_dir: str = "stats"):
        
        self.num_physical_nodes_range = num_physical_nodes_range
        self.max_virtual_nodes_range = max_virtual_nodes_range
        self.bandwidth_levels = bandwidth_levels
        
        # 资源范围
        self.physical_cpu_range = physical_cpu_range
        self.physical_memory_range = physical_memory_range
        self.physical_bandwidth_range = physical_bandwidth_range
        self.virtual_cpu_range = virtual_cpu_range
        self.virtual_memory_range = virtual_memory_range
        self.virtual_bandwidth_range = virtual_bandwidth_range
        
        # 连接概率
        self.physical_connectivity_prob = physical_connectivity_prob
        self.virtual_connectivity_prob = virtual_connectivity_prob
        
        # 训练参数
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # 文件管理
        self.model_dir = model_dir
        self.stats_dir = stats_dir
        
        # 创建目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'mapping_actor_losses': [],
            'bandwidth_actor_losses': [],
            'critic_losses': [],
            'constraint_violations': [],
            'resource_utilization': [],
            'load_balancing': [],
            'bandwidth_satisfaction': []
        }
        
        # 初始化智能体和环境
        self._initialize_agent_and_env()
    
    def _initialize_agent_and_env(self):
        """初始化智能体和环境"""
        # 随机选择节点数量
        self.num_physical_nodes = np.random.randint(*self.num_physical_nodes_range)
        self.max_virtual_nodes = np.random.randint(*self.max_virtual_nodes_range)
        
        # 创建环境
        self.env = TwoStageNetworkSchedulerEnvironment(
            num_physical_nodes=self.num_physical_nodes,
            max_virtual_nodes=self.max_virtual_nodes,
            bandwidth_levels=self.bandwidth_levels,
            physical_cpu_range=self.physical_cpu_range,
            physical_memory_range=self.physical_memory_range,
            physical_bandwidth_range=self.physical_bandwidth_range,
            virtual_cpu_range=self.virtual_cpu_range,
            virtual_memory_range=self.virtual_memory_range,
            virtual_bandwidth_range=self.virtual_bandwidth_range,
            physical_connectivity_prob=self.physical_connectivity_prob,
            virtual_connectivity_prob=self.virtual_connectivity_prob
        )
        
        # 获取状态维度
        state = self.env.reset()
        physical_node_dim = state['physical_features'].size(1)
        virtual_node_dim = state['virtual_features'].size(1)
        
        # 创建智能体
        self.agent = TwoStagePPOAgent(
            physical_node_dim=physical_node_dim,
            virtual_node_dim=virtual_node_dim,
            num_physical_nodes=self.num_physical_nodes,
            max_virtual_nodes=self.max_virtual_nodes,
            bandwidth_levels=self.bandwidth_levels,
            lr=self.lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_ratio=self.clip_ratio,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef
        )
        
        print(f"✅ 智能体和环境初始化完成")
        print(f"   物理节点数: {self.num_physical_nodes}")
        print(f"   最大虚拟节点数: {self.max_virtual_nodes}")
        print(f"   物理节点特征维度: {physical_node_dim}")
        print(f"   虚拟节点特征维度: {virtual_node_dim}")
    
    def train_episode(self):
        """训练一个episode"""
        # 重置环境
        state = self.env.reset()
        
        # 获取动作
        mapping_action, bandwidth_action, mapping_log_prob, bandwidth_log_prob, value, link_indices = self.agent.select_actions(state)
        
        # 执行动作
        next_state, reward, done, info = self.env.step(mapping_action, bandwidth_action)
        
        # 存储经验
        self.agent.store_transition(
            state, mapping_action, bandwidth_action, 
            reward, value, mapping_log_prob, bandwidth_log_prob, done
        )
        
        # 更新网络
        self.agent.update()
        
        # 记录统计信息
        episode_stats = {
            'reward': reward,
            'length': 1,  # 两阶段环境一步完成
            'is_valid': info['is_valid'],
            'constraint_violations': len(info['constraint_violations']),
            'mapping_result': info['mapping_result'],
            'bandwidth_result': info['bandwidth_result']
        }
        
        return episode_stats
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100, eval_interval: int = 50):
        """训练主循环"""
        print(f"🚀 开始两阶段PPO训练")
        print(f"   总episodes: {num_episodes}")
        print(f"   保存间隔: {save_interval}")
        print(f"   评估间隔: {eval_interval}")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="训练进度"):
            # 训练一个episode
            episode_stats = self.train_episode()
            
            # 记录统计信息
            self.training_stats['episode_rewards'].append(episode_stats['reward'])
            self.training_stats['episode_lengths'].append(episode_stats['length'])
            self.training_stats['constraint_violations'].append(episode_stats['constraint_violations'])
            
            # 定期评估
            if (episode + 1) % eval_interval == 0:
                self._evaluate_and_log(episode + 1)
            
            # 定期保存
            if (episode + 1) % save_interval == 0:
                self.save_model(episode + 1)
                self.save_training_stats(episode + 1)
        
        # 最终保存
        self.save_model(num_episodes)
        self.save_training_stats(num_episodes)
        
        training_time = time.time() - start_time
        print(f"\n🎯 训练完成！")
        print(f"   总训练时间: {training_time:.2f}秒")
        print(f"   平均每episode时间: {training_time/num_episodes:.3f}秒")
        
        # 绘制训练曲线
        self._plot_training_curves()
    
    def _evaluate_and_log(self, episode):
        """评估并记录日志"""
        # 计算最近episodes的平均奖励
        recent_rewards = self.training_stats['episode_rewards'][-50:]
        avg_reward = np.mean(recent_rewards)
        
        # 计算约束违反率
        recent_violations = self.training_stats['constraint_violations'][-50:]
        violation_rate = np.mean([1 if v > 0 else 0 for v in recent_violations])
        
        print(f"Episode {episode:4d} | 平均奖励: {avg_reward:6.3f} | 约束违反率: {violation_rate:.2%}")
    
    def save_model(self, episode):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f"two_stage_ppo_model_episode_{episode}.pth")
        
        torch.save({
            'episode': episode,
            'mapping_actor_state_dict': self.agent.mapping_actor.state_dict(),
            'bandwidth_actor_state_dict': self.agent.bandwidth_actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'mapping_optimizer_state_dict': self.agent.mapping_optimizer.state_dict(),
            'bandwidth_optimizer_state_dict': self.agent.bandwidth_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'env_config': {
                'num_physical_nodes': self.num_physical_nodes,
                'max_virtual_nodes': self.max_virtual_nodes,
                'bandwidth_levels': self.bandwidth_levels,
                'physical_cpu_range': self.physical_cpu_range,
                'physical_memory_range': self.physical_memory_range,
                'physical_bandwidth_range': self.physical_bandwidth_range,
                'virtual_cpu_range': self.virtual_cpu_range,
                'virtual_memory_range': self.virtual_memory_range,
                'virtual_bandwidth_range': self.virtual_bandwidth_range,
                'physical_connectivity_prob': self.physical_connectivity_prob,
                'virtual_connectivity_prob': self.virtual_connectivity_prob
            }
        }, model_path)
        
        print(f"💾 模型已保存: {model_path}")
    
    def load_model(self, episode):
        """加载模型"""
        model_path = os.path.join(self.model_dir, f"two_stage_ppo_model_episode_{episode}.pth")
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location=self.agent.device)
        
        self.agent.mapping_actor.load_state_dict(checkpoint['mapping_actor_state_dict'])
        self.agent.bandwidth_actor.load_state_dict(checkpoint['bandwidth_actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.mapping_optimizer.load_state_dict(checkpoint['mapping_optimizer_state_dict'])
        self.agent.bandwidth_optimizer.load_state_dict(checkpoint['bandwidth_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_stats = checkpoint['training_stats']
        
        print(f"📂 模型已加载: {model_path}")
        return True
    
    def save_training_stats(self, episode):
        """保存训练统计"""
        stats_path = os.path.join(self.stats_dir, f"two_stage_training_stats_episode_{episode}.json")
        
        # 转换为可序列化的格式
        serializable_stats = {}
        for key, value in self.training_stats.items():
            if isinstance(value, list):
                serializable_stats[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
            else:
                serializable_stats[key] = value
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 训练统计已保存: {stats_path}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 奖励曲线
            axes[0, 0].plot(self.training_stats['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # 约束违反率
            violation_rates = [1 if v > 0 else 0 for v in self.training_stats['constraint_violations']]
            window_size = 50
            moving_avg = []
            for i in range(len(violation_rates)):
                start = max(0, i - window_size + 1)
                moving_avg.append(np.mean(violation_rates[start:i+1]))
            
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title('Constraint Violation Rate (Moving Average)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Violation Rate')
            axes[0, 1].grid(True)
            
            # 平均奖励（移动平均）
            reward_moving_avg = []
            for i in range(len(self.training_stats['episode_rewards'])):
                start = max(0, i - window_size + 1)
                reward_moving_avg.append(np.mean(self.training_stats['episode_rewards'][start:i+1]))
            
            axes[1, 0].plot(reward_moving_avg)
            axes[1, 0].set_title('Average Reward (Moving Average)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Average Reward')
            axes[1, 0].grid(True)
            
            # 约束违反数量
            axes[1, 1].plot(self.training_stats['constraint_violations'])
            axes[1, 1].set_title('Constraint Violations')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Number of Violations')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = os.path.join(self.stats_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 训练曲线已保存: {plot_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 绘制训练曲线时出错: {e}")
    
    def test_agent(self, num_test_episodes: int = 10):
        """测试智能体"""
        print(f"\n🧪 测试智能体 ({num_test_episodes} episodes)")
        print("=" * 60)
        
        test_stats = {
            'rewards': [],
            'constraint_violations': [],
            'valid_actions': 0,
            'total_actions': 0
        }
        
        for episode in range(num_test_episodes):
            # 重置环境
            state = self.env.reset()
            
            # 获取动作
            mapping_action, bandwidth_action, _, _, _, _ = self.agent.select_actions(state)
            
            # 执行动作
            _, reward, _, info = self.env.step(mapping_action, bandwidth_action)
            
            # 记录统计
            test_stats['rewards'].append(reward)
            test_stats['constraint_violations'].append(len(info['constraint_violations']))
            test_stats['total_actions'] += 1
            
            if info['is_valid']:
                test_stats['valid_actions'] += 1
            
            # 打印结果
            print(f"Episode {episode + 1:2d} | 奖励: {reward:6.3f} | 有效: {info['is_valid']} | 违反: {len(info['constraint_violations'])}")
        
        # 计算统计结果
        avg_reward = np.mean(test_stats['rewards'])
        avg_violations = np.mean(test_stats['constraint_violations'])
        valid_rate = test_stats['valid_actions'] / test_stats['total_actions']
        
        print(f"\n📊 测试结果:")
        print(f"   平均奖励: {avg_reward:.3f}")
        print(f"   平均约束违反: {avg_violations:.2f}")
        print(f"   有效动作率: {valid_rate:.2%}")
        
        return test_stats

def main():
    """主函数"""
    print("🎯 两阶段PPO网络调度器训练")
    print("=" * 60)
    
    # 创建训练器
    trainer = TwoStagePPOTrainer(
        num_physical_nodes_range=(5, 8),
        max_virtual_nodes_range=(3, 6),
        bandwidth_levels=10,
        physical_cpu_range=(50.0, 200.0),
        physical_memory_range=(100.0, 400.0),
        physical_bandwidth_range=(100.0, 1000.0),
        virtual_cpu_range=(10.0, 50.0),
        virtual_memory_range=(20.0, 100.0),
        virtual_bandwidth_range=(10.0, 200.0),
        physical_connectivity_prob=0.3,
        virtual_connectivity_prob=0.4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01
    )
    
    # 开始训练
    trainer.train(num_episodes=500, save_interval=100, eval_interval=50)
    
    # 测试智能体
    trainer.test_agent(num_test_episodes=10)
    
    print(f"\n🎉 训练和测试完成！")

if __name__ == "__main__":
    main() 