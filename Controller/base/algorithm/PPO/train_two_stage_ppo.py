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
                 stats_dir: str = "stats",
                 session_name: str = None):
        
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
        
        # 文件管理 - 创建唯一的会话文件夹
        self.session_name = session_name or self._generate_session_name()
        self.model_dir = os.path.join(model_dir, self.session_name)
        self.stats_dir = os.path.join(stats_dir, self.session_name)
        
        # 创建目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        print(f"📁 创建训练会话: {self.session_name}")
        print(f"   模型目录: {self.model_dir}")
        print(f"   统计目录: {self.stats_dir}")
        
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
        
        # 保存会话配置信息
        self._save_session_config()
    
    def _generate_session_name(self):
        """生成唯一的会话名称"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        import random
        session_id = random.randint(1000, 9999)
        return f"session_{timestamp}_{session_id}"
    
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
        print(f"重置环境")
        state = self.env.reset()
        
        # 获取动作
        print(f"获取动作")
        mapping_action, bandwidth_action, mapping_log_prob, bandwidth_log_prob, value, link_indices = self.agent.select_actions(state)
        
        # 执行动作
        print(f"执行动作")
        next_state, reward, done, info = self.env.step(mapping_action, bandwidth_action)
        
        # 存储经验
        print(f"存储经验")
        self.agent.store_transition(
            state, mapping_action, bandwidth_action, 
            reward, value, mapping_log_prob, bandwidth_log_prob, done
        )
        
        # 更新网络
        print(f"更新网络")
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
        model_path = os.path.join(self.model_dir, f"ppo_model_{self.session_name}_episode_{episode}.pth")
        
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
    
    def load_model(self, episode, session_name=None):
        """加载模型"""
        if session_name is None:
            session_name = self.session_name
        model_path = os.path.join(self.model_dir, f"ppo_model_{session_name}_episode_{episode}.pth")
        
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
        stats_path = os.path.join(self.stats_dir, f"training_stats_{self.session_name}_episode_{episode}.json")
        
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
            plot_path = os.path.join(self.stats_dir, f'training_curves_{self.session_name}.png')
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

    def _save_session_config(self):
        """保存会话配置信息"""
        config_path = os.path.join(self.stats_dir, f'session_config_{self.session_name}.json')
        
        config_data = {
            'timestamp': self.session_name.split('_')[-2], # 从会话名称提取时间戳
            'num_physical_nodes_range': self.num_physical_nodes_range,
            'max_virtual_nodes_range': self.max_virtual_nodes_range,
            'bandwidth_levels': self.bandwidth_levels,
            'physical_cpu_range': self.physical_cpu_range,
            'physical_memory_range': self.physical_memory_range,
            'physical_bandwidth_range': self.physical_bandwidth_range,
            'virtual_cpu_range': self.virtual_cpu_range,
            'virtual_memory_range': self.virtual_memory_range,
            'virtual_bandwidth_range': self.virtual_bandwidth_range,
            'physical_connectivity_prob': self.physical_connectivity_prob,
            'virtual_connectivity_prob': self.virtual_connectivity_prob,
            'lr': self.lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 会话配置已保存: {config_path}")
    
    @staticmethod
    def list_sessions(base_model_dir="models", base_stats_dir="stats"):
        """列出所有可用的训练会话"""
        sessions = []
        
        if os.path.exists(base_model_dir):
            for session_name in os.listdir(base_model_dir):
                session_path = os.path.join(base_model_dir, session_name)
                if os.path.isdir(session_path):
                    # 检查是否有配置文件
                    config_path = os.path.join(base_stats_dir, session_name, f'session_config_{session_name}.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            sessions.append({
                                'name': session_name,
                                'timestamp': config.get('timestamp', 'Unknown'),
                                'config': config
                            })
                        except:
                            sessions.append({
                                'name': session_name,
                                'timestamp': 'Unknown',
                                'config': {}
                            })
        
        return sessions
    
    @staticmethod
    def print_sessions(base_model_dir="models", base_stats_dir="stats"):
        """打印所有可用的训练会话"""
        sessions = TwoStagePPOTrainer.list_sessions(base_model_dir, base_stats_dir)
        
        if not sessions:
            print("📁 没有找到任何训练会话")
            return
        
        print(f"📁 找到 {len(sessions)} 个训练会话:")
        print("=" * 80)
        
        for i, session in enumerate(sessions, 1):
            print(f"{i:2d}. {session['name']}")
            print(f"    时间: {session['timestamp']}")
            if session['config']:
                config = session['config']
                print(f"    物理节点范围: {config.get('num_physical_nodes_range', 'N/A')}")
                print(f"    虚拟节点范围: {config.get('max_virtual_nodes_range', 'N/A')}")
                print(f"    学习率: {config.get('lr', 'N/A')}")
            print()

def main():
    """主函数"""
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # 列出所有会话
        TwoStagePPOTrainer.print_sessions()
        return
    
    print("🎯 两阶段PPO网络调度器训练")
    print("=" * 60)
    
    # 创建训练器
    trainer = TwoStagePPOTrainer(
        num_physical_nodes_range=(5, 8),
        max_virtual_nodes_range=(5, 8),
        bandwidth_levels=10,
        physical_cpu_range=(128, 256),
        physical_memory_range=(300, 500),
        physical_bandwidth_range=(100, 500),
        virtual_cpu_range=(5, 10),
        virtual_memory_range=(20, 50),
        virtual_bandwidth_range=(5, 15),
        physical_connectivity_prob=0.9,
        virtual_connectivity_prob=0.7,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01
    )
    
    # 开始训练
    trainer.train(num_episodes=6000, save_interval=100, eval_interval=50)
    
    # 测试智能体
    # trainer.test_agent(num_test_episodes=10)
    
    print(f"\n🎉 训练和测试完成！")
    print(f"📁 训练结果保存在: {trainer.stats_dir}")
    print(f"💾 模型保存在: {trainer.model_dir}")
    print(f"📋 使用 'python train_two_stage_ppo.py --list' 查看所有训练会话")

if __name__ == "__main__":
    main() 