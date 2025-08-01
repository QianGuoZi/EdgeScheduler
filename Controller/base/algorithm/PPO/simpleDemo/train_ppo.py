#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import json
import os
from collections import deque

# 导入自定义模块
from ppo import PPOAgent, GraphEncoder, Actor, Critic
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler, create_sample_topology, create_sample_virtual_work
from replaybuffer import PPOBuffer, EpisodeBuffer

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 update_epochs: int = 10,
                 batch_size: int = 64,
                 model_dir: str = "models",
                 stats_dir: str = "stats",
                 # 节点数量范围
                 num_physical_nodes_range: Tuple[int, int] = (8, 15),
                 max_virtual_nodes_range: Tuple[int, int] = (5, 12),
                 bandwidth_levels: int = 10, # 带宽等级
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
                 virtual_connectivity_prob: float = 0.4):

        self.hidden_dim = hidden_dim
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.stats_dir = stats_dir
        
        # 资源范围参数
        self.num_physical_nodes_range = num_physical_nodes_range
        self.max_virtual_nodes_range = max_virtual_nodes_range
        self.bandwidth_levels = bandwidth_levels
        self.physical_cpu_range = physical_cpu_range
        self.physical_memory_range = physical_memory_range
        self.physical_bandwidth_range = physical_bandwidth_range
        self.virtual_cpu_range = virtual_cpu_range
        self.virtual_memory_range = virtual_memory_range
        self.virtual_bandwidth_range = virtual_bandwidth_range
        self.physical_connectivity_prob = physical_connectivity_prob
        self.virtual_connectivity_prob = virtual_connectivity_prob
        
        # 创建保存目录
        import os
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # 创建网络拓扑
        self.topology = self._create_custom_physical_topology()
        
        # 创建PPO智能体
        self.agent = PPOAgent(
            physical_node_dim=2,  # CPU, Memory
            virtual_node_dim=2,   # CPU需求, Memory需求
            num_physical_nodes=self.topology.num_nodes,
            bandwidth_levels=bandwidth_levels,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef
        )
        
        # 创建经验缓冲区
        self.buffer = PPOBuffer(buffer_size=10000)
        self.episode_buffer = EpisodeBuffer()
        
        # 训练统计
        self.training_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        
    def _create_custom_physical_topology(self) -> NetworkTopology:
        """创建物理网络拓扑"""
        # 随机选择物理节点数量
        num_physical_nodes = np.random.randint(self.num_physical_nodes_range[0], 
                                             self.num_physical_nodes_range[1] + 1)
        topology = NetworkTopology(num_physical_nodes)
        
        # 设置节点资源
        for i in range(num_physical_nodes):
            cpu = np.random.randint(self.physical_cpu_range[0], self.physical_cpu_range[1] + 1)
            memory = np.random.randint(self.physical_memory_range[0], self.physical_memory_range[1] + 1)
            topology.set_node_resources(i, cpu, memory)
        
        # 创建网络连接（部分连接）
        for i in range(num_physical_nodes):
            for j in range(i + 1, num_physical_nodes):
                if np.random.random() < self.physical_connectivity_prob:
                    bandwidth_1_to_2 = np.random.randint(self.physical_bandwidth_range[0], self.physical_bandwidth_range[1] + 1)
                    bandwidth_2_to_1 = np.random.randint(self.physical_bandwidth_range[0], self.physical_bandwidth_range[1] + 1)
                    topology.add_link(i, j, bandwidth_1_to_2, bandwidth_2_to_1)
        
        return topology
    
    def _create_custom_virtual_work(self) -> VirtualWork:
        """创建虚拟工作"""
        # 随机确定虚拟节点数量
        num_virtual_nodes = np.random.randint(self.max_virtual_nodes_range[0], 
                                            self.max_virtual_nodes_range[1] + 1)
        virtual_work = VirtualWork(num_virtual_nodes)
        
        # 设置节点需求
        for i in range(num_virtual_nodes):
            cpu = np.random.randint(self.virtual_cpu_range[0], self.virtual_cpu_range[1] + 1)
            memory = np.random.randint(self.virtual_memory_range[0], self.virtual_memory_range[1] + 1)
            virtual_work.set_node_requirement(i, cpu, memory)
        
        # 创建虚拟链路（支持不对称带宽）
        for i in range(num_virtual_nodes):
            for j in range(i + 1, num_virtual_nodes):
                if np.random.random() < self.virtual_connectivity_prob:
                    # 方向1: i -> j
                    min_bandwidth_1_to_2 = np.random.randint(self.virtual_bandwidth_range[0], self.virtual_bandwidth_range[1] * 0.5 + 1)
                    max_bandwidth_1_to_2 = np.random.randint(min_bandwidth_1_to_2, self.virtual_bandwidth_range[1] + 1)
                    
                    # 方向2: j -> i
                    min_bandwidth_2_to_1 = np.random.randint(self.virtual_bandwidth_range[0], self.virtual_bandwidth_range[1] * 0.5 + 1)
                    max_bandwidth_2_to_1 = np.random.randint(min_bandwidth_2_to_1, self.virtual_bandwidth_range[1] + 1)
                    
                    virtual_work.add_link_requirement(i, j, min_bandwidth_1_to_2, max_bandwidth_1_to_2,
                                                    min_bandwidth_2_to_1, max_bandwidth_2_to_1)
        
        return virtual_work
    
    def train_episode(self) -> Tuple[float, int]:
        """训练一个episode"""
        # 创建虚拟工作
        virtual_work = self._create_custom_virtual_work()
        
        # 创建调度器
        scheduler = NetworkScheduler(self.topology)
        
        # 重置episode缓冲区
        self.episode_buffer.clear()
        
        episode_reward = 0
        episode_length = 0
        
        # 逐个调度虚拟节点
        for virtual_node_idx in range(virtual_work.num_nodes):
            # 获取当前状态
            state = self._get_state(scheduler, virtual_work, virtual_node_idx)
            
            # 选择动作
            action, log_prob, value = self.agent.select_action(state, virtual_node_idx)
            
            # 执行动作
            mapping_action, bandwidth_action = action
            success = scheduler.schedule_node(virtual_node_idx, mapping_action)
            
            if success:
                # 分配带宽
                allocated_bandwidth = bandwidth_action * 10  # 转换为实际带宽值
                
                # 尝试为所有相关链路分配带宽
                for link_req in virtual_work.link_requirements:
                    if (link_req['from'] == virtual_node_idx or 
                        link_req['to'] == virtual_node_idx):
                        scheduler.allocate_bandwidth(link_req['from'], 
                                                   link_req['to'], 
                                                   allocated_bandwidth)
            
            # 计算奖励
            reward = scheduler.calculate_reward(virtual_work) if success else -10
            
            # 检查是否完成
            done = (virtual_node_idx == virtual_work.num_nodes - 1)
            
            # 存储经验
            self.episode_buffer.store(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            
            if not success:
                break
        
        # 将episode数据添加到主缓冲区
        episode_data = self.episode_buffer.get_episode_data()
        for i in range(len(episode_data['states'])):
            self.buffer.store(
                episode_data['states'][i],
                episode_data['actions'][i],
                episode_data['rewards'][i],
                episode_data['values'][i],
                episode_data['log_probs'][i],
                episode_data['dones'][i]
            )
        
        return episode_reward, episode_length
    
    def _get_state(self, scheduler: NetworkScheduler, 
                   virtual_work: VirtualWork, 
                   current_virtual_node: int) -> Dict:
        """获取当前状态"""
        # 构建物理节点特征
        physical_features = []
        for i in range(self.topology.num_nodes):
            available = self.topology.get_available_resources(i)
            total_cpu = self.topology.node_resources[i]['cpu']
            total_memory = self.topology.node_resources[i]['memory']
            
            features = [
                available['cpu'] / total_cpu,  # CPU利用率
                available['memory'] / total_memory  # 内存利用率
            ]
            physical_features.append(features)
        
        # 构建虚拟节点特征
        virtual_features = []
        for i in range(virtual_work.num_nodes):
            if i in virtual_work.node_requirements:
                req = virtual_work.node_requirements[i]
                features = [
                    req['cpu'] / 100,  # 归一化CPU需求
                    req['memory'] / 200  # 归一化内存需求
                ]
            else:
                features = [0, 0]
            virtual_features.append(features)
        
        # 构建边特征
        physical_edge_index = self._get_physical_edges()
        virtual_edge_index = self._get_virtual_edges(virtual_work)
        
        return {
            'physical_features': torch.tensor(physical_features, dtype=torch.float32),
            'virtual_features': torch.tensor(virtual_features, dtype=torch.float32),
            'physical_edge_index': physical_edge_index,
            'virtual_edge_index': virtual_edge_index,
            'current_virtual_node': current_virtual_node
        }
    
    def _get_physical_edges(self) -> torch.Tensor:
        """获取物理网络边"""
        edges = []
        for link_key in self.topology.links.keys():
            edges.append(list(link_key))
        
        if not edges:
            # 如果没有边，创建部分连接
            num_physical_nodes = self.topology.num_nodes
            for i in range(num_physical_nodes):
                for j in range(i + 1, num_physical_nodes):
                    if np.random.random() < self.physical_connectivity_prob:
                        edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.tensor([[0], [0]], dtype=torch.long)
    
    def _get_virtual_edges(self, virtual_work: VirtualWork) -> torch.Tensor:
        """获取虚拟网络边"""
        edges = []
        for link_req in virtual_work.link_requirements:
            edges.append([link_req['from'], link_req['to']])
        
        if not edges:
            # 如果没有边，创建部分连接
            for i in range(virtual_work.num_nodes):
                for j in range(i + 1, virtual_work.num_nodes):
                    if np.random.random() < self.virtual_connectivity_prob:
                        edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.tensor([[0], [0]], dtype=torch.long)
    
    def update_policy(self):
        """更新策略"""
        if len(self.buffer) < self.batch_size:
            return
        
        # 计算优势函数
        self.buffer.compute_advantages(
            gamma=self.agent.gamma,
            gae_lambda=self.agent.gae_lambda
        )
        
        # 获取批次数据
        batch_data = self.buffer.get_batch(self.batch_size)
        
        # 多次更新
        for epoch in range(self.update_epochs):
            actor_loss, critic_loss, entropy_loss = self._update_networks(batch_data)
            
            # 记录损失
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)
            self.entropy_losses.append(entropy_loss)
        
        # 清空缓冲区
        self.buffer.clear()
    
    def _update_networks(self, batch_data: Dict) -> Tuple[float, float, float]:
        """更新网络"""
        states = batch_data['states']
        actions = batch_data['actions']
        advantages = batch_data['advantages'].to(self.agent.device)
        returns = batch_data['returns'].to(self.agent.device)
        old_log_probs = batch_data['old_log_probs'].to(self.agent.device)
        
        # 计算新的动作概率和价值
        new_log_probs = []
        new_values = []
        
        for i, state in enumerate(states):
            physical_features = state['physical_features'].to(self.agent.device)
            virtual_features = state['virtual_features'].to(self.agent.device)
            physical_edge_index = state['physical_edge_index'].to(self.agent.device)
            virtual_edge_index = state['virtual_edge_index'].to(self.agent.device)
            
            mapping_logits, bandwidth_logits = self.agent.actor(
                physical_features, physical_edge_index, None,
                virtual_features, virtual_edge_index, None,
                state['current_virtual_node']
            )
            
            mapping_probs = F.softmax(mapping_logits, dim=-1)
            bandwidth_probs = F.softmax(bandwidth_logits, dim=-1)
            
            mapping_dist = torch.distributions.Categorical(mapping_probs)
            bandwidth_dist = torch.distributions.Categorical(bandwidth_probs)
            
            mapping_action, bandwidth_action = actions[i]
            mapping_log_prob = mapping_dist.log_prob(torch.tensor(mapping_action).to(self.agent.device))
            bandwidth_log_prob = bandwidth_dist.log_prob(torch.tensor(bandwidth_action).to(self.agent.device))
            
            new_log_probs.append(mapping_log_prob + bandwidth_log_prob)
            
            value = self.agent.critic(physical_features, physical_edge_index, None,
                                    virtual_features, virtual_edge_index, None)
            new_values.append(value)
        
        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(new_values).squeeze()
        
        # 计算比率 - 确保维度匹配
        # old_log_probs是(batch_size, 2)，我们需要提取总的log概率
        if old_log_probs.dim() == 2:
            old_total_log_probs = old_log_probs.sum(dim=1)  # 将两个log概率相加
        else:
            old_total_log_probs = old_log_probs
        
        ratio = torch.exp(new_log_probs - old_total_log_probs)
        
        # PPO损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.agent.clip_ratio, 1 + self.agent.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = F.mse_loss(new_values, returns)
        
        # 熵损失 - 使用最后一个状态的概率分布
        if len(new_log_probs) > 0:
            # 重新计算最后一个状态的概率分布用于熵计算
            last_state = states[-1]
            physical_features = last_state['physical_features'].to(self.agent.device)
            virtual_features = last_state['virtual_features'].to(self.agent.device)
            physical_edge_index = last_state['physical_edge_index'].to(self.agent.device)
            virtual_edge_index = last_state['virtual_edge_index'].to(self.agent.device)
            
            mapping_logits, bandwidth_logits = self.agent.actor(
                physical_features, physical_edge_index, None,
                virtual_features, virtual_edge_index, None,
                last_state['current_virtual_node']
            )
            
            mapping_probs = F.softmax(mapping_logits, dim=-1)
            bandwidth_probs = F.softmax(bandwidth_logits, dim=-1)
            
            entropy_loss = -(mapping_probs * torch.log(mapping_probs + 1e-8)).sum() - \
                          (bandwidth_probs * torch.log(bandwidth_probs + 1e-8)).sum()
        else:
            entropy_loss = torch.tensor(0.0).to(self.agent.device)
        
        # 总损失
        total_loss = (actor_loss + 
                     self.agent.value_loss_coef * value_loss + 
                     self.agent.entropy_coef * entropy_loss)
        
        # 更新网络
        self.agent.actor_optimizer.zero_grad()
        self.agent.critic_optimizer.zero_grad()
        total_loss.backward()
        self.agent.actor_optimizer.step()
        self.agent.critic_optimizer.step()
        
        return actor_loss.item(), value_loss.item(), entropy_loss.item()
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """训练PPO智能体"""
        print(f"开始训练PPO智能体，总episodes: {num_episodes}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # 训练一个episode
            episode_reward, episode_length = self.train_episode()
            
            # 记录统计信息
            self.training_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # 更新策略
            self.update_policy()
            
            # 打印进度
            if episode % 50 == 0:
                avg_reward = np.mean(self.training_rewards[-50:])
                avg_length = np.mean(self.episode_lengths[-50:])
                elapsed_time = time.time() - start_time
                
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                self.save_model(f"ppo_model_episode_{episode}.pth")
                self.save_training_stats(f"training_stats_episode_{episode}.json")
        
        # 最终保存
        self.save_model("ppo_model_final.pth")
        self.save_training_stats("training_stats_final.json")
        
        print(f"训练完成！总用时: {time.time() - start_time:.1f}s")
    
    def save_model(self, filename: str):
        """保存模型"""
        import os
        filepath = os.path.join(self.model_dir, filename)
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filename: str):
        """加载模型"""
        import os
        filepath = os.path.join(self.model_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.agent.device)
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"模型已从 {filepath} 加载")
    
    def save_training_stats(self, filename: str):
        """保存训练统计信息"""
        import os
        filepath = os.path.join(self.stats_dir, filename)
        stats = {
            'training_rewards': self.training_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropy_losses': self.entropy_losses
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"训练统计已保存到: {filepath}")
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        axes[0, 0].plot(self.training_rewards)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode长度曲线
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        
        # Actor损失曲线
        if self.actor_losses:
            axes[1, 0].plot(self.actor_losses)
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Critic损失曲线
        if self.critic_losses:
            axes[1, 1].plot(self.critic_losses)
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练曲线已保存到: {save_path}")

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建训练器
    trainer = PPOTrainer(
        # 物理节点数量范围
        num_physical_nodes_range=(8, 15),
        max_virtual_nodes_range=(5, 12),
        bandwidth_levels=10, # 带宽等级
        hidden_dim=128, # 隐藏层维度
        lr=3e-4, # 学习率
        gamma=0.99, # 折扣因子
        gae_lambda=0.95, # 优势函数参数
        clip_ratio=0.2, # 剪切比率
        value_loss_coef=0.5, # 价值损失系数
        entropy_coef=0.01, # 熵损失系数
        update_epochs=10, # 更新步数
        batch_size=64, # 批量大小
        model_dir="models", # 模型保存目录
        stats_dir="stats", # 统计信息保存目录
        # 物理节点资源范围
        physical_cpu_range=(50.0, 200.0), 
        physical_memory_range=(100.0, 400.0), 
        physical_bandwidth_range=(100.0, 1000.0), 
        # 虚拟节点资源范围
        virtual_cpu_range=(10.0, 50.0),
        virtual_memory_range=(20.0, 100.0),
        virtual_bandwidth_range=(10.0, 200.0),
        # 网络连接概率
        physical_connectivity_prob=0.3,
        virtual_connectivity_prob=0.4
    )
    
    # 开始训练
    trainer.train(num_episodes=1000, save_interval=100)
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    print("训练完成！")

if __name__ == "__main__":
    main() 