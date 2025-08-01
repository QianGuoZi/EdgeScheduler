#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import time

# 导入自定义模块
from ppo import PPOAgent
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler, create_sample_topology, create_sample_virtual_work
from replaybuffer import PPOBuffer, EpisodeBuffer

def test_simple_training():
    """测试简化的训练过程"""
    print("=" * 50)
    print("测试PPO训练过程")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建较小的网络环境
    topology = create_sample_topology(num_nodes=4)
    virtual_work = create_sample_virtual_work(num_nodes=3)
    
    # 创建PPO智能体
    agent = PPOAgent(
        physical_node_dim=2,
        virtual_node_dim=2,
        num_physical_nodes=4,
        bandwidth_levels=8
    )
    
    # 创建缓冲区
    buffer = PPOBuffer(buffer_size=100)
    episode_buffer = EpisodeBuffer()
    
    print("网络环境:")
    print(f"  物理节点数: {topology.num_nodes}")
    print(f"  虚拟节点数: {virtual_work.num_nodes}")
    print(f"  带宽等级数: {agent.actor.bandwidth_levels}")
    
    # 训练几个episodes
    num_episodes = 5
    print(f"\n训练 {num_episodes} 个episodes...")
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # 创建调度器
        scheduler = NetworkScheduler(topology)
        episode_buffer.clear()
        
        episode_reward = 0
        episode_length = 0
        
        # 逐个调度虚拟节点
        for virtual_node_idx in range(virtual_work.num_nodes):
            # 构建状态
            physical_features = []
            for i in range(topology.num_nodes):
                available = topology.get_available_resources(i)
                total_cpu = topology.node_resources[i]['cpu']
                total_memory = topology.node_resources[i]['memory']
                features = [
                    available['cpu'] / total_cpu,
                    available['memory'] / total_memory
                ]
                physical_features.append(features)
            
            virtual_features = []
            for i in range(virtual_work.num_nodes):
                if i in virtual_work.node_requirements:
                    req = virtual_work.node_requirements[i]
                    features = [req['cpu'] / 100, req['memory'] / 200]
                else:
                    features = [0, 0]
                virtual_features.append(features)
            
            state = {
                'physical_features': torch.tensor(physical_features, dtype=torch.float32),
                'virtual_features': torch.tensor(virtual_features, dtype=torch.float32),
                'physical_edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                'virtual_edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                'current_virtual_node': virtual_node_idx
            }
            
            # 选择动作
            action, log_prob, value = agent.select_action(state, virtual_node_idx)
            
            # 执行动作
            mapping_action, bandwidth_action = action
            success = scheduler.schedule_node(virtual_node_idx, mapping_action)
            
            if success:
                # 分配带宽
                allocated_bandwidth = bandwidth_action * 5
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
            episode_buffer.store(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            
            print(f"  虚拟节点{virtual_node_idx} -> 物理节点{mapping_action}, 成功: {success}, 奖励: {reward:.3f}")
            
            if not success:
                break
        
        # 将episode数据添加到主缓冲区
        episode_data = episode_buffer.get_episode_data()
        for i in range(len(episode_data['states'])):
            buffer.store(
                episode_data['states'][i],
                episode_data['actions'][i],
                episode_data['rewards'][i],
                episode_data['values'][i],
                episode_data['log_probs'][i],
                episode_data['dones'][i]
            )
        
        print(f"  Episode奖励: {episode_reward:.3f}, 长度: {episode_length}")
        
        # 如果缓冲区有足够的数据，进行策略更新
        if len(buffer) >= 8:  # 使用较小的batch_size
            print("  进行策略更新...")
            
            # 计算优势函数
            buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)
            
            # 获取批次数据
            batch_data = buffer.get_batch(batch_size=8)
            
            # 更新网络
            states = batch_data['states']
            actions = batch_data['actions']
            advantages = batch_data['advantages'].to(agent.device)
            returns = batch_data['returns'].to(agent.device)
            old_log_probs = batch_data['old_log_probs'].to(agent.device)
            
            # 计算新的动作概率和价值
            new_log_probs = []
            new_values = []
            
            for i, state in enumerate(states):
                physical_features = state['physical_features'].to(agent.device)
                virtual_features = state['virtual_features'].to(agent.device)
                physical_edge_index = state['physical_edge_index'].to(agent.device)
                virtual_edge_index = state['virtual_edge_index'].to(agent.device)
                
                mapping_logits, bandwidth_logits = agent.actor(
                    physical_features, physical_edge_index, None,
                    virtual_features, virtual_edge_index, None,
                    state['current_virtual_node']
                )
                
                mapping_probs = torch.softmax(mapping_logits, dim=-1)
                bandwidth_probs = torch.softmax(bandwidth_logits, dim=-1)
                
                mapping_dist = torch.distributions.Categorical(mapping_probs)
                bandwidth_dist = torch.distributions.Categorical(bandwidth_probs)
                
                mapping_action, bandwidth_action = actions[i]
                mapping_log_prob = mapping_dist.log_prob(torch.tensor(mapping_action).to(agent.device))
                bandwidth_log_prob = bandwidth_dist.log_prob(torch.tensor(bandwidth_action).to(agent.device))
                
                total_log_prob = mapping_log_prob + bandwidth_log_prob
                new_log_probs.append(total_log_prob)
                
                value = agent.critic(physical_features, physical_edge_index, None,
                                   virtual_features, virtual_edge_index, None)
                new_values.append(value)
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - agent.clip_ratio, 1 + agent.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = torch.nn.functional.mse_loss(new_values, returns)
            
            # 熵损失
            entropy_loss = -(mapping_probs * torch.log(mapping_probs + 1e-8)).sum() - \
                          (bandwidth_probs * torch.log(bandwidth_probs + 1e-8)).sum()
            
            # 总损失
            total_loss = (actor_loss + 
                         agent.value_loss_coef * value_loss + 
                         agent.entropy_coef * entropy_loss)
            
            # 更新网络
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            total_loss.backward()
            agent.actor_optimizer.step()
            agent.critic_optimizer.step()
            
            print(f"  Actor损失: {actor_loss.item():.4f}")
            print(f"  Critic损失: {value_loss.item():.4f}")
            print(f"  熵损失: {entropy_loss.item():.4f}")
            
            # 清空缓冲区
            buffer.clear()
    
    print("\n训练完成！")
    
    # 测试训练后的智能体
    print("\n测试训练后的智能体:")
    
    # 创建新的测试环境
    test_topology = create_sample_topology(num_nodes=4)
    test_virtual_work = create_sample_virtual_work(num_nodes=3)
    test_scheduler = NetworkScheduler(test_topology)
    
    for virtual_node_idx in range(test_virtual_work.num_nodes):
        # 构建状态
        physical_features = []
        for i in range(test_topology.num_nodes):
            available = test_topology.get_available_resources(i)
            total_cpu = test_topology.node_resources[i]['cpu']
            total_memory = test_topology.node_resources[i]['memory']
            features = [
                available['cpu'] / total_cpu,
                available['memory'] / total_memory
            ]
            physical_features.append(features)
        
        virtual_features = []
        for i in range(test_virtual_work.num_nodes):
            if i in test_virtual_work.node_requirements:
                req = test_virtual_work.node_requirements[i]
                features = [req['cpu'] / 100, req['memory'] / 200]
            else:
                features = [0, 0]
            virtual_features.append(features)
        
        state = {
            'physical_features': torch.tensor(physical_features, dtype=torch.float32),
            'virtual_features': torch.tensor(virtual_features, dtype=torch.float32),
            'physical_edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            'virtual_edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            'current_virtual_node': virtual_node_idx
        }
        
        # 选择动作
        action, _, _ = agent.select_action(state, virtual_node_idx)
        mapping_action, bandwidth_action = action
        
        # 执行动作
        success = test_scheduler.schedule_node(virtual_node_idx, mapping_action)
        
        if success:
            allocated_bandwidth = bandwidth_action * 5
            for link_req in test_virtual_work.link_requirements:
                if (link_req['from'] == virtual_node_idx or 
                    link_req['to'] == virtual_node_idx):
                    test_scheduler.allocate_bandwidth(link_req['from'], 
                                                   link_req['to'], 
                                                   allocated_bandwidth)
        
        reward = test_scheduler.calculate_reward(test_virtual_work) if success else -10
        
        print(f"  测试 - 虚拟节点{virtual_node_idx} -> 物理节点{mapping_action}, 成功: {success}, 奖励: {reward:.3f}")
    
    final_reward = test_scheduler.calculate_reward(test_virtual_work)
    print(f"  最终测试奖励: {final_reward:.3f}")

if __name__ == "__main__":
    test_simple_training() 