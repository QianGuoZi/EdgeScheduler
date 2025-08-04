#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

# 导入自定义模块
from ppo import PPOAgent
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler, create_sample_topology, create_sample_virtual_work
from train_ppo import PPOTrainer

def demo_basic_scheduling():
    """演示基本的调度功能"""
    print("=" * 50)
    print("演示1: 基本网络调度功能")
    print("=" * 50)
    
    # 创建网络拓扑
    topology = NetworkTopology(num_nodes=5)
    
    # 设置节点资源
    for i in range(5):
        topology.set_node_resources(i, cpu=100, memory=200)
    
    # 添加网络连接（支持不对称带宽）
    topology.add_link(0, 1, 500, 480)  # 0->1: 500, 1->0: 480
    topology.add_link(1, 2, 300, 320)  # 1->2: 300, 2->1: 320
    topology.add_link(2, 3, 400, 380)  # 2->3: 400, 3->2: 380
    topology.add_link(3, 4, 350, 360)  # 3->4: 350, 4->3: 360
    topology.add_link(0, 4, 200, 220)  # 0->4: 200, 4->0: 220
    
    # 创建虚拟工作
    virtual_work = VirtualWork(num_nodes=3)
    virtual_work.set_node_requirement(0, cpu=30, memory=60)
    virtual_work.set_node_requirement(1, cpu=25, memory=50)
    virtual_work.set_node_requirement(2, cpu=35, memory=70)
    
    virtual_work.add_link_requirement(0, 1, 20, 80, 25, 75)  # 0->1: 20-80, 1->0: 25-75
    virtual_work.add_link_requirement(1, 2, 30, 100, 28, 95)  # 1->2: 30-100, 2->1: 28-95
    virtual_work.add_link_requirement(0, 2, 15, 60, 18, 55)  # 0->2: 15-60, 2->0: 18-55
    
    # 创建调度器
    scheduler = NetworkScheduler(topology)
    
    print("物理网络拓扑:")
    print(f"  节点数量: {topology.num_nodes}")
    print(f"  链路数量: {len(topology.physical_links) // 2}")
    print(f"  总CPU资源: {sum(topology.pysical_node_resources[i]['cpu'] for i in range(5))}")
    print(f"  总内存资源: {sum(topology.pysical_node_resources[i]['memory'] for i in range(5))}")
    
    print("\n虚拟工作需求:")
    print(f"  虚拟节点数量: {virtual_work.num_nodes}")
    print(f"  虚拟链路数量: {len(virtual_work.link_requirements)}")
    print(f"  总CPU需求: {sum(virtual_work.node_requirements[i]['cpu'] for i in range(3))}")
    print(f"  总内存需求: {sum(virtual_work.node_requirements[i]['memory'] for i in range(3))}")
    
    # 执行调度
    print("\n执行调度...")
    
    # 手动调度示例
    mapping_success = [
        scheduler.schedule_node(0, 0),  # 虚拟节点0 -> 物理节点0
        scheduler.schedule_node(1, 1),  # 虚拟节点1 -> 物理节点1
        scheduler.schedule_node(2, 2),  # 虚拟节点2 -> 物理节点2
    ]
    
    # 分配带宽
    bandwidth_success = [
        scheduler.allocate_bandwidth(0, 1, 50),  # 节点0到节点1分配50带宽
        scheduler.allocate_bandwidth(1, 2, 70),  # 节点1到节点2分配70带宽
        scheduler.allocate_bandwidth(0, 2, 40),  # 节点0到节点2分配40带宽
    ]
    
    # 获取结果
    result = scheduler.get_scheduling_result()
    reward = scheduler.calculate_reward(virtual_work)
    
    print("\n调度结果:")
    print(f"  节点映射成功: {sum(mapping_success)}/{len(mapping_success)}")
    print(f"  带宽分配成功: {sum(bandwidth_success)}/{len(bandwidth_success)}")
    print(f"  节点映射: {result['node_mapping']}")
    print(f"  带宽分配: {result['bandwidth_allocation']}")
    print(f"  调度奖励: {reward:.3f}")
    
    # 显示资源利用率
    print("\n资源利用率:")
    for i in range(5):
        available = topology.get_available_resources(i)
        total_cpu = topology.pysical_node_resources[i]['cpu']
        total_memory = topology.pysical_node_resources[i]['memory']
        cpu_util = (total_cpu - available['cpu']) / total_cpu * 100
        memory_util = (total_memory - available['memory']) / total_memory * 100
        print(f"  物理节点{i}: CPU利用率 {cpu_util:.1f}%, 内存利用率 {memory_util:.1f}%")

def demo_ppo_training():
    """演示PPO训练过程"""
    print("\n" + "=" * 50)
    print("演示2: PPO算法训练过程")
    print("=" * 50)
    
    # 创建训练器（使用较小的规模进行快速演示）
    trainer = PPOTrainer(
        num_physical_nodes_range=(5, 7),
        max_virtual_nodes_range=(3, 5),
        bandwidth_levels=8,
        hidden_dim=64,
        lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        update_epochs=5,
        batch_size=32
    )
    
    print("开始PPO训练...")
    print("训练参数:")
    print(f"  物理节点范围: {trainer.num_physical_nodes_range}")
    print(f"  虚拟节点范围: {trainer.max_virtual_nodes_range}")
    print(f"  带宽等级数: {trainer.bandwidth_levels}")
    print(f"  隐藏层维度: {trainer.hidden_dim}")
    print(f"  学习率: {trainer.agent.actor_optimizer.param_groups[0]['lr']}")
    
    # 训练少量episodes进行演示
    num_demo_episodes = 50
    print(f"\n训练 {num_demo_episodes} 个episodes...")
    
    start_time = time.time()
    rewards = []
    
    for episode in range(num_demo_episodes):
        # 训练一个episode
        episode_reward, episode_length = trainer.train_episode()
        rewards.append(episode_reward)
        
        # 更新策略
        trainer.update_policy()
        
        # 打印进度
        if episode % 10 == 0:
            avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
            print(f"  Episode {episode}/{num_demo_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Length: {episode_length}")
    
    training_time = time.time() - start_time
    
    print(f"\n训练完成！")
    print(f"  总用时: {training_time:.1f}秒")
    print(f"  平均奖励: {np.mean(rewards):.3f}")
    print(f"  最佳奖励: {np.max(rewards):.3f}")
    print(f"  最差奖励: {np.min(rewards):.3f}")
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.7, label='Episode Reward')
    plt.plot(np.convolve(rewards, np.ones(5)/5, mode='valid'), 
             label='Moving Average (5)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ppo_training_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"训练曲线已保存到: ppo_training_demo.png")

def demo_intelligent_scheduling():
    """演示智能调度效果"""
    print("\n" + "=" * 50)
    print("演示3: 智能调度效果对比")
    print("=" * 50)
    
    # 创建相同的网络环境
    topology = create_sample_topology(num_nodes=8)
    virtual_work = create_sample_virtual_work(num_nodes=5)
    
    print("网络环境:")
    print(f"  物理节点数: {topology.num_nodes}")
    print(f"  虚拟节点数: {virtual_work.num_nodes}")
    
    # 1. 随机调度
    print("\n1. 随机调度策略:")
    random_rewards = []
    
    for trial in range(10):
        scheduler = NetworkScheduler(topology)
        trial_reward = 0
        
        for virtual_node_idx in range(virtual_work.num_nodes):
            # 随机选择物理节点
            physical_node = np.random.randint(0, topology.num_nodes)
            success = scheduler.schedule_node(virtual_node_idx, physical_node)
            
            if success:
                # 随机分配带宽
                bandwidth = np.random.uniform(10, 50)
                for link_req in virtual_work.link_requirements:
                    if (link_req['from'] == virtual_node_idx or 
                        link_req['to'] == virtual_node_idx):
                        scheduler.allocate_bandwidth(link_req['from'], 
                                                   link_req['to'], 
                                                   bandwidth)
                
                reward = scheduler.calculate_reward(virtual_work)
                trial_reward += reward
        
        random_rewards.append(trial_reward)
    
    print(f"  平均奖励: {np.mean(random_rewards):.3f}")
    print(f"  标准差: {np.std(random_rewards):.3f}")
    
    # 2. 贪心调度（选择资源最多的节点）
    print("\n2. 贪心调度策略:")
    greedy_rewards = []
    
    for trial in range(10):
        scheduler = NetworkScheduler(topology)
        trial_reward = 0
        
        for virtual_node_idx in range(virtual_work.num_nodes):
            # 选择可用资源最多的物理节点
            best_node = -1
            max_available = -1
            
            for physical_node in range(topology.num_nodes):
                available = topology.get_available_resources(physical_node)
                total_available = available['cpu'] + available['memory']
                
                if total_available > max_available:
                    # 检查是否满足需求
                    if virtual_node_idx in virtual_work.node_requirements:
                        req = virtual_work.node_requirements[virtual_node_idx]
                        if (available['cpu'] >= req['cpu'] and 
                            available['memory'] >= req['memory']):
                            max_available = total_available
                            best_node = physical_node
            
            if best_node >= 0:
                success = scheduler.schedule_node(virtual_node_idx, best_node)
                if success:
                    # 分配最大带宽
                    bandwidth = 50
                    for link_req in virtual_work.link_requirements:
                        if (link_req['from'] == virtual_node_idx or 
                            link_req['to'] == virtual_node_idx):
                            scheduler.allocate_bandwidth(link_req['from'], 
                                                       link_req['to'], 
                                                       bandwidth)
                    
                    reward = scheduler.calculate_reward(virtual_work)
                    trial_reward += reward
        
        greedy_rewards.append(trial_reward)
    
    print(f"  平均奖励: {np.mean(greedy_rewards):.3f}")
    print(f"  标准差: {np.std(greedy_rewards):.3f}")
    
    # 3. PPO智能调度（使用训练好的模型）
    print("\n3. PPO智能调度策略:")
    
    # 创建PPO智能体
    agent = PPOAgent(
        physical_node_dim=2,
        virtual_node_dim=2,
        num_physical_nodes=8,
        bandwidth_levels=10
    )
    
    ppo_rewards = []
    
    for trial in range(10):
        scheduler = NetworkScheduler(topology)
        trial_reward = 0
        
        for virtual_node_idx in range(virtual_work.num_nodes):
            # 构建状态
            physical_features = []
            for i in range(topology.num_nodes):
                available = topology.get_available_resources(i)
                total_cpu = topology.pysical_node_resources[i]['cpu']
                total_memory = topology.pysical_node_resources[i]['memory']
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
            action, _, _ = agent.select_action(state, virtual_node_idx)
            mapping_action, bandwidth_action = action
            
            # 执行动作
            success = scheduler.schedule_node(virtual_node_idx, mapping_action)
            
            if success:
                allocated_bandwidth = bandwidth_action * 5  # 转换为实际带宽值
                for link_req in virtual_work.link_requirements:
                    if (link_req['from'] == virtual_node_idx or 
                        link_req['to'] == virtual_node_idx):
                        scheduler.allocate_bandwidth(link_req['from'], 
                                                   link_req['to'], 
                                                   allocated_bandwidth)
                
                reward = scheduler.calculate_reward(virtual_work)
                trial_reward += reward
        
        ppo_rewards.append(trial_reward)
    
    print(f"  平均奖励: {np.mean(ppo_rewards):.3f}")
    print(f"  标准差: {np.std(ppo_rewards):.3f}")
    
    # 对比结果
    print("\n策略对比:")
    print(f"  随机调度:     {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}")
    print(f"  贪心调度:     {np.mean(greedy_rewards):.3f} ± {np.std(greedy_rewards):.3f}")
    print(f"  PPO调度:      {np.mean(ppo_rewards):.3f} ± {np.std(ppo_rewards):.3f}")
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    strategies = ['随机调度', '贪心调度', 'PPO调度']
    means = [np.mean(random_rewards), np.mean(greedy_rewards), np.mean(ppo_rewards)]
    stds = [np.std(random_rewards), np.std(greedy_rewards), np.std(ppo_rewards)]
    
    bars = plt.bar(strategies, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('平均奖励')
    plt.title('不同调度策略性能对比')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('scheduling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"对比图已保存到: scheduling_comparison.png")

def main():
    """主函数"""
    print("PPO网络资源调度算法演示")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 演示1: 基本调度功能
        demo_basic_scheduling()
        
        # 演示2: PPO训练过程
        demo_ppo_training()
        
        # 演示3: 智能调度效果对比
        demo_intelligent_scheduling()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 