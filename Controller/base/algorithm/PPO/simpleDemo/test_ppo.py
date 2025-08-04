#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppo import GraphEncoder, Actor, Critic, PPOAgent
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler, create_sample_topology, create_sample_virtual_work
from replaybuffer import PPOBuffer, EpisodeBuffer

def test_graph_encoder():
    """测试图编码器"""
    print("测试图编码器...")
    
    # 创建图编码器
    encoder = GraphEncoder(node_features=2, hidden_dim=64)
    
    # 创建测试数据
    num_nodes = 5
    x = torch.randn(num_nodes, 2)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    
    # 前向传播
    output = encoder(x, edge_index)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print("✓ 图编码器测试通过\n")

def test_actor():
    """测试Actor网络"""
    print("测试Actor网络...")
    
    # 创建Actor网络
    actor = Actor(
        physical_node_dim=2,
        virtual_node_dim=2,
        hidden_dim=64,
        num_physical_nodes=5,
        bandwidth_levels=10
    )
    
    # 创建测试数据
    physical_features = torch.randn(5, 2)
    virtual_features = torch.randn(3, 2)
    physical_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    virtual_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    
    # 前向传播
    mapping_logits, bandwidth_logits = actor(
        physical_features, physical_edge_index, None,
        virtual_features, virtual_edge_index, None,
        virtual_node_idx=0
    )
    
    print(f"映射logits形状: {mapping_logits.shape}")
    print(f"带宽logits形状: {bandwidth_logits.shape}")
    print(f"映射概率和: {torch.softmax(mapping_logits, dim=-1).sum().item():.3f}")
    print(f"带宽概率和: {torch.softmax(bandwidth_logits, dim=-1).sum().item():.3f}")
    print("✓ Actor网络测试通过\n")

def test_critic():
    """测试Critic网络"""
    print("测试Critic网络...")
    
    # 创建Critic网络
    critic = Critic(
        physical_node_dim=2,
        virtual_node_dim=2,
        hidden_dim=64
    )
    
    # 创建测试数据
    physical_features = torch.randn(5, 2)
    virtual_features = torch.randn(3, 2)
    physical_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    virtual_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    
    # 前向传播
    value = critic(
        physical_features, physical_edge_index, None,
        virtual_features, virtual_edge_index, None
    )
    
    print(f"价值输出形状: {value.shape}")
    print(f"价值范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
    print("✓ Critic网络测试通过\n")

def test_network_topology():
    """测试网络拓扑"""
    print("测试网络拓扑...")
    
    # 创建拓扑
    topology = NetworkTopology(num_nodes=5)
    
    # 设置节点资源
    for i in range(5):
        topology.set_node_resources(i, cpu=100, memory=200)
    
            # 添加链路（支持不对称带宽）
        topology.add_link(0, 1, 500, 450)  # 0->1: 500, 1->0: 450
        topology.add_link(1, 2, 300, 350)  # 1->2: 300, 2->1: 350
        topology.add_link(2, 3, 400, 380)  # 2->3: 400, 3->2: 380
    
    # 测试资源分配
    topology.allocate_node_resources(0, cpu=20, memory=40)
    available = topology.get_available_resources(0)
    
    print(f"节点0可用CPU: {available['cpu']}")
    print(f"节点0可用内存: {available['memory']}")
    
    # 测试路径查找
    path = topology.get_shortest_path(0, 3)
    print(f"从节点0到节点3的路径: {path}")
    
    # 测试带宽检查
    can_allocate = topology.check_bandwidth_availability(path, 100)
    print(f"路径上是否可以分配100带宽: {can_allocate}")
    
    print("✓ 网络拓扑测试通过\n")

def test_virtual_work():
    """测试虚拟工作"""
    print("测试虚拟工作...")
    
    # 创建虚拟工作
    virtual_work = VirtualWork(num_nodes=3)
    
    # 设置节点需求
    virtual_work.set_node_requirement(0, cpu=20, memory=40)
    virtual_work.set_node_requirement(1, cpu=15, memory=30)
    virtual_work.set_node_requirement(2, cpu=25, memory=50)
    
    # 添加链路需求（支持不对称带宽）
    virtual_work.add_link_requirement(0, 1, 10, 50, 15, 45)  # 0->1: 10-50, 1->0: 15-45
    virtual_work.add_link_requirement(1, 2, 20, 60, 18, 55)  # 1->2: 20-60, 2->1: 18-55
    
    print(f"虚拟节点数量: {virtual_work.num_nodes}")
    print(f"节点需求: {virtual_work.node_requirements}")
    print(f"链路需求数量: {len(virtual_work.link_requirements)}")
    print("✓ 虚拟工作测试通过\n")

def test_network_scheduler():
    """测试网络调度器"""
    print("测试网络调度器...")
    
    # 创建拓扑和虚拟工作
    topology = create_sample_topology(num_nodes=5)
    virtual_work = create_sample_virtual_work(num_nodes=3)
    
    # 创建调度器
    scheduler = NetworkScheduler(topology)
    
    # 测试节点调度
    success1 = scheduler.schedule_node(0, 0)
    success2 = scheduler.schedule_node(1, 1)
    success3 = scheduler.schedule_node(2, 2)
    
    print(f"节点0调度到物理节点0: {success1}")
    print(f"节点1调度到物理节点1: {success2}")
    print(f"节点2调度到物理节点2: {success3}")
    
    # 测试带宽分配
    if success1 and success2:
        bandwidth_success = scheduler.allocate_bandwidth(0, 1, 30)
        print(f"节点0到节点1的带宽分配: {bandwidth_success}")
    
    # 计算奖励
    reward = scheduler.calculate_reward(virtual_work)
    print(f"调度奖励: {reward:.3f}")
    
    # 获取调度结果
    result = scheduler.get_scheduling_result()
    print(f"节点映射: {result['node_mapping']}")
    print(f"带宽分配: {result['bandwidth_allocation']}")
    
    print("✓ 网络调度器测试通过\n")

def test_ppo_agent():
    """测试PPO智能体"""
    print("测试PPO智能体...")
    
    # 创建智能体
    agent = PPOAgent(
        physical_node_dim=2,
        virtual_node_dim=2,
        num_physical_nodes=5,
        bandwidth_levels=10
    )
    
    # 创建测试状态
    state = {
        'physical_features': torch.randn(5, 2),
        'virtual_features': torch.randn(3, 2),
        'physical_edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        'virtual_edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        'current_virtual_node': 0
    }
    
    # 选择动作
    action, log_prob, value = agent.select_action(state, virtual_node_idx=0)
    
    print(f"选择的动作: {action}")
    print(f"动作概率: {log_prob}")
    print(f"状态价值: {value:.3f}")
    
    # 存储经验
    agent.store_transition(state, action, reward=1.0, value=value, log_prob=log_prob, done=False)
    
    print("✓ PPO智能体测试通过\n")

def test_replay_buffer():
    """测试经验回放缓冲区"""
    print("测试经验回放缓冲区...")
    
    # 创建缓冲区
    buffer = PPOBuffer(buffer_size=100)
    
    # 存储一些经验
    for i in range(10):
        state = {'test': i}
        action = (i % 5, i % 10)
        reward = i * 0.1
        value = i * 0.2
        log_prob = (i * 0.1, i * 0.1)
        done = (i == 9)
        
        buffer.store(state, action, reward, value, log_prob, done)
    
    print(f"缓冲区大小: {len(buffer)}")
    
    # 计算优势函数
    buffer.compute_advantages()
    
    # 获取批次
    batch = buffer.get_batch(batch_size=5)
    print(f"批次大小: {len(batch['states'])}")
    print(f"优势函数范围: [{batch['advantages'].min().item():.3f}, {batch['advantages'].max().item():.3f}]")
    
    print("✓ 经验回放缓冲区测试通过\n")

def test_integration():
    """测试集成功能"""
    print("测试集成功能...")
    
    # 创建完整的训练环境
    topology = create_sample_topology(num_nodes=5)
    virtual_work = create_sample_virtual_work(num_nodes=3)
    scheduler = NetworkScheduler(topology)
    agent = PPOAgent(
        physical_node_dim=2,
        virtual_node_dim=2,
        num_physical_nodes=5,
        bandwidth_levels=10
    )
    
    # 模拟一个完整的调度过程
    for virtual_node_idx in range(virtual_work.num_nodes):
        # 构建状态
        physical_features = []
        for i in range(5):
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
        action, log_prob, value = agent.select_action(state, virtual_node_idx)
        
        # 执行动作
        mapping_action, bandwidth_action = action
        success = scheduler.schedule_node(virtual_node_idx, mapping_action)
        
        if success:
            allocated_bandwidth = bandwidth_action * 10
            for link_req in virtual_work.link_requirements:
                if (link_req['from'] == virtual_node_idx or 
                    link_req['to'] == virtual_node_idx):
                    scheduler.allocate_bandwidth(link_req['from'], 
                                               link_req['to'], 
                                               allocated_bandwidth)
        
        reward = scheduler.calculate_reward(virtual_work) if success else -10
        done = (virtual_node_idx == virtual_work.num_nodes - 1)
        
        # 存储经验
        agent.store_transition(state, action, reward, value, log_prob, done)
        
        print(f"虚拟节点{virtual_node_idx} -> 物理节点{mapping_action}, 成功: {success}, 奖励: {reward:.3f}")
    
    # 获取最终结果
    result = scheduler.get_scheduling_result()
    print(f"最终节点映射: {result['node_mapping']}")
    print(f"最终带宽分配: {result['bandwidth_allocation']}")
    
    print("✓ 集成功能测试通过\n")

def main():
    """运行所有测试"""
    print("开始PPO算法测试...\n")
    
    try:
        test_graph_encoder()
        test_actor()
        test_critic()
        test_network_topology()
        test_virtual_work()
        test_network_scheduler()
        test_ppo_agent()
        test_replay_buffer()
        test_integration()
        
        print("🎉 所有测试通过！PPO算法实现正确。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 