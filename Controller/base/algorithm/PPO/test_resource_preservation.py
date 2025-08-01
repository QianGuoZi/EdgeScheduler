#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from two_stage_environment import TwoStageNetworkSchedulerEnvironment

def test_resource_preservation():
    """测试资源保留功能"""
    print("🧪 测试资源保留功能")
    print("=" * 60)
    
    # 创建环境
    env = TwoStageNetworkSchedulerEnvironment(
        num_physical_nodes=3,
        max_virtual_nodes=4,
        bandwidth_levels=5,
        use_network_scheduler=True
    )
    
    # 重置环境
    state = env.reset()
    
    print("📊 初始物理节点状态:")
    physical_features = state['physical_features'].numpy()
    for i in range(env.num_physical_nodes):
        total_cpu = physical_features[i][0]
        total_memory = physical_features[i][1]
        cpu_usage = physical_features[i][2]
        memory_usage = physical_features[i][3]
        used_cpu = total_cpu * cpu_usage
        used_memory = total_memory * memory_usage
        
        print(f"   节点{i}: 总CPU={total_cpu}, 总内存={total_memory}")
        print(f"           CPU使用率={cpu_usage:.2%}, 已用CPU={used_cpu:.1f}")
        print(f"           内存使用率={memory_usage:.2%}, 已用内存={used_memory:.1f}")
    
    print(f"\n📊 初始网络调度器状态:")
    if env.network_topology:
        for i in range(env.num_physical_nodes):
            available = env.network_topology.get_available_resources(i)
            resources = env.network_topology.node_resources[i]
            print(f"   节点{i}: 可用CPU={available['cpu']:.1f}, 可用内存={available['memory']:.1f}")
            print(f"          已用CPU={resources['used_cpu']:.1f}, 已用内存={resources['used_memory']:.1f}")
    
    print(f"\n📊 虚拟工作需求:")
    virtual_features = state['virtual_features'].numpy()
    for i in range(env.virtual_work['num_nodes']):
        cpu_demand = virtual_features[i][0]
        memory_demand = virtual_features[i][1]
        print(f"   虚拟节点{i}: CPU需求={cpu_demand}, 内存需求={memory_demand}")
    
    # 测试第一次调度
    print(f"\n🔄 第一次调度测试:")
    mapping_action = np.array([0, 1, 2])  # 映射到不同节点
    bandwidth_action = np.array([2, 3])   # 带宽等级
    
    # 执行调度
    next_state, reward, done, info = env.step(mapping_action, bandwidth_action)
    
    print(f"   调度结果: 有效={info['is_valid']}, 奖励={reward:.4f}")
    
    # 检查调度后的资源状态
    print(f"\n📊 第一次调度后的资源状态:")
    if env.network_topology:
        for i in range(env.num_physical_nodes):
            available = env.network_topology.get_available_resources(i)
            resources = env.network_topology.node_resources[i]
            print(f"   节点{i}: 可用CPU={available['cpu']:.1f}, 可用内存={available['memory']:.1f}")
            print(f"          已用CPU={resources['used_cpu']:.1f}, 已用内存={resources['used_memory']:.1f}")
    
    # 测试第二次调度（应该重置到初始状态）
    print(f"\n🔄 第二次调度测试（应该重置到初始状态）:")
    mapping_action2 = np.array([1, 2, 0])  # 不同的映射
    bandwidth_action2 = np.array([1, 4])   # 不同的带宽
    
    # 执行调度
    next_state2, reward2, done2, info2 = env.step(mapping_action2, bandwidth_action2)
    
    print(f"   调度结果: 有效={info2['is_valid']}, 奖励={reward2:.4f}")
    
    # 检查第二次调度后的资源状态
    print(f"\n📊 第二次调度后的资源状态:")
    if env.network_topology:
        for i in range(env.num_physical_nodes):
            available = env.network_topology.get_available_resources(i)
            resources = env.network_topology.node_resources[i]
            print(f"   节点{i}: 可用CPU={available['cpu']:.1f}, 可用内存={available['memory']:.1f}")
            print(f"          已用CPU={resources['used_cpu']:.1f}, 已用内存={resources['used_memory']:.1f}")
    
    # 验证资源是否恢复到初始状态
    print(f"\n✅ 验证资源恢复:")
    initial_physical_features = state['physical_features'].numpy()
    current_physical_features = next_state2['physical_features'].numpy()
    
    for i in range(env.num_physical_nodes):
        initial_cpu_usage = initial_physical_features[i][2]
        current_cpu_usage = current_physical_features[i][2]
        initial_memory_usage = initial_physical_features[i][3]
        current_memory_usage = current_physical_features[i][3]
        
        cpu_match = abs(initial_cpu_usage - current_cpu_usage) < 0.01
        memory_match = abs(initial_memory_usage - current_memory_usage) < 0.01
        
        print(f"   节点{i}: CPU使用率匹配={cpu_match}, 内存使用率匹配={memory_match}")
        if not (cpu_match and memory_match):
            print(f"      初始: CPU={initial_cpu_usage:.2%}, 内存={initial_memory_usage:.2%}")
            print(f"      当前: CPU={current_cpu_usage:.2%}, 内存={current_memory_usage:.2%}")
    
    print(f"\n🎯 资源保留测试完成！")

def test_multiple_scheduling_attempts():
    """测试多次调度尝试"""
    print(f"\n🧪 测试多次调度尝试")
    print("=" * 60)
    
    # 创建环境
    env = TwoStageNetworkSchedulerEnvironment(
        num_physical_nodes=5,
        max_virtual_nodes=3,
        bandwidth_levels=5,
        use_network_scheduler=True
    )
    
    # 重置环境
    state = env.reset()
    
    print("📊 初始资源状态:")
    if env.network_topology:
        for i in range(env.num_physical_nodes):
            available = env.network_topology.get_available_resources(i)
            print(f"   节点{i}: 可用CPU={available['cpu']:.1f}, 可用内存={available['memory']:.1f}")
    
    # 尝试多次不同的调度
    mapping_attempts = [
        np.array([0, 1, 2]),  # 尝试1
        np.array([1, 2, 3]),  # 尝试2
        np.array([2, 3, 4]),  # 尝试3
        np.array([0, 2, 4]),  # 尝试4
    ]
    
    bandwidth_attempts = [
        np.array([2, 3]),     # 尝试1
        np.array([1, 4]),     # 尝试2
        np.array([3, 2]),     # 尝试3
        np.array([4, 1]),     # 尝试4
    ]
    
    for attempt in range(len(mapping_attempts)):
        print(f"\n🔄 调度尝试 {attempt + 1}:")
        
        # 执行调度
        next_state, reward, done, info = env.step(mapping_attempts[attempt], bandwidth_attempts[attempt])
        
        print(f"   映射: {mapping_attempts[attempt]}")
        print(f"   带宽: {bandwidth_attempts[attempt]}")
        print(f"   结果: 有效={info['is_valid']}, 奖励={reward:.4f}")
        
        # 检查资源状态
        if env.network_topology:
            print(f"   资源状态:")
            for i in range(env.num_physical_nodes):
                available = env.network_topology.get_available_resources(i)
                resources = env.network_topology.node_resources[i]
                print(f"     节点{i}: 可用CPU={available['cpu']:.1f}, 已用CPU={resources['used_cpu']:.1f}")
    
    print(f"\n✅ 多次调度尝试测试完成！")

if __name__ == "__main__":
    test_resource_preservation()
    test_multiple_scheduling_attempts() 