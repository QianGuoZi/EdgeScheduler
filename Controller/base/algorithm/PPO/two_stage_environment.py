#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler

class TwoStageNetworkSchedulerEnvironment:
    """支持两阶段动作的网络调度环境，集成network_scheduler功能"""
    
    def __init__(self, 
                 num_physical_nodes: int = 10,
                 max_virtual_nodes: int = 8,
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
                 # 是否使用network_scheduler
                 use_network_scheduler: bool = True):
        
        self.num_physical_nodes = num_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
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
        
        # 是否使用network_scheduler
        self.use_network_scheduler = use_network_scheduler
        
        # 带宽等级到实际带宽的映射
        self.bandwidth_mapping = self._create_bandwidth_mapping()
        
        # 环境状态
        self.physical_state = None
        self.virtual_work = None
        self.current_step = 0
        self.max_steps = 1  # 两阶段：一步完成所有映射和带宽分配
        
        # 调度结果
        self.mapping_result = None
        self.bandwidth_result = None
        
        # network_scheduler相关对象
        self.network_topology = None
        self.virtual_work_obj = None
        self.network_scheduler = None
    
    def _create_bandwidth_mapping(self):
        """创建带宽等级到实际带宽的映射"""
        # 10个等级，从最小到最大带宽
        min_bandwidth = self.virtual_bandwidth_range[0]
        max_bandwidth = self.virtual_bandwidth_range[1]
        
        bandwidths = np.linspace(min_bandwidth, max_bandwidth, self.bandwidth_levels)
        return {i: int(bandwidths[i]) for i in range(self.bandwidth_levels)}
    
    def _initialize_network_scheduler(self):
        """初始化network_scheduler相关对象"""
        # 创建网络拓扑
        self.network_topology = NetworkTopology(self.num_physical_nodes)
        
        # 设置物理节点资源（包含已使用资源）
        physical_features = self.physical_state['features'].numpy()
        for i in range(self.num_physical_nodes):
            total_cpu = physical_features[i][0]
            total_memory = physical_features[i][1]
            cpu_usage = physical_features[i][2]      # 当前CPU使用率
            memory_usage = physical_features[i][3]   # 当前内存使用率
            
            # 计算已使用的资源
            used_cpu = total_cpu * cpu_usage
            used_memory = total_memory * memory_usage
            
            # 设置资源（包含已使用量）
            self.network_topology.set_node_resources(i, total_cpu, total_memory, used_cpu, used_memory)
        
        # 设置物理网络连接
        physical_edges = self.physical_state['edges'].numpy()
        physical_edge_features = self.physical_state['edge_features'].numpy()
        
        for i, (src, dst) in enumerate(physical_edges.T):
            bandwidth = physical_edge_features[i][0]
            bandwidth_usage = physical_edge_features[i][1]  # 当前带宽使用率
            
            # 计算已使用的带宽
            used_bandwidth = bandwidth * bandwidth_usage
            
            # 设置链路（包含已使用带宽） 对称链路
            self.network_topology.add_link(src, dst, bandwidth, bandwidth, used_bandwidth, used_bandwidth)
        
        # 创建虚拟工作对象
        num_virtual_nodes = self.virtual_work['num_nodes']
        self.virtual_work_obj = VirtualWork(num_virtual_nodes)
        
        # 设置虚拟节点需求
        virtual_features = self.virtual_work['features'].numpy()
        for i in range(num_virtual_nodes):
            cpu_demand = virtual_features[i][0]
            memory_demand = virtual_features[i][1]
            self.virtual_work_obj.set_node_requirement(i, cpu_demand, memory_demand)
        
        # 设置虚拟链路需求
        virtual_edges = self.virtual_work['edges'].numpy()
        virtual_edge_features = self.virtual_work['edge_features'].numpy()
        
        for i, (src, dst) in enumerate(virtual_edges.T):
            min_bandwidth = virtual_edge_features[i][0]
            max_bandwidth = virtual_edge_features[i][1]
            # 假设对称带宽需求
            self.virtual_work_obj.add_link_requirement(src, dst, min_bandwidth, max_bandwidth, min_bandwidth, max_bandwidth)
        
        # 创建网络调度器
        self.network_scheduler = NetworkScheduler(self.network_topology)
    
    def reset(self, physical_state=None, virtual_work=None):
        """
        重置环境
        
        Args:
            physical_state: 物理网络状态
            virtual_work: 虚拟工作需求
        """
        self.current_step = 0
        
        # 生成或使用提供的物理状态
        if physical_state is None:
            self.physical_state = self._generate_physical_state()
        else:
            self.physical_state = physical_state
        
        # 生成或使用提供的虚拟工作
        if virtual_work is None:
            self.virtual_work = self._generate_virtual_work()
        else:
            self.virtual_work = virtual_work
        
        # 重置调度结果
        self.mapping_result = None
        self.bandwidth_result = None
        
        # 如果使用network_scheduler，初始化相关对象
        if self.use_network_scheduler:
            self._initialize_network_scheduler()
        
        return self._get_state()
    
    def _generate_physical_state(self):
        """生成随机物理网络状态"""
        # 物理节点特征：CPU, 内存, 当前CPU使用率, 当前内存使用率
        physical_features = []
        
        for i in range(self.num_physical_nodes):
            cpu = np.random.randint(*self.physical_cpu_range)
            memory = np.random.randint(*self.physical_memory_range)
            cpu_usage = np.random.uniform(0.1, 0.8)  # 当前使用率
            memory_usage = np.random.uniform(0.1, 0.8)
            
            physical_features.append([cpu, memory, cpu_usage, memory_usage])
        
        # 物理网络边
        physical_edges = self._get_physical_edges()
        
        # 物理网络边特征：带宽, 当前带宽使用率
        physical_edge_features = []
        for edge in physical_edges.T:
            bandwidth = np.random.randint(*self.physical_bandwidth_range)
            bandwidth_usage = np.random.uniform(0.1, 0.7)
            physical_edge_features.append([bandwidth, bandwidth_usage])
        
        return {
            'features': torch.tensor(physical_features, dtype=torch.float32),
            'edges': physical_edges,
            'edge_features': torch.tensor(physical_edge_features, dtype=torch.float32)
        }
    
    def _generate_virtual_work(self):
        """生成随机虚拟工作需求"""
        num_virtual_nodes = np.random.randint(2, self.max_virtual_nodes + 1)
        
        # 虚拟节点特征：CPU需求, 内存需求
        virtual_features = []
        
        for i in range(num_virtual_nodes):
            cpu_demand = np.random.randint(*self.virtual_cpu_range)
            memory_demand = np.random.randint(*self.virtual_memory_range)
            
            virtual_features.append([cpu_demand, memory_demand])
        
        # 虚拟网络边
        virtual_edges = self._get_virtual_edges(num_virtual_nodes)
        
        # 虚拟网络边特征：最小带宽需求, 最大带宽需求
        virtual_edge_features = []
        for edge in virtual_edges.T:
            min_bandwidth = np.random.randint(self.virtual_bandwidth_range[0], 
                                            int(self.virtual_bandwidth_range[1] * 0.5) + 1)
            max_bandwidth = np.random.randint(min_bandwidth, self.virtual_bandwidth_range[1] + 1)
            virtual_edge_features.append([min_bandwidth, max_bandwidth])
        
        return {
            'features': torch.tensor(virtual_features, dtype=torch.float32),
            'edges': virtual_edges,
            'edge_features': torch.tensor(virtual_edge_features, dtype=torch.float32),
            'num_nodes': num_virtual_nodes
        }
    
    def _get_physical_edges(self):
        """获取物理网络边（部分连接）"""
        edges = []
        # 使用部分连接
        for i in range(self.num_physical_nodes):
            for j in range(i + 1, self.num_physical_nodes):
                if np.random.random() < self.physical_connectivity_prob:
                    edges.append([i, j])
                    edges.append([j, i])  # 添加反向边
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.tensor([[0], [0]], dtype=torch.long)
    
    def _get_virtual_edges(self, num_virtual_nodes):
        """获取虚拟网络边"""
        edges = []
        # 使用部分连接
        for i in range(num_virtual_nodes):
            for j in range(i + 1, num_virtual_nodes):
                if np.random.random() < self.virtual_connectivity_prob:
                    edges.append([i, j])
                    edges.append([j, i])  # 添加反向边
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.tensor([[0], [0]], dtype=torch.long)
    
    def _get_state(self):
        """获取当前状态"""
        return {
            'physical_features': self.physical_state['features'],
            'physical_edge_index': self.physical_state['edges'],
            'physical_edge_attr': self.physical_state['edge_features'],
            'virtual_features': self.virtual_work['features'],
            'virtual_edge_index': self.virtual_work['edges'],
            'virtual_edge_attr': self.virtual_work['edge_features']
        }
    
    def step(self, mapping_action, bandwidth_action):
        """
        执行两阶段动作
        
        Args:
            mapping_action: 映射动作 [num_virtual_nodes] (物理节点索引)
            bandwidth_action: 带宽动作 [num_links] (带宽等级)
        
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.current_step += 1
        
        # 存储结果
        self.mapping_result = mapping_action
        self.bandwidth_result = bandwidth_action
        
        # 如果使用network_scheduler，先重置调度器
        if self.use_network_scheduler:
            self.network_scheduler.reset()
        
        # 验证动作的有效性
        is_valid, constraint_violations = self._validate_actions(mapping_action, bandwidth_action)
        
        if not is_valid:
            # 无效动作给予负奖励
            reward = -10.0
            info = {
                'constraint_violations': constraint_violations,
                'mapping_result': mapping_action,
                'bandwidth_result': bandwidth_action,
                'is_valid': False
            }
        else:
            # 如果使用network_scheduler，执行调度
            if self.use_network_scheduler:
                self._execute_network_scheduler_actions(mapping_action, bandwidth_action)
            
            # 计算奖励
            reward = self._calculate_reward(mapping_action, bandwidth_action)
            info = {
                'constraint_violations': [],
                'mapping_result': mapping_action,
                'bandwidth_result': bandwidth_action,
                'is_valid': True
            }
        
        # 环境结束
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done, info
    
    def _execute_network_scheduler_actions(self, mapping_action, bandwidth_action):
        """使用network_scheduler执行调度动作"""
        # 执行节点映射
        for virtual_node, physical_node in enumerate(mapping_action):
            success = self.network_scheduler.schedule_node(virtual_node, physical_node)
            if not success:
                print(f"警告：虚拟节点{virtual_node}映射到物理节点{physical_node}失败")
        
        # 执行带宽分配
        virtual_edges = self.virtual_work['edges'].numpy()
        for i, (src, dst) in enumerate(virtual_edges.T):
            if i < len(bandwidth_action):
                allocated_bandwidth = self.bandwidth_mapping[bandwidth_action[i]]
                success = self.network_scheduler.allocate_bandwidth(src, dst, allocated_bandwidth)
                if not success:
                    print(f"警告：虚拟链路({src},{dst})带宽分配{allocated_bandwidth}失败")
    
    def _validate_actions(self, mapping_action, bandwidth_action):
        """验证动作的有效性"""
        constraint_violations = []
        
        # 检查映射动作
        if len(mapping_action) != self.virtual_work['num_nodes']:
            constraint_violations.append("映射动作长度不匹配")
            return False, constraint_violations
        
        # 检查物理节点索引范围
        if np.any(mapping_action < 0) or np.any(mapping_action >= self.num_physical_nodes):
            constraint_violations.append("物理节点索引超出范围")
            return False, constraint_violations
        
        # 如果使用network_scheduler，使用其验证逻辑
        if self.use_network_scheduler:
            return self._validate_actions_with_network_scheduler(mapping_action, bandwidth_action)
        
        # 原有的验证逻辑
        # 检查资源约束
        physical_features = self.physical_state['features'].numpy()
        virtual_features = self.virtual_work['features'].numpy()
        
        for i, physical_node_idx in enumerate(mapping_action):
            # CPU约束
            required_cpu = virtual_features[i][0]
            available_cpu = physical_features[physical_node_idx][0] * (1 - physical_features[physical_node_idx][2])
            
            if required_cpu > available_cpu:
                constraint_violations.append(f"节点{i}的CPU需求({required_cpu})超过物理节点{physical_node_idx}的可用CPU({available_cpu:.1f})")
            
            # 内存约束
            required_memory = virtual_features[i][1]
            available_memory = physical_features[physical_node_idx][1] * (1 - physical_features[physical_node_idx][3])
            
            if required_memory > available_memory:
                constraint_violations.append(f"节点{i}的内存需求({required_memory})超过物理节点{physical_node_idx}的可用内存({available_memory:.1f})")
        
        # 检查带宽约束
        if len(bandwidth_action) > 0:
            virtual_edges = self.virtual_work['edges'].numpy()
            virtual_edge_features = self.virtual_work['edge_features'].numpy()
            physical_edges = self.physical_state['edges'].numpy()
            physical_edge_features = self.physical_state['edge_features'].numpy()
            
            for i, (src, dst) in enumerate(virtual_edges.T):
                if i >= len(bandwidth_action):
                    break
                
                # 获取分配的带宽
                allocated_bandwidth = self.bandwidth_mapping[bandwidth_action[i]]
                
                # 检查带宽需求约束
                min_required = virtual_edge_features[i][0]
                max_required = virtual_edge_features[i][1]
                
                if allocated_bandwidth < min_required:
                    constraint_violations.append(f"链路({src},{dst})的带宽分配({allocated_bandwidth})低于最小需求({min_required})")
                
                if allocated_bandwidth > max_required:
                    constraint_violations.append(f"链路({src},{dst})的带宽分配({allocated_bandwidth})超过最大需求({max_required})")
                
                # 检查物理路径带宽约束（简化版本）
                src_physical = mapping_action[src]
                dst_physical = mapping_action[dst]
                
                if src_physical != dst_physical:
                    # 需要检查物理路径上的带宽
                    # 这里简化处理，假设直接连接
                    for j, (p_src, p_dst) in enumerate(physical_edges.T):
                        if (p_src == src_physical and p_dst == dst_physical) or \
                           (p_src == dst_physical and p_dst == src_physical):
                            available_bandwidth = physical_edge_features[j][0] * (1 - physical_edge_features[j][1])
                            if allocated_bandwidth > available_bandwidth:
                                constraint_violations.append(f"物理链路({p_src},{p_dst})的可用带宽({available_bandwidth})不足以支持分配的带宽({allocated_bandwidth})")
                            break
        
        return len(constraint_violations) == 0, constraint_violations
    
    def _validate_actions_with_network_scheduler(self, mapping_action, bandwidth_action):
        """使用network_scheduler验证动作的有效性"""
        constraint_violations = []
        
        # 检查映射动作长度
        if len(mapping_action) != self.virtual_work['num_nodes']:
            constraint_violations.append("映射动作长度不匹配")
            return False, constraint_violations
        
        # 检查物理节点索引范围
        if np.any(mapping_action < 0) or np.any(mapping_action >= self.num_physical_nodes):
            constraint_violations.append("物理节点索引超出范围")
            return False, constraint_violations
        
        # 使用network_scheduler验证节点映射
        for virtual_node, physical_node in enumerate(mapping_action):
            # 检查节点资源是否足够
            if not self.network_scheduler._check_node_resources(virtual_node, physical_node):
                constraint_violations.append(f"虚拟节点{virtual_node}的资源需求超过物理节点{physical_node}的可用资源")
        
        # 检查带宽约束
        virtual_edges = self.virtual_work['edges'].numpy()
        for i, (src, dst) in enumerate(virtual_edges.T):
            if i < len(bandwidth_action):
                allocated_bandwidth = self.bandwidth_mapping[bandwidth_action[i]]
                
                # 检查带宽需求约束
                virtual_edge_features = self.virtual_work['edge_features'].numpy()
                min_required = virtual_edge_features[i][0]
                max_required = virtual_edge_features[i][1]
                
                if allocated_bandwidth < min_required:
                    constraint_violations.append(f"链路({src},{dst})的带宽分配({allocated_bandwidth})低于最小需求({min_required})")
                
                if allocated_bandwidth > max_required:
                    constraint_violations.append(f"链路({src},{dst})的带宽分配({allocated_bandwidth})超过最大需求({max_required})")
                
                # 检查物理路径带宽约束
                src_physical = mapping_action[src]
                dst_physical = mapping_action[dst]
                
                if src_physical != dst_physical:
                    # 获取最短路径
                    path = self.network_topology.get_shortest_path(src_physical, dst_physical)
                    if not path:
                        constraint_violations.append(f"虚拟链路({src},{dst})映射的物理节点({src_physical},{dst_physical})之间无路径")
                    elif not self.network_topology.check_bandwidth_availability(path, allocated_bandwidth):
                        constraint_violations.append(f"虚拟链路({src},{dst})的带宽需求({allocated_bandwidth})超过物理路径的可用带宽")
        
        return len(constraint_violations) == 0, constraint_violations
    
    def _calculate_reward(self, mapping_action, bandwidth_action):
        """计算奖励"""
        # 如果使用network_scheduler，使用其奖励计算
        if self.use_network_scheduler:
            return self.network_scheduler.calculate_reward(self.virtual_work_obj)
        
        # 原有的奖励计算逻辑
        reward = 0.0
        
        # 1. 资源利用率奖励
        resource_utilization_reward = self._calculate_resource_utilization(mapping_action)
        reward += resource_utilization_reward * 0.3
        
        # 2. 负载均衡奖励
        load_balancing_reward = self._calculate_load_balancing(mapping_action)
        reward += load_balancing_reward * 0.3
        
        # 3. 带宽满足度奖励
        bandwidth_satisfaction_reward = self._calculate_bandwidth_satisfaction(bandwidth_action)
        reward += bandwidth_satisfaction_reward * 0.4
        
        return reward
    
    def _calculate_resource_utilization(self, mapping_action):
        """计算资源利用率"""
        physical_features = self.physical_state['features'].numpy()
        virtual_features = self.virtual_work['features'].numpy()
        
        # 计算每个物理节点的资源使用情况
        node_utilizations = []
        
        for physical_node_idx in range(self.num_physical_nodes):
            # 找到映射到此物理节点的虚拟节点
            mapped_virtual_nodes = [i for i, p_idx in enumerate(mapping_action) if p_idx == physical_node_idx]
            
            if not mapped_virtual_nodes:
                continue
            
            # 计算CPU利用率
            total_cpu_demand = sum(virtual_features[i][0] for i in mapped_virtual_nodes)
            cpu_utilization = total_cpu_demand / physical_features[physical_node_idx][0]
            
            # 计算内存利用率
            total_memory_demand = sum(virtual_features[i][1] for i in mapped_virtual_nodes)
            memory_utilization = total_memory_demand / physical_features[physical_node_idx][1]
            
            # 综合利用率
            avg_utilization = (cpu_utilization + memory_utilization) / 2
            node_utilizations.append(avg_utilization)
        
        # 返回平均利用率（理想值在0.7-0.8之间）
        if node_utilizations:
            avg_utilization = np.mean(node_utilizations)
            # 奖励在0.7-0.8之间的利用率
            if 0.7 <= avg_utilization <= 0.8:
                return 1.0
            else:
                return max(0, 1.0 - abs(avg_utilization - 0.75) * 2)
        else:
            return 0.0
    
    def _calculate_load_balancing(self, mapping_action):
        """计算负载均衡度"""
        physical_features = self.physical_state['features'].numpy()
        virtual_features = self.virtual_work['features'].numpy()
        
        # 计算每个物理节点的负载
        node_loads = []
        
        for physical_node_idx in range(self.num_physical_nodes):
            mapped_virtual_nodes = [i for i, p_idx in enumerate(mapping_action) if p_idx == physical_node_idx]
            
            if not mapped_virtual_nodes:
                node_loads.append(0.0)
                continue
            
            # 计算综合负载
            total_cpu_demand = sum(virtual_features[i][0] for i in mapped_virtual_nodes)
            total_memory_demand = sum(virtual_features[i][1] for i in mapped_virtual_nodes)
            
            cpu_load = total_cpu_demand / physical_features[physical_node_idx][0]
            memory_load = total_memory_demand / physical_features[physical_node_idx][1]
            
            avg_load = (cpu_load + memory_load) / 2
            node_loads.append(avg_load)
        
        # 计算负载均衡度（负载方差越小越好）
        if node_loads:
            load_variance = np.var(node_loads)
            # 奖励低方差
            return max(0, 1.0 - load_variance * 10)
        else:
            return 0.0
    
    def _calculate_bandwidth_satisfaction(self, bandwidth_action):
        """计算带宽满足度"""
        if len(bandwidth_action) == 0:
            return 1.0
        
        virtual_edge_features = self.virtual_work['edge_features'].numpy()
        satisfaction_scores = []
        
        for i, bandwidth_level in enumerate(bandwidth_action):
            if i >= len(virtual_edge_features):
                break
            
            allocated_bandwidth = self.bandwidth_mapping[bandwidth_level]
            min_required = virtual_edge_features[i][0]
            max_required = virtual_edge_features[i][1]
            
            # 计算满足度（在最小和最大需求之间，越接近最大需求越好）
            if allocated_bandwidth < min_required:
                satisfaction = 0.0
            elif allocated_bandwidth > max_required:
                satisfaction = 1.0  # 超过最大需求也是好的
            else:
                # 在范围内，越接近最大需求越好
                satisfaction = (allocated_bandwidth - min_required) / (max_required - min_required)
            
            satisfaction_scores.append(satisfaction)
        
        return np.mean(satisfaction_scores) if satisfaction_scores else 0.0
    
    def get_scheduling_result(self):
        """获取调度结果"""
        result = {
            'mapping': self.mapping_result,
            'bandwidth': self.bandwidth_result,
            'physical_state': self.physical_state,
            'virtual_work': self.virtual_work
        }
        
        # 如果使用network_scheduler，添加其调度结果
        if self.use_network_scheduler and self.network_scheduler:
            network_result = self.network_scheduler.get_scheduling_result()
            result['network_scheduler_result'] = network_result
        
        return result

def test_two_stage_environment():
    """测试两阶段环境"""
    print("🧪 测试两阶段环境")
    print("=" * 60)
    
    # 测试不使用network_scheduler的情况
    print("📋 测试1: 不使用network_scheduler")
    print("-" * 40)
    
    env1 = TwoStageNetworkSchedulerEnvironment(
        num_physical_nodes=5,
        max_virtual_nodes=4,
        bandwidth_levels=10,
        use_network_scheduler=False
    )
    
    print(f"✅ 环境创建成功 (use_network_scheduler=False)")
    print(f"   物理节点数: {env1.num_physical_nodes}")
    print(f"   最大虚拟节点数: {env1.max_virtual_nodes}")
    print(f"   带宽等级数: {env1.bandwidth_levels}")
    
    # 重置环境
    state1 = env1.reset()
    
    print(f"\n📊 环境状态:")
    print(f"   物理节点特征: {state1['physical_features'].shape}")
    print(f"   物理边: {state1['physical_edge_index'].shape}")
    print(f"   虚拟节点特征: {state1['virtual_features'].shape}")
    print(f"   虚拟边: {state1['virtual_edge_index'].shape}")
    
    # 测试动作
    num_virtual_nodes = state1['virtual_features'].size(0)
    num_links = state1['virtual_edge_index'].size(1) // 2  # 无向图，除以2
    
    # 随机动作
    mapping_action = np.random.randint(0, env1.num_physical_nodes, num_virtual_nodes)
    bandwidth_action = np.random.randint(0, env1.bandwidth_levels, num_links)
    
    print(f"\n🎯 测试动作:")
    print(f"   映射动作: {mapping_action}")
    print(f"   带宽动作: {bandwidth_action}")
    
    # 执行动作
    next_state1, reward1, done1, info1 = env1.step(mapping_action, bandwidth_action)
    
    print(f"\n📈 执行结果:")
    print(f"   奖励: {reward1:.4f}")
    print(f"   是否结束: {done1}")
    print(f"   是否有效: {info1['is_valid']}")
    print(f"   约束违反: {info1['constraint_violations']}")
    
    # 获取调度结果
    result1 = env1.get_scheduling_result()
    print(f"\n📋 调度结果:")
    print(f"   映射结果: {result1['mapping']}")
    print(f"   带宽结果: {result1['bandwidth']}")
    
    # 测试使用network_scheduler的情况
    print(f"\n📋 测试2: 使用network_scheduler")
    print("-" * 40)
    
    env2 = TwoStageNetworkSchedulerEnvironment(
        num_physical_nodes=5,
        max_virtual_nodes=4,
        bandwidth_levels=10,
        use_network_scheduler=True
    )
    
    print(f"✅ 环境创建成功 (use_network_scheduler=True)")
    
    # 重置环境
    state2 = env2.reset()
    
    print(f"\n📊 环境状态:")
    print(f"   物理节点特征: {state2['physical_features'].shape}")
    print(f"   物理边: {state2['physical_edge_index'].shape}")
    print(f"   虚拟节点特征: {state2['virtual_features'].shape}")
    print(f"   虚拟边: {state2['virtual_edge_index'].shape}")
    
    # 使用相同的动作进行对比
    print(f"\n🎯 测试动作 (与测试1相同):")
    print(f"   映射动作: {mapping_action}")
    print(f"   带宽动作: {bandwidth_action}")
    
    # 执行动作
    next_state2, reward2, done2, info2 = env2.step(mapping_action, bandwidth_action)
    
    print(f"\n📈 执行结果:")
    print(f"   奖励: {reward2:.4f}")
    print(f"   是否结束: {done2}")
    print(f"   是否有效: {info2['is_valid']}")
    print(f"   约束违反: {info2['constraint_violations']}")
    
    # 获取调度结果
    result2 = env2.get_scheduling_result()
    print(f"\n📋 调度结果:")
    print(f"   映射结果: {result2['mapping']}")
    print(f"   带宽结果: {result2['bandwidth']}")
    
    # 显示network_scheduler的结果
    if 'network_scheduler_result' in result2:
        network_result = result2['network_scheduler_result']
        print(f"\n🔧 Network Scheduler结果:")
        print(f"   节点映射: {network_result['node_mapping']}")
        print(f"   带宽分配: {network_result['bandwidth_allocation']}")
        print(f"   已调度节点: {network_result['scheduled_nodes']}")
    
    # 对比结果
    print(f"\n📊 对比结果:")
    print(f"   不使用network_scheduler的奖励: {reward1:.4f}")
    print(f"   使用network_scheduler的奖励: {reward2:.4f}")
    print(f"   奖励差异: {reward2 - reward1:.4f}")
    
    print(f"\n🎯 两阶段环境测试完成！")

def demonstrate_network_scheduler_integration():
    """演示network_scheduler集成的详细用法"""
    print("\n🔧 演示Network Scheduler集成")
    print("=" * 60)
    
    # 创建使用network_scheduler的环境
    env = TwoStageNetworkSchedulerEnvironment(
        num_physical_nodes=6,
        max_virtual_nodes=4,
        bandwidth_levels=8,
        use_network_scheduler=True,
        physical_connectivity_prob=0.5,
        virtual_connectivity_prob=0.6,
        physical_cpu_range=(100, 300),
        physical_memory_range=(200, 600),
        physical_bandwidth_range=(200, 800),
        virtual_cpu_range=(20, 80),
        virtual_memory_range=(40, 150),
        virtual_bandwidth_range=(20, 300)
    )
    
    print("✅ 创建集成环境成功")
    print(f"   使用network_scheduler: {env.use_network_scheduler}")
    
    # 重置环境
    state = env.reset()
    
    print(f"\n📊 网络拓扑信息:")
    print(f"   物理节点数: {env.num_physical_nodes}")
    print(f"   虚拟节点数: {env.virtual_work['num_nodes']}")
    print(f"   物理边数: {state['physical_edge_index'].size(1) // 2}")
    print(f"   虚拟边数: {state['virtual_edge_index'].size(1) // 2}")
    
    # 显示物理网络拓扑
    print(f"\n🏗️ 物理网络拓扑:")
    physical_edges = state['physical_edge_index'].numpy()
    for i, (src, dst) in enumerate(physical_edges.T):
        if src < dst:  # 避免重复显示
            bandwidth = state['physical_edge_attr'][i][0].item()
            print(f"   物理链路 {src} <-> {dst}: 带宽 {int(bandwidth)}")
    
    # 显示虚拟工作需求
    print(f"\n💻 虚拟工作需求:")
    virtual_features = state['virtual_features'].numpy()
    for i in range(env.virtual_work['num_nodes']):
        cpu = virtual_features[i][0]
        memory = virtual_features[i][1]
        print(f"   虚拟节点 {i}: CPU={int(cpu)}, 内存={int(memory)}")
    
    virtual_edges = state['virtual_edge_index'].numpy()
    virtual_edge_features = state['virtual_edge_attr'].numpy()
    for i, (src, dst) in enumerate(virtual_edges.T):
        if src < dst:  # 避免重复显示
            min_bw = virtual_edge_features[i][0].item()
            max_bw = virtual_edge_features[i][1].item()
            print(f"   虚拟链路 {src} <-> {dst}: 带宽需求 [{int(min_bw)}, {int(max_bw)}]")
    
    # 生成合理的调度动作
    print(f"\n🎯 生成合理的调度动作:")
    
    # 节点映射：基于资源需求进行简单映射
    mapping_action = []
    physical_features = state['physical_features'].numpy()
    
    for i in range(env.virtual_work['num_nodes']):
        # 找到资源最匹配的物理节点
        best_node = 0
        best_score = float('inf')
        
        for j in range(env.num_physical_nodes):
            # 计算资源匹配度
            cpu_available = physical_features[j][0] * (1 - physical_features[j][2])
            memory_available = physical_features[j][1] * (1 - physical_features[j][3])
            
            cpu_needed = virtual_features[i][0]
            memory_needed = virtual_features[i][1]
            
            if cpu_available >= cpu_needed and memory_available >= memory_needed:
                # 计算资源利用率（越接近1越好）
                cpu_util = cpu_needed / cpu_available
                memory_util = memory_needed / memory_available
                score = abs(cpu_util - 0.7) + abs(memory_util - 0.7)  # 目标70%利用率
                
                if score < best_score:
                    best_score = score
                    best_node = j
        
        mapping_action.append(best_node)
        print(f"   虚拟节点 {i} -> 物理节点 {best_node}")
    
    # 带宽分配：基于需求分配合适的带宽等级
    bandwidth_action = []
    for i, (src, dst) in enumerate(virtual_edges.T):
        if src < dst:  # 避免重复
            min_bw = virtual_edge_features[i][0].item()
            max_bw = virtual_edge_features[i][1].item()
            target_bw = (min_bw + max_bw) / 2  # 目标带宽
            
            # 找到最接近的带宽等级
            best_level = 0
            best_diff = float('inf')
            
            for level in range(env.bandwidth_levels):
                allocated_bw = env.bandwidth_mapping[level]
                diff = abs(allocated_bw - target_bw)
                
                if diff < best_diff:
                    best_diff = diff
                    best_level = level
            
            bandwidth_action.append(best_level)
            allocated_bw = env.bandwidth_mapping[best_level]
            print(f"   虚拟链路 ({src},{dst}): 等级{best_level} -> 带宽{allocated_bw}")
    
    # 执行调度
    print(f"\n🚀 执行调度:")
    # 转换为numpy数组
    mapping_action = np.array(mapping_action)
    bandwidth_action = np.array(bandwidth_action)
    next_state, reward, done, info = env.step(mapping_action, bandwidth_action)
    
    print(f"   奖励: {reward:.4f}")
    print(f"   是否有效: {info['is_valid']}")
    print(f"   约束违反: {info['constraint_violations']}")
    
    # 获取详细结果
    result = env.get_scheduling_result()
    
    print(f"\n📋 调度结果详情:")
    print(f"   节点映射: {result['mapping']}")
    print(f"   带宽分配: {result['bandwidth']}")
    
    if 'network_scheduler_result' in result:
        network_result = result['network_scheduler_result']
        print(f"\n🔧 Network Scheduler详细结果:")
        print(f"   节点映射: {network_result['node_mapping']}")
        print(f"   带宽分配: {network_result['bandwidth_allocation']}")
        print(f"   已调度节点: {network_result['scheduled_nodes']}")
        
        # 显示网络利用率
        if env.network_topology:
            network_util = env.network_topology.get_network_utilization()
            print(f"   网络带宽利用率: {network_util['bandwidth_utilization']:.2%}")
    
    print(f"\n✅ Network Scheduler集成演示完成！")

if __name__ == "__main__":
    test_two_stage_environment()
    demonstrate_network_scheduler_integration() 