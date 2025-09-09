#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler

class SmartBandwidthAllocator:
    """
    智能带宽分配器 - 基于heuristic_algorithm中的智能带宽分配策略
    """
    
    def __init__(self):
        pass
    
    def allocate_bandwidth(self, virtual_work: VirtualWork, 
                          node_mapping: Dict[int, int],
                          network_scheduler: NetworkScheduler) -> bool:
        """
        智能带宽分配：根据链路重要性和资源可用性
        
        Args:
            virtual_work: 虚拟工作对象
            node_mapping: 节点映射 {virtual_node: physical_node}
            network_scheduler: 网络调度器
            
        Returns:
            bool: 是否所有链路都成功分配带宽
        """
        all_success = True
        
        for link_req in virtual_work.link_requirements:
            src = link_req['from']
            dst = link_req['to']
            min_bandwidth = link_req['min_bandwidth_1_to_2']
            max_bandwidth = link_req['max_bandwidth_1_to_2']
            
            # 检查节点是否已映射
            if src not in node_mapping or dst not in node_mapping:
                all_success = False
                continue
            
            physical_src = node_mapping[src]
            physical_dst = node_mapping[dst]
            
            # 如果映射到同一物理节点，分配最大带宽（无实际消耗）
            if physical_src == physical_dst:
                success = network_scheduler.allocate_bandwidth(src, dst, max_bandwidth)
                if not success:
                    all_success = False
                continue
            
            # 计算链路重要性（基于虚拟网络中的度）
            src_degree = sum(1 for req in virtual_work.link_requirements 
                           if req['from'] == src or req['to'] == src)
            dst_degree = sum(1 for req in virtual_work.link_requirements 
                           if req['from'] == dst or req['to'] == dst)
            
            total_nodes = len(virtual_work.node_requirements)
            link_importance = (src_degree + dst_degree) / (2.0 * max(1, total_nodes - 1))
            
            # 根据重要性确定目标带宽
            if link_importance > 0.7:  # 重要链路
                target_bandwidth = min_bandwidth + 0.8 * (max_bandwidth - min_bandwidth)
            elif link_importance > 0.4:  # 中等重要链路
                target_bandwidth = min_bandwidth + 0.5 * (max_bandwidth - min_bandwidth)
            else:  # 低重要性链路
                target_bandwidth = min_bandwidth + 0.2 * (max_bandwidth - min_bandwidth)
            
            # 尝试分配带宽（从目标带宽开始，逐步降低到最小需求）
            bandwidth_levels = [target_bandwidth, 
                              (min_bandwidth + target_bandwidth) / 2, 
                              min_bandwidth]
            
            success = False
            for try_bandwidth in bandwidth_levels:
                # 检查物理路径可用带宽
                path = network_scheduler.topology.get_shortest_path(physical_src, physical_dst)
                if not path:
                    break
                
                # 检查路径上所有链路的可用带宽
                path_available = True
                for i in range(len(path) - 1):
                    available_bw = network_scheduler.topology.get_available_bandwidth(path[i], path[i+1])
                    if available_bw < try_bandwidth:
                        path_available = False
                        break
                
                if path_available:
                    success = network_scheduler.allocate_bandwidth(src, dst, try_bandwidth)
                    if success:
                        print(f"🔗 链路({src},{dst}) 分配带宽: {try_bandwidth:.1f} (重要性: {link_importance:.2f})")
                        break
            
            if not success:
                print(f"❌ 链路({src},{dst}) 带宽分配失败")
                all_success = False
        
        return all_success


class A2CNetworkEnvironment:
    """
    A2C网络调度环境
    专注于任务节点到物理节点的映射，包含预调度模块和负载均衡奖励
    """
    
    def __init__(self, 
                 num_physical_nodes: int = 10,
                 # 物理节点资源范围
                 physical_cpu_range: Tuple[int, int] = (50, 100),
                 physical_memory_range: Tuple[int, int] = (50, 100),
                 physical_bandwidth_range: Tuple[int, int] = (100, 1000),
                 # 任务节点资源范围
                 task_cpu_range: Tuple[int, int] = (10, 50),
                 task_memory_range: Tuple[int, int] = (10, 50),
                 task_bandwidth_range: Tuple[int, int] = (10, 100),
                 # 网络连接概率
                 physical_connectivity_prob: float = 1.0,
                 task_connectivity_prob: float = 0.4,
                 # 任务节点数范围
                 task_nodes_range: Tuple[int, int] = (3, 6),
                 # 物理资源初始使用率
                 initial_usage_range: Tuple[float, float] = (0.1, 0.5),
                 seed: int = None):
        
        # 设置随机种子
        if seed is not None:
            self._set_random_seed(seed)
            print(f"🌱 A2C环境设置随机种子: {seed}")
        
        self.seed = seed
        
        # 环境参数
        self.num_physical_nodes = num_physical_nodes
        self.physical_cpu_range = physical_cpu_range
        self.physical_memory_range = physical_memory_range
        self.physical_bandwidth_range = physical_bandwidth_range
        self.task_cpu_range = task_cpu_range
        self.task_memory_range = task_memory_range
        self.task_bandwidth_range = task_bandwidth_range
        self.physical_connectivity_prob = physical_connectivity_prob
        self.task_connectivity_prob = task_connectivity_prob
        self.task_nodes_range = task_nodes_range
        self.initial_usage_range = initial_usage_range
        
        # 环境状态
        self.current_task_index = 0
        self.task_mapping = {}  # {task_node: physical_node}
        self.task_queue = []    # 待调度的任务列表
        
        # 网络组件
        self.network_topology = None
        self.virtual_work = None
        self.network_scheduler = None
        self.bandwidth_allocator = SmartBandwidthAllocator()
        
        # 统计信息
        self.episode_stats = {
            'accepted_tasks': 0,
            'rejected_tasks': 0,
            'total_tasks': 0,
            'load_balance_rewards': [],
            'rejection_penalties': [],
            'final_bandwidth_success': False
        }
        
        print(f"✅ A2C环境初始化完成")
        print(f"   物理节点数: {num_physical_nodes}")
        print(f"   任务节点范围: {task_nodes_range}")
    
    def _set_random_seed(self, seed: int):
        """设置环境的随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"🔒 A2C环境随机种子设置完成: {seed}")
    
    def reset(self) -> Dict:
        """重置环境，开始新的episode"""
        self.current_task_index = 0
        self.task_mapping = {}
        
        # 生成物理拓扑
        self._generate_physical_topology()
        
        # 生成任务队列
        self._generate_task_queue()
        
        # 初始化网络调度器
        self._initialize_network_scheduler()
        
        # 重置统计信息
        self.episode_stats = {
            'accepted_tasks': 0,
            'rejected_tasks': 0,
            'total_tasks': len(self.task_queue),
            'load_balance_rewards': [],
            'rejection_penalties': [],
            'final_bandwidth_success': False
        }
        
        print(f"🔄 A2C环境重置完成:")
        print(f"   物理节点数: {self.num_physical_nodes}")
        print(f"   任务数量: {len(self.task_queue)}")
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一个调度步骤
        
        Args:
            action: 选择的物理节点索引
            
        Returns:
            next_state: 下一个状态
            reward: 即时奖励
            done: 是否完成
            info: 额外信息
        """
        if self.current_task_index >= len(self.task_queue):
            return self._get_state(), 0.0, True, {'error': 'No more tasks'}
        
        current_task = self.task_queue[self.current_task_index]
        task_node_id = current_task['id']
        # 确保 physical_node_id 是Python整数，不是tensor
        physical_node_id = int(action)
        
        print(f"📍 调度步骤 {self.current_task_index}: 任务{task_node_id} -> 物理节点{physical_node_id}")
        
        # 预调度检查：验证资源是否满足需求
        is_feasible, violation_info = self._pre_schedule_check(current_task, physical_node_id)
        
        reward = 0.0
        info = {
            'task_id': task_node_id,
            'physical_node': physical_node_id,
            'is_feasible': is_feasible,
            'violation_info': violation_info
        }
        
        if is_feasible:
            # 执行映射
            self.task_mapping[task_node_id] = physical_node_id
            
            # 在网络调度器中执行映射
            success = self.network_scheduler.schedule_node(task_node_id, physical_node_id)
            if success:
                self.episode_stats['accepted_tasks'] += 1
                print(f"✅ 任务{task_node_id}映射成功")
            else:
                print(f"⚠️ NetworkScheduler映射失败")
            
            # 计算负载均衡奖励
            load_balance_reward = self._calculate_load_balance_reward()
            reward = load_balance_reward
            
            self.episode_stats['load_balance_rewards'].append(load_balance_reward)
            info['load_balance_reward'] = load_balance_reward
            
        else:
            # 任务被拒绝
            self.episode_stats['rejected_tasks'] += 1
            rejection_penalty = -1.0
            reward = rejection_penalty
            
            self.episode_stats['rejection_penalties'].append(rejection_penalty)
            info['rejection_penalty'] = rejection_penalty
            
            print(f"❌ 任务{task_node_id}被拒绝: {violation_info}")
        
        # 移动到下一个任务
        self.current_task_index += 1
        
        # 检查是否完成所有任务
        done = self.current_task_index >= len(self.task_queue)
        
        if done:
            # Episode结束，进行带宽分配
            final_reward = self._finalize_episode()
            reward += final_reward
            info['final_reward'] = final_reward
            
            print(f"🏁 Episode结束")
            print(f"   接受任务: {self.episode_stats['accepted_tasks']}/{self.episode_stats['total_tasks']}")
            print(f"   拒绝任务: {self.episode_stats['rejected_tasks']}")
            print(f"   最终奖励: {final_reward:.3f}")
            print(f"   总奖励: {reward:.3f}")
        
        return self._get_state(), reward, done, info
    
    def _pre_schedule_check(self, task: Dict, physical_node_id: int) -> Tuple[bool, str]:
        """
        预调度检查：验证物理节点资源是否满足任务需求
        
        Args:
            task: 任务信息
            physical_node_id: 物理节点ID
            
        Returns:
            is_feasible: 是否可行
            violation_info: 违约信息
        """
        if physical_node_id < 0 or physical_node_id >= self.num_physical_nodes:
            return False, f"物理节点ID{physical_node_id}超出范围"
        
        
        available_resources = self.network_scheduler.topology.get_available_resources(physical_node_id)
        
        # 检查CPU资源
        if task['cpu_demand'] > available_resources['cpu']:
            return False, f"CPU资源不足: 需要{task['cpu_demand']}, 可用{available_resources['cpu']:.1f}"
        
        # 检查内存资源
        if task['memory_demand'] > available_resources['memory']:
            return False, f"内存资源不足: 需要{task['memory_demand']}, 可用{available_resources['memory']:.1f}"
        
        return True, ""
    
    def _calculate_load_balance_reward(self) -> float:
        """
        计算负载均衡奖励
        
        公式：
        - τ: 每个物理节点的最大可用资源量
        - ϕ: 所有物理节点的资源均值
        - ω = ϕ / (τ · M), M是物理节点数
        - R = ω_cpu + ω_memory
        """
        cpu_available = []
        memory_available = []
        
        # 收集所有物理节点的可用资源
        for node_id in range(self.num_physical_nodes):
            available = self.network_scheduler.topology.get_available_resources(node_id)
            cpu_available.append(available['cpu'])
            memory_available.append(available['memory'])
        
        # 计算最大可用资源量τ
        tau_cpu = max(cpu_available) if cpu_available else 1.0
        tau_memory = max(memory_available) if memory_available else 1.0
        
        # 计算所有物理节点的资源均值ϕ
        phi_cpu = np.mean(cpu_available) if cpu_available else 0.0
        phi_memory = np.mean(memory_available) if memory_available else 0.0
        
        # 计算ω = ϕ / (τ · M)
        M = self.num_physical_nodes
        omega_cpu = phi_cpu / (tau_cpu * M) if tau_cpu > 0 else 0.0
        omega_memory = phi_memory / (tau_memory * M) if tau_memory > 0 else 0.0
        
        # 最终奖励R = ω_cpu + ω_memory
        reward = omega_cpu + omega_memory
        
        return reward
    
    def _finalize_episode(self) -> float:
        """
        完成episode，进行带宽分配并计算最终奖励
        
        Returns:
            final_reward: 最终奖励
        """
        if not self.task_mapping:
            return -0.5  # 没有成功映射任何任务
        
        # 使用智能带宽分配器进行带宽分配
        bandwidth_success = self.bandwidth_allocator.allocate_bandwidth(
            self.virtual_work, self.task_mapping, self.network_scheduler
        )
        
        self.episode_stats['final_bandwidth_success'] = bandwidth_success
        
        if bandwidth_success:
            print("✅ 智能带宽分配成功")
            return 0.5  # 带宽分配成功奖励
        else:
            print("❌ 智能带宽分配失败")
            return -0.3  # 带宽分配失败惩罚
    
    def _get_state(self) -> Dict:
        """
        获取当前状态
        
        Returns:
            state: 包含物理节点可用资源和当前任务需求的状态
        """
        # 获取所有物理节点的可用资源
        physical_resources = []
        for node_id in range(self.num_physical_nodes):
            available = self.network_scheduler.topology.get_available_resources(node_id)
            # 创建副本，避免与原始数据的意外关联
            cpu_available = float(available['cpu'])
            memory_available = float(available['memory'])
            physical_resources.append([cpu_available, memory_available])

        
        physical_resources_tensor = torch.tensor(physical_resources, dtype=torch.float32)
        
        # 获取当前任务的资源需求
        if self.current_task_index < len(self.task_queue):
            current_task = self.task_queue[self.current_task_index]
            task_requirements = torch.tensor([
                current_task['cpu_demand'],
                current_task['memory_demand']
            ], dtype=torch.float32)
        else:
            # 没有更多任务，用零填充
            task_requirements = torch.zeros(2, dtype=torch.float32)
        
        # 计算有效动作掩码（哪些物理节点可以满足当前任务需求）
        valid_actions = torch.zeros(self.num_physical_nodes, dtype=torch.bool)
        if self.current_task_index < len(self.task_queue):
            current_task = self.task_queue[self.current_task_index]
            for node_id in range(self.num_physical_nodes):
                is_feasible, _ = self._pre_schedule_check(current_task, node_id)
                valid_actions[node_id] = is_feasible
        
        return {
            'physical_resources': physical_resources_tensor,
            'task_requirements': task_requirements,
            'valid_actions': valid_actions,
            'current_task_index': self.current_task_index,
            'total_tasks': len(self.task_queue),
            'task_mapping': self.task_mapping.copy()
        }
    
    def _generate_physical_topology(self):
        """生成物理网络拓扑"""
        self.network_topology = NetworkTopology(self.num_physical_nodes)
        
        # 设置物理节点资源
        for i in range(self.num_physical_nodes):
            # 确保物理资源范围有效
            cpu_min, cpu_max = self.physical_cpu_range
            if cpu_min >= cpu_max:
                cpu_max = cpu_min + 1
            
            mem_min, mem_max = self.physical_memory_range
            if mem_min >= mem_max:
                mem_max = mem_min + 1
            
            cpu_total = np.random.randint(cpu_min, cpu_max + 1)
            memory_total = np.random.randint(mem_min, mem_max + 1)
            
            # 初始使用量
            cpu_usage_rate = np.random.uniform(*self.initial_usage_range)
            memory_usage_rate = np.random.uniform(*self.initial_usage_range)
            
            cpu_used = cpu_total * cpu_usage_rate
            memory_used = memory_total * memory_usage_rate
            
            try:
                self.network_topology.set_node_resources(i, cpu_total, memory_total, cpu_used, memory_used)
            except Exception as e:
                # 使用备用配置重试
                backup_cpu = 100
                backup_memory = 100
                backup_cpu_used = backup_cpu * 0.2
                backup_memory_used = backup_memory * 0.2
                
                print(f"🔧 使用备用配置重试节点{i}...")
                self.network_topology.set_node_resources(i, backup_cpu, backup_memory, backup_cpu_used, backup_memory_used)
                print(f"✅ 节点{i}使用备用配置设置成功")
        
        # 创建物理网络连接
        for i in range(self.num_physical_nodes):
            for j in range(i + 1, self.num_physical_nodes):
                if np.random.random() < self.physical_connectivity_prob:
                    # 确保带宽范围有效
                    bw_min, bw_max = self.physical_bandwidth_range
                    if bw_min >= bw_max:
                        bw_max = bw_min + 1
                    
                    bandwidth = np.random.randint(bw_min, bw_max + 1)
                    usage_rate = np.random.uniform(0.1, 0.3)
                    used_bandwidth = bandwidth * usage_rate
                    
                    self.network_topology.add_link(i, j, bandwidth, bandwidth, 
                                                 used_bandwidth, used_bandwidth)
    
    def _generate_task_queue(self):
        """生成任务队列"""
        # 确保范围有效
        min_tasks, max_tasks = self.task_nodes_range
        if min_tasks >= max_tasks:
            max_tasks = min_tasks + 1
        
        num_tasks = np.random.randint(min_tasks, max_tasks + 1)
        self.task_queue = []
        
        for i in range(num_tasks):
            # 确保CPU和内存范围有效
            cpu_min, cpu_max = self.task_cpu_range
            if cpu_min >= cpu_max:
                cpu_max = cpu_min + 1
            
            mem_min, mem_max = self.task_memory_range
            if mem_min >= mem_max:
                mem_max = mem_min + 1
            
            task = {
                'id': i,
                'cpu_demand': np.random.randint(cpu_min, cpu_max + 1),
                'memory_demand': np.random.randint(mem_min, mem_max + 1),
            }
            self.task_queue.append(task)
        
        # 生成任务之间的连接需求（用于后续带宽分配）
        self.task_links = []
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                if np.random.random() < self.task_connectivity_prob:
                    # 确保任务带宽范围有效
                    task_bw_min, task_bw_max = self.task_bandwidth_range
                    if task_bw_min >= task_bw_max:
                        task_bw_max = task_bw_min + 1
                    
                    # 生成最小带宽需求
                    mid_point = int(task_bw_max * 0.6)
                    if mid_point <= task_bw_min:
                        mid_point = task_bw_min + 1
                    
                    min_bandwidth = np.random.randint(task_bw_min, mid_point)
                    max_bandwidth = np.random.randint(min_bandwidth, task_bw_max + 1)
                    
                    self.task_links.append({
                        'from': i,
                        'to': j,
                        'min_bandwidth': min_bandwidth,
                        'max_bandwidth': max_bandwidth
                    })
    
    def _initialize_network_scheduler(self):
        """初始化网络调度器"""
        # 创建虚拟工作对象
        num_tasks = len(self.task_queue)
        self.virtual_work = VirtualWork(num_tasks)
        
        # 设置任务节点需求
        for task in self.task_queue:
            self.virtual_work.set_node_requirement(
                task['id'], task['cpu_demand'], task['memory_demand']
            )
        
        # 添加任务链路需求
        for link in self.task_links:
            self.virtual_work.add_link_requirement(
                link['from'], link['to'],
                link['min_bandwidth'], link['max_bandwidth'],
                link['min_bandwidth'], link['max_bandwidth']
            )
        
        # 创建网络调度器
        self.network_scheduler = NetworkScheduler(self.network_topology)
        self.network_scheduler.add_virtual_work(self.virtual_work, work_id="a2c_work")
