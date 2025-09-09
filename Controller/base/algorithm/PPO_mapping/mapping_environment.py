#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple, Dict, List
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler
import torch.nn as nn

class MappingNetworkSchedulerEnvironment:
    """
    PPO_mapping网络调度环境
    Actor只负责节点调度，带宽分配使用贪心策略自动完成
    """
    
    def __init__(self, 
                 num_physical_nodes: int = 5,
                 max_virtual_nodes: int = 6,
                 # 物理节点资源范围
                 physical_cpu_range: Tuple[int, int] = (40, 80),
                 physical_memory_range: Tuple[int, int] = (40, 80),
                 physical_bandwidth_range: Tuple[int, int] = (40, 80),
                 # 虚拟节点资源范围
                 virtual_cpu_range: Tuple[int, int] = (12, 25),
                 virtual_memory_range: Tuple[int, int] = (12, 25),
                 virtual_bandwidth_range: Tuple[int, int] = (8, 20),
                 # 网络连接概率
                 physical_connectivity_prob: float = 0.8,
                 virtual_connectivity_prob: float = 0.7,
                 # 虚拟节点数范围
                 virtual_nodes_range: Tuple[int, int] = (4, 6),
                 # 随机种子
                 seed: int = None,
                 curriculum_enabled: bool = True):
        
        # 设置随机种子
        if seed is not None:
            self._set_random_seed(seed)
            print(f"🌱 Mapping环境设置随机种子: {seed}")
        
        self.seed = seed
        
        # 环境参数
        self.num_physical_nodes = num_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
        
        # 资源范围
        self.physical_cpu_range = physical_cpu_range
        self.physical_memory_range = physical_memory_range
        self.physical_bandwidth_range = physical_bandwidth_range
        self.virtual_nodes_range = virtual_nodes_range
        self.virtual_cpu_range = virtual_cpu_range
        self.virtual_memory_range = virtual_memory_range
        self.virtual_bandwidth_range = virtual_bandwidth_range
        
        # 连接概率
        self.physical_connectivity_prob = physical_connectivity_prob
        self.virtual_connectivity_prob = virtual_connectivity_prob
        
        self.curriculum_enabled = curriculum_enabled
        
        # 决策状态
        self.current_step = 0
        self.current_virtual_node = 0
        
        # 映射结果
        self.node_mapping = None  # [-1, -1, -1, ...] -1表示未映射
        
        # 环境状态
        self.physical_state = None
        self.virtual_work = None
        
        # network_scheduler相关对象
        self.network_topology = None
        self.virtual_work_obj = None
        self.network_scheduler = None
        
        # 统计信息
        self.episode_stats = {
            'mapping_rewards': [],
            'step_rewards': [],
            'resource_utilizations': [],
            'constraint_violations': [],
            'bandwidth_allocations': []
        }

        self.episode_count = 0
        self.success_history = []  # 记录最近的成功率
        self.history_window = 20  # 计算成功率的窗口大小
        self.difficulty_level = 1.0  # 当前难度等级 (0.5-2.0)
        self.min_difficulty = 0.5
        self.max_difficulty = 2.0
        self.difficulty_adjustment_rate = 0.1
        
    def _set_random_seed(self, seed: int):
        """设置环境的随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        print(f"🔒 Mapping环境随机种子设置完成: {seed}")
    
    def reset(self):
        """重置环境，开始新的episode"""
        # 在新episode开始前调整难度
        self._adjust_difficulty()
        
        self.current_step = 0
        self.current_virtual_node = 0
        
        # 生成网络状态（会受到难度调整的影响）
        self.physical_state = self._generate_physical_state()
        self.virtual_work = self._generate_virtual_work()
        
        # 初始化映射结果
        num_virtual_nodes = self.virtual_work['num_nodes']
        self.node_mapping = [-1] * num_virtual_nodes  # -1表示未映射
        
        # 初始化网络调度器
        self._initialize_network_scheduler()
        
        # 重置统计信息
        self.episode_stats = {
            'mapping_rewards': [],
            'step_rewards': [],
            'resource_utilizations': [],
            'constraint_violations': [],
            'bandwidth_allocations': []
        }
        
        print(f"🔄 Mapping环境重置完成:")
        print(f"   虚拟节点数: {num_virtual_nodes}")
        print(f"   物理节点数: {self.num_physical_nodes}")
        print(f"   预计总步数: {num_virtual_nodes}")
        
        return self._get_state()
    
    def step(self, action):
        """执行一个决策步骤 - 只进行节点映射"""
        print(f"📍 映射步骤 {self.current_step}: 虚拟节点{self.current_virtual_node} -> 物理节点{action}")
        
        # 验证动作有效性
        is_valid, constraint_violations = self._validate_mapping_action(
            self.current_virtual_node, action)
        
        if not is_valid:
            # 无效动作，给予小惩罚并继续
            reward = -0.1
            info = {
                'is_valid': False,
                'constraint_violations': constraint_violations,
                'current_virtual_node': self.current_virtual_node,
                'action': action
            }
            print(f"❌ 映射动作无效: {constraint_violations}")
        else:
            # 执行映射
            self.node_mapping[self.current_virtual_node] = action
            
            # 在network_scheduler中执行映射
            success = self.network_scheduler.schedule_node(self.current_virtual_node, action)
            if not success:
                print(f"⚠️ NetworkScheduler映射失败")
            
            # 计算即时奖励
            reward = self._calculate_mapping_reward(action)
            
            info = {
                'is_valid': True,
                'constraint_violations': [],
                'current_virtual_node': self.current_virtual_node,
                'action': action,
                'node_mapping': self.node_mapping.copy(),
                'mapping_reward': reward
            }
            print(f"✅ 映射成功，即时奖励: {reward:.3f}")
        
        # 更新状态
        self.current_step += 1
        self.current_virtual_node += 1
        
        # 检查是否完成所有节点映射
        done = self.current_virtual_node >= self.virtual_work['num_nodes']
        
        if done:
            # 检查是否所有节点都成功映射
            unmapped_nodes = [i for i, mapping in enumerate(self.node_mapping) if mapping == -1]
            if unmapped_nodes:
                print(f"❌ 映射阶段结束，但有{len(unmapped_nodes)}个节点未成功映射: {unmapped_nodes}")
                final_reward = -1.0  # 映射失败的惩罚
            else:
                # 映射成功，使用贪心策略分配带宽
                print(f"🎯 节点映射完成，开始贪心带宽分配")
                bandwidth_success = self._greedy_bandwidth_allocation()
                
                if bandwidth_success:
                    # 计算最终奖励
                    final_reward = self._calculate_final_reward()
                    print(f"✅ 贪心带宽分配成功")
                else:
                    final_reward = -0.5  # 带宽分配失败的惩罚
                    print(f"❌ 贪心带宽分配失败")
            
            reward += final_reward  # 加上最终奖励
            info['final_reward'] = final_reward
            
            # 记录成功与否用于Curriculum Learning
            episode_success = final_reward > 0.0  # 正奖励视为成功
            self.success_history.append(episode_success)
            self.episode_count += 1
            
            # 保持历史记录在窗口大小内
            if len(self.success_history) > self.history_window * 2:
                self.success_history = self.success_history[-self.history_window:]
            
            print(f"🏁 Episode {self.episode_count}结束，最终奖励: {final_reward:.3f}, 总奖励: {reward:.3f}, 成功: {episode_success}")
        
        # 更新统计信息
        self.episode_stats['mapping_rewards'].append(reward)
        self.episode_stats['step_rewards'].append(reward)
        
        return self._get_state(), reward, done, info
    
    def _greedy_bandwidth_allocation(self):
        """
        贪心策略带宽分配
        对每条虚拟链路，选择满足需求的最小带宽分配
        """
        virtual_edges = self.virtual_work['edges']
        virtual_edge_features = self.virtual_work['edge_features']
        
        all_allocations_successful = True
        
        for i in range(virtual_edges.size(1)):
            src = virtual_edges[0, i].item()
            dst = virtual_edges[1, i].item()
            min_bandwidth = virtual_edge_features[i, 0].item()
            max_bandwidth = virtual_edge_features[i, 1].item()
            
            # 获取映射的物理节点
            physical_src = self.node_mapping[src]
            physical_dst = self.node_mapping[dst]
            
            # 如果映射到同一物理节点，自动获得无限带宽
            if physical_src == physical_dst:
                # 分配最大需求带宽（虽然实际上不消耗物理资源）
                allocated_bandwidth = max_bandwidth
                success = self.network_scheduler.allocate_bandwidth(src, dst, allocated_bandwidth)
                print(f"🔗 虚拟链路({src},{dst}) 同节点映射，分配带宽: {allocated_bandwidth}")
            else:
                # 贪心策略：尝试分配最小满足需求的带宽
                allocated_bandwidth = None
                
                # 从最小需求开始，逐步增加，找到可行的带宽分配
                for try_bandwidth in [min_bandwidth, (min_bandwidth + max_bandwidth) // 2, max_bandwidth]:
                    # 检查物理路径上的带宽是否足够
                    path = self.network_scheduler.topology.get_shortest_path(physical_src, physical_dst)
                    if not path:
                        print(f"❌ 虚拟链路({src},{dst}): 物理路径不可达 {physical_src} -> {physical_dst}")
                        success = False
                        break
                    
                    # 检查路径上的带宽
                    bandwidth_sufficient = True
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j + 1]
                        available = self.network_scheduler.topology.get_available_bandwidth(u, v)
                        if try_bandwidth > available:
                            bandwidth_sufficient = False
                            break
                    
                    if bandwidth_sufficient:
                        # 分配带宽
                        success = self.network_scheduler.allocate_bandwidth(src, dst, try_bandwidth)
                        if success:
                            allocated_bandwidth = try_bandwidth
                            print(f"🔗 虚拟链路({src},{dst}) 贪心分配带宽: {allocated_bandwidth} (路径: {' -> '.join(map(str, path))})")
                            break
                
                if allocated_bandwidth is None:
                    print(f"❌ 虚拟链路({src},{dst}): 无法分配满足需求的带宽")
                    success = False
            
            if not success:
                all_allocations_successful = False
            
            # 记录带宽分配统计
            self.episode_stats['bandwidth_allocations'].append({
                'virtual_link': (src, dst),
                'required': (min_bandwidth, max_bandwidth),
                'allocated': allocated_bandwidth if success else 0,
                'success': success
            })
        
        return all_allocations_successful
    
    def _calculate_mapping_reward(self, physical_node_action):
        """计算节点映射的即时奖励"""
        # 基础成功奖励
        base_reward = 1.0
        
        # 资源利用效率奖励
        virtual_node = self.current_virtual_node
        virtual_features = self.virtual_work['features']
        cpu_demand = virtual_features[virtual_node, 0].item()
        memory_demand = virtual_features[virtual_node, 1].item()
        
        # 获取物理节点可用资源
        available_resources = self.network_scheduler.topology.get_available_resources(physical_node_action)
        total_resources = self.network_scheduler.topology.node_resources[physical_node_action]
        
        # 计算映射后的资源利用率
        cpu_utilization = (total_resources['cpu'] - available_resources['cpu'] + cpu_demand) / total_resources['cpu']
        memory_utilization = (total_resources['memory'] - available_resources['memory'] + memory_demand) / total_resources['memory']
        
        # 奖励适中的资源利用率 (0.3-0.8 范围内给高奖励)
        cpu_efficiency = 1.0 - abs(cpu_utilization - 0.55) / 0.45 if cpu_utilization <= 1.0 else 0.0
        memory_efficiency = 1.0 - abs(memory_utilization - 0.55) / 0.45 if memory_utilization <= 1.0 else 0.0
        
        efficiency_reward = 0.5 * (cpu_efficiency + memory_efficiency)
        
        # 同节点映射潜力奖励
        colocation_reward = self._calculate_colocation_potential_reward(physical_node_action)
        
        total_reward = base_reward + efficiency_reward + colocation_reward
        
        return max(0.0, total_reward)  # 确保奖励非负
    
    def _calculate_colocation_potential_reward(self, physical_node_action):
        """计算同节点映射潜力奖励"""
        current_vnode = self.current_virtual_node
        virtual_edges = self.virtual_work['edges'].numpy()
        
        # 找出与当前虚拟节点连接的所有虚拟节点
        connected_vnodes = set()
        for i, (src, dst) in enumerate(virtual_edges.T):
            if src == current_vnode:
                connected_vnodes.add(dst)
            elif dst == current_vnode:
                connected_vnodes.add(src)
        
        # 计算如果映射到该物理节点的同节点映射收益
        colocation_benefit = 0.0
        for vnode in connected_vnodes:
            if vnode < len(self.node_mapping) and self.node_mapping[vnode] == physical_node_action:
                # 这个虚拟节点已经映射到该物理节点，存在同节点映射机会
                colocation_benefit += 1.0
        
        # 归一化奖励
        if connected_vnodes:
            normalized_benefit = colocation_benefit / len(connected_vnodes)
            reward = normalized_benefit * 0.8  # 同节点映射潜力奖励权重
            if reward > 0:
                print(f"🔗 同节点映射潜力奖励: {reward:.3f} (连接{len(connected_vnodes)}个节点，{colocation_benefit}个已在物理节点{physical_node_action})")
            return reward
        else:
            return 0.0
    
    def _calculate_final_reward(self):
        """计算episode结束时的最终奖励"""
        try:
            # 使用新的简化奖励函数
            if hasattr(self.network_scheduler, 'calculate_simple_reward'):
                final_reward = self.network_scheduler.calculate_simple_reward(self.virtual_work_obj)
                print(f"🎯 使用简化奖励函数: {final_reward:.3f}")
                
                # 获取增强奖励组件用于调试
                if hasattr(self.network_scheduler, 'get_simple_reward_components'):
                    components = self.network_scheduler.get_simple_reward_components(self.virtual_work_obj)
                    print(f"   映射成功率: {components.get('mapping_success_rate', 0):.3f}")
                    print(f"   负载均衡: {components.get('load_balance_reward', 0):.3f}")
                    print(f"   资源效率: {components.get('resource_efficiency', 0):.3f}")
                    print(f"   带宽满足度: {components.get('bandwidth_satisfaction', 0):.3f}")
                    print(f"   路径优化: {components.get('path_length_penalty', 0):.3f}")
                
                return final_reward
            else:
                # 回退到旧的奖励计算
                components = self.network_scheduler.calculate_reward_components(self.virtual_work_obj)
                load_balance = components.get('load_balance', 0.0)
                bandwidth_satisfaction = components.get('bandwidth_satisfaction', 0.0)
                final_reward = 0.5 * (load_balance + bandwidth_satisfaction)
                print(f"🔙 使用旧奖励函数: {final_reward:.3f}")
                return final_reward
        except Exception as e:
            print(f"⚠️ 计算最终奖励失败: {e}")
            return 0.0
    
    def _get_state(self):
        """获取当前状态表示"""
        # 基础状态
        state = {
            'physical_features': self.physical_state['features'],
            'physical_edges': self.physical_state['edges'],
            'physical_edge_features': self.physical_state['edge_features'],
            'virtual_features': self.virtual_work['features'],
            'virtual_edges': self.virtual_work['edges'],
            'virtual_edge_features': self.virtual_work['edge_features'],
            
            # 决策状态信息
            'current_step': self.current_step,
            'current_virtual_node': self.current_virtual_node,
            'node_mapping': self.node_mapping.copy(),
            
            # 辅助信息
            'num_physical_nodes': self.num_physical_nodes,
            'num_virtual_nodes': self.virtual_work['num_nodes'],
            'num_virtual_links': self.virtual_work['edges'].size(1),
        }
        
        return state
    
    # ============= 以下是从原环境复制的辅助函数 =============
    
    def _generate_physical_state(self):
        """生成随机物理网络状态"""
        # 先生成边，以便计算每个节点的链路带宽特征
        physical_edges = self._get_physical_edges()
        
        # 生成边特征
        physical_edge_features = []
        for edge in physical_edges.T:
            bandwidth = np.random.randint(*self.physical_bandwidth_range)
            bandwidth_usage = np.random.uniform(0.1, 0.3)  # 较低的初始使用率
            physical_edge_features.append([bandwidth, bandwidth_usage])
        
        # 计算每个物理节点连接的所有链路可用带宽的均值
        physical_link_bandwidth_means = [0.0] * self.num_physical_nodes
        for i in range(self.num_physical_nodes):
            connected_bandwidths = []
            for j, edge in enumerate(physical_edges.T):
                src, dst = edge[0], edge[1]
                if src == i:  # 该节点作为源节点的出边
                    total_bandwidth = physical_edge_features[j][0]
                    bandwidth_usage = physical_edge_features[j][1]
                    available_bandwidth = total_bandwidth * (1 - bandwidth_usage)
                    connected_bandwidths.append(available_bandwidth)
            
            # 计算均值，如果没有连接则为0
            if connected_bandwidths:
                physical_link_bandwidth_means[i] = np.mean(connected_bandwidths)
            else:
                physical_link_bandwidth_means[i] = 0.0
        
        physical_features = []
        for i in range(self.num_physical_nodes):
            cpu = np.random.randint(*self.physical_cpu_range)
            memory = np.random.randint(*self.physical_memory_range)
            cpu_usage = np.random.uniform(0.1, 0.3)  # 较低的初始使用率
            memory_usage = np.random.uniform(0.1, 0.3)
            avg_available_bandwidth = physical_link_bandwidth_means[i]  # 连接的链路可用带宽均值
            
            physical_features.append([cpu, memory, cpu_usage, memory_usage, avg_available_bandwidth])

        return {
            'features': torch.tensor(physical_features, dtype=torch.float32),
            'edges': physical_edges,
            'edge_features': torch.tensor(physical_edge_features, dtype=torch.float32),
            'num_nodes': self.num_physical_nodes
        }
    
    def _adjust_difficulty(self):
        """根据历史成功率调整难度"""
        if not self.curriculum_enabled or len(self.success_history) < self.history_window:
            return
        
        recent_success_rate = np.mean(self.success_history[-self.history_window:])
        
        # 根据成功率调整难度
        if recent_success_rate > 0.85:  # 太简单，增加难度
            self.difficulty_level = min(self.max_difficulty, 
                                      self.difficulty_level + self.difficulty_adjustment_rate)
            print(f"📈 难度调整: {self.difficulty_level:.2f} (成功率: {recent_success_rate:.2%})")
        elif recent_success_rate < 0.4:  # 太难，降低难度
            self.difficulty_level = max(self.min_difficulty, 
                                      self.difficulty_level - self.difficulty_adjustment_rate)
            print(f"📉 难度调整: {self.difficulty_level:.2f} (成功率: {recent_success_rate:.2%})")
    
    def _generate_virtual_work(self):
        """生成随机虚拟工作需求（带难度调整）"""
        # 根据难度调整虚拟节点数量
        base_min, base_max = self.virtual_nodes_range
        adjusted_min = max(2, int(base_min * self.difficulty_level))
        adjusted_max = min(self.max_virtual_nodes, int(base_max * self.difficulty_level))
        
        # 确保 adjusted_min <= adjusted_max
        if adjusted_min > adjusted_max:
            adjusted_min = adjusted_max
            
        num_virtual_nodes = np.random.randint(adjusted_min, adjusted_max + 1)
        
        # 先生成边，以便计算每个节点的链路带宽特征
        virtual_edges = self._get_virtual_edges(num_virtual_nodes)
        
        # 生成边特征
        virtual_edge_features = []
        for edge in virtual_edges.T:
            min_bandwidth = np.random.randint(self.virtual_bandwidth_range[0], 
                                            int(self.virtual_bandwidth_range[1] * 0.6) + 1)
            max_bandwidth = np.random.randint(min_bandwidth, self.virtual_bandwidth_range[1] + 1)
            virtual_edge_features.append([min_bandwidth, max_bandwidth])
        
        # 计算每个虚拟节点连接的所有链路带宽区间(min+max)/2的均值
        virtual_link_bandwidth_means = [0.0] * num_virtual_nodes
        for i in range(num_virtual_nodes):
            connected_bandwidth_midpoints = []
            for j, edge in enumerate(virtual_edges.T):
                src, dst = edge[0], edge[1]
                if src == i:  # 该节点作为源节点的出边
                    min_bandwidth = virtual_edge_features[j][0]
                    max_bandwidth = virtual_edge_features[j][1]
                    bandwidth_midpoint = (min_bandwidth + max_bandwidth) / 2
                    connected_bandwidth_midpoints.append(bandwidth_midpoint)
            
            # 计算均值，如果没有连接则为0
            if connected_bandwidth_midpoints:
                virtual_link_bandwidth_means[i] = np.mean(connected_bandwidth_midpoints)
            else:
                virtual_link_bandwidth_means[i] = 0.0
        
        virtual_features = []
        for i in range(num_virtual_nodes):
            # 根据难度调整资源需求
            cpu_base_min, cpu_base_max = self.virtual_cpu_range
            cpu_min = max(1, int(cpu_base_min * self.difficulty_level))
            cpu_max = max(cpu_min, int(cpu_base_max * self.difficulty_level))
            cpu_demand = np.random.randint(cpu_min, cpu_max + 1)
            
            mem_base_min, mem_base_max = self.virtual_memory_range
            mem_min = max(1, int(mem_base_min * self.difficulty_level))
            mem_max = max(mem_min, int(mem_base_max * self.difficulty_level))
            memory_demand = np.random.randint(mem_min, mem_max + 1)
            avg_bandwidth_requirement = virtual_link_bandwidth_means[i]  # 连接的链路带宽需求均值
            
            virtual_features.append([cpu_demand, memory_demand, avg_bandwidth_requirement])

        return {
            'features': torch.tensor(virtual_features, dtype=torch.float32),
            'edges': virtual_edges,
            'edge_features': torch.tensor(virtual_edge_features, dtype=torch.float32),
            'num_nodes': num_virtual_nodes
        }
    
    def _get_physical_edges(self):
        """获取物理网络边"""
        edges = []
        
        # 确保连通性：创建一个环
        for i in range(self.num_physical_nodes):
            next_node = (i + 1) % self.num_physical_nodes
            edges.extend([[i, next_node], [next_node, i]])
        
        # 添加额外连接
        for i in range(self.num_physical_nodes):
            for j in range(i + 1, self.num_physical_nodes):
                if [i, j] not in edges and np.random.random() < self.physical_connectivity_prob:
                    edges.extend([[i, j], [j, i]])
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _get_virtual_edges(self, num_virtual_nodes):
        """获取虚拟网络边"""
        edges = []
        
        # 确保连通性：创建一个环
        for i in range(num_virtual_nodes):
            next_node = (i + 1) % num_virtual_nodes
            edges.extend([[i, next_node], [next_node, i]])
        
        # 添加额外连接
        for i in range(num_virtual_nodes):
            for j in range(i + 1, num_virtual_nodes):
                if [i, j] not in edges and np.random.random() < self.virtual_connectivity_prob:
                    edges.extend([[i, j], [j, i]])
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _initialize_network_scheduler(self):
        """初始化network_scheduler"""
        self.network_topology = NetworkTopology(self.num_physical_nodes)
        
        # 设置物理节点资源
        physical_features = self.physical_state['features'].numpy()
        for i in range(self.num_physical_nodes):
            total_cpu = physical_features[i][0]
            total_memory = physical_features[i][1]
            cpu_usage = physical_features[i][2]
            memory_usage = physical_features[i][3]
            
            used_cpu = total_cpu * cpu_usage
            used_memory = total_memory * memory_usage
            
            self.network_topology.set_node_resources(i, total_cpu, total_memory, used_cpu, used_memory)
        
        # 设置物理网络连接
        physical_edges = self.physical_state['edges'].numpy()
        physical_edge_features = self.physical_state['edge_features'].numpy()
        
        for i, (src, dst) in enumerate(physical_edges.T):
            bandwidth = physical_edge_features[i][0]
            bandwidth_usage = physical_edge_features[i][1]
            used_bandwidth = bandwidth * bandwidth_usage
            
            self.network_topology.add_link(src, dst, bandwidth, bandwidth, used_bandwidth, used_bandwidth)
        
        # 创建虚拟工作对象
        num_virtual_nodes = self.virtual_work['num_nodes']
        self.virtual_work_obj = VirtualWork(num_virtual_nodes)
        
        virtual_features = self.virtual_work['features'].numpy()
        for i in range(num_virtual_nodes):
            cpu_demand = virtual_features[i][0]
            memory_demand = virtual_features[i][1]
            self.virtual_work_obj.set_node_requirement(i, cpu_demand, memory_demand)
        
        virtual_edges = self.virtual_work['edges'].numpy()
        virtual_edge_features = self.virtual_work['edge_features'].numpy()
        
        for i, (src, dst) in enumerate(virtual_edges.T):
            min_bandwidth = virtual_edge_features[i][0]
            max_bandwidth = virtual_edge_features[i][1]
            self.virtual_work_obj.add_link_requirement(src, dst, min_bandwidth, max_bandwidth, min_bandwidth, max_bandwidth)
        
        # 创建网络调度器
        self.network_scheduler = NetworkScheduler(self.network_topology)
        self.network_scheduler.add_virtual_work(self.virtual_work_obj, work_id="work_1")
    
    def _validate_mapping_action(self, virtual_node, physical_node):
        """验证节点映射动作的有效性"""
        constraint_violations = []
        
        # 检查物理节点索引范围
        if physical_node < 0 or physical_node >= self.num_physical_nodes:
            constraint_violations.append(f"物理节点索引{physical_node}超出范围[0, {self.num_physical_nodes-1}]")
            return False, constraint_violations
        
        # 检查资源是否足够
        virtual_features = self.virtual_work['features']
        cpu_demand = virtual_features[virtual_node, 0].item()
        memory_demand = virtual_features[virtual_node, 1].item()
        
        available_resources = self.network_scheduler.topology.get_available_resources(physical_node)
        
        if cpu_demand > available_resources['cpu']:
            constraint_violations.append(f"CPU资源不足: 需要{cpu_demand}, 可用{available_resources['cpu']:.1f}")
            return False, constraint_violations
        
        if memory_demand > available_resources['memory']:
            constraint_violations.append(f"内存资源不足: 需要{memory_demand}, 可用{available_resources['memory']:.1f}")
            return False, constraint_violations
        
        return True, []
