#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple, Dict, List
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler
import torch.nn as nn

class SequentialNetworkSchedulerEnvironment:
    """Sequential网络调度环境，将决策过程分解为多个步骤"""
    
    def __init__(self, 
                 num_physical_nodes: int = 5,
                 max_virtual_nodes: int = 6,  # 增加到6个虚拟节点
                 bandwidth_levels: int = 5,  # 简化带宽等级
                 # 物理节点资源范围 - 减少物理资源
                 physical_cpu_range: Tuple[int, int] = (40, 80),  # 从(50,100)减少到(40,80)
                 physical_memory_range: Tuple[int, int] = (40, 80),  # 从(50,100)减少到(40,80)
                 physical_bandwidth_range: Tuple[int, int] = (40, 80),  # 从(50,100)减少到(40,80)
                 # 虚拟节点资源范围 - 增加虚拟需求
                 virtual_cpu_range: Tuple[int, int] = (12, 25),  # 从(8,15)增加到(12,25)
                 virtual_memory_range: Tuple[int, int] = (12, 25),  # 从(8,15)增加到(12,25)
                 virtual_bandwidth_range: Tuple[int, int] = (8, 20),  # 从(5,15)增加到(8,20)
                 # 网络连接概率
                 physical_connectivity_prob: float = 0.8,
                 virtual_connectivity_prob: float = 0.7,
                 # 虚拟节点数范围 - 增加虚拟节点数量
                 virtual_nodes_range: Tuple[int, int] = (4, 6),  # 从(3,4)增加到(4,6)
                 # 随机种子
                 seed: int = None,
                 curriculum_enabled: bool = True):
        
        # 设置随机种子
        if seed is not None:
            self._set_random_seed(seed)
            print(f"🌱 Sequential环境设置随机种子: {seed}")
        
        self.seed = seed
        
        # 环境参数
        self.num_physical_nodes = num_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
        self.bandwidth_levels = bandwidth_levels
        
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
        
        # Sequential决策状态
        self.current_step = 0
        self.current_virtual_node = 0
        self.current_link_index = 0
        self.mapping_phase = True  # True: 映射阶段, False: 带宽分配阶段
        
        # 部分决策结果
        self.partial_mapping = None  # [-1, -1, -1, ...] -1表示未映射
        self.partial_bandwidth = None  # [0, 0, 0, ...] 0表示未分配
        
        # 环境状态
        self.physical_state = None
        self.virtual_work = None
        self.bandwidth_mapping = None
        
        # network_scheduler相关对象
        self.network_topology = None
        self.virtual_work_obj = None
        self.network_scheduler = None
        
    
        
        # 统计信息
        self.episode_stats = {
            'mapping_rewards': [],
            'bandwidth_rewards': [],
            'step_rewards': [],
            'resource_utilizations': [],
            'constraint_violations': []
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
        print(f"🔒 Sequential环境随机种子设置完成: {seed}")
    
    def reset(self):
        """重置环境，开始新的episode"""
        # 在新episode开始前调整难度
        self._adjust_difficulty()
        
        self.current_step = 0
        self.current_virtual_node = 0
        self.current_link_index = 0
        self.mapping_phase = True
        
        # 生成网络状态（会受到难度调整的影响）
        self.physical_state = self._generate_physical_state()
        self.virtual_work = self._generate_virtual_work()
        
        # 初始化部分决策结果
        num_virtual_nodes = self.virtual_work['num_nodes']
        self.partial_mapping = [-1] * num_virtual_nodes  # -1表示未映射
        
        # 计算虚拟链路数量
        num_virtual_links = self.virtual_work['edges'].size(1)
        self.partial_bandwidth = [0] * num_virtual_links  # 0表示未分配
        
        # 创建带宽映射
        self.bandwidth_mapping = self._create_bandwidth_mapping()
        
        # 初始化网络调度器
        self._initialize_network_scheduler()
        
        # 重置统计信息
        self.episode_stats = {
            'mapping_rewards': [],
            'bandwidth_rewards': [],
            'step_rewards': [],
            'resource_utilizations': [],
            'constraint_violations': []
        }
        
        print(f"🔄 Sequential环境重置完成:")
        print(f"   虚拟节点数: {num_virtual_nodes}")
        print(f"   虚拟链路数: {num_virtual_links}")
        print(f"   物理节点数: {self.num_physical_nodes}")
        print(f"   预计总步数: {num_virtual_nodes + num_virtual_links}")
        
        return self._get_state()
    
    def step(self, action):
        """执行一个决策步骤"""
        if self.mapping_phase:
            return self._step_mapping(action)
        else:
            return self._step_bandwidth(action)
    
    def _step_mapping(self, physical_node_action):
        """执行节点映射步骤"""
        print(f"📍 映射步骤 {self.current_step}: 虚拟节点{self.current_virtual_node} -> 物理节点{physical_node_action}")
        
        # 验证动作有效性
        is_valid, constraint_violations = self._validate_mapping_action(
            self.current_virtual_node, physical_node_action)
        
        if not is_valid:
            # 无效动作，给予小惩罚并继续
            reward = -0.1
            info = {
                'is_valid': False,
                'constraint_violations': constraint_violations,
                'phase': 'mapping',
                'current_virtual_node': self.current_virtual_node,
                'action': physical_node_action
            }
            print(f"❌ 映射动作无效: {constraint_violations}")
        else:
            # 执行映射
            self.partial_mapping[self.current_virtual_node] = physical_node_action
            
            # 在network_scheduler中执行映射
            success = self.network_scheduler.schedule_node(self.current_virtual_node, physical_node_action)
            if not success:
                print(f"⚠️ NetworkScheduler映射失败")
            
            # 计算即时奖励
            reward = self._calculate_mapping_reward(physical_node_action)
            
            info = {
                'is_valid': True,
                'constraint_violations': [],
                'phase': 'mapping',
                'current_virtual_node': self.current_virtual_node,
                'action': physical_node_action,
                'partial_mapping': self.partial_mapping.copy(),
                'mapping_reward': reward
            }
            print(f"✅ 映射成功，即时奖励: {reward:.3f}")
        
        # 更新状态
        self.current_step += 1
        self.current_virtual_node += 1
        
        # 检查是否完成所有节点映射
        if self.current_virtual_node >= self.virtual_work['num_nodes']:
            # 检查是否所有节点都成功映射
            unmapped_nodes = [i for i, mapping in enumerate(self.partial_mapping) if mapping == -1]
            if unmapped_nodes:
                print(f"❌ 映射阶段结束，但有{len(unmapped_nodes)}个节点未成功映射: {unmapped_nodes}")
                print(f"🚫 Episode提前结束")
                # 不进入带宽分配阶段，直接结束episode
            else:
                self.mapping_phase = False
                self.current_link_index = 0
                print(f"🎯 节点映射阶段完成，进入带宽分配阶段")
        
        # 检查episode是否结束
        done = self._check_episode_done()
        
        # 如果episode结束（由于映射失败），计算最终奖励
        if done:
            final_reward = self._calculate_final_reward()
            reward += final_reward  # 加上最终奖励
            info['final_reward'] = final_reward
            
            # 记录成功与否用于Curriculum Learning
            episode_success = final_reward > 0.0  # 正奖励视为成功
            self.success_history.append(episode_success)
            self.episode_count += 1
            
            # 保持历史记录在窗口大小内
            if len(self.success_history) > self.history_window * 2:
                self.success_history = self.success_history[-self.history_window:]
            
            print(f"🏁 Episode {self.episode_count}结束（映射阶段），最终奖励: {final_reward:.3f}, 总奖励: {reward:.3f}, 成功: {episode_success}")
        
        # 更新统计信息
        self.episode_stats['mapping_rewards'].append(reward)
        self.episode_stats['step_rewards'].append(reward)
        
        return self._get_state(), reward, done, info
    
    def _step_bandwidth(self, bandwidth_level_action):
        """执行带宽分配步骤"""
        virtual_edges = self.virtual_work['edges']
        current_link_src = virtual_edges[0, self.current_link_index].item()
        current_link_dst = virtual_edges[1, self.current_link_index].item()
        
        # print(f"🔗 带宽步骤 {self.current_step}: 链路({current_link_src},{current_link_dst}) -> 等级{bandwidth_level_action}")
        
        # 验证动作有效性
        is_valid, constraint_violations = self._validate_bandwidth_action(
            self.current_link_index, bandwidth_level_action)
        
        if not is_valid:
            reward = -0.1
            info = {
                'is_valid': False,
                'constraint_violations': constraint_violations,
                'phase': 'bandwidth',
                'current_link': (current_link_src, current_link_dst),
                'action': bandwidth_level_action
            }
            # print(f"❌ 带宽动作无效: {constraint_violations}")
        else:
            # 执行带宽分配
            self.partial_bandwidth[self.current_link_index] = bandwidth_level_action
            
            # 获取实际带宽值
            link_key = f"{current_link_src}_{current_link_dst}"
            actual_bandwidth = self.bandwidth_mapping[link_key][bandwidth_level_action]
            
            # 在network_scheduler中执行带宽分配
            success = self.network_scheduler.allocate_bandwidth(
                current_link_src, current_link_dst, actual_bandwidth)
            if not success:
                print(f"⚠️ NetworkScheduler带宽分配失败")
            
            # 计算即时奖励
            reward = self._calculate_bandwidth_reward(bandwidth_level_action)
            
            info = {
                'is_valid': True,
                'constraint_violations': [],
                'phase': 'bandwidth',
                'current_link': (current_link_src, current_link_dst),
                'action': bandwidth_level_action,
                'actual_bandwidth': actual_bandwidth,
                'partial_bandwidth': self.partial_bandwidth.copy(),
                'bandwidth_reward': reward
            }
            # print(f"✅ 带宽分配成功，实际带宽: {actual_bandwidth}, 即时奖励: {reward:.3f}")
        
        # 更新状态
        self.current_step += 1
        self.current_link_index += 1
        
        # 检查episode是否结束
        done = self._check_episode_done()
        
        # 如果episode结束，计算最终奖励
        if done:
            final_reward = self._calculate_final_reward()
            reward += final_reward  # 加上最终奖励
            info['final_reward'] = final_reward
            
            # 记录成功与否用于Curriculum Learning
            episode_success = final_reward > 0.0  # 正奖励视为成功
            self.success_history.append(episode_success)
            self.episode_count += 1
            
            # 保持历史记录在窗口大小内
            if len(self.success_history) > self.history_window * 2:
                self.success_history = self.success_history[-self.history_window:]
            
            # print(f"🏁 Episode {self.episode_count}结束，最终奖励: {final_reward:.3f}, 总奖励: {reward:.3f}, 成功: {episode_success}")
        
        # 更新统计信息
        self.episode_stats['bandwidth_rewards'].append(reward)
        self.episode_stats['step_rewards'].append(reward)
        
        return self._get_state(), reward, done, info
    
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
        
        total_reward = base_reward + efficiency_reward
        
        return max(0.0, total_reward)  # 确保奖励非负
    
    def _calculate_bandwidth_reward(self, bandwidth_level_action):
        """计算带宽分配的即时奖励"""
        current_link_index = self.current_link_index
        virtual_edge_features = self.virtual_work['edge_features']
        min_bandwidth = virtual_edge_features[current_link_index, 0].item()
        max_bandwidth = virtual_edge_features[current_link_index, 1].item()
        
        # 获取实际分配的带宽
        virtual_edges = self.virtual_work['edges']
        src = virtual_edges[0, current_link_index].item()
        dst = virtual_edges[1, current_link_index].item()
        link_key = f"{src}_{dst}"
        allocated_bandwidth = self.bandwidth_mapping[link_key][bandwidth_level_action]
        
        # 计算满足度奖励
        if allocated_bandwidth >= min_bandwidth:
            if allocated_bandwidth <= max_bandwidth:
                # 在需求范围内，线性奖励
                satisfaction = (allocated_bandwidth - min_bandwidth) / (max_bandwidth - min_bandwidth) if max_bandwidth > min_bandwidth else 1.0
            else:
                # 超过最大需求也给予奖励，但略低
                satisfaction = 1.0
        else:
            # 未满足最小需求
            satisfaction = 0.1
        
        return satisfaction
    
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
    
    def _check_episode_done(self):
        """检查episode是否结束"""
        # 如果还在映射阶段，检查是否所有节点都尝试映射过了
        if self.mapping_phase:
            # 如果所有节点都尝试过映射，检查是否有未成功映射的节点
            if self.current_virtual_node >= self.virtual_work['num_nodes']:
                unmapped_nodes = [i for i, mapping in enumerate(self.partial_mapping) if mapping == -1]
                if unmapped_nodes:
                    # 有未映射的节点，episode结束
                    return True
            return False
        
        # 如果在带宽分配阶段，检查是否完成所有链路
        num_virtual_links = self.virtual_work['edges'].size(1)
        return self.current_link_index >= num_virtual_links
    
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
            'bandwidth_mapping': self.bandwidth_mapping,
            
            # Sequential决策状态信息
            'current_step': self.current_step,
            'mapping_phase': self.mapping_phase,
            'current_virtual_node': self.current_virtual_node if self.mapping_phase else -1,
            'current_link_index': self.current_link_index if not self.mapping_phase else -1,
            'partial_mapping': self.partial_mapping.copy(),
            'partial_bandwidth': self.partial_bandwidth.copy(),
            
            # 辅助信息
            'num_physical_nodes': self.num_physical_nodes,
            'num_virtual_nodes': self.virtual_work['num_nodes'],
            'num_virtual_links': self.virtual_work['edges'].size(1)
        }
        
        return state
    
    # ============= 以下是从原环境复制的辅助函数 =============
    
    def _generate_physical_state(self):
        """生成随机物理网络状态"""
        physical_features = []
        
        for i in range(self.num_physical_nodes):
            cpu = np.random.randint(*self.physical_cpu_range)
            memory = np.random.randint(*self.physical_memory_range)
            cpu_usage = np.random.uniform(0.1, 0.3)  # 较低的初始使用率
            memory_usage = np.random.uniform(0.1, 0.3)
            
            physical_features.append([cpu, memory, cpu_usage, memory_usage])
        
        physical_edges = self._get_physical_edges()
        
        physical_edge_features = []
        for edge in physical_edges.T:
            bandwidth = np.random.randint(*self.physical_bandwidth_range)
            bandwidth_usage = np.random.uniform(0.1, 0.3)  # 较低的初始使用率
            physical_edge_features.append([bandwidth, bandwidth_usage])

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
            virtual_features.append([cpu_demand, memory_demand])
        
        virtual_edges = self._get_virtual_edges(num_virtual_nodes)
        
        virtual_edge_features = []
        for edge in virtual_edges.T:
            min_bandwidth = np.random.randint(self.virtual_bandwidth_range[0], 
                                            int(self.virtual_bandwidth_range[1] * 0.6) + 1)
            max_bandwidth = np.random.randint(min_bandwidth, self.virtual_bandwidth_range[1] + 1)
            virtual_edge_features.append([min_bandwidth, max_bandwidth])

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
    
    def _create_bandwidth_mapping(self):
        """创建带宽等级映射"""
        virtual_edges = self.virtual_work['edges'].numpy()
        virtual_edge_features = self.virtual_work['edge_features'].numpy()
        
        link_bandwidth_mappings = {}
        
        for i, (src, dst) in enumerate(virtual_edges.T):
            min_bandwidth = virtual_edge_features[i][0]
            max_bandwidth = virtual_edge_features[i][1]
            
            # 创建等级到带宽的映射
            bandwidths = np.linspace(min_bandwidth, max_bandwidth, self.bandwidth_levels).astype(int)
            link_mapping = {level: int(bandwidths[level]) for level in range(self.bandwidth_levels)}
            
            link_key = f"{src}_{dst}"
            link_bandwidth_mappings[link_key] = link_mapping
        
        return link_bandwidth_mappings
    
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
    
    def _validate_bandwidth_action(self, link_index, bandwidth_level):
        """验证带宽分配动作的有效性"""
        constraint_violations = []
        
        # 检查带宽等级范围
        if bandwidth_level < 0 or bandwidth_level >= self.bandwidth_levels:
            constraint_violations.append(f"带宽等级{bandwidth_level}超出范围[0, {self.bandwidth_levels-1}]")
            return False, constraint_violations
        
        # 获取链路信息
        virtual_edges = self.virtual_work['edges']
        src = virtual_edges[0, link_index].item()
        dst = virtual_edges[1, link_index].item()
        
        # 检查节点是否已映射
        if self.partial_mapping[src] == -1 or self.partial_mapping[dst] == -1:
            constraint_violations.append(f"链路端点未完成映射: src={src}, dst={dst}")
            return False, constraint_violations
        
        # 如果映射到同一物理节点，无需检查带宽约束
        physical_src = self.partial_mapping[src]
        physical_dst = self.partial_mapping[dst]
        if physical_src == physical_dst:
            return True, []
        
        # 检查带宽约束
        link_key = f"{src}_{dst}"
        if link_key not in self.bandwidth_mapping:
            constraint_violations.append(f"链路{link_key}的带宽映射不存在")
            return False, constraint_violations
        
        required_bandwidth = self.bandwidth_mapping[link_key][bandwidth_level]
        
        # 获取物理路径
        path = self.network_scheduler.topology.get_shortest_path(physical_src, physical_dst)
        if not path:
            constraint_violations.append(f"物理路径不可达: {physical_src} -> {physical_dst}")
            return False, constraint_violations
        
        # 检查路径上的带宽是否足够
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            available = self.network_scheduler.topology.get_available_bandwidth(u, v)
            if required_bandwidth > available:
                constraint_violations.append(f"物理链路({u},{v})带宽不足: 需要{required_bandwidth}, 可用{available:.1f}")
                return False, constraint_violations
        
        return True, []


# 测试函数
def test_sequential_environment():
    """测试Sequential环境"""
    print("🧪 开始测试Sequential环境")
    
    env = SequentialNetworkSchedulerEnvironment(
        num_physical_nodes=3,
        max_virtual_nodes=3,
        virtual_nodes_range=(2, 3),
        bandwidth_levels=3,
        seed=42
    )
    
    # 测试一个完整的episode
    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print(f"\n📊 初始状态:")
    print(f"   虚拟节点数: {state['num_virtual_nodes']}")
    print(f"   虚拟链路数: {state['num_virtual_links']}")
    print(f"   物理节点数: {state['num_physical_nodes']}")
    
    while not done and step < 20:  # 防止无限循环
        print(f"\n--- Step {step} ---")
        print(f"Phase: {'Mapping' if state['mapping_phase'] else 'Bandwidth'}")
        
        if state['mapping_phase']:
            # 映射阶段：随机选择物理节点
            action = np.random.randint(0, state['num_physical_nodes'])
            print(f"Mapping action: 虚拟节点{state['current_virtual_node']} -> 物理节点{action}")
        else:
            # 带宽阶段：随机选择带宽等级
            action = np.random.randint(0, env.bandwidth_levels)
            print(f"Bandwidth action: 链路{state['current_link_index']} -> 等级{action}")
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Reward: {reward:.3f}, Done: {done}")
        print(f"Valid: {info.get('is_valid', True)}")
        
        state = next_state
        step += 1
    
    print(f"\n🎯 Episode完成!")
    print(f"   总步数: {step}")
    print(f"   总奖励: {total_reward:.3f}")
    print(f"   最终映射: {env.partial_mapping}")
    print(f"   最终带宽: {env.partial_bandwidth}")

if __name__ == "__main__":
    test_sequential_environment()