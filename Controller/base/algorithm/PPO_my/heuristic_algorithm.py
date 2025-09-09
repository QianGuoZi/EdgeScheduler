#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from sequential_environment import SequentialNetworkSchedulerEnvironment



class NaiveResourceHeuristicAgent:
    """
    天真资源启发式算法（更笨拙的版本）
    
    策略：
    1. 映射阶段：随机选择前几个可用节点，不做优化
    2. 带宽阶段：使用固定的中等带宽等级
    """
    
    def __init__(self):
        self.name = "NaiveResourceHeuristic"
        self.preferred_bandwidth_level = 2  # 固定使用中等带宽等级
    
    def select_mapping_action(self, env: SequentialNetworkSchedulerEnvironment, state: Dict) -> int:
        """选择映射动作：在前几个可用节点中随机选择"""
        current_virtual_node = state.get('current_virtual_node', 0)
        virtual_features = state['virtual_features']
        
        # 获取当前虚拟节点的资源需求
        cpu_demand = virtual_features[current_virtual_node, 0].item()
        memory_demand = virtual_features[current_virtual_node, 1].item()
        
        # 找到所有可能的候选节点
        candidates = []
        for physical_node in range(state.get('num_physical_nodes', env.num_physical_nodes)):
            is_valid, _ = env._validate_mapping_action(current_virtual_node, physical_node)
            if not is_valid:
                continue
            
            available_resources = env.network_scheduler.topology.get_available_resources(physical_node)
            
            # 只要资源勉强够用就加入候选
            if available_resources['cpu'] >= cpu_demand and available_resources['memory'] >= memory_demand:
                candidates.append(physical_node)
        
        # 天真策略：如果有候选，随机选择前面的几个
        if candidates:
            # 只考虑前面的候选节点（最多3个），这样选择会比较局限
            limited_candidates = candidates[:min(3, len(candidates))]
            return limited_candidates[current_virtual_node % len(limited_candidates)]
        
        # 如果没有合适的候选，选择第一个可用的
        for physical_node in range(state.get('num_physical_nodes', env.num_physical_nodes)):
            is_valid, _ = env._validate_mapping_action(current_virtual_node, physical_node)
            if is_valid:
                return physical_node
        
        return 0
    
    def select_bandwidth_action(self, env: SequentialNetworkSchedulerEnvironment, state: Dict) -> int:
        """选择带宽动作：使用固定的中等带宽等级"""
        current_link_index = state.get('current_link_index', 0)
        
        # 天真策略：总是尝试使用固定的中等带宽等级
        target_level = min(self.preferred_bandwidth_level, env.bandwidth_levels - 1)
        is_valid, _ = env._validate_bandwidth_action(current_link_index, target_level)
        if is_valid:
            return target_level
        
        # 如果固定等级不可用，降级尝试
        for level in range(target_level - 1, -1, -1):
            is_valid, _ = env._validate_bandwidth_action(current_link_index, level)
            if is_valid:
                return level
        
        # 如果低等级都不行，尝试高等级
        for level in range(target_level + 1, env.bandwidth_levels):
            is_valid, _ = env._validate_bandwidth_action(current_link_index, level)
            if is_valid:
                return level
        
        return 0


class SmartLoadBalanceHeuristicAgent:
    """
    智能负载均衡启发式算法
    
    策略：
    1. 映射阶段：综合考虑负载均衡、资源效率和连接性
    2. 带宽阶段：根据链路重要性和可用资源动态分配带宽
    """
    
    def __init__(self):
        self.name = "SmartLoadBalanceHeuristic"
        self.node_load_history = {}  # 记录每个物理节点的负载历史
    
    def select_mapping_action(self, env: SequentialNetworkSchedulerEnvironment, state: Dict) -> int:
        """智能映射选择：综合负载均衡、资源效率和网络连接性"""
        current_virtual_node = state.get('current_virtual_node', 0)
        virtual_features = state['virtual_features']
        partial_mapping = state.get('partial_mapping', [])
        
        # 获取当前虚拟节点的资源需求
        cpu_demand = virtual_features[current_virtual_node, 0].item()
        memory_demand = virtual_features[current_virtual_node, 1].item()
        
        best_node = -1
        best_score = -1
        
        # 计算当前已映射节点的负载分布
        mapped_nodes = [node for node in partial_mapping if node != -1]
        node_usage_count = {}
        for node in mapped_nodes:
            node_usage_count[node] = node_usage_count.get(node, 0) + 1
        
        # 遍历所有物理节点
        for physical_node in range(state.get('num_physical_nodes', env.num_physical_nodes)):
            # 检查基本可行性
            is_valid, _ = env._validate_mapping_action(current_virtual_node, physical_node)
            if not is_valid:
                continue
            
            # 获取资源信息
            available_resources = env.network_scheduler.topology.get_available_resources(physical_node)
            total_resources = env.network_scheduler.topology.node_resources[physical_node]
            
            if available_resources['cpu'] < cpu_demand or available_resources['memory'] < memory_demand:
                continue
            
            # 1. 资源效率分数
            cpu_utilization = (total_resources['cpu'] - available_resources['cpu'] + cpu_demand) / total_resources['cpu']
            memory_utilization = (total_resources['memory'] - available_resources['memory'] + memory_demand) / total_resources['memory']
            
            # 理想利用率在50%-75%之间
            cpu_efficiency = max(0, 1.0 - abs(cpu_utilization - 0.625) / 0.375)
            memory_efficiency = max(0, 1.0 - abs(memory_utilization - 0.625) / 0.375)
            resource_score = (cpu_efficiency + memory_efficiency) / 2.0
            
            # 2. 负载均衡分数
            current_load = node_usage_count.get(physical_node, 0)
            avg_load = len(mapped_nodes) / max(1, state.get('num_physical_nodes', env.num_physical_nodes))
            load_balance_score = max(0, 1.0 - abs(current_load - avg_load) / max(1, avg_load))
            
            # 3. 连接性分数（与已映射节点的连接程度）
            connectivity_score = 0.0
            if mapped_nodes:
                connected_count = 0
                for mapped_node in set(mapped_nodes):
                    # 检查是否有直接连接
                    if env.network_scheduler.topology.get_available_bandwidth(physical_node, mapped_node) > 0:
                        connected_count += 1
                connectivity_score = connected_count / len(set(mapped_nodes))
            else:
                connectivity_score = 1.0  # 第一个节点没有连接性考虑
            
            # 4. 资源余量分数（避免选择资源紧张的节点）
            cpu_margin = available_resources['cpu'] / total_resources['cpu']
            memory_margin = available_resources['memory'] / total_resources['memory']
            margin_score = (cpu_margin + memory_margin) / 2.0
            
            # 综合评分
            total_score = (0.35 * resource_score + 
                          0.25 * load_balance_score + 
                          0.25 * connectivity_score + 
                          0.15 * margin_score)
            
            if total_score > best_score:
                best_score = total_score
                best_node = physical_node
        
        # 更新负载历史
        if best_node != -1:
            self.node_load_history[best_node] = self.node_load_history.get(best_node, 0) + 1
        
        # 回退策略
        if best_node == -1:
            for physical_node in range(state.get('num_physical_nodes', env.num_physical_nodes)):
                is_valid, _ = env._validate_mapping_action(current_virtual_node, physical_node)
                if is_valid:
                    return physical_node
            return 0
        
        return best_node
    
    def select_bandwidth_action(self, env: SequentialNetworkSchedulerEnvironment, state: Dict) -> int:
        """智能带宽分配：根据链路重要性和资源可用性"""
        current_link_index = state.get('current_link_index', 0)
        virtual_edge_features = state['virtual_edge_features']
        virtual_edges = state['virtual_edges']
        
        # 获取链路信息
        src = virtual_edges[0, current_link_index].item()
        dst = virtual_edges[1, current_link_index].item()
        min_bandwidth = virtual_edge_features[current_link_index, 0].item()
        max_bandwidth = virtual_edge_features[current_link_index, 1].item()
        
        # 获取物理映射
        partial_mapping = state.get('partial_mapping', [])
        if src >= len(partial_mapping) or dst >= len(partial_mapping):
            return 0
        
        physical_src = partial_mapping[src]
        physical_dst = partial_mapping[dst]
        
        # 如果映射到同一物理节点，选择最高带宽等级
        if physical_src == physical_dst:
            return env.bandwidth_levels - 1
        
        # 获取带宽映射
        link_key = f"{src}_{dst}"
        bandwidth_mapping = state.get('bandwidth_mapping', {})
        
        if link_key not in bandwidth_mapping:
            return 0
        
        link_bandwidths = bandwidth_mapping[link_key]
        
        # 计算链路重要性（基于虚拟网络中的度）
        num_virtual_nodes = state.get('num_virtual_nodes', 0)
        src_degree = sum(1 for i in range(virtual_edges.size(1)) 
                        if virtual_edges[0, i].item() == src or virtual_edges[1, i].item() == src)
        dst_degree = sum(1 for i in range(virtual_edges.size(1)) 
                        if virtual_edges[0, i].item() == dst or virtual_edges[1, i].item() == dst)
        
        link_importance = (src_degree + dst_degree) / (2.0 * max(1, num_virtual_nodes - 1))
        
        # 根据重要性调整目标带宽
        bandwidth_demand_ratio = (max_bandwidth - min_bandwidth) / max(1, max_bandwidth)
        if link_importance > 0.7:  # 重要链路
            target_bandwidth = min_bandwidth + 0.8 * (max_bandwidth - min_bandwidth)
        elif link_importance > 0.4:  # 中等重要链路
            target_bandwidth = min_bandwidth + 0.5 * (max_bandwidth - min_bandwidth)
        else:  # 低重要性链路
            target_bandwidth = min_bandwidth + 0.2 * (max_bandwidth - min_bandwidth)
        
        # 选择最接近目标带宽的等级
        best_level = 0
        best_diff = float('inf')
        
        for level in range(env.bandwidth_levels):
            if level in link_bandwidths:
                allocated_bandwidth = link_bandwidths[level]
                
                # 必须满足最小需求
                if allocated_bandwidth < min_bandwidth:
                    continue
                
                # 验证可行性
                is_valid, _ = env._validate_bandwidth_action(current_link_index, level)
                if not is_valid:
                    continue
                
                # 计算与目标带宽的差距
                diff = abs(allocated_bandwidth - target_bandwidth)
                if diff < best_diff:
                    best_diff = diff
                    best_level = level
        
        return best_level


class ModerateResourceHeuristicAgent:
    """
    中等资源启发式算法（介于naive和smart之间）
    
    策略：
    1. 映射阶段：考虑资源利用率和简单的负载均衡，但不考虑复杂的连接性
    2. 带宽阶段：根据需求动态调整，但使用简化的策略
    """
    
    def __init__(self):
        self.name = "ModerateResourceHeuristic"
        self.node_usage_count = {}  # 简单记录节点使用次数
    
    def select_mapping_action(self, env: SequentialNetworkSchedulerEnvironment, state: Dict) -> int:
        """中等智能映射选择：考虑资源利用率和简单负载均衡"""
        current_virtual_node = state.get('current_virtual_node', 0)
        virtual_features = state['virtual_features']
        partial_mapping = state.get('partial_mapping', [])
        
        # 获取当前虚拟节点的资源需求
        cpu_demand = virtual_features[current_virtual_node, 0].item()
        memory_demand = virtual_features[current_virtual_node, 1].item()
        
        # 找到所有满足资源需求的候选节点
        candidates = []
        candidate_scores = []
        
        # 计算已映射节点的使用情况
        mapped_nodes = [node for node in partial_mapping if node != -1]
        usage_count = {}
        for node in mapped_nodes:
            usage_count[node] = usage_count.get(node, 0) + 1
        
        for physical_node in range(state.get('num_physical_nodes', env.num_physical_nodes)):
            # 检查基本可行性
            is_valid, _ = env._validate_mapping_action(current_virtual_node, physical_node)
            if not is_valid:
                continue
            
            # 获取资源信息
            available_resources = env.network_scheduler.topology.get_available_resources(physical_node)
            total_resources = env.network_scheduler.topology.node_resources[physical_node]
            
            # 必须满足资源需求
            if available_resources['cpu'] < cpu_demand or available_resources['memory'] < memory_demand:
                continue
            
            # 计算简单的评分
            score = 0.0
            
            # 1. 资源充裕度评分（比naive聪明：考虑剩余资源比例）
            cpu_available_ratio = available_resources['cpu'] / total_resources['cpu']
            memory_available_ratio = available_resources['memory'] / total_resources['memory']
            resource_abundance = (cpu_available_ratio + memory_available_ratio) / 2.0
            score += 0.4 * resource_abundance  # 权重40%
            
            # 2. 简单负载均衡（比naive聪明：避免重复使用相同节点）
            current_usage = usage_count.get(physical_node, 0)
            if len(mapped_nodes) > 0:
                avg_usage = len(mapped_nodes) / state.get('num_physical_nodes', env.num_physical_nodes)
                # 偏向使用较少的节点
                load_balance_score = max(0, 1.0 - (current_usage / max(1, avg_usage + 1)))
            else:
                load_balance_score = 1.0  # 第一个节点没有负载均衡考虑
            score += 0.3 * load_balance_score  # 权重30%
            
            # 3. 资源匹配度（比naive聪明：避免过度分配资源）
            cpu_match = 1.0 - abs(cpu_demand - available_resources['cpu']) / total_resources['cpu']
            memory_match = 1.0 - abs(memory_demand - available_resources['memory']) / total_resources['memory']
            resource_match = (cpu_match + memory_match) / 2.0
            score += 0.3 * resource_match  # 权重30%
            
            candidates.append(physical_node)
            candidate_scores.append(score)
        
        # 选择评分最高的节点
        if candidates:
            best_idx = candidate_scores.index(max(candidate_scores))
            best_node = candidates[best_idx]
            
            # 更新使用计数
            self.node_usage_count[best_node] = self.node_usage_count.get(best_node, 0) + 1
            return best_node
        
        # 回退策略：选择第一个可用的
        for physical_node in range(state.get('num_physical_nodes', env.num_physical_nodes)):
            is_valid, _ = env._validate_mapping_action(current_virtual_node, physical_node)
            if is_valid:
                return physical_node
        
        return 0
    
    def select_bandwidth_action(self, env: SequentialNetworkSchedulerEnvironment, state: Dict) -> int:
        """中等智能带宽分配：根据需求动态调整但使用简化策略"""
        current_link_index = state.get('current_link_index', 0)
        virtual_edge_features = state['virtual_edge_features']
        virtual_edges = state['virtual_edges']
        
        # 获取链路信息
        src = virtual_edges[0, current_link_index].item()
        dst = virtual_edges[1, current_link_index].item()
        min_bandwidth = virtual_edge_features[current_link_index, 0].item()
        max_bandwidth = virtual_edge_features[current_link_index, 1].item()
        
        # 获取物理映射
        partial_mapping = state.get('partial_mapping', [])
        if src >= len(partial_mapping) or dst >= len(partial_mapping):
            return 0
        
        physical_src = partial_mapping[src]
        physical_dst = partial_mapping[dst]
        
        # 如果映射到同一物理节点，选择最高带宽等级
        if physical_src == physical_dst:
            return env.bandwidth_levels - 1
        
        # 获取带宽映射
        link_key = f"{src}_{dst}"
        bandwidth_mapping = state.get('bandwidth_mapping', {})
        
        if link_key not in bandwidth_mapping:
            return 0
        
        link_bandwidths = bandwidth_mapping[link_key]
        
        # 简化的带宽需求计算（比naive聪明：考虑实际需求）
        bandwidth_range = max_bandwidth - min_bandwidth
        if bandwidth_range > 0:
            # 根据需求范围选择合适的带宽等级
            # 高需求范围：选择较高等级；低需求范围：选择较低等级
            demand_ratio = bandwidth_range / max_bandwidth
            if demand_ratio > 0.5:  # 高需求变化
                target_bandwidth = min_bandwidth + 0.7 * bandwidth_range
            elif demand_ratio > 0.2:  # 中等需求变化
                target_bandwidth = min_bandwidth + 0.5 * bandwidth_range
            else:  # 低需求变化
                target_bandwidth = min_bandwidth + 0.3 * bandwidth_range
        else:
            target_bandwidth = min_bandwidth
        
        # 选择最接近目标带宽的可行等级
        best_level = 0
        best_diff = float('inf')
        
        for level in range(env.bandwidth_levels):
            if level in link_bandwidths:
                allocated_bandwidth = link_bandwidths[level]
                
                # 必须满足最小需求
                if allocated_bandwidth < min_bandwidth:
                    continue
                
                # 验证可行性
                is_valid, _ = env._validate_bandwidth_action(current_link_index, level)
                if not is_valid:
                    continue
                
                # 计算与目标带宽的差距
                diff = abs(allocated_bandwidth - target_bandwidth)
                if diff < best_diff:
                    best_diff = diff
                    best_level = level
        
        return best_level


class FlexiTaskHeuristicAgent:
    """
    FlexiTask启发式算法
    
    基于Kubernetes调度思想的智能调度算法:
    1. Predicates（预选）：资源过滤 + 负载阈值检查
    2. Priorities（优选）：NLL（资源空闲度）+ NBB（资源均衡度）- H（热度惩罚）
    """
    
    def __init__(self):
        self.name = "FlexiTaskHeuristic"
        
        # 负载阈值配置
        self.load_threshold_2min = 0.65   # 2分钟平均利用率阈值
        self.load_threshold_30min = 0.75  # 30分钟峰值利用率阈值
        
        # 优选阶段权重配置
        self.w1 = 0.4  # NLL权重（资源空闲度）
        self.w2 = 0.4  # NBB权重（资源均衡度）
        self.w3 = 0.2  # H权重（热度惩罚）
        
        # 节点历史记录
        self.node_load_history = {}      # 节点负载历史
        self.node_selection_history = {} # 节点选择历史（用于计算热度）
        self.node_resource_history = {}  # 节点资源使用历史
        
        # 时间窗口配置（模拟）
        self.current_time = time.time()
        self.time_windows = {
            '1min': 60,
            '2min': 120,
            '5min': 300,
            '30min': 1800
        }
    
    def _update_node_history(self, physical_node: int, cpu_usage: float, memory_usage: float):
        """更新节点历史记录"""
        current_time = time.time()
        
        # 初始化节点历史
        if physical_node not in self.node_resource_history:
            self.node_resource_history[physical_node] = {
                'cpu_usage': [],
                'memory_usage': [],
                'timestamps': []
            }
        
        # 添加当前记录
        history = self.node_resource_history[physical_node]
        history['cpu_usage'].append(cpu_usage)
        history['memory_usage'].append(memory_usage)
        history['timestamps'].append(current_time)
        
        # 清理过期数据（保留30分钟内的数据）
        cutoff_time = current_time - self.time_windows['30min']
        while history['timestamps'] and history['timestamps'][0] < cutoff_time:
            history['cpu_usage'].pop(0)
            history['memory_usage'].pop(0)
            history['timestamps'].pop(0)
    
    def _get_node_load_stats(self, physical_node: int) -> Dict[str, float]:
        """获取节点负载统计信息"""
        current_time = time.time()
        
        if physical_node not in self.node_resource_history:
            return {
                'avg_cpu_2min': 0.0,
                'avg_memory_2min': 0.0,
                'peak_cpu_30min': 0.0,
                'peak_memory_30min': 0.0
            }
        
        history = self.node_resource_history[physical_node]
        if not history['timestamps']:
            return {
                'avg_cpu_2min': 0.0,
                'avg_memory_2min': 0.0,
                'peak_cpu_30min': 0.0,
                'peak_memory_30min': 0.0
            }
        
        # 2分钟平均值
        cutoff_2min = current_time - self.time_windows['2min']
        recent_cpu = [cpu for cpu, ts in zip(history['cpu_usage'], history['timestamps']) if ts >= cutoff_2min]
        recent_memory = [mem for mem, ts in zip(history['memory_usage'], history['timestamps']) if ts >= cutoff_2min]
        
        avg_cpu_2min = np.mean(recent_cpu) if recent_cpu else 0.0
        avg_memory_2min = np.mean(recent_memory) if recent_memory else 0.0
        
        # 30分钟峰值
        peak_cpu_30min = max(history['cpu_usage']) if history['cpu_usage'] else 0.0
        peak_memory_30min = max(history['memory_usage']) if history['memory_usage'] else 0.0
        
        return {
            'avg_cpu_2min': avg_cpu_2min,
            'avg_memory_2min': avg_memory_2min,
            'peak_cpu_30min': peak_cpu_30min,
            'peak_memory_30min': peak_memory_30min
        }
    
    def _get_node_hotness(self, physical_node: int) -> float:
        """计算节点热度（最近被选中的频次）"""
        current_time = time.time()
        
        if physical_node not in self.node_selection_history:
            return 0.0
        
        selection_times = self.node_selection_history[physical_node]
        
        # 统计1分钟内和5分钟内的选择次数
        cutoff_1min = current_time - self.time_windows['1min']
        cutoff_5min = current_time - self.time_windows['5min']
        
        count_1min = sum(1 for ts in selection_times if ts >= cutoff_1min)
        count_5min = sum(1 for ts in selection_times if ts >= cutoff_5min)
        
        # 热度计算：最近时间窗口内选择次数的加权和
        hotness = 0.7 * count_1min + 0.3 * count_5min
        return hotness
    
    def _predicates_filter(self, env: SequentialNetworkSchedulerEnvironment, 
                          state: Dict, cpu_demand: float, memory_demand: float) -> List[int]:
        """Predicates预选阶段：过滤不符合条件的节点"""
        candidates = []
        
        for physical_node in range(state.get('num_physical_nodes', env.num_physical_nodes)):
            # 1. 基本可行性检查
            is_valid, _ = env._validate_mapping_action(state.get('current_virtual_node', 0), physical_node)
            if not is_valid:
                continue
            
            # 2. 资源需求检查
            available_resources = env.network_scheduler.topology.get_available_resources(physical_node)
            if available_resources['cpu'] < cpu_demand or available_resources['memory'] < memory_demand:
                continue
            
            # 3. 负载阈值检查
            total_resources = env.network_scheduler.topology.node_resources[physical_node]
            current_cpu_util = (total_resources['cpu'] - available_resources['cpu']) / total_resources['cpu']
            current_memory_util = (total_resources['memory'] - available_resources['memory']) / total_resources['memory']
            
            # 更新历史记录
            self._update_node_history(physical_node, current_cpu_util, current_memory_util)
            
            # 获取负载统计
            load_stats = self._get_node_load_stats(physical_node)
            
            # 检查2分钟平均利用率阈值
            avg_util_2min = (load_stats['avg_cpu_2min'] + load_stats['avg_memory_2min']) / 2.0
            if avg_util_2min > self.load_threshold_2min:
                continue
            
            # 检查30分钟峰值利用率阈值
            peak_util_30min = max(load_stats['peak_cpu_30min'], load_stats['peak_memory_30min'])
            if peak_util_30min > self.load_threshold_30min:
                continue
            
            candidates.append(physical_node)
        
        return candidates
    
    def _calculate_nll_score(self, physical_node: int, env: SequentialNetworkSchedulerEnvironment) -> float:
        """计算节点资源空闲度分数NLL"""
        available_resources = env.network_scheduler.topology.get_available_resources(physical_node)
        total_resources = env.network_scheduler.topology.node_resources[physical_node]
        
        # 当前资源剩余率
        cpu_available_ratio = available_resources['cpu'] / total_resources['cpu']
        memory_available_ratio = available_resources['memory'] / total_resources['memory']
        
        # 获取历史统计信息
        load_stats = self._get_node_load_stats(physical_node)
        
        # 历史平均剩余率（基于2分钟平均利用率）
        hist_cpu_available = 1.0 - load_stats['avg_cpu_2min']
        hist_memory_available = 1.0 - load_stats['avg_memory_2min']
        
        # 综合当前和历史数据（权重：当前70%，历史30%）
        cpu_score = 0.7 * cpu_available_ratio + 0.3 * hist_cpu_available
        memory_score = 0.7 * memory_available_ratio + 0.3 * hist_memory_available
        
        # 加权求和（CPU和内存等权重）
        nll_score = (cpu_score + memory_score) / 2.0
        return nll_score
    
    def _calculate_nbb_score(self, physical_node: int, env: SequentialNetworkSchedulerEnvironment,
                           cpu_demand: float, memory_demand: float) -> float:
        """计算Pod部署后资源均衡度分数NBB"""
        available_resources = env.network_scheduler.topology.get_available_resources(physical_node)
        total_resources = env.network_scheduler.topology.node_resources[physical_node]
        
        # 计算部署后的资源利用率
        after_cpu_util = (total_resources['cpu'] - available_resources['cpu'] + cpu_demand) / total_resources['cpu']
        after_memory_util = (total_resources['memory'] - available_resources['memory'] + memory_demand) / total_resources['memory']
        
        # 计算均衡度：利用率的标准差越小越均衡
        utilizations = [after_cpu_util, after_memory_util]
        std_dev = np.std(utilizations)
        
        # 将标准差转换为分数：标准差越小分数越高
        # 使用指数衰减函数：score = exp(-k * std_dev)
        nbb_score = np.exp(-5.0 * std_dev)  # k=5.0是调节参数
        
        return nbb_score
    
    def _update_selection_history(self, physical_node: int):
        """更新节点选择历史"""
        current_time = time.time()
        
        if physical_node not in self.node_selection_history:
            self.node_selection_history[physical_node] = []
        
        # 添加当前选择时间
        self.node_selection_history[physical_node].append(current_time)
        
        # 清理过期数据（保留5分钟内的数据）
        cutoff_time = current_time - self.time_windows['5min']
        self.node_selection_history[physical_node] = [
            ts for ts in self.node_selection_history[physical_node] if ts >= cutoff_time
        ]
    
    def select_mapping_action(self, env: SequentialNetworkSchedulerEnvironment, state: Dict) -> int:
        """FlexiTask映射选择：Predicates + Priorities两阶段调度"""
        current_virtual_node = state.get('current_virtual_node', 0)
        virtual_features = state['virtual_features']
        
        # 获取当前虚拟节点的资源需求
        cpu_demand = virtual_features[current_virtual_node, 0].item()
        memory_demand = virtual_features[current_virtual_node, 1].item()
        
        # 第一阶段：Predicates预选
        candidates = self._predicates_filter(env, state, cpu_demand, memory_demand)
        
        if not candidates:
            # 如果预选阶段没有候选节点，回退到基本策略
            for physical_node in range(state.get('num_physical_nodes', env.num_physical_nodes)):
                is_valid, _ = env._validate_mapping_action(current_virtual_node, physical_node)
                if is_valid:
                    available_resources = env.network_scheduler.topology.get_available_resources(physical_node)
                    if available_resources['cpu'] >= cpu_demand and available_resources['memory'] >= memory_demand:
                        self._update_selection_history(physical_node)
                        return physical_node
            return 0
        
        # 第二阶段：Priorities优选
        best_node = -1
        best_score = -1
        
        for physical_node in candidates:
            # 计算NLL分数（资源空闲度）
            nll_score = self._calculate_nll_score(physical_node, env)
            
            # 计算NBB分数（资源均衡度）
            nbb_score = self._calculate_nbb_score(physical_node, env, cpu_demand, memory_demand)
            
            # 计算热度惩罚
            hotness = self._get_node_hotness(physical_node)
            
            # 综合评分：NFF = w1·NLL + w2·NBB - w3·H
            final_score = (self.w1 * nll_score + 
                          self.w2 * nbb_score - 
                          self.w3 * hotness)
            
            if final_score > best_score:
                best_score = final_score
                best_node = physical_node
        
        # 更新选择历史
        if best_node != -1:
            self._update_selection_history(best_node)
            return best_node
        
        # 最终回退
        return candidates[0] if candidates else 0
    
    def select_bandwidth_action(self, env: SequentialNetworkSchedulerEnvironment, state: Dict) -> int:
        """使用中等智能的带宽分配方法"""
        current_link_index = state.get('current_link_index', 0)
        virtual_edge_features = state['virtual_edge_features']
        virtual_edges = state['virtual_edges']
        
        # 获取链路信息
        src = virtual_edges[0, current_link_index].item()
        dst = virtual_edges[1, current_link_index].item()
        min_bandwidth = virtual_edge_features[current_link_index, 0].item()
        max_bandwidth = virtual_edge_features[current_link_index, 1].item()
        
        # 获取物理映射
        partial_mapping = state.get('partial_mapping', [])
        if src >= len(partial_mapping) or dst >= len(partial_mapping):
            return 0
        
        physical_src = partial_mapping[src]
        physical_dst = partial_mapping[dst]
        
        # 如果映射到同一物理节点，选择最高带宽等级
        if physical_src == physical_dst:
            return env.bandwidth_levels - 1
        
        # 获取带宽映射
        link_key = f"{src}_{dst}"
        bandwidth_mapping = state.get('bandwidth_mapping', {})
        
        if link_key not in bandwidth_mapping:
            return 0
        
        link_bandwidths = bandwidth_mapping[link_key]
        
        # 简化的带宽需求计算（采用中等智能策略）
        bandwidth_range = max_bandwidth - min_bandwidth
        if bandwidth_range > 0:
            # 根据需求范围选择合适的带宽等级
            demand_ratio = bandwidth_range / max_bandwidth
            if demand_ratio > 0.5:  # 高需求变化
                target_bandwidth = min_bandwidth + 0.7 * bandwidth_range
            elif demand_ratio > 0.2:  # 中等需求变化
                target_bandwidth = min_bandwidth + 0.5 * bandwidth_range
            else:  # 低需求变化
                target_bandwidth = min_bandwidth + 0.3 * bandwidth_range
        else:
            target_bandwidth = min_bandwidth
        
        # 选择最接近目标带宽的可行等级
        best_level = 0
        best_diff = float('inf')
        
        for level in range(env.bandwidth_levels):
            if level in link_bandwidths:
                allocated_bandwidth = link_bandwidths[level]
                
                # 必须满足最小需求
                if allocated_bandwidth < min_bandwidth:
                    continue
                
                # 验证可行性
                is_valid, _ = env._validate_bandwidth_action(current_link_index, level)
                if not is_valid:
                    continue
                
                # 计算与目标带宽的差距
                diff = abs(allocated_bandwidth - target_bandwidth)
                if diff < best_diff:
                    best_diff = diff
                    best_level = level
        
        return best_level


def create_heuristic_agent(algorithm_type: str = "naive"):
    """创建启发式算法代理"""
    if algorithm_type.lower() == "naive":
        return NaiveResourceHeuristicAgent()
    elif algorithm_type.lower() == "moderate":
        return ModerateResourceHeuristicAgent()
    elif algorithm_type.lower() == "smart":
        return SmartLoadBalanceHeuristicAgent()
    elif algorithm_type.lower() == "flexitask":
        return FlexiTaskHeuristicAgent()
    else:
        raise ValueError(f"Unknown heuristic algorithm type: {algorithm_type}. Available: 'naive', 'moderate', 'smart', 'flexitask'")


def run_heuristic_episode(env: SequentialNetworkSchedulerEnvironment,
                         heuristic_agent,
                         max_steps: int = 50) -> Tuple[float, bool, Dict]:
    """使用启发式算法运行一个Episode"""
    state = env.reset()
    
    # 在reset后集成原始奖励计算器（需要导入）
    try:
        from original_reward import integrate_with_network_scheduler
        if hasattr(env, 'network_scheduler') and env.network_scheduler is not None:
            integrate_with_network_scheduler(env.network_scheduler)
    except Exception as e:
        print(f"Warning: Failed to integrate original reward calculator in heuristic: {e}")
    
    done = False
    total_reward = 0.0
    steps = 0
    
    # 记录详细信息
    episode_info = {
        'num_virtual_nodes': state.get('num_virtual_nodes', 0),
        'num_virtual_links': state.get('num_virtual_links', 0),
        'algorithm': heuristic_agent.name,
        'steps': 0
    }

    while not done and steps < max_steps:
        if state.get('mapping_phase', True):
            # 映射阶段
            action = heuristic_agent.select_mapping_action(env, state)
        else:
            # 带宽分配阶段
            action = heuristic_agent.select_bandwidth_action(env, state)

        state, reward, done, info = env.step(int(action))
        total_reward += float(reward)
        steps += 1

    # 成功判定：所有虚拟节点映射完成且总奖励为正
    all_nodes_mapped = all(node != -1 for node in (env.partial_mapping or []))
    success = bool(all_nodes_mapped and total_reward > 0)
    
    # 计算L和D_BW指标
    load_balance_degree = float('nan')
    bandwidth_satisfaction = float('nan')
    
    if (hasattr(env, 'network_scheduler') and env.network_scheduler is not None and 
        hasattr(env.network_scheduler, 'get_original_reward_components')):
        try:
            if hasattr(env, 'virtual_work_obj') and env.virtual_work_obj is not None:
                components = env.network_scheduler.get_original_reward_components(env.virtual_work_obj)
                load_balance_degree = components.get('L', float('nan'))
                bandwidth_satisfaction = components.get('D_BW', float('nan'))
        except Exception as e:
            print(f"Warning: Failed to calculate L and D_BW for Heuristic: {e}")
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward,
        'load_balance_degree': load_balance_degree,
        'bandwidth_satisfaction': bandwidth_satisfaction,
        'final_mapping': env.partial_mapping.copy() if env.partial_mapping else [],
        'final_bandwidth': env.partial_bandwidth.copy() if env.partial_bandwidth else []
    })
    
    return total_reward, success, episode_info


# 测试函数
def test_heuristic_algorithms():
    """测试启发式算法"""
    print("🧪 测试启发式算法")
    
    from sequential_environment import SequentialNetworkSchedulerEnvironment
    
    # 创建测试环境
    env = SequentialNetworkSchedulerEnvironment(
        num_physical_nodes=5,
        max_virtual_nodes=4,
        virtual_nodes_range=(3, 4),
        bandwidth_levels=5,
        seed=42
    )
    
    # 测试四种启发式算法
    algorithms = ["naive", "moderate", "smart", "flexitask"]
    
    for algo_type in algorithms:
        print(f"\n🔍 测试 {algo_type} 启发式算法:")
        
        agent = create_heuristic_agent(algo_type)
        total_rewards = []
        success_count = 0
        
        for episode in range(5):
            reward, success, info = run_heuristic_episode(env, agent)
            total_rewards.append(reward)
            if success:
                success_count += 1
            
            print(f"   Episode {episode + 1}: 奖励={reward:.3f}, 成功={success}, 步数={info['steps']}")
        
        avg_reward = np.mean(total_rewards)
        success_rate = success_count / 5
        print(f"   平均奖励: {avg_reward:.3f}, 成功率: {success_rate:.1%}")


if __name__ == "__main__":
    test_heuristic_algorithms()
