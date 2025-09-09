#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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
        
        # 如果映射到同一物理节点，选择最低带宽等级
        if physical_src == physical_dst:
            return 0
        
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


def create_heuristic_agent(algorithm_type: str = "naive"):
    """创建启发式算法代理"""
    if algorithm_type.lower() == "naive":
        return NaiveResourceHeuristicAgent()
    elif algorithm_type.lower() == "smart":
        return SmartLoadBalanceHeuristicAgent()
    else:
        raise ValueError(f"Unknown heuristic algorithm type: {algorithm_type}. Available: 'naive', 'smart'")


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
    
    # 测试两种启发式算法
    algorithms = ["naive", "smart"]
    
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
