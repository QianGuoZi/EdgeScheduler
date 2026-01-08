#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级启发式集成方案
只修改奖励函数和添加少量状态特征，最小化对原始PPO架构的改动
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from network_scheduler import VirtualWork

from sequential_environment import SequentialNetworkSchedulerEnvironment


class NewHeuristicEnvironment(SequentialNetworkSchedulerEnvironment):
    """
    轻量级启发式引导环境
    只修改奖励函数和状态表示，保持原有架构
    """
    
    def __init__(self, 
                 # 继承原有环境的所有参数
                 num_physical_nodes: int = 5,
                 max_virtual_nodes: int = 6,
                 bandwidth_levels: int = 5,
                 physical_cpu_range: Tuple[int, int] = (40, 80),
                 physical_memory_range: Tuple[int, int] = (40, 80),
                 physical_bandwidth_range: Tuple[int, int] = (40, 80),
                 virtual_cpu_range: Tuple[int, int] = (12, 25),
                 virtual_memory_range: Tuple[int, int] = (12, 25),
                 virtual_bandwidth_range: Tuple[int, int] = (8, 20),
                 physical_connectivity_prob: float = 0.8,
                 virtual_connectivity_prob: float = 0.7,
                 virtual_nodes_range: Tuple[int, int] = (4, 6),
                 seed: int = None,
                 curriculum_enabled: bool = True,
                 # 外部VirtualWork支持（用于调度模式）
                 use_external_virtual_work: bool = False,
                 external_virtual_work: Optional['VirtualWork'] = None,
                 # 轻量级启发式参数
                 heuristic_reward_weight: float = 0.3,
                 enable_load_balance_reward: bool = True,
                 enable_resource_efficiency_reward: bool = True,
                 enable_progress_reward: bool = True):
        
        # 调用父类构造函数
        super().__init__(
            num_physical_nodes=num_physical_nodes,
            max_virtual_nodes=max_virtual_nodes,
            bandwidth_levels=bandwidth_levels,
            physical_cpu_range=physical_cpu_range,
            physical_memory_range=physical_memory_range,
            physical_bandwidth_range=physical_bandwidth_range,
            virtual_cpu_range=virtual_cpu_range,
            virtual_memory_range=virtual_memory_range,
            virtual_bandwidth_range=virtual_bandwidth_range,
            physical_connectivity_prob=physical_connectivity_prob,
            virtual_connectivity_prob=virtual_connectivity_prob,
            virtual_nodes_range=virtual_nodes_range,
            seed=seed,
            curriculum_enabled=curriculum_enabled,
            use_external_virtual_work=use_external_virtual_work,
            external_virtual_work=external_virtual_work
        )
        
        # 轻量级启发式参数
        self.heuristic_reward_weight = heuristic_reward_weight
        self.enable_load_balance_reward = enable_load_balance_reward
        self.enable_resource_efficiency_reward = enable_resource_efficiency_reward
        self.enable_progress_reward = enable_progress_reward
        
        # 启发式状态跟踪（轻量级）
        self.load_balance_history = []
        self.resource_efficiency_history = []
        
        print(f"🎯 轻量级启发式环境初始化完成")
        print(f"   启发式奖励权重: {heuristic_reward_weight}")
        print(f"   负载均衡奖励: {enable_load_balance_reward}")
        print(f"   资源效率奖励: {enable_resource_efficiency_reward}")
        print(f"   进度奖励: {enable_progress_reward}")
    
    def _get_state(self):
        """
        增强的状态获取 - 添加少量启发式特征
        在原有状态基础上添加关键的启发式指标
        """
        # 获取原始状态
        original_state = super()._get_state()
        
        return original_state
    
    def _calculate_current_load_balance(self) -> float:
        """计算当前负载均衡程度（0-1，越高越均衡）"""
        if not self.partial_mapping:
            return 1.0
        
        # 统计每个物理节点的负载
        node_loads = {}
        for physical_node in self.partial_mapping:
            if physical_node != -1:
                node_loads[physical_node] = node_loads.get(physical_node, 0) + 1
        
        if not node_loads:
            return 1.0
        
        # 计算负载方差（越小越均衡）
        loads = list(node_loads.values())
        mean_load = np.mean(loads)
        load_variance = np.var(loads) if len(loads) > 1 else 0
        
        # 转换为0-1分数，方差越小分数越高
        max_possible_variance = mean_load ** 2  # 最坏情况：所有负载集中在一个节点
        if max_possible_variance > 0:
            balance_score = 1.0 - (load_variance / max_possible_variance)
        else:
            balance_score = 1.0
        
        return max(0.0, min(1.0, balance_score))
    
    def _calculate_current_resource_efficiency(self) -> float:
        """计算当前资源效率（0-1，越高越好）"""
        if not hasattr(self, 'physical_state') or self.physical_state is None:
            return 0.5
        
        physical_features = self.physical_state['features']
        efficiency_scores = []
        
        for i in range(physical_features.size(0)):
            cpu_utilization = physical_features[i, 2].item()
            memory_utilization = physical_features[i, 3].item()
            
            # 理想利用率在50%-75%之间，最优点62.5%
            ideal_util = 0.625
            cpu_efficiency = 1.0 - abs(cpu_utilization - ideal_util) / 0.625
            memory_efficiency = 1.0 - abs(memory_utilization - ideal_util) / 0.625
            
            node_efficiency = (cpu_efficiency + memory_efficiency) / 2.0
            efficiency_scores.append(max(0.0, node_efficiency))
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.5
    
    def _calculate_resource_pressure(self) -> float:
        """计算资源压力指标（0-1，越高压力越大）"""
        if not hasattr(self, 'physical_state') or self.physical_state is None:
            return 0.5
        
        physical_features = self.physical_state['features']
        pressure_scores = []
        
        for i in range(physical_features.size(0)):
            cpu_utilization = physical_features[i, 2].item()
            memory_utilization = physical_features[i, 3].item()
            
            # 资源压力 = 平均利用率
            node_pressure = (cpu_utilization + memory_utilization) / 2.0
            pressure_scores.append(node_pressure)
        
        return np.mean(pressure_scores) if pressure_scores else 0.5
    
    def _calculate_simple_connectivity(self) -> float:
        """计算简化的连通性指标"""
        if not self.partial_mapping:
            return 1.0
        
        mapped_nodes = [node for node in self.partial_mapping if node != -1]
        if len(mapped_nodes) <= 1:
            return 1.0
        
        # 简化的连通性：计算已映射节点间的"距离"
        total_distance = 0
        count = 0
        
        for i, node1 in enumerate(mapped_nodes):
            for node2 in mapped_nodes[i+1:]:
                # 使用节点ID差值作为简化的距离度量
                distance = abs(node1 - node2)
                total_distance += distance
                count += 1
        
        if count == 0:
            return 1.0
        
        # 平均距离越小，连通性越好
        avg_distance = total_distance / count
        max_distance = self.num_physical_nodes - 1
        
        connectivity = 1.0 - (avg_distance / max_distance) if max_distance > 0 else 1.0
        return max(0.0, min(1.0, connectivity))
    
    def _calculate_mapping_reward(self, physical_node_action: int) -> float:
        """
        增强的映射奖励计算
        在原始奖励基础上添加启发式奖励组件
        """
        # 计算基础奖励
        base_reward = super()._calculate_mapping_reward(physical_node_action)
        
        # 计算启发式奖励组件
        heuristic_reward = 0.0
        
        if self.enable_load_balance_reward:
            load_balance_reward = self._calculate_load_balance_reward(physical_node_action)
            heuristic_reward += 0.4 * load_balance_reward
        
        if self.enable_resource_efficiency_reward:
            efficiency_reward = self._calculate_resource_efficiency_reward(physical_node_action)
            heuristic_reward += 0.6 * efficiency_reward
        
        # 融合基础奖励和启发式奖励
        total_reward = base_reward + self.heuristic_reward_weight * heuristic_reward
        
        return total_reward
    
    def _calculate_load_balance_reward(self, physical_node: int) -> float:
        """计算负载均衡奖励"""
        if not self.partial_mapping:
            return 0.0
        
        # 计算选择该节点前后的负载均衡变化
        before_balance = self._calculate_current_load_balance()
        
        # 模拟选择该节点后的负载分布
        node_loads = {}
        for node in self.partial_mapping:
            if node != -1:
                node_loads[node] = node_loads.get(node, 0) + 1
        
        # 添加当前选择
        node_loads[physical_node] = node_loads.get(physical_node, 0) + 1
        
        # 计算新的负载均衡
        loads = list(node_loads.values())
        mean_load = np.mean(loads)
        load_variance = np.var(loads) if len(loads) > 1 else 0
        max_possible_variance = mean_load ** 2
        
        if max_possible_variance > 0:
            after_balance = 1.0 - (load_variance / max_possible_variance)
        else:
            after_balance = 1.0
        
        after_balance = max(0.0, min(1.0, after_balance))
        
        # 奖励 = 负载均衡的改善程度
        balance_improvement = after_balance - before_balance
        
        # 转换为奖励信号 (-1 到 +1)
        return 2.0 * balance_improvement
    
    def _calculate_resource_efficiency_reward(self, physical_node: int) -> float:
        """计算资源效率奖励"""
        if not hasattr(self, 'physical_state') or self.physical_state is None:
            return 0.0
        
        physical_features = self.physical_state['features']
        virtual_features = self.virtual_work['features']
        
        if physical_node >= physical_features.size(0):
            return -1.0
        
        current_virtual_node = self.current_virtual_node
        if current_virtual_node >= virtual_features.size(0):
            return -1.0
        
        # 获取资源需求和当前利用率
        cpu_demand = virtual_features[current_virtual_node, 0].item()
        memory_demand = virtual_features[current_virtual_node, 1].item()
        
        cpu_total = physical_features[physical_node, 0].item()
        memory_total = physical_features[physical_node, 1].item()
        cpu_utilization = physical_features[physical_node, 2].item()
        memory_utilization = physical_features[physical_node, 3].item()
        
        # 检查资源可行性
        cpu_available = cpu_total * (1 - cpu_utilization)
        memory_available = memory_total * (1 - memory_utilization)
        
        if cpu_available < cpu_demand or memory_available < memory_demand:
            return -1.0  # 资源不足，严重惩罚
        
        # 计算映射后的利用率
        new_cpu_util = (cpu_total * cpu_utilization + cpu_demand) / cpu_total
        new_memory_util = (memory_total * memory_utilization + memory_demand) / memory_total
        
        # 理想利用率在50%-75%之间
        ideal_util = 0.625
        cpu_efficiency = 1.0 - abs(new_cpu_util - ideal_util) / 0.625
        memory_efficiency = 1.0 - abs(new_memory_util - ideal_util) / 0.625
        
        # 综合效率分数
        efficiency_score = (cpu_efficiency + memory_efficiency) / 2.0
        efficiency_score = max(0.0, min(1.0, efficiency_score))
        
        # 转换为奖励信号 (-1 到 +1)
        return 2.0 * efficiency_score - 1.0
    
    def _calculate_bandwidth_reward(self, bandwidth_level_action: int) -> float:
        """
        增强的带宽奖励计算
        """
        # 计算基础奖励
        base_reward = super()._calculate_bandwidth_reward(bandwidth_level_action)
        
        # 计算启发式带宽奖励
        heuristic_reward = self._calculate_bandwidth_efficiency_reward(bandwidth_level_action)
        
        # 融合奖励
        total_reward = base_reward + self.heuristic_reward_weight * heuristic_reward
        
        return total_reward
    
    def _calculate_bandwidth_efficiency_reward(self, bandwidth_level: int) -> float:
        """计算带宽效率奖励"""
        current_link_index = self.current_link_index
        virtual_edge_features = self.virtual_work['edge_features']
        virtual_edges = self.virtual_work['edges']
        
        if current_link_index >= virtual_edge_features.size(0):
            return -1.0
        
        # 获取链路信息
        src = virtual_edges[0, current_link_index].item()
        dst = virtual_edges[1, current_link_index].item()
        min_bandwidth = virtual_edge_features[current_link_index, 0].item()
        max_bandwidth = virtual_edge_features[current_link_index, 1].item()
        
        # 计算链路重要性（简化版本）
        link_importance = self._calculate_simple_link_importance(src, dst)
        
        # 根据重要性确定理想带宽等级
        if link_importance > 0.7:
            ideal_ratio = 0.8  # 重要链路需要高带宽
        elif link_importance > 0.4:
            ideal_ratio = 0.5  # 中等重要链路
        else:
            ideal_ratio = 0.2  # 低重要性链路
        
        # 计算当前带宽等级的合理性
        max_level = self.bandwidth_levels - 1
        if max_level == 0:
            return 0.0
        
        current_ratio = bandwidth_level / max_level
        
        # 计算与理想比例的匹配度
        match_score = 1.0 - abs(current_ratio - ideal_ratio) / max(ideal_ratio, 1 - ideal_ratio)
        match_score = max(0.0, min(1.0, match_score))
        
        # 转换为奖励信号
        return 2.0 * match_score - 1.0
    
    def _calculate_simple_link_importance(self, src: int, dst: int) -> float:
        """计算简化的链路重要性"""
        virtual_edges = self.virtual_work['edges']
        num_virtual_nodes = self.virtual_work['num_nodes']
        
        # 计算节点度数
        src_degree = 0
        dst_degree = 0
        
        for i in range(virtual_edges.size(1)):
            edge_src = virtual_edges[0, i].item()
            edge_dst = virtual_edges[1, i].item()
            
            if edge_src == src or edge_dst == src:
                src_degree += 1
            if edge_src == dst or edge_dst == dst:
                dst_degree += 1
        
        # 重要性 = 平均度数 / 最大可能度数
        avg_degree = (src_degree + dst_degree) / 2.0
        max_degree = num_virtual_nodes - 1 if num_virtual_nodes > 1 else 1
        
        importance = avg_degree / max_degree
        return min(1.0, importance)
    
    def step(self, action):
        """增强的step方法，添加进度奖励"""
        # 执行父类的step
        state, reward, done, info = super().step(action)
        
        # 添加进度奖励
        if self.enable_progress_reward:
            progress_reward = self._calculate_progress_reward()
            reward += progress_reward * 0.1  # 小权重的进度奖励
            
            if 'progress_reward' not in info:
                info['progress_reward'] = progress_reward
        
        # 记录启发式指标
        self._record_heuristic_metrics(state)
        
        return state, reward, done, info
    
    def _calculate_progress_reward(self) -> float:
        """计算进度奖励"""
        progress_reward = 0.0
        
        if self.mapping_phase:
            # 映射阶段：映射的节点越多，进度奖励越高
            if self.partial_mapping:
                mapped_count = sum(1 for x in self.partial_mapping if x != -1)
                total_nodes = len(self.partial_mapping)
                progress = mapped_count / total_nodes if total_nodes > 0 else 0
                progress_reward = progress * 0.5
        else:
            # 带宽阶段：分配的链路越多，进度奖励越高
            if hasattr(self, 'partial_bandwidth'):
                allocated_count = sum(1 for x in self.partial_bandwidth if x > 0)
                total_links = len(self.partial_bandwidth)
                progress = allocated_count / total_links if total_links > 0 else 0
                progress_reward = progress * 0.5
        
        return progress_reward
    
    def _record_heuristic_metrics(self, state: Dict):
        """记录启发式指标用于分析"""
        if 'load_balance_score' in state:
            self.load_balance_history.append(state['load_balance_score'])
        
        if 'resource_efficiency' in state:
            self.resource_efficiency_history.append(state['resource_efficiency'])
        
        # 限制历史长度
        max_history = 100
        if len(self.load_balance_history) > max_history:
            self.load_balance_history = self.load_balance_history[-max_history:]
        if len(self.resource_efficiency_history) > max_history:
            self.resource_efficiency_history = self.resource_efficiency_history[-max_history:]
    
    def get_heuristic_metrics_summary(self) -> Dict:
        """获取启发式指标摘要"""
        summary = {}
        
        if self.load_balance_history:
            summary['load_balance_mean'] = np.mean(self.load_balance_history)
            summary['load_balance_trend'] = np.mean(self.load_balance_history[-10:]) - np.mean(self.load_balance_history[:10]) if len(self.load_balance_history) >= 20 else 0
        
        if self.resource_efficiency_history:
            summary['resource_efficiency_mean'] = np.mean(self.resource_efficiency_history)
            summary['resource_efficiency_trend'] = np.mean(self.resource_efficiency_history[-10:]) - np.mean(self.resource_efficiency_history[:10]) if len(self.resource_efficiency_history) >= 20 else 0
        
        return summary
    
    def reset(self, *args, **kwargs):
        """重置环境时清理历史记录
        
        兼容父类的reset签名，以支持external_virtual_work等关键字参数
        """
        state = super().reset(*args, **kwargs)
        
        # 清理部分历史记录，保留一些用于趋势分析
        if len(self.load_balance_history) > 50:
            self.load_balance_history = self.load_balance_history[-25:]
        if len(self.resource_efficiency_history) > 50:
            self.resource_efficiency_history = self.resource_efficiency_history[-25:]
        
        return state


# 便捷的创建函数
def create_lightweight_heuristic_environment(**kwargs):
    """创建轻量级启发式环境"""
    return NewHeuristicEnvironment(**kwargs)


# 测试函数
def test_lightweight_integration():
    """测试轻量级集成"""
    print("🧪 测试轻量级启发式集成")
    
    # 创建环境
    env = NewHeuristicEnvironment(
        num_physical_nodes=5,
        max_virtual_nodes=4,
        bandwidth_levels=5,
        virtual_nodes_range=(3, 4),
        heuristic_reward_weight=0.3,
        seed=42
    )
    
    # 运行测试episode
    state = env.reset()
    print(f"✅ 环境创建成功")
    print(f"   初始状态特征数: {len(state)}")
    
    # 检查新增的启发式特征
    heuristic_features = ['load_balance_score', 'resource_efficiency', 'mapping_progress', 
                         'resource_pressure', 'connectivity_score']
    
    print(f"\n🔍 启发式特征检查:")
    for feature in heuristic_features:
        if feature in state:
            print(f"   ✅ {feature}: {state[feature]:.3f}")
        else:
            print(f"   ❌ {feature}: 缺失")
    
    # 运行几个步骤
    total_reward = 0
    step_count = 0
    
    while not env.mapping_phase == False and step_count < 8:  # 只测试映射阶段
        # 随机动作
        action = np.random.randint(0, env.num_physical_nodes)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        print(f"   步骤 {step_count}: 动作={action}, 奖励={reward:.3f}")
        
        state = next_state
        
        if done:
            break
    
    print(f"\n📊 测试结果:")
    print(f"   总步数: {step_count}")
    print(f"   总奖励: {total_reward:.3f}")
    print(f"   平均奖励: {total_reward/step_count:.3f}")
    
    # 获取启发式指标摘要
    metrics = env.get_heuristic_metrics_summary()
    print(f"\n📈 启发式指标摘要:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.3f}")
    
    print("✅ 轻量级集成测试完成")


if __name__ == "__main__":
    test_lightweight_integration()
