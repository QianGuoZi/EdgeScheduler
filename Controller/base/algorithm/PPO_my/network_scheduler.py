import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import heapq
from collections import defaultdict
import torch

class NetworkTopology:
    """网络拓扑管理类"""
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.graph = nx.Graph()
        
        # 添加节点
        for i in range(num_nodes):
            self.graph.add_node(i)
        
        # 物理链路信息
        self.links = {}  # (node1, node2) -> bandwidth_info
        self.node_resources = {}  # node_id -> {cpu, memory}
        
    def add_link(self, node1: int, node2: int, bandwidth_1_to_2: int, bandwidth_2_to_1: int, 
                 used_bandwidth_1_to_2: int = 0, used_bandwidth_2_to_1: int = 0):
        """添加链路，支持不对称带宽和初始使用量"""
        self.graph.add_edge(node1, node2)
        # 从node1到node2的链路
        self.links[(node1, node2)] = {
            'bandwidth': bandwidth_1_to_2,
            'used_bandwidth': used_bandwidth_1_to_2  # 支持设置初始已使用量
        }
        # 从node2到node1的链路
        self.links[(node2, node1)] = {
            'bandwidth': bandwidth_2_to_1,
            'used_bandwidth': used_bandwidth_2_to_1  # 支持设置初始已使用量
        }
    
    def set_node_resources(self, node_id: int, cpu: int, memory: int, used_cpu: int = 0, used_memory: int = 0):
        """设置节点资源"""
        self.node_resources[node_id] = {
            'cpu': cpu,
            'memory': memory,
            'used_cpu': used_cpu,      # 支持设置初始已使用量
            'used_memory': used_memory # 支持设置初始已使用量
        }
    
    def get_shortest_path(self, source: int, target: int) -> List[int]:
        """使用Dijkstra算法获取最短路径"""
        try:
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return []
    
    def check_bandwidth_availability(self, path: List[int], 
                                   required_bandwidth: int) -> bool:
        """检查路径上的带宽是否足够"""
        if len(path) < 2:
            return True
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            link_key = (node1, node2)
            
            if link_key not in self.links:
                return False
            
            link = self.links[link_key]
            available_bandwidth = link['bandwidth'] - link['used_bandwidth']
            
            if available_bandwidth < required_bandwidth:
                return False
        
        return True
    
    def allocate_bandwidth(self, path: List[int], bandwidth: float) -> bool:
        """分配带宽，返回是否分配成功"""
        if len(path) < 2:
            return False
        
        # 检查路径上所有链路的可用带宽是否足够
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            link_key = (node1, node2)
            if link_key not in self.links:
                return False
            
            available_bandwidth = self.get_available_bandwidth(node1, node2)
            if available_bandwidth < bandwidth:
                return False
        
        # 分配带宽
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            link_key = (node1, node2)
            self.links[link_key]['used_bandwidth'] += bandwidth
        
        return True
    
    def release_bandwidth(self, path: List[int], bandwidth: float):
        """释放带宽"""
        if len(path) < 2:
            return
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            link_key = (node1, node2)
            self.links[link_key]['used_bandwidth'] = max(0, self.links[link_key]['used_bandwidth'] - bandwidth)
    
    def allocate_node_resources(self, node_id: int, cpu: float, memory: float) -> bool:
        """分配节点资源，返回是否分配成功"""
        if node_id not in self.node_resources:
            return False
        
        # 检查可用资源是否足够
        available_resources = self.get_available_resources(node_id)
        if available_resources['cpu'] < cpu or available_resources['memory'] < memory:
            return False
        
        # 分配资源
        self.node_resources[node_id]['used_cpu'] += cpu
        self.node_resources[node_id]['used_memory'] += memory
        return True
    
    def release_node_resources(self, node_id: int, cpu: float, memory: float):
        """释放节点资源"""
        if node_id in self.node_resources:
            self.node_resources[node_id]['used_cpu'] = max(0, self.node_resources[node_id]['used_cpu'] - cpu)
            self.node_resources[node_id]['used_memory'] = max(0, self.node_resources[node_id]['used_memory'] - memory)
    
    def get_available_resources(self, node_id: int) -> Dict[str, float]:
        """获取节点可用资源"""
        if node_id not in self.node_resources:
            return {'cpu': 0, 'memory': 0}
        
        resources = self.node_resources[node_id]
        return {
            'cpu': resources['cpu'] - resources['used_cpu'],
            'memory': resources['memory'] - resources['used_memory']
        }
    
    def get_network_utilization(self) -> Dict[str, float]:
        """获取网络利用率"""
        total_bandwidth = 0
        used_bandwidth = 0
        
        for link_info in self.links.values():
            total_bandwidth += link_info['bandwidth']
            used_bandwidth += link_info['used_bandwidth']
        
        return {
            'bandwidth_utilization': used_bandwidth / total_bandwidth if total_bandwidth > 0 else 0
        }

    def get_available_bandwidth(self, node1: int, node2: int) -> float:
        """获取链路的可用带宽"""
        link_key = (node1, node2)
        if link_key not in self.links:
            return 0.0
        
        link = self.links[link_key]
        return link['bandwidth'] - link['used_bandwidth']
    
    def get_link_utilization(self, node1: int, node2: int) -> Dict[str, float]:
        """获取链路的利用率信息"""
        link_key = (node1, node2)
        if link_key not in self.links:
            return {'total_bandwidth': 0, 'used_bandwidth': 0, 'utilization': 0}
        
        link = self.links[link_key]
        utilization = link['used_bandwidth'] / link['bandwidth'] if link['bandwidth'] > 0 else 0
        
        return {
            'total_bandwidth': link['bandwidth'],
            'used_bandwidth': link['used_bandwidth'],
            'utilization': utilization
        }

class VirtualWork:
    """虚拟工作类"""
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.node_requirements = {}  # node_id -> {cpu, memory}
        self.link_requirements = []  # [{from, to, min_bandwidth, max_bandwidth}]
        
    def set_node_requirement(self, node_id: int, cpu: int, memory: int):
        """设置节点需求"""
        self.node_requirements[node_id] = {
            'cpu': cpu,
            'memory': memory
        }
    
    def add_link_requirement(self, from_node: int, to_node: int, 
                           min_bandwidth_1_to_2: int, max_bandwidth_1_to_2: int,
                           min_bandwidth_2_to_1: int, max_bandwidth_2_to_1: int):
        """添加链路需求，支持不对称带宽"""
        self.link_requirements.append({
            'from': from_node,
            'to': to_node,
            'min_bandwidth_1_to_2': min_bandwidth_1_to_2,
            'max_bandwidth_1_to_2': max_bandwidth_1_to_2,
            'min_bandwidth_2_to_1': min_bandwidth_2_to_1,
            'max_bandwidth_2_to_1': max_bandwidth_2_to_1
        })

class NetworkScheduler:
    """网络调度器"""
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.node_mapping = {}  # virtual_node -> physical_node
        self.bandwidth_allocation = {}  # (virtual_from, virtual_to) -> bandwidth
        self.virtual_works = {}  # virtual_work_id -> VirtualWork
        self.scheduled_nodes = set()
        # 新增：保存多个VirtualWork实例的列表
        self.virtual_work_list = []  # List[VirtualWork]
        
    def add_virtual_work(self, virtual_work: VirtualWork, work_id: str = None):
        """添加虚拟工作到调度器"""
        if work_id is None:
            work_id = f"work_{len(self.virtual_work_list)}"
        
        self.virtual_works[work_id] = virtual_work
        self.virtual_work_list.append(virtual_work)
        
    def clear_virtual_works(self):
        """清空所有虚拟工作"""
        self.virtual_works.clear()
        self.virtual_work_list.clear()
        
    def schedule_node(self, virtual_node: int, physical_node: int) -> bool:
        """调度虚拟节点到物理节点"""
        if virtual_node in self.scheduled_nodes:
            return False
        
        # 检查物理节点资源是否足够
        if not self._check_node_resources(virtual_node, physical_node):
            return False
        
        # 执行映射
        self.node_mapping[virtual_node] = physical_node
        self.scheduled_nodes.add(virtual_node)
        
        # 分配资源
        self._allocate_node_resources(virtual_node, physical_node)
        
        return True
    
    def allocate_bandwidth(self, virtual_from: int, virtual_to: int, 
                          bandwidth: int) -> bool:
        """分配虚拟链路带宽"""
        if virtual_from not in self.scheduled_nodes or virtual_to not in self.scheduled_nodes:
            return False
        
        physical_from = self.node_mapping[virtual_from]
        physical_to = self.node_mapping[virtual_to]
        
        # 如果映射到同一物理节点，带宽消耗为0
        if physical_from == physical_to:
            self.bandwidth_allocation[(virtual_from, virtual_to)] = bandwidth
            return True
        
        # 获取最短路径
        path = self.topology.get_shortest_path(physical_from, physical_to)
        if not path:
            # print(f"❌ 路径不存在: 物理节点 {physical_from} -> {physical_to}")
            return False
        
        # 检查带宽是否足够（考虑路径方向）
        # 在检查前，打印路径上每条链路的可用带宽
        # print(f"🔍 检查路径 {path} 的带宽可用性 (需求: {bandwidth}mbps):")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            available = self.topology.get_available_bandwidth(u, v)
            link_info = self.topology.links.get((u, v), {})
            total_bw = link_info.get('bandwidth', 0)
            used_bw = link_info.get('used_bandwidth', 0)
            # print(f"  链路({u},{v}): 总={total_bw}mbps, 已用={used_bw}mbps, 可用={available}mbps")
        
        if not self.topology.check_bandwidth_availability(path, bandwidth):
            # print(f"❌ 路径 {path} 带宽不足 (需求: {bandwidth}mbps)")
            return False
        
        # 分配带宽
        success = self.topology.allocate_bandwidth(path, bandwidth)
        if not success:
            # print(f"❌ 警告：路径 {path} 上的带宽分配失败")
            return False
        
        # 分配成功后，打印更新后的带宽状态
        # print(f"✅ 带宽分配成功，路径 {path} 已分配 {bandwidth}mbps:")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            available = self.topology.get_available_bandwidth(u, v)
            link_info = self.topology.links.get((u, v), {})
            total_bw = link_info.get('bandwidth', 0)
            used_bw = link_info.get('used_bandwidth', 0)
            # print(f"  链路({u},{v}): 总={total_bw}mbps, 已用={used_bw}mbps, 可用={available}mbps")
        
        self.bandwidth_allocation[(virtual_from, virtual_to)] = bandwidth
        
        return True
    
    def _check_bandwidth_availability(self, virtual_from: int, virtual_to: int, 
                                   required_bandwidth: int, mapping_action: List[int] = None) -> bool:
        """
        检查两个虚拟节点之间的链路带宽资源是否足够
        
        Args:
            virtual_from: 源虚拟节点ID
            virtual_to: 目标虚拟节点ID
            required_bandwidth: 需要的带宽
            mapping_action: 映射动作列表 [num_virtual_nodes]，如果提供则使用此映射，否则使用self.node_mapping
            
        Returns:
            bool: True表示带宽足够，False表示带宽不足
        """
        # 确定使用哪个映射
        if mapping_action is not None:
            # 使用提供的mapping_action
            if virtual_from >= len(mapping_action) or virtual_to >= len(mapping_action) or virtual_from < 0 or virtual_to < 0:
                # print(f"警告：虚拟节点 {virtual_from} 或 {virtual_to} 超出映射动作范围")
                return False
            
            # 检查虚拟节点是否已映射（映射值不为None）
            if mapping_action[virtual_from] is None or mapping_action[virtual_to] is None:
                # print(f"警告：虚拟节点 {virtual_from} 或 {virtual_to} 尚未映射")
                return False
            
            physical_from = mapping_action[virtual_from]
            physical_to = mapping_action[virtual_to]
        else:
            # 使用原有的scheduled_nodes和node_mapping
            if virtual_from not in self.scheduled_nodes or virtual_to not in self.scheduled_nodes:
                # print(f"警告：虚拟节点 {virtual_from} 或 {virtual_to} 尚未调度")
                return False
            
            physical_from = self.node_mapping[virtual_from]
            physical_to = self.node_mapping[virtual_to]
        
        # 如果映射到同一物理节点，带宽消耗为0，总是返回True
        if physical_from == physical_to:
            # print(f"虚拟节点 {virtual_from} 和 {virtual_to} 映射到同一物理节点 {physical_from}，带宽需求为0")
            return True
        
        # 获取最短路径
        path = self.topology.get_shortest_path(physical_from, physical_to)
        if not path:
            # print(f"警告：物理节点 {physical_from} 和 {physical_to} 之间无路径")
            return False
        
        # 检查路径上的带宽是否足够
        if not self.topology.check_bandwidth_availability(path, required_bandwidth):
            # print(f"警告：路径 {path} 上的可用带宽不足以支持需求带宽 {required_bandwidth}")
            return False
        
        # print(f"✅ 虚拟链路 ({virtual_from}, {virtual_to}) 的带宽需求 {required_bandwidth} 可以满足")
        return True
    
    def _check_node_resources(self, virtual_node: int, physical_node: int) -> bool:
        """检查节点资源是否足够"""
        # 从virtual_work_list中查找虚拟节点的资源需求
        for virtual_work in self.virtual_work_list:
            if virtual_node in virtual_work.node_requirements:
                available = self.topology.get_available_resources(physical_node)
                required = virtual_work.node_requirements[virtual_node]
                # print(f"检查节点资源是否足够 (从virtual_work):")
                # print(f"virtual_node: {virtual_node}, required: {required} ")
                # print(f"physical_node: {physical_node}, available: {available}")
                
                return (available['cpu'] >= required['cpu'] and
                        available['memory'] >= required['memory'])
        
        # 如果都找不到，返回False
        # print(f"警告：找不到虚拟节点 {virtual_node} 的资源需求信息")
        return False
    
    def _allocate_node_resources(self, virtual_node: int, physical_node: int) -> bool:
        """分配节点资源，返回是否分配成功"""
        # # 首先检查topology中是否有该虚拟节点的资源信息（向后兼容）
        # if virtual_node in self.topology.node_resources:
        #     required = self.topology.node_resources[virtual_node]
        #     self.topology.allocate_node_resources(physical_node, 
        #                                         required['cpu'], 
        #                                         required['memory'])
        #     return
        
        # 从virtual_work_list中查找虚拟节点的资源需求
        for virtual_work in self.virtual_work_list:
            if virtual_node in virtual_work.node_requirements:
                required = virtual_work.node_requirements[virtual_node]
                success = self.topology.allocate_node_resources(physical_node, 
                                                             required['cpu'], 
                                                             required['memory'])
                if not success:
                    # print(f"警告：物理节点 {physical_node} 资源不足，无法分配虚拟节点 {virtual_node} 的资源需求")
                    return False
                return True
        
        # print(f"警告：找不到虚拟节点 {virtual_node} 的资源需求信息，无法分配资源")
        return False
    
    def get_scheduling_result(self) -> Dict:
        """获取调度结果"""
        return {
            'node_mapping': self.node_mapping.copy(),
            'bandwidth_allocation': self.bandwidth_allocation.copy(),
            'scheduled_nodes': list(self.scheduled_nodes),
            'virtual_works_count': len(self.virtual_work_list)
        }
    
    def calculate_reward_components(self, virtual_work: VirtualWork) -> Dict[str, float]:
        """
        计算奖励的各个组件（用于调试和分析）
        
        Args:
            virtual_work: 虚拟工作对象
            
        Returns:
            Dict[str, float]: 包含各个奖励组件的字典
        """
        if len(self.scheduled_nodes) == 0:
            return {
                'load_balance': 0,
                'bandwidth_satisfaction': 0,
                'resource_utilization': 0,
                'network_efficiency': 0,
                'concentration_penalty': 0,
                'overload_penalty': 0,
                'total_reward': 0
            }
        
        # 1. 资源负载均衡奖励（CPU、内存、网络带宽）
        cpu_utilizations = []
        memory_utilizations = []
        bandwidth_utilizations = []
        
        for node_id in range(self.topology.num_nodes):
            available = self.topology.get_available_resources(node_id)
            total_cpu = self.topology.node_resources[node_id]['cpu']
            total_memory = self.topology.node_resources[node_id]['memory']
            
            cpu_utilizations.append(1 - available['cpu'] / total_cpu)
            memory_utilizations.append(1 - available['memory'] / total_memory)
        
        # 计算网络带宽负载均衡
        for node1 in range(self.topology.num_nodes):
            for node2 in range(node1 + 1, self.topology.num_nodes):
                link_key = (node1, node2)
                if link_key in self.topology.links:
                    link = self.topology.links[link_key]
                    if link['bandwidth'] > 0:
                        utilization = link['used_bandwidth'] / link['bandwidth']
                        bandwidth_utilizations.append(utilization)
        
        # 计算负载均衡奖励（标准差越小越好）
        cpu_balance = 1 - np.std(cpu_utilizations) if cpu_utilizations else 1.0
        memory_balance = 1 - np.std(memory_utilizations) if memory_utilizations else 1.0
        bandwidth_balance = 1 - np.std(bandwidth_utilizations) if bandwidth_utilizations else 1.0
        
        # 确保负载均衡奖励在0-1范围内
        cpu_balance = np.clip(cpu_balance, 0, 1)
        memory_balance = np.clip(memory_balance, 0, 1)
        bandwidth_balance = np.clip(bandwidth_balance, 0, 1)
        
        # 综合负载均衡奖励
        load_balance_reward = (0.4 * cpu_balance + 0.4 * memory_balance + 0.2 * bandwidth_balance)
        
        # 2. 带宽满足度奖励
        bandwidth_satisfaction = 0
        satisfied_links = 0
        same_node_bandwidth_bonus = 0  # 同一节点映射的带宽奖励
        
        for link_req in virtual_work.link_requirements:
            from_node = link_req['from']
            to_node = link_req['to']
            physical_from = self.node_mapping[from_node]
            physical_to = self.node_mapping[to_node]
            if from_node in self.scheduled_nodes and to_node in self.scheduled_nodes:
                if physical_from == physical_to:
                    # 同一物理节点映射，给予最高带宽奖励
                    same_node_bandwidth_bonus += 1.0
                    satisfied_links += 1
                    continue
                
                # 检查正向链路 (from_node -> to_node)
                allocated_1_to_2 = self.bandwidth_allocation.get((from_node, to_node), 0)
                min_req_1_to_2 = link_req['min_bandwidth_1_to_2']
                max_req_1_to_2 = link_req['max_bandwidth_1_to_2']
                
                # 检查反向链路 (to_node -> from_node)
                allocated_2_to_1 = self.bandwidth_allocation.get((to_node, from_node), 0)
                min_req_2_to_1 = link_req['min_bandwidth_2_to_1']
                max_req_2_to_1 = link_req['max_bandwidth_2_to_1']
                
                # 计算两个方向的满足度
                satisfaction_1_to_2 = 0
                satisfaction_2_to_1 = 0
                
                if min_req_1_to_2 <= allocated_1_to_2 <= max_req_1_to_2:
                    if max_req_1_to_2 > min_req_1_to_2:
                        satisfaction_1_to_2 = (allocated_1_to_2 - min_req_1_to_2) / (max_req_1_to_2 - min_req_1_to_2)
                    else:
                        satisfaction_1_to_2 = 1.0
                elif allocated_1_to_2 > max_req_1_to_2:
                    satisfaction_1_to_2 = 1.0  # 超过最大需求也是好的
                else:
                    satisfaction_1_to_2 = 0.0  # 低于最小需求
                
                if min_req_2_to_1 <= allocated_2_to_1 <= max_req_2_to_1:
                    if max_req_2_to_1 > min_req_2_to_1:
                        satisfaction_2_to_1 = (allocated_2_to_1 - min_req_2_to_1) / (max_req_2_to_1 - min_req_2_to_1)
                    else:
                        satisfaction_2_to_1 = 1.0
                elif allocated_2_to_1 > max_req_2_to_1:
                    satisfaction_2_to_1 = 1.0
                else:
                    satisfaction_2_to_1 = 0.0
                
                # 取两个方向的平均满足度
                avg_satisfaction = (satisfaction_1_to_2 + satisfaction_2_to_1) / 2
                bandwidth_satisfaction += avg_satisfaction
                satisfied_links += 1
        
        # 归一化带宽满足度
        if satisfied_links > 0:
            bandwidth_satisfaction = (bandwidth_satisfaction + same_node_bandwidth_bonus) / satisfied_links
        bandwidth_satisfaction = np.clip(bandwidth_satisfaction, 0, 1)
        
                # 5. 负载过载惩罚
        overload_penalty = 0
        overload_count = 0
        
        # 节点资源过载惩罚（CPU和内存）
        for node_id in range(self.topology.num_nodes):
            available = self.topology.get_available_resources(node_id)
            total_cpu = self.topology.node_resources[node_id]['cpu']
            total_memory = self.topology.node_resources[node_id]['memory']
            
            cpu_usage = 1 - available['cpu'] / total_cpu
            memory_usage = 1 - available['memory'] / total_memory
            
            # 如果CPU或内存使用率超过85%，给予惩罚
            if cpu_usage > 0.85 or memory_usage > 0.85:
                overload_score = max(cpu_usage - 0.85, memory_usage - 0.85) / 0.15
                overload_penalty += overload_score
                overload_count += 1
        
        # 带宽过载惩罚
        bandwidth_overload_count = 0
        for node1 in range(self.topology.num_nodes):
            for node2 in range(node1 + 1, self.topology.num_nodes):
                link_key = (node1, node2)
                if link_key in self.topology.links:
                    link = self.topology.links[link_key]
                    if link['bandwidth'] > 0:
                        bandwidth_usage = link['used_bandwidth'] / link['bandwidth']
                        
                        # 如果带宽使用率超过85%，给予惩罚
                        if bandwidth_usage > 0.85:
                            bandwidth_overload_score = (bandwidth_usage - 0.85) / 0.15
                            overload_penalty += bandwidth_overload_score
                            overload_count += 1
                            bandwidth_overload_count += 1
        
        if overload_count > 0:
            overload_penalty /= overload_count
        overload_penalty = np.clip(overload_penalty, 0, 1)
    
        # 计算总奖励
        total_reward = (0.6 * load_balance_reward + 
                       0.4 * bandwidth_satisfaction)
        
        total_reward = np.clip(total_reward, 0, 1)
        
        return {
            'load_balance': load_balance_reward,
            'bandwidth_satisfaction': bandwidth_satisfaction,
            'total_reward': total_reward,
            'same_node_bandwidth_bonus': same_node_bandwidth_bonus,
            'overload_penalty': overload_penalty
        }
    
    def reset(self):
        """重置调度器（只重置当前调度的资源，保留原有使用量）"""
        # 只释放当前调度器分配的资源，不重置整个拓扑
        for virtual_node, physical_node in self.node_mapping.items():
            # 首先检查topology中是否有该虚拟节点的资源信息（向后兼容）
            if virtual_node in self.topology.node_resources:
                required = self.topology.node_resources[virtual_node]
                self.topology.release_node_resources(physical_node, 
                                                   required['cpu'], 
                                                   required['memory'])
            else:
                # 从virtual_work_list中查找虚拟节点的资源需求
                for virtual_work in self.virtual_work_list:
                    if virtual_node in virtual_work.node_requirements:
                        required = virtual_work.node_requirements[virtual_node]
                        self.topology.release_node_resources(physical_node, 
                                                           required['cpu'], 
                                                           required['memory'])
                        break
        
        # 释放当前调度的带宽
        for (virtual_from, virtual_to), bandwidth in self.bandwidth_allocation.items():
            if virtual_from in self.node_mapping and virtual_to in self.node_mapping:
                physical_from = self.node_mapping[virtual_from]
                physical_to = self.node_mapping[virtual_to]
                
                if physical_from != physical_to:
                    path = self.topology.get_shortest_path(physical_from, physical_to)
                    self.topology.release_bandwidth(path, bandwidth)
        
        # 清空当前调度的状态
        self.node_mapping = {}
        self.bandwidth_allocation = {}
        self.scheduled_nodes = set()
        # 注意：不清空virtual_work_list，因为虚拟工作信息应该保留
    
    def calculate_simple_reward(self, virtual_work: VirtualWork) -> float:
        """
        计算增强的奖励函数，包含负载均衡和效率优化
        解决100%成功率和奖励趋势不明显的问题
        
        Returns:
            float: 增强的奖励值 [-1, 1]
        """
        if len(self.scheduled_nodes) == 0:
            return -0.5  # 空映射给予负奖励
        
        # 1. 映射成功率奖励（基础奖励）
        total_virtual_nodes = len([node for node in self.virtual_work_list[0].node_requirements.keys()])
        mapping_success_rate = len(self.scheduled_nodes) / total_virtual_nodes if total_virtual_nodes > 0 else 0.0
        
        # 如果映射不完整，给予惩罚
        if mapping_success_rate < 1.0:
            return -0.5 * (1.0 - mapping_success_rate)
        
        mapped_physical_nodes = set(self.node_mapping.values())
        if mapped_physical_nodes:
            # 1. 资源负载均衡奖励（CPU、内存、网络带宽）
            cpu_utilizations = []
            memory_utilizations = []
            bandwidth_utilizations = []
            
            for node_id in range(self.topology.num_nodes):
                available = self.topology.get_available_resources(node_id)
                total_cpu = self.topology.node_resources[node_id]['cpu']
                total_memory = self.topology.node_resources[node_id]['memory']
                
                cpu_utilizations.append(1 - available['cpu'] / total_cpu)
                memory_utilizations.append(1 - available['memory'] / total_memory)
            
            # 计算网络带宽负载均衡
            for node1 in range(self.topology.num_nodes):
                for node2 in range(node1 + 1, self.topology.num_nodes):
                    link_key = (node1, node2)
                    if link_key in self.topology.links:
                        link = self.topology.links[link_key]
                        if link['bandwidth'] > 0:
                            utilization = link['used_bandwidth'] / link['bandwidth']
                            bandwidth_utilizations.append(utilization)
            
            # 计算负载均衡奖励（标准差越小越好）
            cpu_balance = 1 - np.std(cpu_utilizations) if cpu_utilizations else 1.0
            memory_balance = 1 - np.std(memory_utilizations) if memory_utilizations else 1.0
            bandwidth_balance = 1 - np.std(bandwidth_utilizations) if bandwidth_utilizations else 1.0
            
            # 确保负载均衡奖励在0-1范围内
            cpu_balance = np.clip(cpu_balance, 0, 1)
            memory_balance = np.clip(memory_balance, 0, 1)
            bandwidth_balance = np.clip(bandwidth_balance, 0, 1)
            
            # 综合负载均衡奖励
            load_balance_reward = (0.4 * cpu_balance + 0.4 * memory_balance + 0.2 * bandwidth_balance)

            # 3. 资源利用效率奖励（奖励高效利用）
            mean_cpu_util = np.mean(cpu_utilizations)
            mean_memory_util = np.mean(memory_utilizations)
            
            # 目标利用率在60%-80%之间
            cpu_efficiency = 1.0 - abs(mean_cpu_util - 0.7) / 0.3 if mean_cpu_util <= 1.0 else -0.5
            memory_efficiency = 1.0 - abs(mean_memory_util - 0.7) / 0.3 if mean_memory_util <= 1.0 else -0.5
            
            resource_efficiency = (cpu_efficiency + memory_efficiency) / 2.0

        # 4. 带宽满足度和路径长度奖励
        bandwidth_satisfaction = 0.0
        path_length_penalty = 0.0
        satisfied_links = 0
        total_path_length = 0
        
        for link_req in virtual_work.link_requirements:
            from_node = link_req['from']
            to_node = link_req['to']
            
            if from_node in self.scheduled_nodes and to_node in self.scheduled_nodes:
                physical_from = self.node_mapping[from_node]
                physical_to = self.node_mapping[to_node]
                
                # 同一物理节点映射给予高奖励
                if physical_from == physical_to:
                    bandwidth_satisfaction += 1.0
                    path_length_penalty += 0.0  # 路径长度为0，最优
                    satisfied_links += 1
                    continue
                
                # 计算路径长度（使用最短路径）
                try:
                    path = self.topology.shortest_path(physical_from, physical_to)
                    path_length = len(path) - 1 if path else float('inf')
                    # 路径越短越好，长度为1最优，每增加一跳降低奖励
                    path_penalty = min(1.0, path_length / 3.0)  # 3跳以上给最大惩罚
                    total_path_length += path_penalty
                except:
                    path_penalty = 1.0  # 无路径，最大惩罚
                    total_path_length += path_penalty
                
                # 检查带宽分配
                allocated = self.bandwidth_allocation.get((from_node, to_node), 0)
                min_req = link_req['min_bandwidth_1_to_2']
                max_req = link_req['max_bandwidth_1_to_2']
                
                if allocated >= min_req:
                    if max_req > min_req:
                        satisfaction = min(1.0, (allocated - min_req) / (max_req - min_req))
                    else:
                        satisfaction = 1.0
                    bandwidth_satisfaction += satisfaction
                else:
                    bandwidth_satisfaction += -0.2  # 未满足最小需求给予负奖励
                
                satisfied_links += 1
        
        if satisfied_links > 0:
            bandwidth_satisfaction /= satisfied_links
            path_length_penalty = 1.0 - (total_path_length / satisfied_links)  # 转换为奖励
        
        # 5. 综合奖励计算（增强权重分配）
        # 30% 负载均衡 + 30% 资源效率 + 25% 带宽满足度 + 15% 路径优化
        total_reward = (0.3 * load_balance_reward + 
                       0.3 * resource_efficiency + 
                       0.25 * bandwidth_satisfaction +
                       0.15 * path_length_penalty)
        
        return min(1.0, max(0.0, total_reward))  # 确保在[0,1]范围内
    
    def get_simple_reward_components(self, virtual_work: VirtualWork) -> Dict[str, float]:
        """
        获取增强奖励的各个组件（用于调试）
        """
        if len(self.scheduled_nodes) == 0:
            return {
                'mapping_success_rate': 0.0,
                'load_balance_reward': 0.0,
                'resource_efficiency': 0.0,
                'bandwidth_satisfaction': 0.0,
                'path_length_penalty': 0.0,
                'total_reward': -0.5
            }
        
        # 复制增强奖励的计算逻辑
        total_virtual_nodes = len([node for node in self.virtual_work_list[0].node_requirements.keys()])
        mapping_success_rate = len(self.scheduled_nodes) / total_virtual_nodes if total_virtual_nodes > 0 else 0.0
        
        if mapping_success_rate < 1.0:
            return {
                'mapping_success_rate': mapping_success_rate,
                'load_balance_reward': 0.0,
                'resource_efficiency': 0.0,
                'bandwidth_satisfaction': 0.0,
                'path_length_penalty': 0.0,
                'total_reward': -0.5 * (1.0 - mapping_success_rate)
            }
        
        # 负载均衡和资源效率计算
        load_balance_reward = 0.0
        resource_efficiency = 0.0
        mapped_physical_nodes = set(self.node_mapping.values())
        
        # if mapped_physical_nodes:
        #     import numpy as np
        #     cpu_utilizations = []
        #     memory_utilizations = []
            
        #     for physical_node in mapped_physical_nodes:
        #         available = self.topology.get_available_resources(physical_node)
        #         total_resources = self.topology.node_resources[physical_node]
                
        #         cpu_util = 1 - available['cpu'] / total_resources['cpu']
        #         memory_util = 1 - available['memory'] / total_resources['memory']
                
        #         cpu_utilizations.append(cpu_util)
        #         memory_utilizations.append(memory_util)
            
        #     # 负载均衡度
        #     cpu_std = np.std(cpu_utilizations) if len(cpu_utilizations) > 1 else 0.0
        #     memory_std = np.std(memory_utilizations) if len(memory_utilizations) > 1 else 0.0
        #     load_balance_reward = 1.0 - (cpu_std + memory_std)
            
        #     # 资源效率
        #     mean_cpu_util = np.mean(cpu_utilizations)
        #     mean_memory_util = np.mean(memory_utilizations)
        #     cpu_efficiency = 1.0 - abs(mean_cpu_util - 0.7) / 0.3 if mean_cpu_util <= 1.0 else -0.5
        #     memory_efficiency = 1.0 - abs(mean_memory_util - 0.7) / 0.3 if mean_memory_util <= 1.0 else -0.5
        #     resource_efficiency = (cpu_efficiency + memory_efficiency) / 2.0
        if mapped_physical_nodes:
            # 1. 资源负载均衡奖励（CPU、内存、网络带宽）
            cpu_utilizations = []
            memory_utilizations = []
            bandwidth_utilizations = []
            
            for node_id in range(self.topology.num_nodes):
                available = self.topology.get_available_resources(node_id)
                total_cpu = self.topology.node_resources[node_id]['cpu']
                total_memory = self.topology.node_resources[node_id]['memory']
                
                cpu_utilizations.append(1 - available['cpu'] / total_cpu)
                memory_utilizations.append(1 - available['memory'] / total_memory)
            
            # 计算网络带宽负载均衡
            for node1 in range(self.topology.num_nodes):
                for node2 in range(node1 + 1, self.topology.num_nodes):
                    link_key = (node1, node2)
                    if link_key in self.topology.links:
                        link = self.topology.links[link_key]
                        if link['bandwidth'] > 0:
                            utilization = link['used_bandwidth'] / link['bandwidth']
                            bandwidth_utilizations.append(utilization)
            
            # 计算负载均衡奖励（标准差越小越好）
            cpu_balance = 1 - np.std(cpu_utilizations) if cpu_utilizations else 1.0
            memory_balance = 1 - np.std(memory_utilizations) if memory_utilizations else 1.0
            bandwidth_balance = 1 - np.std(bandwidth_utilizations) if bandwidth_utilizations else 1.0
            
            # 确保负载均衡奖励在0-1范围内
            cpu_balance = np.clip(cpu_balance, 0, 1)
            memory_balance = np.clip(memory_balance, 0, 1)
            bandwidth_balance = np.clip(bandwidth_balance, 0, 1)
            
            # 综合负载均衡奖励
            load_balance_reward = (0.4 * cpu_balance + 0.4 * memory_balance + 0.2 * bandwidth_balance)

            # 3. 资源利用效率奖励（奖励高效利用）
            mean_cpu_util = np.mean(cpu_utilizations)
            mean_memory_util = np.mean(memory_utilizations)
            
            # 目标利用率在60%-80%之间
            cpu_efficiency = 1.0 - abs(mean_cpu_util - 0.7) / 0.3 if mean_cpu_util <= 1.0 else -0.5
            memory_efficiency = 1.0 - abs(mean_memory_util - 0.7) / 0.3 if mean_memory_util <= 1.0 else -0.5
            
            resource_efficiency = (cpu_efficiency + memory_efficiency) / 2.0
            
        # 带宽满足度和路径长度计算
        bandwidth_satisfaction = 0.0
        path_length_penalty = 0.0
        satisfied_links = 0
        total_path_length = 0
        
        for link_req in virtual_work.link_requirements:
            from_node = link_req['from']
            to_node = link_req['to']
            
            if from_node in self.scheduled_nodes and to_node in self.scheduled_nodes:
                physical_from = self.node_mapping[from_node]
                physical_to = self.node_mapping[to_node]
                
                if physical_from == physical_to:
                    bandwidth_satisfaction += 1.0
                    satisfied_links += 1
                    continue
                
                # 路径长度
                try:
                    path = self.topology.shortest_path(physical_from, physical_to)
                    path_length = len(path) - 1 if path else float('inf')
                    path_penalty = min(1.0, path_length / 3.0)
                    total_path_length += path_penalty
                except:
                    total_path_length += 1.0
                
                # 带宽分配
                allocated = self.bandwidth_allocation.get((from_node, to_node), 0)
                min_req = link_req['min_bandwidth_1_to_2']
                max_req = link_req['max_bandwidth_1_to_2']
                
                if allocated >= min_req:
                    if max_req > min_req:
                        satisfaction = min(1.0, (allocated - min_req) / (max_req - min_req))
                    else:
                        satisfaction = 1.0
                    bandwidth_satisfaction += satisfaction
                else:
                    bandwidth_satisfaction += -0.2
                
                satisfied_links += 1
        
        if satisfied_links > 0:
            bandwidth_satisfaction /= satisfied_links
            path_length_penalty = 1.0 - (total_path_length / satisfied_links)
        
        total_reward = (0.3 * load_balance_reward + 
                       0.3 * resource_efficiency + 
                       0.25 * bandwidth_satisfaction +
                       0.15 * path_length_penalty)
        
        return {
            'mapping_success_rate': mapping_success_rate,
            'load_balance_reward': load_balance_reward,
            'resource_efficiency': resource_efficiency,
            'bandwidth_satisfaction': bandwidth_satisfaction,
            'path_length_penalty': path_length_penalty,
            'total_reward': min(1.0, max(0.0, total_reward))
        }

def create_sample_topology(num_nodes: int = 10, 
                          cpu_range: Tuple[int, int] = (50, 100),
                          memory_range: Tuple[int, int] = (100, 200),
                          bandwidth_range: Tuple[int, int] = (100, 500),
                          connectivity_prob: float = 0.3,
                          seed: int = None) -> NetworkTopology:
    """创建示例网络拓扑"""
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    topology = NetworkTopology(num_nodes)
    
    # 设置节点资源
    for i in range(num_nodes):
        cpu = np.random.randint(cpu_range[0], cpu_range[1])
        memory = np.random.randint(memory_range[0], memory_range[1])
        topology.set_node_resources(i, cpu, memory)
    
    # 创建网络连接（随机拓扑，支持不对称带宽）
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < connectivity_prob:
                bandwidth_1_to_2 = np.random.randint(bandwidth_range[0], bandwidth_range[1])
                bandwidth_2_to_1 = np.random.randint(bandwidth_range[0], bandwidth_range[1])
                topology.add_link(i, j, bandwidth_1_to_2, bandwidth_2_to_1)
    
    return topology

def create_sample_virtual_work(num_nodes: int = 8,
                              cpu_range: Tuple[int, int] = (10, 30),
                              memory_range: Tuple[int, int] = (20, 50),
                              bandwidth_range: Tuple[int, int] = (10, 100),
                              connectivity_prob: float = 0.4,
                              seed: int = None) -> VirtualWork:
    """创建示例虚拟工作"""
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    virtual_work = VirtualWork(num_nodes)
    
    # 设置节点需求
    for i in range(num_nodes):
        cpu = np.random.randint(cpu_range[0], cpu_range[1])
        memory = np.random.randint(memory_range[0], memory_range[1])
        virtual_work.set_node_requirement(i, cpu, memory)
    
    # 创建虚拟链路（支持不对称带宽）
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < connectivity_prob:
                # 方向1: i -> j
                min_bandwidth_1_to_2 = np.random.randint(bandwidth_range[0], bandwidth_range[1] * 0.5)
                max_bandwidth_1_to_2 = np.random.randint(min_bandwidth_1_to_2, bandwidth_range[1])
                
                # 方向2: j -> i
                min_bandwidth_2_to_1 = np.random.randint(bandwidth_range[0], bandwidth_range[1] * 0.5)
                max_bandwidth_2_to_1 = np.random.randint(min_bandwidth_2_to_1, bandwidth_range[1])
                
                virtual_work.add_link_requirement(i, j, min_bandwidth_1_to_2, max_bandwidth_1_to_2,
                                                min_bandwidth_2_to_1, max_bandwidth_2_to_1)
    
    return virtual_work

 