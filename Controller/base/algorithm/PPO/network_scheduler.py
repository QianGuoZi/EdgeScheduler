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
    
    def allocate_bandwidth(self, path: List[int], bandwidth: float):
        """分配带宽"""
        if len(path) < 2:
            return
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            link_key = (node1, node2)
            self.links[link_key]['used_bandwidth'] += bandwidth
    
    def release_bandwidth(self, path: List[int], bandwidth: float):
        """释放带宽"""
        if len(path) < 2:
            return
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            link_key = (node1, node2)
            self.links[link_key]['used_bandwidth'] = max(0, self.links[link_key]['used_bandwidth'] - bandwidth)
    
    def allocate_node_resources(self, node_id: int, cpu: float, memory: float):
        """分配节点资源"""
        if node_id in self.node_resources:
            self.node_resources[node_id]['used_cpu'] += cpu
            self.node_resources[node_id]['used_memory'] += memory
    
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
            return False
        
        # 检查带宽是否足够（考虑路径方向）
        if not self.topology.check_bandwidth_availability(path, bandwidth):
            return False
        
        # 分配带宽
        self.topology.allocate_bandwidth(path, bandwidth)
        self.bandwidth_allocation[(virtual_from, virtual_to)] = bandwidth
        
        return True
    
    def _check_bandwidth_availability(self, virtual_from: int, virtual_to: int, 
                                   required_bandwidth: int) -> bool:
        """
        检查两个虚拟节点之间的链路带宽资源是否足够
        
        Args:
            virtual_from: 源虚拟节点ID
            virtual_to: 目标虚拟节点ID
            required_bandwidth: 需要的带宽
            
        Returns:
            bool: True表示带宽足够，False表示带宽不足
        """
        # 检查虚拟节点是否已经调度
        if virtual_from not in self.scheduled_nodes or virtual_to not in self.scheduled_nodes:
            print(f"警告：虚拟节点 {virtual_from} 或 {virtual_to} 尚未调度")
            return False
        
        # 获取对应的物理节点
        physical_from = self.node_mapping[virtual_from]
        physical_to = self.node_mapping[virtual_to]
        
        # 如果映射到同一物理节点，带宽消耗为0，总是返回True
        if physical_from == physical_to:
            print(f"虚拟节点 {virtual_from} 和 {virtual_to} 映射到同一物理节点 {physical_from}，带宽需求为0")
            return True
        
        # 获取最短路径
        path = self.topology.get_shortest_path(physical_from, physical_to)
        if not path:
            print(f"警告：物理节点 {physical_from} 和 {physical_to} 之间无路径")
            return False
        
        # 检查路径上的带宽是否足够
        if not self.topology.check_bandwidth_availability(path, required_bandwidth):
            print(f"警告：路径 {path} 上的可用带宽不足以支持需求带宽 {required_bandwidth}")
            return False
        
        print(f"✅ 虚拟链路 ({virtual_from}, {virtual_to}) 的带宽需求 {required_bandwidth} 可以满足")
        return True
    
    def _check_node_resources(self, virtual_node: int, physical_node: int) -> bool:
        """检查节点资源是否足够"""
        # 首先检查topology中是否有该虚拟节点的资源信息（向后兼容）
        # if virtual_node in self.topology.node_resources:
        #     available = self.topology.get_available_resources(physical_node)
        #     required = self.topology.node_resources[virtual_node]
        #     print(f"检查节点资源是否足够 (从topology):")
        #     print(f"virtual_node: {virtual_node}, available: {available}")
        #     print(f"physical_node: {physical_node}, required: {required}")
            
        #     return (available['cpu'] >= required['cpu'] and 
        #             available['memory'] >= required['memory'])
        
        # 从virtual_work_list中查找虚拟节点的资源需求
        for virtual_work in self.virtual_work_list:
            if virtual_node in virtual_work.node_requirements:
                available = self.topology.get_available_resources(physical_node)
                required = virtual_work.node_requirements[virtual_node]
                print(f"检查节点资源是否足够 (从virtual_work):")
                print(f"virtual_node: {virtual_node}, available: {available}")
                print(f"physical_node: {physical_node}, required: {required}")
                
                return (available['cpu'] >= required['cpu'] and
                        available['memory'] >= required['memory'])
        
        # 如果都找不到，返回False
        print(f"警告：找不到虚拟节点 {virtual_node} 的资源需求信息")
        return False
    
    def _allocate_node_resources(self, virtual_node: int, physical_node: int):
        """分配节点资源"""
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
                self.topology.allocate_node_resources(physical_node, 
                                                    required['cpu'], 
                                                    required['memory'])
                return
        
        print(f"警告：找不到虚拟节点 {virtual_node} 的资源需求信息，无法分配资源")
    
    def get_scheduling_result(self) -> Dict:
        """获取调度结果"""
        return {
            'node_mapping': self.node_mapping.copy(),
            'bandwidth_allocation': self.bandwidth_allocation.copy(),
            'scheduled_nodes': list(self.scheduled_nodes),
            'virtual_works_count': len(self.virtual_work_list)
        }
    
    def calculate_reward(self, virtual_work: VirtualWork) -> float:
        """计算调度奖励"""
        if len(self.scheduled_nodes) == 0:
            return 0
        
        # 1. 资源负载均衡奖励
        cpu_utilizations = []
        memory_utilizations = []
        
        for node_id in range(self.topology.num_nodes):
            available = self.topology.get_available_resources(node_id)
            total_cpu = self.topology.node_resources[node_id]['cpu']
            total_memory = self.topology.node_resources[node_id]['memory']
            
            cpu_utilizations.append(1 - available['cpu'] / total_cpu)
            memory_utilizations.append(1 - available['memory'] / total_memory)
        
        cpu_balance = 1 - np.std(cpu_utilizations)
        memory_balance = 1 - np.std(memory_utilizations)
        
        # 2. 带宽满足度奖励（支持不对称带宽）
        bandwidth_satisfaction = 0
        satisfied_links = 0
        
        for link_req in virtual_work.link_requirements:
            from_node = link_req['from']
            to_node = link_req['to']
            
            if from_node in self.scheduled_nodes and to_node in self.scheduled_nodes:
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
                        satisfaction_1_to_2 = 1.0  # 如果最小和最大需求相同，则完全满足
                
                if min_req_2_to_1 <= allocated_2_to_1 <= max_req_2_to_1:
                    if max_req_2_to_1 > min_req_2_to_1:
                        satisfaction_2_to_1 = (allocated_2_to_1 - min_req_2_to_1) / (max_req_2_to_1 - min_req_2_to_1)
                    else:
                        satisfaction_2_to_1 = 1.0  # 如果最小和最大需求相同，则完全满足
                
                # 取两个方向的平均满足度
                avg_satisfaction = (satisfaction_1_to_2 + satisfaction_2_to_1) / 2
                bandwidth_satisfaction += avg_satisfaction
                satisfied_links += 1
        
        if satisfied_links > 0:
            bandwidth_satisfaction /= satisfied_links
        
        # 3. 网络效率奖励
        network_efficiency = 0
        total_links = 0
        
        for v1 in self.scheduled_nodes:
            for v2 in self.scheduled_nodes:
                if v1 != v2:
                    p1 = self.node_mapping[v1]
                    p2 = self.node_mapping[v2]
                    if p1 == p2:  # 同一物理节点
                        network_efficiency += 1
                    total_links += 1
        
        if total_links > 0:
            network_efficiency /= total_links
        
        # 4. 网络利用率奖励
        network_util = self.topology.get_network_utilization()
        network_utilization_balance = network_util['bandwidth_utilization']
        
        # 综合奖励
        reward = (0.25 * cpu_balance + 
                 0.25 * memory_balance + 
                 0.25 * bandwidth_satisfaction + 
                 0.15 * network_efficiency + 
                 0.10 * network_utilization_balance)
        
        return reward
    
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

def create_sample_topology(num_nodes: int = 10, 
                          cpu_range: Tuple[int, int] = (50, 100),
                          memory_range: Tuple[int, int] = (100, 200),
                          bandwidth_range: Tuple[int, int] = (100, 500),
                          connectivity_prob: float = 0.3) -> NetworkTopology:
    """创建示例网络拓扑"""
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
                              connectivity_prob: float = 0.4) -> VirtualWork:
    """创建示例虚拟工作"""
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