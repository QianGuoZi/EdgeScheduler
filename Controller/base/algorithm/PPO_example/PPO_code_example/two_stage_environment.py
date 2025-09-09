#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple, Dict, List
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler
import torch.nn as nn

class AdvancedGCNFeatureExtractor(nn.Module):
    """高级图卷积特征提取器"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super(AdvancedGCNFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 简单的图卷积层（不使用PyTorch Geometric）
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers.append(layer)
    
    def forward(self, node_features, edge_index, edge_attr):
        """
        前向传播
        
        Args:
            node_features: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
        """
        x = node_features
        
        for i, layer in enumerate(self.layers):
            # 图卷积操作：聚合邻居信息
            x_new = layer(x)
            
            # 简单的邻居聚合
            if edge_index.size(1) > 0:
                neighbor_agg = torch.zeros_like(x_new)
                for j in range(edge_index.size(1)):
                    src, dst = edge_index[0, j], edge_index[1, j]
                    neighbor_agg[src] += x_new[dst]
                    neighbor_agg[dst] += x_new[src]
                
                # 归一化
                neighbor_agg = neighbor_agg / (edge_index.size(1) + 1e-8)
                
                # 结合自身特征和邻居特征
                x = torch.relu(x_new + 0.5 * neighbor_agg)
            else:
                x = torch.relu(x_new)
        
        return x

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
                 use_network_scheduler: bool = True,
                 # 新增：虚拟节点数范围
                 virtual_nodes_range: Tuple[int, int] = None,
                 # 新增：随机种子
                 seed: int = None):
        
        # 新增：设置随机种子
        if seed is not None:
            self._set_random_seed(seed)
            print(f"🌱 环境设置随机种子: {seed}")
        
        self.seed = seed
        
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
        
        # 是否使用network_scheduler
        self.use_network_scheduler = use_network_scheduler
        
        # 新增：高级GCN特征提取器
        self.physical_gcn_extractor = None
        self.virtual_gcn_extractor = None
        
        # 环境状态
        self.physical_state = None
        self.virtual_work = None
        self.bandwidth_mapping = None  # 将在reset中创建
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
        """创建每个虚拟链路的独立带宽映射"""
        # 获取虚拟链路信息
        virtual_edges = self.virtual_work['edges'].numpy()
        virtual_edge_features = self.virtual_work['edge_features'].numpy()
        
        # 为每个虚拟链路创建独立的带宽映射
        link_bandwidth_mappings = {}
        
        for i, (src, dst) in enumerate(virtual_edges.T):
            # 从虚拟链路特征中获取该链路的带宽需求范围
            min_bandwidth = virtual_edge_features[i][0]  # 最小带宽需求
            max_bandwidth = virtual_edge_features[i][1]  # 最大带宽需求
            
            # 为该链路创建带宽映射
            link_range = (min_bandwidth, max_bandwidth)
            link_mapping = self.create_bandwidth_mapping(link_range, self.bandwidth_levels)
            
            # 使用链路索引作为键
            link_key = f"{src}_{dst}"
            link_bandwidth_mappings[link_key] = link_mapping
            
            # print(f"链路 {src}->{dst}: 带宽范围 [{min_bandwidth}, {max_bandwidth}], 映射: {link_mapping}")
        
        return link_bandwidth_mappings
    
    @staticmethod
    def create_bandwidth_mapping(bandwidth_range: Tuple[int, int], levels: int) -> Dict[int, int]:
        """
        创建带宽等级到实际带宽的映射（通用函数）
        
        Args:
            bandwidth_range: 带宽范围，格式为 (min_bandwidth, max_bandwidth)
            levels: 带宽等级数量
            
        Returns:
            Dict[int, int]: 等级到带宽值的映射字典
        """
        min_bandwidth, max_bandwidth = bandwidth_range
        
        # 生成等间距的带宽值并转换为整数
        bandwidths = np.linspace(min_bandwidth, max_bandwidth, levels).astype(int)
        # print(f"bandwidths: {bandwidths}")
        
        # 创建等级到带宽值的映射
        return {i: int(bandwidths[i]) for i in range(levels)}
    
    @staticmethod
    def get_bandwidth_value_for_link(link_bandwidth_mappings: Dict[str, Dict[int, int]], 
                                   link_index: int, virtual_edges: np.ndarray, 
                                   level: int) -> int:
        """
        根据链路索引和等级获取具体的带宽值
        
        Args:
            link_bandwidth_mappings: 所有链路的带宽映射字典
            link_index: 链路在虚拟边数组中的索引
            virtual_edges: 虚拟边数组
            level: 具体的带宽等级 (0 到 levels-1)
            
        Returns:
            int: 对应的带宽值
        """
        if level < 0:
            raise ValueError(f"带宽等级 {level} 不能为负数")
        
        # 获取链路信息
        src, dst = virtual_edges[:, link_index]
        link_key = f"{src}_{dst}"
        
        if link_key not in link_bandwidth_mappings:
            raise ValueError(f"链路 {link_key} 的带宽映射不存在")
        
        link_mapping = link_bandwidth_mappings[link_key]
        
        if level >= len(link_mapping):
            raise ValueError(f"带宽等级 {level} 超出有效范围 [0, {len(link_mapping)-1}]")
        
        return link_mapping[level]
    
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
            # print(f"add physical link: src: {src}, dst: {dst}, bandwidth: {bandwidth}, used_bandwidth: {used_bandwidth}")
        
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
        self.network_scheduler.add_virtual_work(self.virtual_work_obj, work_id="work_1")

    def _set_random_seed(self, seed: int):
        """
        设置环境的随机种子
        
        Args:
            seed: 随机种子值
        """
        import random
        import numpy as np
        
        # 设置Python内置random模块的种子
        random.seed(seed)
        
        # 设置numpy的随机种子
        np.random.seed(seed)
        
        print(f"🔒 环境随机种子设置完成: {seed}")
    
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
        
        # 新增：初始化GCN特征提取器
        if self.physical_gcn_extractor is None:
            physical_feature_dim = self.physical_state['features'].size(1)
            self.physical_gcn_extractor = AdvancedGCNFeatureExtractor(
                input_dim=physical_feature_dim, 
                hidden_dim=64, 
                num_layers=3
            )
        
        if self.virtual_gcn_extractor is None:
            virtual_feature_dim = self.virtual_work['features'].size(1)
            self.virtual_gcn_extractor = AdvancedGCNFeatureExtractor(
                input_dim=virtual_feature_dim, 
                hidden_dim=64, 
                num_layers=3
            )
        
        # 创建链路特定的带宽映射
        self.bandwidth_mapping = self._create_bandwidth_mapping()
        
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
            cpu_usage = np.random.uniform(0.1, 0.5)  # 当前使用率
            memory_usage = np.random.uniform(0.1, 0.5)
            
            physical_features.append([cpu, memory, cpu_usage, memory_usage])
        
        # 物理网络边
        physical_edges = self._get_physical_edges()
        
        # 物理网络边特征：带宽, 当前带宽使用率
        physical_edge_features = []
        for edge in physical_edges.T:
            bandwidth = np.random.randint(*self.physical_bandwidth_range)
            bandwidth_usage = np.random.uniform(0.1, 0.5)
            physical_edge_features.append([bandwidth, bandwidth_usage])
        
        # print(f"reset physical_state")
        # print(f"num_physical_nodes: {self.num_physical_nodes}")
        # print(f"physical_features: {physical_features}")
        # print(f"physical_edges: {physical_edges}")
        # print(f"physical_edge_features: {physical_edge_features}")

        return {
            'features': torch.tensor(physical_features, dtype=torch.float32),
            'edges': physical_edges,
            'edge_features': torch.tensor(physical_edge_features, dtype=torch.float32),
            'num_nodes': self.num_physical_nodes
        }
    
    def _generate_virtual_work(self):
        """生成随机虚拟工作需求"""
        # 使用传入的虚拟节点数范围生成随机节点数
        if hasattr(self, 'virtual_nodes_range'):
            num_virtual_nodes = np.random.randint(self.virtual_nodes_range[0], self.virtual_nodes_range[1] + 1)
        else:
            # 保持向后兼容性
            num_virtual_nodes = np.random.randint(3, self.max_virtual_nodes + 1)
        
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

        # print(f"reset virtual_work")
        # print(f"num_virtual_nodes: {num_virtual_nodes}")
        # print(f"virtual_features: {virtual_features}")
        # print(f"virtual_edges: {virtual_edges}")
        # print(f"virtual_edge_features: {virtual_edge_features}")

        return {
            'features': torch.tensor(virtual_features, dtype=torch.float32),
            'edges': virtual_edges,
            'edge_features': torch.tensor(virtual_edge_features, dtype=torch.float32),
            'num_nodes': num_virtual_nodes
        }
    
    def _get_physical_edges(self):
        """获取物理网络边，确保每个节点都至少有一条边"""
        edges = []
        connected_nodes = set()
        
        # 第一步：确保每个节点都至少有一条边
        for i in range(self.num_physical_nodes):
            if i not in connected_nodes:
                # 为未连接的节点寻找连接
                if i == 0:
                    # 第一个节点连接到下一个节点
                    if self.num_physical_nodes > 1:
                        edges.append([i, 1])
                        edges.append([1, i])  # 添加反向边
                        connected_nodes.add(i)
                        connected_nodes.add(1)
                else:
                    # 其他节点连接到前一个节点（如果前一个节点未连接）或第一个节点
                    if i - 1 not in connected_nodes:
                        edges.append([i, 0])
                        edges.append([0, i])  # 添加反向边
                        connected_nodes.add(i)
                        connected_nodes.add(0)
                    else:
                        edges.append([i, i - 1])
                        edges.append([i - 1, i])  # 添加反向边
                        connected_nodes.add(i)
        
        # 第二步：添加额外的随机连接
        for i in range(self.num_physical_nodes):
            for j in range(i + 1, self.num_physical_nodes):
                # 避免重复添加已有的边
                if [i, j] not in edges and np.random.random() < self.physical_connectivity_prob:
                    edges.append([i, j])
                    edges.append([j, i])  # 添加反向边
        
        # 确保至少有一条边
        if not edges:
            edges = [[0, 1], [1, 0]]  # 默认连接节点0和1
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _get_virtual_edges(self, num_virtual_nodes):
        """获取虚拟网络边，确保每个节点都至少有一条边"""
        edges = []
        connected_nodes = set()
        
        # 第一步：确保每个节点都至少有一条边
        for i in range(num_virtual_nodes):
            if i not in connected_nodes:
                # 为未连接的节点寻找连接
                if i == 0:
                    # 第一个节点连接到下一个节点
                    if num_virtual_nodes > 1:
                        edges.append([i, 1])
                        edges.append([1, i])  # 添加反向边
                        connected_nodes.add(i)
                        connected_nodes.add(1)
                else:
                    # 其他节点连接到前一个节点（如果前一个节点未连接）或第一个节点
                    if i - 1 not in connected_nodes:
                        edges.append([i, 0])
                        edges.append([0, i])  # 添加反向边
                        connected_nodes.add(i)
                        connected_nodes.add(0)
                    else:
                        edges.append([i, i - 1])
                        edges.append([i - 1, i])  # 添加反向边
                        connected_nodes.add(i)
        
        # 第二步：添加额外的随机连接
        for i in range(num_virtual_nodes):
            for j in range(i + 1, num_virtual_nodes):
                # 避免重复添加已有的边
                if [i, j] not in edges and np.random.random() < self.virtual_connectivity_prob:
                    edges.append([i, j])
                    edges.append([j, i])  # 添加反向边
        
        # 确保至少有一条边
        if not edges:
            edges = [[0, 1], [1, 0]]  # 默认连接节点0和1
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _compute_gcn_features(self, node_features, edge_index, edge_attr):
        """
        计算图卷积特征
        
        Args:
            node_features: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
            
        Returns:
            gcn_features: 图卷积特征 [num_nodes, gcn_dim]
        """
        # 简单的图卷积特征计算（不使用PyTorch Geometric，避免依赖问题）
        num_nodes = node_features.size(0)
        gcn_dim = 64  # GCN特征维度
        
        # 初始化GCN特征
        gcn_features = torch.zeros(num_nodes, gcn_dim, dtype=torch.float32)
        
        # 计算每个节点的邻居聚合特征
        for node in range(num_nodes):
            # 找到该节点的邻居
            neighbors = []
            for i in range(edge_index.size(1)):
                if edge_index[0, i] == node:
                    neighbors.append(edge_index[1, i].item())
                elif edge_index[1, i] == node:
                    neighbors.append(edge_index[0, i].item())
            
            if neighbors:
                # 聚合邻居特征
                neighbor_features = node_features[neighbors]
                neighbor_edge_features = []
                
                # 获取相关的边特征
                for i in range(edge_index.size(1)):
                    if (edge_index[0, i] == node and edge_index[1, i] in neighbors) or \
                       (edge_index[1, i] == node and edge_index[0, i] in neighbors):
                        neighbor_edge_features.append(edge_attr[i])
                
                if neighbor_edge_features:
                    neighbor_edge_features = torch.stack(neighbor_edge_features)
                    # 简单的特征聚合：节点特征 + 边特征的平均
                    aggregated_features = torch.cat([
                        node_features[node],
                        neighbor_features.mean(dim=0),
                        neighbor_edge_features.mean(dim=0)
                    ])
                    
                    # 投影到GCN特征维度（使用简单的线性变换）
                    if aggregated_features.size(0) <= gcn_dim:
                        gcn_features[node, :aggregated_features.size(0)] = aggregated_features
                    else:
                        # 如果特征维度超过GCN维度，进行降维
                        gcn_features[node] = aggregated_features[:gcn_dim]
            else:
                # 孤立节点，使用自身特征
                if node_features[node].size(0) <= gcn_dim:
                    gcn_features[node, :node_features[node].size(0)] = node_features[node]
                else:
                    gcn_features[node] = node_features[node][:gcn_dim]
        
        return gcn_features
    
    def _get_state(self):
        """获取当前状态"""
        # 使用高级GCN特征提取器计算图卷积特征
        with torch.no_grad():
            # 确保GCN特征提取器在正确的设备上
            device = self.physical_state['features'].device
            if self.physical_gcn_extractor is not None:
                self.physical_gcn_extractor = self.physical_gcn_extractor.to(device)
            if self.virtual_gcn_extractor is not None:
                self.virtual_gcn_extractor = self.virtual_gcn_extractor.to(device)
            
            # 确保输入数据是torch.tensor类型
            physical_features = self.physical_state['features']
            physical_edges = self.physical_state['edges']
            physical_edge_features = self.physical_state['edge_features']
            physical_num_nodes = self.physical_state['num_nodes']
            
            virtual_features = self.virtual_work['features']
            virtual_edges = self.virtual_work['edges']
            virtual_edge_features = self.virtual_work['edge_features']
            virtual_num_nodes = self.virtual_work['num_nodes']
            
            # 如果边索引不是torch.tensor，转换为torch.tensor
            if not isinstance(physical_edges, torch.Tensor):
                physical_edges = torch.tensor(physical_edges, dtype=torch.long, device=device)
            if not isinstance(virtual_edges, torch.Tensor):
                virtual_edges = torch.tensor(virtual_edges, dtype=torch.long, device=device)
            
            physical_gcn_features = self.physical_gcn_extractor.forward(
                physical_features, 
                physical_edges, 
                physical_edge_features,
                # physical_num_nodes
            )
            virtual_gcn_features = self.virtual_gcn_extractor.forward(
                virtual_features, 
                virtual_edges, 
                virtual_edge_features,
                # virtual_num_nodes
            )
        
        return {
            'physical_features': self.physical_state['features'],
            'physical_edges': self.physical_state['edges'],
            'physical_edge_features': self.physical_state['edge_features'],
            'physical_num_nodes': physical_num_nodes,
            'physical_gcn_features': physical_gcn_features,  # 新增：物理网络GCN特征
            'virtual_features': self.virtual_work['features'],
            'virtual_edges': self.virtual_work['edges'],
            'virtual_edge_features': self.virtual_work['edge_features'],
            'virtual_num_nodes': virtual_num_nodes,
            'virtual_gcn_features': virtual_gcn_features,  # 新增：虚拟网络GCN特征
            'bandwidth_mapping': self.bandwidth_mapping
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
                'is_valid': False,
                # 无效动作时奖励组件都为0
                'load_balance_reward': 0.0,
                'bandwidth_satisfaction_reward': 0.0,
                'total_reward': reward
            }
        else:
            # 如果使用network_scheduler，执行调度
            if self.use_network_scheduler:
                self._execute_network_scheduler_actions(mapping_action, bandwidth_action)
            
            # 计算奖励
            reward = self._calculate_reward(mapping_action, bandwidth_action)
            
            # 获取奖励组件信息
            load_balance_reward = 0.0
            bandwidth_satisfaction_reward = 0.0
            
            if self.use_network_scheduler and hasattr(self.network_scheduler, 'calculate_reward_components'):
                components = self.network_scheduler.calculate_reward_components(self.virtual_work_obj)
                load_balance_reward = components.get('load_balance', 0.0)
                bandwidth_satisfaction_reward = components.get('bandwidth_satisfaction', 0.0)
            
            info = {
                'constraint_violations': [],
                'mapping_result': mapping_action,
                'bandwidth_result': bandwidth_action,
                'is_valid': True,
                'load_balance_reward': load_balance_reward,
                'bandwidth_satisfaction_reward': bandwidth_satisfaction_reward,
                'total_reward': reward
            }
        print(f"TwoStageNetworkSchedulerEnvironment step 执行两阶段动作")
        print(f"TwoStageNetworkSchedulerEnvironment step reward: {reward}")
        print(f"TwoStageNetworkSchedulerEnvironment step mapping_action: {mapping_action}")
        print(f"TwoStageNetworkSchedulerEnvironment step bandwidth_action: {bandwidth_action}")
        print(f"TwoStageNetworkSchedulerEnvironment step info: {info}")

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
                # 使用链路特定的带宽映射
                link_key = f"{src}_{dst}"
                if link_key in self.bandwidth_mapping:
                    allocated_bandwidth = self.bandwidth_mapping[link_key][bandwidth_action[i]]
                    success = self.network_scheduler.allocate_bandwidth(src, dst, allocated_bandwidth)
                    if not success:
                        print(f"警告：虚拟链路({src},{dst})带宽分配{allocated_bandwidth}失败")
                else:
                    print(f"警告：找不到链路 {link_key} 的带宽映射")
    
    def _validate_actions(self, mapping_action, bandwidth_action):
        """验证动作的有效性"""
        constraint_violations = []
        
        # 检查映射动作
        if len(mapping_action) != self.virtual_work['num_nodes']:
            constraint_violations.append("映射动作长度不匹配")
            return False, constraint_violations
        
        # 检查物理节点索引范围
        mapping_action = np.array(mapping_action)
        if np.any(mapping_action < 0) or np.any(mapping_action >= self.num_physical_nodes):
            constraint_violations.append("物理节点索引超出范围")
            return False, constraint_violations
        
        # 如果使用network_scheduler，使用其验证逻辑
        if self.use_network_scheduler:
            return self._validate_actions_with_network_scheduler(mapping_action, bandwidth_action)
        
        return len(constraint_violations) == 0, constraint_violations
    
    def _validate_actions_with_network_scheduler(self, mapping_action, bandwidth_action):
        """使用network_scheduler验证动作的有效性"""
        constraint_violations = []
        
        # 检查映射动作长度
        if len(mapping_action) != self.virtual_work['num_nodes']:
            constraint_violations.append("映射动作长度不匹配")
            return False, constraint_violations
        
        # 检查物理节点索引范围
        mapping_action = np.array(mapping_action)
        if np.any(mapping_action < 0) or np.any(mapping_action >= self.num_physical_nodes):
            constraint_violations.append("物理节点索引超出范围")
            return False, constraint_violations
        
        # 🚀 修改：先统计每个物理节点的总资源需求，再统一验证
        physical_node_total_demands = {}
        
        # 第一步：统计每个物理节点上准备映射的虚拟节点的总资源需求
        for virtual_node, physical_node in enumerate(mapping_action):
            if physical_node not in physical_node_total_demands:
                physical_node_total_demands[physical_node] = {'cpu': 0, 'memory': 0, 'virtual_nodes': []}
            
            # 获取虚拟节点的资源需求
            for virtual_work in self.network_scheduler.virtual_work_list:
                if virtual_node in virtual_work.node_requirements:
                    req = virtual_work.node_requirements[virtual_node]
                    physical_node_total_demands[physical_node]['cpu'] += req['cpu']
                    physical_node_total_demands[physical_node]['memory'] += req['memory']
                    physical_node_total_demands[physical_node]['virtual_nodes'].append(virtual_node)
                    break
        
        # 第二步：验证每个物理节点的总资源需求是否超过可用资源
        for physical_node, demands in physical_node_total_demands.items():
            # 获取物理节点的总资源
            total_cpu = self.network_scheduler.topology.node_resources[physical_node]['cpu']
            total_memory = self.network_scheduler.topology.node_resources[physical_node]['memory']
            
            # 获取物理节点的已使用资源
            available_resources = self.network_scheduler.topology.get_available_resources(physical_node)
            used_cpu = total_cpu - available_resources['cpu']
            used_memory = total_memory - available_resources['memory']
            
            # 计算映射后的总资源需求
            total_cpu_after_mapping = used_cpu + demands['cpu']
            total_memory_after_mapping = used_memory + demands['memory']
            
            # 检查是否超过物理节点的总资源
            if total_cpu_after_mapping > total_cpu:
                constraint_violations.append(
                    f"资源不足：物理节点{physical_node}CPU资源不足 "
                    f"(已用: {used_cpu:.1f}, 新增需求: {demands['cpu']:.1f}, 总计: {total_cpu_after_mapping:.1f}/{total_cpu})"
                )
                return False, constraint_violations
            
            if total_memory_after_mapping > total_memory:
                constraint_violations.append(
                    f"资源不足：物理节点{physical_node}内存资源不足 "
                    f"(已用: {used_memory:.1f}, 新增需求: {demands['memory']:.1f}, 总计: {total_memory_after_mapping:.1f}/{total_memory})"
                )
                return False, constraint_violations
            
            # 计算资源利用率
            cpu_utilization = total_cpu_after_mapping / total_cpu
            memory_utilization = total_memory_after_mapping / total_memory
            
            # 记录映射信息用于后续分析
            print(f"🔍 物理节点{physical_node}资源分析:")
            print(f"  - 映射的虚拟节点: {demands['virtual_nodes']}")
            print(f"  - CPU: 已用{used_cpu:.1f} + 新增{demands['cpu']:.1f} = {total_cpu_after_mapping:.1f}/{total_cpu} ({cpu_utilization:.1%})")
            print(f"  - 内存: 已用{used_memory:.1f} + 新增{demands['memory']:.1f} = {total_memory_after_mapping:.1f}/{total_memory} ({memory_utilization:.1%})")
            
            # # 检查负载是否合理（软约束）
            # if cpu_utilization > 0.95 or memory_utilization > 0.95:
            #     constraint_violations.append(f"严重过载：物理节点{physical_node}负载过高 (CPU: {cpu_utilization:.1%}, Memory: {memory_utilization:.1%})")
            #     # return False, constraint_violations
            # elif cpu_utilization > 0.85 or memory_utilization > 0.85:
            #     constraint_violations.append(f"高负载警告：物理节点{physical_node}负载较高 (CPU: {cpu_utilization:.1%}, Memory: {memory_utilization:.1%})")
            # elif cpu_utilization > 0.80 or memory_utilization > 0.80:
            #     constraint_violations.append(f"负载警告：物理节点{physical_node}负载中等 (CPU: {cpu_utilization:.1%}, Memory: {memory_utilization:.1%})")
        
        # 检查带宽约束（按物理链路聚合总需求后统一校验）
        virtual_edges = self.virtual_work['edges'].numpy()
        physical_link_total_demands = {}  # {(u,v): total_required_bw}

        for i, (src, dst) in enumerate(virtual_edges.T):
            if i < len(bandwidth_action):
                # 使用链路特定的带宽映射
                link_key = f"{src}_{dst}"
                if link_key in self.bandwidth_mapping:
                    required_bandwidth = self.bandwidth_mapping[link_key][bandwidth_action[i]]

                    # 如果映射到同一物理节点，带宽需求为0，跳过
                    physical_from = mapping_action[src]
                    physical_to = mapping_action[dst]
                    if physical_from == physical_to:
                        continue

                    # 计算物理最短路径
                    path = self.network_scheduler.topology.get_shortest_path(physical_from, physical_to)
                    if not path or len(path) < 2:
                        constraint_violations.append(
                            f"物理路径不可达：虚拟链路({src},{dst}) 映射到物理({physical_from}->{physical_to}) 无可用路径")
                        return False, constraint_violations

                    # 将该虚拟链路的需求累加到路径上的每条物理链路
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j + 1]
                        key = (u, v)
                        physical_link_total_demands[key] = physical_link_total_demands.get(key, 0.0) + float(required_bandwidth)
                else:
                    print(f"警告：找不到链路 {link_key} 的带宽映射")

        # 统一校验每条物理链路的总需求是否超过可用带宽
        for (u, v), total_required in physical_link_total_demands.items():
            available = self.network_scheduler.topology.get_available_bandwidth(u, v)
            if total_required > available:
                constraint_violations.append(
                    f"带宽不足：物理链路({u}->{v}) 可用带宽 {available:.1f} 小于需求总和 {total_required:.1f}")
                return False, constraint_violations

            # 软约束提示：检查分配后的利用率水平
            link_info = self.network_scheduler.topology.links.get((u, v))
            if link_info is not None:
                total_bw = float(link_info['bandwidth'])
                used_bw = float(link_info['used_bandwidth'])
                utilization_after = (used_bw + total_required) / (total_bw + 1e-8) if total_bw > 0 else 1.0
                # if utilization_after > 0.95:
                #     constraint_violations.append(
                #         f"严重带宽过载风险：物理链路({u}->{v}) 预计利用率 {utilization_after:.1%}")
                #     # return False, constraint_violations
                # elif utilization_after > 0.85:
                #     constraint_violations.append(
                #         f"带宽高负载警告：物理链路({u}->{v}) 预计利用率 {utilization_after:.1%}")
                # elif utilization_after > 0.80:
                #     constraint_violations.append(
                #         f"带宽中等负载提示：物理链路({u}->{v}) 预计利用率 {utilization_after:.1%}")
        
        return True, constraint_violations
    
    def _calculate_reward(self, mapping_action, bandwidth_action):
        """计算奖励（归一化版本）"""
        # 如果使用network_scheduler，使用其归一化奖励计算
        if self.use_network_scheduler:
            reward = self.network_scheduler.calculate_reward_2(self.virtual_work_obj)
            
            # 添加调试信息
            print(f"🔍 奖励计算详情:")
            print(f"   - 网络调度器奖励: {reward:.4f}")
            print(f"   - 映射动作: {mapping_action}")
            print(f"   - 带宽动作: {bandwidth_action}")
            
            # 分析各个组件
            if hasattr(self.network_scheduler, 'calculate_reward_components'):
                components = self.network_scheduler.calculate_reward_components(self.virtual_work_obj)
                print(f"   - 负载均衡奖励: {components.get('load_balance', 0):.4f}")
                print(f"   - 带宽满足度: {components.get('bandwidth_satisfaction', 0):.4f}")
                print(f"   - 同一节点带宽奖励: {components.get('same_node_bandwidth_bonus', 0):.4f}")
                print(f"   - 负载过载惩罚: {components.get('overload_penalty', 0):.4f}")
                print(f"   - 总奖励: {components.get('total_reward', 0):.4f}")
            
            return reward