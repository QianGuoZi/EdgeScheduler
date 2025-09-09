#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random
from datetime import datetime
import torch.nn as nn

from two_stage_actor_design import TwoStagePPOAgent
from two_stage_environment import TwoStageNetworkSchedulerEnvironment

class RandomAllocationAlgorithm:
    """随机分配算法"""
    
    def __init__(self, num_physical_nodes: int, max_virtual_nodes: int, bandwidth_levels: int):
        self.num_physical_nodes = num_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
        self.bandwidth_levels = bandwidth_levels
    
    def allocate(self, physical_state, virtual_work):
        """随机分配虚拟节点到物理节点，随机分配带宽"""
        num_virtual_nodes = virtual_work['features'].shape[0]
        
        # 随机映射虚拟节点到物理节点
        mapping_action = []
        for _ in range(num_virtual_nodes):
            physical_node_idx = random.randint(0, self.num_physical_nodes - 1)
            mapping_action.append(physical_node_idx)
        
        # 随机分配带宽等级
        num_virtual_edges = virtual_work['edges'].shape[1]
        bandwidth_action = []
        for _ in range(num_virtual_edges):
            bandwidth_level = random.randint(0, self.bandwidth_levels - 1)
            bandwidth_action.append(bandwidth_level)
        
        return np.array(mapping_action), np.array(bandwidth_action)

class PPOvsRandomTester:
    """PPO vs 随机算法对比测试器"""
    
    def __init__(self, 
                 session_name: str,
                 model_dir: str = "models",
                 stats_dir: str = "stats",
                 results_dir: str = "test_results",
                 num_trials: int = 10,
                 seed: int = None):
        
        self.session_name = session_name
        self.model_dir = model_dir
        self.stats_dir = stats_dir
        self.results_dir = results_dir

        # 设置随机种子
        self.seed = seed if seed is not None else random.randint(1, 10000)
        self._set_random_seed(self.seed)
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 模型和环境参数
        self.ppo_agent = None
        self.env = None
        self.random_algorithm = None
        
        # 测试结果
        self.test_results = {
            'ppo': {'load_balancing': [], 'bandwidth_satisfaction': []},
            'random': {'load_balancing': [], 'bandwidth_satisfaction': []}
        }
        
        # 测试配置
        self.virtual_node_counts = list(range(3, 9))  # 3-8个虚拟节点
        self.num_trials = 10  # 每个配置测试10次
        
        # 资源范围配置
        self.physical_cpu_range = (50, 100)
        self.physical_memory_range = (50, 100)
        self.physical_bandwidth_range = (50, 100)
        self.virtual_cpu_range = (8, 10)
        self.virtual_memory_range = (8, 10)
        self.virtual_bandwidth_range = (5, 15)
        self.physical_connectivity_prob = 0.9
        self.virtual_connectivity_prob = 0.7
        self.bandwidth_levels = 10
    
    def _set_random_seed(self, seed: int):
        """设置所有相关的随机种子"""
        import random
        import numpy as np
        import torch
        
        # 设置Python内置random模块的种子
        random.seed(seed)
        
        # 设置numpy的随机种子
        np.random.seed(seed)
        
        # 设置PyTorch的随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 设置PyTorch的确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 设置环境变量
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print(f"测试器随机种子设置完成: {seed}")
    
    def load_ppo_model(self):
        """加载训练好的PPO模型"""
        print("🔄 加载PPO模型...")
        
        # 加载配置文件
        config_path = os.path.join(self.stats_dir, self.session_name, f"session_config_{self.session_name}.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载最新的模型文件
        model_dir = os.path.join(self.model_dir, self.session_name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"找不到模型目录: {model_dir}")
        
        model_files = [f for f in os.listdir(model_dir) if f.startswith(f"ppo_model_{self.session_name}_episode_") and f.endswith(".pth")]
        if not model_files:
            raise FileNotFoundError(f"在目录 {model_dir} 中找不到模型文件")
        
        # 选择最新的模型
        model_files.sort(key=lambda x: int(x.split('_episode_')[1].split('.')[0]), reverse=True)
        latest_model_file = model_files[0]
        model_path = os.path.join(model_dir, latest_model_file)
        
        print(f"📂 找到模型文件: {latest_model_file}")
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 检测模型配置
        model_config = self._detect_model_config(checkpoint)
        
        # 创建智能体
        self.ppo_agent = TwoStagePPOAgent(
            physical_node_dim=model_config['physical_node_dim'],
            virtual_node_dim=model_config['virtual_node_dim'],
            max_physical_nodes=model_config['num_physical_nodes'],
            max_virtual_nodes=model_config['max_virtual_nodes'],
            bandwidth_levels=self.bandwidth_levels,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01
        )
        
        # 检测模型版本和兼容性
        model_version = self._detect_model_version(checkpoint)
        print(f"🔍 检测到模型版本: {model_version}")
        
        # 加载模型权重
        # 使用strict=False来兼容新旧模型（新模型包含GCN投影层）
        try:
            self.ppo_agent.mapping_actor.load_state_dict(checkpoint['mapping_actor_state_dict'], strict=False)
            self.ppo_agent.bandwidth_actor.load_state_dict(checkpoint['bandwidth_actor_state_dict'], strict=False)
            self.ppo_agent.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
            print("✅ 模型权重加载成功（兼容模式）")
        except Exception as e:
            print(f"⚠️ 模型加载警告: {e}")
            print("🔄 尝试严格模式加载...")
            # 如果兼容模式失败，尝试严格模式
            self.ppo_agent.mapping_actor.load_state_dict(checkpoint['mapping_actor_state_dict'])
            self.ppo_agent.bandwidth_actor.load_state_dict(checkpoint['bandwidth_actor_state_dict'])
            self.ppo_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 设置为评估模式
        self.ppo_agent.mapping_actor.eval()
        self.ppo_agent.bandwidth_actor.eval()
        self.ppo_agent.critic.eval()
        
        # 如果是旧模型，初始化新的GCN投影层
        if model_version == "v1.0 (基础版本)":
            print("🔄 检测到旧模型，初始化新的GCN投影层...")
            self._initialize_gcn_layers_for_old_model()
        
        print(f"✅ PPO模型加载成功: {model_path}")
        print(f"   物理节点数: {model_config['num_physical_nodes']}")
        print(f"   最大虚拟节点数: {model_config['max_virtual_nodes']}")
        
        return model_config
    
    def _initialize_gcn_layers_for_old_model(self):
        """为旧模型初始化新的GCN投影层"""
        print("🔧 初始化GCN投影层...")
        
        # 获取隐藏维度
        mapping_state = self.ppo_agent.mapping_actor.state_dict()
        if 'global_mapping_head.0.weight' in mapping_state:
            hidden_dim = mapping_state['global_mapping_head.0.weight'].shape[0] // 2
        else:
            hidden_dim = 128  # 默认值
        
        # 初始化物理GCN投影层
        if self.ppo_agent.mapping_actor.physical_gcn_proj is None:
            self.ppo_agent.mapping_actor.physical_gcn_proj = nn.Linear(64, hidden_dim).to(self.ppo_agent.device)
            self.ppo_agent.mapping_actor.add_module('physical_gcn_proj', self.ppo_agent.mapping_actor.physical_gcn_proj)
        
        # 初始化虚拟GCN投影层
        if self.ppo_agent.mapping_actor.virtual_gcn_proj is None:
            self.ppo_agent.mapping_actor.virtual_gcn_proj = nn.Linear(64, hidden_dim).to(self.ppo_agent.device)
            self.ppo_agent.mapping_actor.add_module('virtual_gcn_proj', self.ppo_agent.mapping_actor.virtual_gcn_proj)
        
        # 对BandwidthActor做同样的处理
        if self.ppo_agent.bandwidth_actor.physical_gcn_proj is None:
            self.ppo_agent.bandwidth_actor.physical_gcn_proj = nn.Linear(64, hidden_dim).to(self.ppo_agent.device)
            self.ppo_agent.bandwidth_actor.add_module('physical_gcn_proj', self.ppo_agent.bandwidth_actor.physical_gcn_proj)
        
        if self.ppo_agent.bandwidth_actor.virtual_gcn_proj is None:
            self.ppo_agent.bandwidth_actor.virtual_gcn_proj = nn.Linear(64, hidden_dim).to(self.ppo_agent.device)
            self.ppo_agent.bandwidth_actor.add_module('virtual_gcn_proj', self.ppo_agent.bandwidth_actor.virtual_gcn_proj)
        
        # 对Critic做同样的处理
        if self.ppo_agent.critic.physical_gcn_proj is None:
            self.ppo_agent.critic.physical_gcn_proj = nn.Linear(64, hidden_dim).to(self.ppo_agent.device)
            self.ppo_agent.critic.add_module('physical_gcn_proj', self.ppo_agent.critic.physical_gcn_proj)
        
        if self.ppo_agent.critic.virtual_gcn_proj is None:
            self.ppo_agent.critic.virtual_gcn_proj = nn.Linear(64, hidden_dim).to(self.ppo_agent.device)
            self.ppo_agent.critic.add_module('virtual_gcn_proj', self.ppo_agent.critic.virtual_gcn_proj)
        
        print("✅ GCN投影层初始化完成")
    
    def _detect_model_version(self, checkpoint):
        """检测模型版本和兼容性"""
        mapping_state = checkpoint['mapping_actor_state_dict']
        bandwidth_state = checkpoint['bandwidth_actor_state_dict']
        critic_state = checkpoint['critic_state_dict']
        
        # 检查是否包含GCN投影层
        has_gcn_layers = (
            'physical_gcn_proj.weight' in mapping_state or
            'virtual_gcn_proj.weight' in mapping_state or
            'physical_gcn_proj.weight' in bandwidth_state or
            'virtual_gcn_proj.weight' in bandwidth_state or
            'physical_gcn_proj.weight' in critic_state or
            'virtual_gcn_proj.weight' in critic_state
        )
        
        if has_gcn_layers:
            return "v2.0 (支持GCN特征)"
        else:
            return "v1.0 (基础版本)"
    
    def _detect_model_config(self, checkpoint):
        """从checkpoint中检测模型配置"""
        print("🔍 检测模型配置...")
        
        # 检测物理节点数量
        mapping_state = checkpoint['mapping_actor_state_dict']
        if 'global_mapping_head.6.weight' in mapping_state:
            num_physical_nodes = mapping_state['global_mapping_head.6.weight'].shape[0]
        else:
            num_physical_nodes = 10
        
        # 检测物理节点特征维度
        if 'physical_encoder.conv_layers.0.lin.weight' in mapping_state:
            physical_node_dim = mapping_state['physical_encoder.conv_layers.0.lin.weight'].shape[1]
        else:
            physical_node_dim = 3
        
        # 检测虚拟节点特征维度
        if 'virtual_encoder.conv_layers.0.lin.weight' in mapping_state:
            virtual_node_dim = mapping_state['virtual_encoder.conv_layers.0.lin.weight'].shape[1]
        else:
            virtual_node_dim = 3
        
        # 检测最大虚拟节点数
        bandwidth_state = checkpoint['bandwidth_actor_state_dict']
        if 'global_bandwidth_head.6.weight' in bandwidth_state:
            max_virtual_nodes = bandwidth_state['global_bandwidth_head.6.weight'].shape[0]
        else:
            max_virtual_nodes = 8
        
        return {
            'num_physical_nodes': num_physical_nodes,
            'max_virtual_nodes': max_virtual_nodes,
            'physical_node_dim': physical_node_dim,
            'virtual_node_dim': virtual_node_dim
        }
    
    def initialize_environment(self, model_config):
        """初始化测试环境"""
        print("🔄 初始化测试环境...")
        
        self.env = TwoStageNetworkSchedulerEnvironment(
            num_physical_nodes=model_config['num_physical_nodes'],
            max_virtual_nodes=model_config['max_virtual_nodes'],
            bandwidth_levels=self.bandwidth_levels,
            physical_cpu_range=self.physical_cpu_range,
            physical_memory_range=self.physical_memory_range,
            physical_bandwidth_range=self.physical_bandwidth_range,
            virtual_cpu_range=self.virtual_cpu_range,
            virtual_memory_range=self.virtual_memory_range,
            virtual_bandwidth_range=self.virtual_bandwidth_range,
            physical_connectivity_prob=self.physical_connectivity_prob,
            virtual_connectivity_prob=self.virtual_connectivity_prob,
            use_network_scheduler=True,
            seed=self.seed  # 传递种子到环境
        )
        
        # 初始化随机算法
        self.random_algorithm = RandomAllocationAlgorithm(
            num_physical_nodes=model_config['num_physical_nodes'],
            max_virtual_nodes=model_config['max_virtual_nodes'],
            bandwidth_levels=self.bandwidth_levels
        )
        
        print("✅ 环境初始化完成")
    
    def calculate_load_balancing_score(self, mapping_action, physical_state, virtual_work):
        """计算负载均衡度分数"""
        physical_features = physical_state['features'].numpy()
        virtual_features = virtual_work['features'].numpy()
        
        # 计算每个物理节点的负载
        node_loads = []
        for physical_node_idx in range(physical_features.shape[0]):
            mapped_virtual_nodes = [i for i, p_idx in enumerate(mapping_action) if p_idx == physical_node_idx]
            
            current_cpu_usage = physical_features[physical_node_idx][2]      # 索引2：CPU使用率
            current_memory_usage = physical_features[physical_node_idx][3]   # 索引3：内存使用率

            if not mapped_virtual_nodes:
                current_load = (current_cpu_usage + current_memory_usage) / 2
                node_loads.append(current_load)
                continue
            
            # 计算综合负载
            total_cpu_demand = sum(virtual_features[i][0] for i in mapped_virtual_nodes)
            total_memory_demand = sum(virtual_features[i][1] for i in mapped_virtual_nodes)
            
             # 原有负载 + 新增需求
            cpu_load = current_cpu_usage + (total_cpu_demand / physical_features[physical_node_idx][0])
            memory_load = current_memory_usage + (total_memory_demand / physical_features[physical_node_idx][1])
            
            cpu_load = min(cpu_load, 1.0)
            memory_load = min(memory_load, 1.0)
            
            avg_load = (cpu_load + memory_load) / 2
            node_loads.append(avg_load)
        
        # 计算负载均衡度（负载方差越小越好）
        if node_loads:
            load_variance = np.var(node_loads)
            return max(0, 1.0 - load_variance * 10)
        else:
            return 0.0
    
    def calculate_load_balancing_score_with_bandwidth(self, mapping_action, bandwidth_action, physical_state, virtual_work):
        """计算负载均衡度分数 - 包含CPU、内存和带宽资源"""
        physical_features = physical_state['features'].numpy()
        virtual_features = virtual_work['features'].numpy()
        physical_edges = physical_state['edges'].numpy()
        physical_edge_features = physical_state['edge_features'].numpy()
        virtual_edges = virtual_work['edges'].numpy()
        virtual_edge_features = virtual_work['edge_features'].numpy()
        
        # 计算每个物理节点的综合负载（CPU + 内存 + 带宽）
        node_loads = []
        
        for physical_node_idx in range(physical_features.shape[0]):
            mapped_virtual_nodes = [i for i, p_idx in enumerate(mapping_action) if p_idx == physical_node_idx]
            
            # 获取物理节点当前的CPU和内存使用率
            current_cpu_usage = physical_features[physical_node_idx][2]      # 索引2：CPU使用率
            current_memory_usage = physical_features[physical_node_idx][3]   # 索引3：内存使用率
            
            # 初始化带宽负载
            bandwidth_load = 0.0
            
            if not mapped_virtual_nodes:
                # 即使没有虚拟节点映射，也要考虑当前负载
                current_load = (current_cpu_usage + current_memory_usage) / 2
                node_loads.append(current_load)
                continue
            
            # 计算CPU和内存负载：原有负载 + 新增需求
            total_cpu_demand = sum(virtual_features[i][0] for i in mapped_virtual_nodes)
            total_memory_demand = sum(virtual_features[i][1] for i in mapped_virtual_nodes)
            
            cpu_load = current_cpu_usage + (total_cpu_demand / physical_features[physical_node_idx][0])
            memory_load = current_memory_usage + (total_memory_demand / physical_features[physical_node_idx][1])
            
            # 确保负载不超过100%
            cpu_load = min(cpu_load, 1.0)
            memory_load = min(memory_load, 1.0)
            
            # 计算带宽负载：考虑映射到此物理节点的虚拟链路
            if len(bandwidth_action) > 0:
                bandwidth_load = self._calculate_node_bandwidth_load(
                    physical_node_idx, mapped_virtual_nodes, 
                    mapping_action, bandwidth_action,
                    physical_edges, physical_edge_features,
                    virtual_edges, virtual_edge_features
                )
            
            # 综合负载：CPU(40%) + 内存(40%) + 带宽(20%)
            # 可以根据实际需求调整权重
            cpu_weight = 0.4
            memory_weight = 0.4
            bandwidth_weight = 0.2
            
            avg_load = (cpu_load * cpu_weight + 
                    memory_load * memory_weight + 
                    bandwidth_load * bandwidth_weight)
            
            node_loads.append(avg_load)
        
        # 计算负载均衡度（负载方差越小越好）
        if node_loads:
            load_variance = np.var(node_loads)
            return max(0, 1.0 - load_variance * 10)
        else:
            return 0.0

    def _calculate_node_bandwidth_load(self, physical_node_idx, mapped_virtual_nodes, 
                                    mapping_action, bandwidth_action,
                                    physical_edges, physical_edge_features,
                                    virtual_edges, virtual_edge_features):
        """计算物理节点的带宽负载"""
        bandwidth_load = 0.0
        
        # 遍历所有虚拟链路
        for i, (src, dst) in enumerate(virtual_edges.T):
            if i >= len(bandwidth_action):
                break
                
            # 检查此虚拟链路是否涉及映射到此物理节点的虚拟节点
            src_physical = mapping_action[src]
            dst_physical = mapping_action[dst]
            
            if src_physical == physical_node_idx or dst_physical == physical_node_idx:
                # 获取分配的带宽
                bandwidth_level = bandwidth_action[i]
                allocated_bandwidth = self._get_allocated_bandwidth(
                    bandwidth_level, i, virtual_edges, virtual_edge_features
                )
                
                # 计算此链路的带宽负载贡献
                # 这里需要根据物理链路的实际带宽容量来归一化
                # 找到对应的物理链路
                physical_link_bandwidth = self._find_physical_link_bandwidth(
                    src_physical, dst_physical, physical_edges, physical_edge_features
                )
                
                if physical_link_bandwidth > 0:
                    # 归一化带宽负载到0-1范围
                    link_bandwidth_load = allocated_bandwidth / physical_link_bandwidth
                    bandwidth_load += link_bandwidth_load
        
        # 确保带宽负载不超过1.0
        bandwidth_load = min(bandwidth_load, 1.0)
        return bandwidth_load

    def _get_allocated_bandwidth(self, bandwidth_level, link_index, virtual_edges, virtual_edge_features):
        """
        根据带宽等级、链路索引和虚拟链路特征获取分配的带宽值
        
        Args:
            bandwidth_level: 带宽等级 (0 到 bandwidth_levels-1)
            link_index: 链路在虚拟边数组中的索引
            virtual_edges: 虚拟边数组 [2, num_virtual_edges]
            virtual_edge_features: 虚拟边特征数组 [num_virtual_edges, 2] (min_bandwidth, max_bandwidth)
            
        Returns:
            int: 对应的带宽值
        """
        if link_index >= virtual_edge_features.shape[0]:
            return 0
        
        # 获取该链路的最小和最大带宽需求
        min_bandwidth = virtual_edge_features[link_index][0]  # 最小带宽需求
        max_bandwidth = virtual_edge_features[link_index][1]  # 最大带宽需求
        
        # 计算带宽等级对应的实际带宽值
        # 使用线性插值在最小和最大带宽之间分配
        if self.bandwidth_levels > 1:
            # 计算每个等级对应的带宽值
            bandwidth_step = (max_bandwidth - min_bandwidth) / (self.bandwidth_levels - 1)
            allocated_bandwidth = min_bandwidth + (bandwidth_level * bandwidth_step)
        else:
            # 如果只有一个等级，使用最大带宽
            allocated_bandwidth = max_bandwidth
        
        return int(allocated_bandwidth)
    
    def _find_physical_link_bandwidth(self, src_physical, dst_physical, physical_edges, physical_edge_features):
        """找到两个物理节点之间链路的带宽容量"""
        for i, (src, dst) in enumerate(physical_edges.T):
            if (src == src_physical and dst == dst_physical) or \
            (src == dst_physical and dst == src_physical):
                return physical_edge_features[i][0]  # 返回总带宽容量
        
        # 如果没有找到直接连接，返回0
        return 0
    
    def calculate_bandwidth_satisfaction_score(self, mapping_action, bandwidth_action, physical_state, virtual_work):
        """计算带宽满意度分数"""
        try:
            virtual_edges = virtual_work['edges'].numpy()
            if len(bandwidth_action) == 0 or virtual_edges.size == 0:
                return 0.0
            
            edge_satisfaction_scores = []
            same_node_bandwidth_bonus = 0
            satisfied_links = 0
            
            for edge_idx in range(min(len(bandwidth_action), virtual_edges.shape[1])):
                src_virtual = virtual_edges[0, edge_idx]
                dst_virtual = virtual_edges[1, edge_idx]
                
                if src_virtual < len(mapping_action) and dst_virtual < len(mapping_action):
                    src_physical = mapping_action[src_virtual]
                    dst_physical = mapping_action[dst_virtual]
                    
                    # 检查物理连接
                    has_physical_connection = self._check_physical_connection(src_physical, dst_physical, physical_state)
                    
                    if has_physical_connection:
                        if src_physical == dst_physical:
                            # 同一物理节点映射
                            same_node_bandwidth_bonus += 1.0
                            satisfied_links += 1
                            edge_satisfaction_scores.append(1.0)
                            continue
                        
                        # 获取带宽等级
                        bandwidth_level = bandwidth_action[edge_idx]
                        if isinstance(bandwidth_level, torch.Tensor):
                            bandwidth_level = bandwidth_level.item()
                        
                        # 计算带宽满意度
                        if edge_idx < virtual_work['edge_features'].shape[0]:
                            min_bandwidth = virtual_work['edge_features'][edge_idx][0]
                            max_bandwidth = virtual_work['edge_features'][edge_idx][1]
                            
                            # 使用带宽等级计算满意度
                            allocated_bandwidth = (bandwidth_level + 1) * 10  # 简化的带宽计算
                            
                            if min_bandwidth <= allocated_bandwidth <= max_bandwidth:
                                if max_bandwidth > min_bandwidth:
                                    satisfaction = (allocated_bandwidth - min_bandwidth) / (max_bandwidth - min_bandwidth)
                                else:
                                    satisfaction = 1.0
                            elif allocated_bandwidth > max_bandwidth:
                                satisfaction = 1.0
                            else:
                                satisfaction = 0.0
                            
                            edge_satisfaction_scores.append(satisfaction)
                            satisfied_links += 1
                        else:
                            satisfaction = (bandwidth_level + 1) / self.bandwidth_levels
                            edge_satisfaction_scores.append(satisfaction)
                            satisfied_links += 1
                    else:
                        edge_satisfaction_scores.append(0.0)
                else:
                    edge_satisfaction_scores.append(0.0)
            
            # 计算最终满意度
            if satisfied_links > 0:
                avg_satisfaction = np.mean(edge_satisfaction_scores) if edge_satisfaction_scores else 0.0
                final_satisfaction = (avg_satisfaction + same_node_bandwidth_bonus) / satisfied_links
                return np.clip(final_satisfaction, 0, 1)
            else:
                return 0.0
                
        except Exception as e:
            print(f"⚠️  计算带宽满意度时出错: {e}")
            return 0.0
    
    def _check_physical_connection(self, src_physical, dst_physical, physical_state):
        """检查两个物理节点是否存在连接"""
        # 尝试不同的边信息键
        for key in ['physical_edge_index', 'edges', 'edge_index']:
            if key in physical_state:
                edge_index = physical_state[key]
                if isinstance(edge_index, torch.Tensor):
                    edge_index = edge_index.cpu().numpy()
                
                if len(edge_index.shape) == 2 and edge_index.shape[0] == 2:
                    src_nodes = edge_index[0]
                    dst_nodes = edge_index[1]
                    
                    has_forward = np.any((src_nodes == src_physical) & (dst_nodes == dst_physical))
                    has_backward = np.any((src_nodes == dst_physical) & (dst_nodes == src_physical))
                    
                    if has_forward or has_backward:
                        return True
        
        # 如果没有边信息，假设所有节点都连接
        return True
    
    def run_single_test(self, num_virtual_nodes: int):
        """运行单次测试"""
        print(f"🔄 测试 {num_virtual_nodes} 个虚拟节点...")
        
        ppo_load_scores = []
        ppo_bandwidth_scores = []
        random_load_scores = []
        random_bandwidth_scores = []
        
        for trial in range(self.num_trials):
            # 重置环境
            self.env.reset()
            
            # 获取状态
            physical_state = self.env.physical_state
            virtual_work = self.env.virtual_work
            
            # 确保虚拟节点数量符合要求
            while virtual_work['features'].shape[0] != num_virtual_nodes:
                self.env.reset()
                physical_state = self.env.physical_state
                virtual_work = self.env.virtual_work
            
            try:
                # PPO推理
                with torch.no_grad():
                    state = self.env._get_state()
                    
                    # 调整状态形状
                    adapted_state = self._adapt_state_for_ppo(state, num_virtual_nodes)
                    
                    # 执行PPO推理
                    mapping_action, bandwidth_action, _, _, _, _, _, _ = self.ppo_agent.select_actions(
                        adapted_state, temperature=0.1
                    )
                    
                    # 验证动作
                    mapping_action = self._validate_mapping_action(mapping_action, num_virtual_nodes, physical_state['features'].size(0))
                    bandwidth_action = self._validate_bandwidth_action(bandwidth_action, virtual_work['edges'].size(1))
                    
                    # 计算分数
                    # ppo_load_score = self.calculate_load_balancing_score(mapping_action, physical_state, virtual_work)
                    ppo_load_score = self.calculate_load_balancing_score_with_bandwidth(
                        mapping_action, bandwidth_action, physical_state, virtual_work
                    )
                    ppo_bandwidth_score = self.calculate_bandwidth_satisfaction_score(mapping_action, bandwidth_action, physical_state, virtual_work)
                    
                    ppo_load_scores.append(ppo_load_score)
                    ppo_bandwidth_scores.append(ppo_bandwidth_score)
                    
            except Exception as e:
                print(f"⚠️  PPO推理失败 (trial {trial}): {e}")
                ppo_load_scores.append(0.0)
                ppo_bandwidth_scores.append(0.0)
            
            try:
                # 随机算法
                mapping_action, bandwidth_action = self.random_algorithm.allocate(physical_state, virtual_work)
                
                # random_load_score = self.calculate_load_balancing_score(mapping_action, physical_state, virtual_work)
                random_load_score = self.calculate_load_balancing_score_with_bandwidth(
                    mapping_action, bandwidth_action, physical_state, virtual_work
                )
                random_bandwidth_score = self.calculate_bandwidth_satisfaction_score(mapping_action, bandwidth_action, physical_state, virtual_work)
                
                random_load_scores.append(random_load_score)
                random_bandwidth_scores.append(random_bandwidth_score)
                
            except Exception as e:
                print(f"⚠️  随机算法失败 (trial {trial}): {e}")
                random_load_scores.append(0.0)
                random_bandwidth_scores.append(0.0)
        
        # 计算统计结果
        result = {
            'num_virtual_nodes': num_virtual_nodes,
            'ppo': {
                'load_balancing': {
                    'scores': ppo_load_scores,
                    'mean': np.mean(ppo_load_scores),
                    'std': np.std(ppo_load_scores)
                },
                'bandwidth_satisfaction': {
                    'scores': ppo_bandwidth_scores,
                    'mean': np.mean(ppo_bandwidth_scores),
                    'std': np.std(ppo_bandwidth_scores)
                }
            },
            'random': {
                'load_balancing': {
                    'scores': random_load_scores,
                    'mean': np.mean(random_load_scores),
                    'std': np.std(random_load_scores)
                },
                'bandwidth_satisfaction': {
                    'scores': random_bandwidth_scores,
                    'mean': np.mean(random_bandwidth_scores),
                    'std': np.std(random_bandwidth_scores)
                }
            }
        }
        
        print(f"✅ 测试完成:")
        print(f"   PPO - 负载均衡: {result['ppo']['load_balancing']['mean']:.4f} ± {result['ppo']['load_balancing']['std']:.4f}")
        print(f"   PPO - 带宽满意度: {result['ppo']['bandwidth_satisfaction']['mean']:.4f} ± {result['ppo']['bandwidth_satisfaction']['std']:.4f}")
        print(f"   Random - 负载均衡: {result['random']['load_balancing']['mean']:.4f} ± {result['random']['load_balancing']['std']:.4f}")
        print(f"   Random - 带宽满意度: {result['random']['bandwidth_satisfaction']['mean']:.4f} ± {result['random']['bandwidth_satisfaction']['std']:.4f}")
        
        return result
    
    def _adapt_state_for_ppo(self, state, num_virtual_nodes):
        """调整状态以适配PPO模型"""
        adapted_state = {}
        
        # 物理节点特征
        adapted_state['physical_features'] = state['physical_features']
        
        # 虚拟节点特征
        adapted_state['virtual_features'] = state['virtual_features'][:num_virtual_nodes]
        
        # 物理边
        adapted_state['physical_edge_index'] = state['physical_edge_index']
        adapted_state['physical_edge_attr'] = state['physical_edge_attr']
        
        # 虚拟边
        if state['virtual_edge_index'].size(1) > 0:
            valid_edges = []
            for i in range(state['virtual_edge_index'].size(1)):
                src, dst = state['virtual_edge_index'][0, i], state['virtual_edge_index'][1, i]
                if src < num_virtual_nodes and dst < num_virtual_nodes:
                    valid_edges.append(i)
            
            if valid_edges:
                adapted_state['virtual_edge_index'] = state['virtual_edge_index'][:, valid_edges]
                adapted_state['virtual_edge_attr'] = state['virtual_edge_attr'][valid_edges]
            else:
                device = state['virtual_edge_index'].device
                dtype = state['virtual_edge_index'].dtype
                adapted_state['virtual_edge_index'] = torch.empty((2, 0), dtype=dtype, device=device)
                adapted_state['virtual_edge_attr'] = torch.empty((0, state['virtual_edge_attr'].size(1)), 
                                                              dtype=state['virtual_edge_attr'].dtype, device=device)
        else:
            adapted_state['virtual_edge_index'] = state['virtual_edge_index']
            adapted_state['virtual_edge_attr'] = state['virtual_edge_attr']
        
        return adapted_state
    
    def _validate_mapping_action(self, mapping_action, expected_virtual_nodes, max_physical_nodes):
        """验证映射动作"""
        if isinstance(mapping_action, torch.Tensor):
            mapping_action = mapping_action.cpu().numpy()
        
        if len(mapping_action) != expected_virtual_nodes:
            if len(mapping_action) > expected_virtual_nodes:
                mapping_action = mapping_action[:expected_virtual_nodes]
            else:
                padding_size = expected_virtual_nodes - len(mapping_action)
                padding = np.random.randint(0, max_physical_nodes, padding_size)
                mapping_action = np.concatenate([mapping_action, padding])
        
        mapping_action = np.clip(mapping_action, 0, max_physical_nodes - 1)
        return mapping_action
    
    def _validate_bandwidth_action(self, bandwidth_action, expected_edges):
        """验证带宽动作"""
        if isinstance(bandwidth_action, torch.Tensor):
            bandwidth_action = bandwidth_action.cpu().numpy()
        
        if len(bandwidth_action) != expected_edges:
            if len(bandwidth_action) > expected_edges:
                bandwidth_action = bandwidth_action[:expected_edges]
            else:
                padding_size = expected_edges - len(bandwidth_action)
                padding = np.random.randint(0, self.bandwidth_levels, padding_size)
                bandwidth_action = np.concatenate([bandwidth_action, padding])
        
        bandwidth_action = np.clip(bandwidth_action, 0, self.bandwidth_levels - 1)
        return bandwidth_action
    
    def run_comparison_test(self):
        """运行完整的对比测试"""
        print("🚀 开始PPO vs 随机算法对比测试")
        print("=" * 60)
        
        try:
            # 加载PPO模型
            model_config = self.load_ppo_model()
            
            # 初始化环境
            self.initialize_environment(model_config)
            
            # 运行测试
            all_results = []
            
            for num_virtual_nodes in self.virtual_node_counts:
                print(f"\n📊 测试配置: {num_virtual_nodes} 个虚拟节点")
                result = self.run_single_test(num_virtual_nodes)
                all_results.append(result)
                
                # 保存中间结果
                self.test_results['ppo']['load_balancing'].append(result['ppo']['load_balancing']['mean'])
                self.test_results['ppo']['bandwidth_satisfaction'].append(result['ppo']['bandwidth_satisfaction']['mean'])
                self.test_results['random']['load_balancing'].append(result['random']['load_balancing']['mean'])
                self.test_results['random']['bandwidth_satisfaction'].append(result['random']['bandwidth_satisfaction']['mean'])
            
            # 保存结果
            self._save_results(all_results)
            
            # 生成图表
            self._generate_plots()
            
            # 打印摘要
            self._print_summary()
            
            print("✅ 对比测试完成！")
            return all_results
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self, results):
        """保存测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.results_dir, f"ppo_vs_random_results_{timestamp}.json")
        
        # 转换numpy类型为Python原生类型
        serializable_results = []
        for result in results:
            serializable_result = {
                'num_virtual_nodes': result['num_virtual_nodes'],
                'seed': self.seed,  # 保存种子信息
                'ppo': {
                    'load_balancing': {
                        'scores': [float(score) for score in result['ppo']['load_balancing']['scores']],
                        'mean': float(result['ppo']['load_balancing']['mean']),
                        'std': float(result['ppo']['load_balancing']['std'])
                    },
                    'bandwidth_satisfaction': {
                        'scores': [float(score) for score in result['ppo']['bandwidth_satisfaction']['scores']],
                        'mean': float(result['ppo']['bandwidth_satisfaction']['mean']),
                        'std': float(result['ppo']['bandwidth_satisfaction']['std'])
                    }
                },
                'random': {
                    'load_balancing': {
                        'scores': [float(score) for score in result['random']['load_balancing']['scores']],
                        'mean': float(result['random']['load_balancing']['mean']),
                        'std': float(result['random']['load_balancing']['std'])
                    },
                    'bandwidth_satisfaction': {
                        'scores': [float(score) for score in result['random']['bandwidth_satisfaction']['scores']],
                        'mean': float(result['random']['bandwidth_satisfaction']['mean']),
                        'std': float(result['random']['bandwidth_satisfaction']['std'])
                    }
                }
            }
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存到: {results_file}")
    
    def _generate_plots(self):
        """生成对比图表"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 图1: 负载均衡度比较
        ax1.plot(self.virtual_node_counts, self.test_results['ppo']['load_balancing'], 
                'o-', label='PPO', linewidth=2, markersize=8, color='#2E86AB')
        ax1.plot(self.virtual_node_counts, self.test_results['random']['load_balancing'], 
                's-', label='Random', linewidth=2, markersize=8, color='#A23B72')
        
        ax1.set_xlabel('virtual nodes', fontsize=12)
        ax1.set_ylabel('load balancing score (higher is better)', fontsize=12)
        ax1.set_title('PPO vs Random: load balancing comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 图2: 带宽满意度比较
        ax2.plot(self.virtual_node_counts, self.test_results['ppo']['bandwidth_satisfaction'], 
                'o-', label='PPO', linewidth=2, markersize=8, color='#2E86AB')
        ax2.plot(self.virtual_node_counts, self.test_results['random']['bandwidth_satisfaction'], 
                's-', label='Random', linewidth=2, markersize=8, color='#A23B72')
        
        ax2.set_xlabel('virtual nodes', fontsize=12)
        ax2.set_ylabel('bandwidth satisfaction score (higher is better)', fontsize=12)
        ax2.set_title('PPO vs Random: bandwidth satisfaction comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 图3: 负载均衡度性能提升
        load_improvements = []
        for ppo_score, random_score in zip(self.test_results['ppo']['load_balancing'], self.test_results['random']['load_balancing']):
            if random_score > 0:
                improvement = ((ppo_score - random_score) / random_score) * 100
                load_improvements.append(improvement)
            else:
                load_improvements.append(0.0)
        
        bars1 = ax3.bar(self.virtual_node_counts, load_improvements, 
                       color=['#2E86AB' if x >= 0 else '#A23B72' for x in load_improvements],
                        alpha=0.7)
        
        ax3.set_xlabel('virtual nodes', fontsize=12)
        ax3.set_ylabel('performance improvement (%)', fontsize=12)
        ax3.set_title('Load balancing performance improvement\n(positive=PPO better, negative=Random better)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 图4: 带宽满意度性能提升
        bandwidth_improvements = []
        for ppo_score, random_score in zip(self.test_results['ppo']['bandwidth_satisfaction'], self.test_results['random']['bandwidth_satisfaction']):
            if random_score > 0:
                improvement = ((ppo_score - random_score) / random_score) * 100
                bandwidth_improvements.append(improvement)
            else:
                bandwidth_improvements.append(0.0)
        
        bars2 = ax4.bar(self.virtual_node_counts, bandwidth_improvements, 
                        color=['#2E86AB' if x >= 0 else '#A23B72' for x in bandwidth_improvements],
                         alpha=0.7)
        
        ax4.set_xlabel('virtual nodes', fontsize=12)
        ax4.set_ylabel('performance improvement (%)', fontsize=12)
        ax4.set_title('Bandwidth satisfaction performance improvement\n(positive=PPO better, negative=Random better)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(self.results_dir, f"ppo_vs_random_plots_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 图表已保存到: {plot_file}")
        
        plt.show()
    
    def _print_summary(self):
        """打印测试结果摘要"""
        print("\n📋 测试结果摘要")
        print("=" * 80)
        print(f"🔒 随机种子: {self.seed}")  # 显示种子信息
        print(f"{'虚拟节点数':<12} {'负载均衡度':<20} {'带宽满意度':<20} {'综合性能':<20}")
        print(f"{'':<12} {'PPO':<8} {'Random':<8} {'提升%':<4} {'PPO':<8} {'Random':<8} {'提升%':<4}")
        print("-" * 80)
        
        for i, (ppo_load, random_load, ppo_bw, random_bw) in enumerate(zip(
            self.test_results['ppo']['load_balancing'], 
            self.test_results['random']['load_balancing'],
            self.test_results['ppo']['bandwidth_satisfaction'],
            self.test_results['random']['bandwidth_satisfaction']
        )):
            virtual_nodes = self.virtual_node_counts[i]
            
            # 计算性能提升
            load_improvement = ((ppo_load - random_load) / random_load) * 100 if random_load > 0 else 0
            bw_improvement = ((ppo_bw - random_bw) / random_bw) * 100 if random_bw > 0 else 0
            
            # 综合性能
            overall_ppo = (ppo_load + ppo_bw) / 2
            overall_random = (random_load + random_bw) / 2
            overall_improvement = ((overall_ppo - overall_random) / overall_random) * 100 if overall_random > 0 else 0
            
            print(f"{virtual_nodes:<12} {ppo_load:<8.4f} {random_load:<8.4f} {load_improvement:<6.1f}% {ppo_bw:<8.4f} {random_bw:<8.4f} {bw_improvement:<6.1f}% {overall_improvement:<6.1f}%")
        
        # 总体统计
        avg_ppo_load = np.mean(self.test_results['ppo']['load_balancing'])
        avg_random_load = np.mean(self.test_results['random']['load_balancing'])
        avg_ppo_bw = np.mean(self.test_results['ppo']['bandwidth_satisfaction'])
        avg_random_bw = np.mean(self.test_results['random']['bandwidth_satisfaction'])
        
        overall_load_improvement = ((avg_ppo_load - avg_random_load) / avg_random_load) * 100 if avg_random_load > 0 else 0
        overall_bw_improvement = ((avg_ppo_bw - avg_random_bw) / avg_random_bw) * 100 if avg_random_bw > 0 else 0
        
        overall_ppo_avg = (avg_ppo_load + avg_ppo_bw) / 2
        overall_random_avg = (avg_random_load + avg_random_bw) / 2
        overall_improvement = ((overall_ppo_avg - overall_random_avg) / overall_random_avg) * 100 if overall_random_avg > 0 else 0
        
        print("-" * 80)
        print(f"{'总体平均':<12} {avg_ppo_load:<8.4f} {avg_random_load:<8.4f} {overall_load_improvement:<6.1f}% {avg_ppo_bw:<8.4f} {avg_random_bw:<8.4f} {overall_bw_improvement:<6.1f}% {overall_improvement:<6.1f}%")
        
        print("\n📊 性能分析:")
        if overall_load_improvement > 0:
            print(f"   🟢 负载均衡度: PPO比随机算法平均提升 {overall_load_improvement:.1f}%")
        else:
            print(f"   🔴 负载均衡度: PPO比随机算法平均下降 {abs(overall_load_improvement):.1f}%")
            
        if overall_bw_improvement > 0:
            print(f"   🟢 带宽满意度: PPO比随机算法平均提升 {overall_bw_improvement:.1f}%")
        else:
            print(f"   🔴 带宽满意度: PPO比随机算法平均下降 {abs(overall_bw_improvement):.1f}%")
            
        if overall_improvement > 0:
            print(f"   🟢 综合性能: PPO比随机算法平均提升 {overall_improvement:.1f}%")
        else:
            print(f"   🔴 综合性能: PPO比随机算法平均下降 {abs(overall_improvement):.1f}%")

def main():
    """主函数"""
    print("🔬 PPO vs 随机算法负载均衡对比测试")
    print("=" * 60)
    
    # 检查目录
    model_dir = "models"
    stats_dir = "stats"
    
    if not os.path.exists(model_dir):
        print(f"❌ 找不到模型目录: {model_dir}")
        return
    
    if not os.path.exists(stats_dir):
        print(f"❌ 找不到统计目录: {stats_dir}")
        return
    
    # 查找可用会话
    sessions = []
    for item in os.listdir(stats_dir):
        session_path = os.path.join(stats_dir, item)
        config_path = os.path.join(session_path, f"session_config_{item}.json")
        if os.path.isdir(session_path) and os.path.exists(config_path):
            model_session_path = os.path.join(model_dir, item)
            if os.path.exists(model_session_path):
                model_files = [f for f in os.listdir(model_session_path) if f.startswith(f"ppo_model_{item}_episode_") and f.endswith(".pth")]
                if model_files:
                    sessions.append(item)
    
    if not sessions:
        print("❌ 没有找到可用的训练会话")
        return
    
    print("📁 可用的训练会话:")
    for i, session in enumerate(sessions):
        print(f"  {i+1}. {session}")
    
    # 选择会话
    if len(sessions) == 1:
        selected_session = sessions[0]
        print(f"✅ 自动选择唯一可用会话: {selected_session}")
    else:
        try:
            choice = int(input(f"\n请选择会话 (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                selected_session = sessions[choice]
            else:
                print("❌ 无效选择")
                return
        except ValueError:
            print("❌ 请输入有效数字")
            return
    # 设置随机种子
    seed = 42  # 你可以修改这个值来获得不同的随机结果
    print(f"🔒 使用随机种子: {seed}")

    # 创建测试器并运行测试
    tester = PPOvsRandomTester(session_name=selected_session, num_trials=30, seed=seed)
    results = tester.run_comparison_test()

if __name__ == "__main__":
    main()
