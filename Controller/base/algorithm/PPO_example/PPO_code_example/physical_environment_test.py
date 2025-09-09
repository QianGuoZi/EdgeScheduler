#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime
import argparse
try:
    from scipy.interpolate import make_interp_spline
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy不可用，将使用简单的移动平均进行曲线平滑")

from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler, create_sample_topology, create_sample_virtual_work

PPO_AVAILABLE = True
# 尝试导入PPO相关模块
from two_stage_actor_design import TwoStagePPOAgent

class RandomAlgorithm:
    """随机调度算法"""
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
    
    def schedule(self, virtual_work: VirtualWork) -> Tuple[Dict[int, int], Dict[Tuple[int, int], float]]:
        """
        随机调度虚拟工作
        
        Args:
            virtual_work: 虚拟工作对象
            
        Returns:
            Tuple[Dict[int, int], Dict[Tuple[int, int], float]]: (节点映射, 带宽分配)
        """
        scheduler = NetworkScheduler(self.topology)
        scheduler.add_virtual_work(virtual_work)
        
        # 随机映射虚拟节点到物理节点
        node_mapping = {}
        for virtual_node in range(virtual_work.num_nodes):
            # 随机选择一个物理节点
            physical_node = random.randint(0, self.topology.num_nodes - 1)
            node_mapping[virtual_node] = physical_node
            
            # 尝试调度节点
            if not scheduler.schedule_node(virtual_node, physical_node):
                # 如果调度失败，尝试其他节点
                available_nodes = list(range(self.topology.num_nodes))
                random.shuffle(available_nodes)
                
                for node in available_nodes:
                    if scheduler.schedule_node(virtual_node, node):
                        node_mapping[virtual_node] = node
                        break
        
        # 随机分配带宽
        bandwidth_allocation = {}
        for link_req in virtual_work.link_requirements:
            from_node = link_req['from']
            to_node = link_req['to']
            
            if from_node in node_mapping and to_node in node_mapping:
                # 随机选择带宽值（在最小和最大需求之间）
                min_bw_1_to_2 = link_req['min_bandwidth_1_to_2']
                max_bw_1_to_2 = link_req['max_bandwidth_1_to_2']
                min_bw_2_to_1 = link_req['min_bandwidth_2_to_1']
                max_bw_2_to_1 = link_req['max_bandwidth_2_to_1']
                
                # 随机分配带宽
                if max_bw_1_to_2 > min_bw_1_to_2:
                    bandwidth_1_to_2 = random.uniform(min_bw_1_to_2, max_bw_1_to_2)
                else:
                    bandwidth_1_to_2 = min_bw_1_to_2
                
                if max_bw_2_to_1 > min_bw_2_to_1:
                    bandwidth_2_to_1 = random.uniform(min_bw_2_to_1, max_bw_2_to_1)
                else:
                    bandwidth_2_to_1 = min_bw_2_to_1
                
                # 尝试分配带宽
                if scheduler.allocate_bandwidth(from_node, to_node, bandwidth_1_to_2):
                    bandwidth_allocation[(from_node, to_node)] = bandwidth_1_to_2
                
                if scheduler.allocate_bandwidth(to_node, from_node, bandwidth_2_to_1):
                    bandwidth_allocation[(to_node, from_node)] = bandwidth_2_to_1
        
        return node_mapping, bandwidth_allocation



class PhysicalEnvironmentTester:
    """物理环境测试器"""
    
    def __init__(self, 
                 session_name: str = None,
                 model_dir: str = "models",
                 stats_dir: str = "stats",
                 results_dir: str = "physical_test_results",
                 num_trials: int = 10,
                 smooth_method: str = 'gaussian',
                 smooth_window: int = 3,
                 seed: int = None):
        
        self.session_name = session_name
        self.model_dir = model_dir
        self.stats_dir = stats_dir
        self.results_dir = results_dir
        self.num_trials = num_trials
        self.smooth_method = smooth_method
        self.smooth_window = smooth_window

        # 设置随机种子
        self.seed = seed if seed is not None else random.randint(1, 10000)
        self._set_random_seed(self.seed)
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 测试结果
        self.test_results = {
            'random': {'load_balancing': [], 'bandwidth_satisfaction': []}
        }
        
        # 如果PPO可用且有session_name，添加PPO结果
        if PPO_AVAILABLE and session_name:
            self.test_results['ppo'] = {'load_balancing': [], 'bandwidth_satisfaction': []}
        
        # 算法实例
        self.random_algorithm = None
        self.ppo_agent = None
        
        # 统计信息
        self.stats = {
            'total_works': 0,
            'successful_random': 0,
            'failed_random': 0
        }
        
        # 如果PPO可用且有session_name，添加PPO统计
        if PPO_AVAILABLE and session_name:
            self.stats['successful_ppo'] = 0
            self.stats['failed_ppo'] = 0
        
        # PPO相关配置
        self.bandwidth_levels = 10
        self.model_config = None
    
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
        
        print(f"物理环境测试器随机种子设置完成: {seed}")

    def run_test(self, 
                 num_physical_nodes: int = 10,
                 num_virtual_works: int = 20,
                 physical_cpu_range: Tuple[int, int] = (50, 200),
                 physical_memory_range: Tuple[int, int] = (100, 400),
                 physical_bandwidth_range: Tuple[int, int] = (100, 1000),
                 virtual_cpu_range: Tuple[int, int] = (10, 50),
                 virtual_memory_range: Tuple[int, int] = (20, 100),
                 virtual_bandwidth_range: Tuple[int, int] = (10, 200),
                 physical_connectivity_prob: float = 0.3,
                 virtual_connectivity_prob: float = 0.4,
                 virtual_nodes_range: Tuple[int, int] = (3, 8)):
        """
        运行测试
        
        Args:
            num_physical_nodes: 物理节点数量
            num_virtual_works: 虚拟工作数量
            physical_cpu_range: 物理节点CPU范围
            physical_memory_range: 物理节点内存范围
            physical_bandwidth_range: 物理链路带宽范围
            virtual_cpu_range: 虚拟节点CPU范围
            virtual_memory_range: 虚拟节点内存范围
            virtual_bandwidth_range: 虚拟链路带宽范围
            physical_connectivity_prob: 物理网络连接概率
            virtual_connectivity_prob: 虚拟网络连接概率
            virtual_nodes_range: 虚拟节点数量范围
        """
        print(f"🚀 开始物理环境测试...")
        print(f"📊 配置参数:")
        print(f"   - 物理节点数: {num_physical_nodes}")
        print(f"   - 虚拟工作数: {num_virtual_works}")
        print(f"   - 虚拟节点范围: {virtual_nodes_range[0]}-{virtual_nodes_range[1]}")
        print(f"   - 测试次数: {self.num_trials}")
        
        # 创建物理环境
        topology = create_sample_topology(
            num_nodes=num_physical_nodes,
            cpu_range=physical_cpu_range,
            memory_range=physical_memory_range,
            bandwidth_range=physical_bandwidth_range,
            connectivity_prob=physical_connectivity_prob,
            seed=self.seed
        )
        
        # 初始化算法
        self.random_algorithm = RandomAlgorithm(topology)
        
        # 如果PPO可用，尝试加载PPO模型
        if PPO_AVAILABLE:
            # 如果没有提供session_name，让用户选择
            if not self.session_name:
                print("🤖 PPO模块可用，但未指定会话名称")
                choice = input("是否要选择PPO训练会话进行测试？(y/n): ").strip().lower()
                if choice in ['y', 'yes', '是']:
                    self.session_name = self.select_session()
                    if not self.session_name:
                        print("⚠️  未选择会话，将只使用随机算法和Two Stage算法")
                        self.ppo_agent = None
            
            # 如果有session_name，尝试加载PPO模型
            if self.session_name:
                try:
                    self.load_ppo_model()
                    print("✅ PPO模型加载成功")
                    
                    # 确保PPO结果字典存在
                    if 'ppo' not in self.test_results:
                        self.test_results['ppo'] = {'load_balancing': [], 'bandwidth_satisfaction': []}
                    if 'successful_ppo' not in self.stats:
                        self.stats['successful_ppo'] = 0
                        self.stats['failed_ppo'] = 0
                        
                except Exception as e:
                    print(f"⚠️  PPO模型加载失败: {e}")
                    print("   将只使用随机算法和Two Stage算法")
                    self.ppo_agent = None
                    self.session_name = None
        
        # 运行测试
        for trial in range(self.num_trials):
            print(f"\n🔄 测试轮次 {trial + 1}/{self.num_trials}")
            
            # 重置物理环境（保持拓扑结构，重置资源使用量）
            self._reset_environment(topology)
            
            # 生成虚拟工作序列
            virtual_works = []
            for i in range(num_virtual_works):
                num_vnodes = random.randint(virtual_nodes_range[0], virtual_nodes_range[1])
                virtual_work = create_sample_virtual_work(
                    num_nodes=num_vnodes,
                    cpu_range=virtual_cpu_range,
                    memory_range=virtual_memory_range,
                    bandwidth_range=virtual_bandwidth_range,
                    connectivity_prob=virtual_connectivity_prob
                )
                virtual_works.append(virtual_work)
            
            # 逐个处理虚拟工作
            for work_idx, virtual_work in enumerate(virtual_works):
                print(f"   📝 处理虚拟工作 {work_idx + 1}/{num_virtual_works} (节点数: {virtual_work.num_nodes})")
                
                # 随机算法调度
                try:
                    # 为随机算法创建干净的环境副本
                    random_topology = self._copy_topology(topology)
                    random_mapping, random_bandwidth = self.random_algorithm.schedule(virtual_work)
                    # 在随机算法专属拓扑上执行部署更新（资源扣减）
                    self._update_environment_resources(virtual_work, random_mapping, random_bandwidth, random_topology)
                    # 基于已更新资源的拓扑进行评估
                    random_score = self._evaluate_scheduling(virtual_work, random_mapping, random_bandwidth, random_topology)
                    self.test_results['random']['load_balancing'].append(random_score['load_balancing'])
                    self.test_results['random']['bandwidth_satisfaction'].append(random_score['bandwidth_satisfaction'])
                    self.stats['successful_random'] += 1
                    print(f"     📊 随机算法评分: 负载均衡={random_score['load_balancing']:.4f}, 带宽满足度={random_score['bandwidth_satisfaction']:.4f}")
                except Exception as e:
                    print(f"     ⚠️  随机算法失败: {e}")
                    self.test_results['random']['load_balancing'].append(0.0)
                    self.test_results['random']['bandwidth_satisfaction'].append(0.0)
                    self.stats['failed_random'] += 1
                
                # PPO算法调度（如果可用）
                if PPO_AVAILABLE and self.ppo_agent is not None and 'ppo' in self.test_results:
                    try:
                        print(f"     🤖 执行PPO算法调度...")
                        # 为PPO算法创建干净的环境副本
                        ppo_topology = self._copy_topology(topology)
                        ppo_mapping, ppo_bandwidth = self._run_ppo_scheduling(virtual_work, ppo_topology)
                        # 在PPO专属拓扑上执行部署更新（资源扣减），与随机算法一致
                        self._update_environment_resources(virtual_work, ppo_mapping, ppo_bandwidth, ppo_topology)
                        print(f"     ✅ PPO调度完成: 节点映射={len(ppo_mapping)}, 带宽分配={len(ppo_bandwidth)}")
                        # 基于已更新资源的拓扑进行评估
                        ppo_score = self._evaluate_scheduling(virtual_work, ppo_mapping, ppo_bandwidth, ppo_topology)
                        self.test_results['ppo']['load_balancing'].append(ppo_score['load_balancing'])
                        self.test_results['ppo']['bandwidth_satisfaction'].append(ppo_score['bandwidth_satisfaction'])
                        self.stats['successful_ppo'] += 1
                        print(f"     📊 PPO评分: 负载均衡={ppo_score['load_balancing']:.4f}, 带宽满足度={ppo_score['bandwidth_satisfaction']:.4f}")
                    except Exception as e:
                        print(f"     ⚠️  PPO算法失败: {e}")
                        import traceback
                        traceback.print_exc()
                        self.test_results['ppo']['load_balancing'].append(0.0)
                        self.test_results['ppo']['bandwidth_satisfaction'].append(0.0)
                        self.stats['failed_ppo'] += 1
                
                self.stats['total_works'] += 1
                
                # 不再需要更新物理环境资源状态，因为每个算法使用独立的环境副本
        
        # 保存结果
        self._save_results()
        
        # 生成报告
        self._generate_report()
    
    def _reset_environment(self, topology: NetworkTopology):
        """重置物理环境资源使用量"""
        # 重置节点资源使用量
        for node_id in topology.node_resources:
            topology.node_resources[node_id]['used_cpu'] = 0
            topology.node_resources[node_id]['used_memory'] = 0
        
        # 重置链路带宽使用量
        for link_key in topology.links:
            topology.links[link_key]['used_bandwidth'] = 0
    
    def _copy_topology(self, topology: NetworkTopology) -> NetworkTopology:
        """创建拓扑的副本，用于独立算法测试"""
        # 创建新的拓扑对象
        new_topology = NetworkTopology(topology.num_nodes)
        
        # 复制节点资源
        for node_id, resources in topology.node_resources.items():
            new_topology.node_resources[node_id] = {
                'cpu': resources['cpu'],
                'memory': resources['memory'],
                'used_cpu': 0,  # 重置为0
                'used_memory': 0  # 重置为0
            }
        
        # 复制链路
        for link_key, link in topology.links.items():
            new_topology.links[link_key] = {
                'bandwidth': link['bandwidth'],
                'used_bandwidth': 0  # 重置为0
            }
        
        return new_topology
    
    def _smooth_curve(self, x_data: List[int], y_data: List[float], 
                      method: str = 'gaussian', window_size: int = 3) -> Tuple[List[float], List[float]]:
        """
        平滑曲线数据
        
        Args:
            x_data: x轴数据
            y_data: y轴数据
            method: 平滑方法 ('gaussian', 'spline', 'moving_average')
            window_size: 移动平均窗口大小
            
        Returns:
            Tuple[List[float], List[float]]: 平滑后的x和y数据
        """
        if len(y_data) < 3:
            return x_data, y_data
        
        y_smooth = y_data.copy()
        
        if method == 'gaussian' and SCIPY_AVAILABLE:
            # 高斯滤波平滑
            y_smooth = gaussian_filter1d(y_data, sigma=0.8).tolist()
        elif method == 'spline' and SCIPY_AVAILABLE and len(y_data) >= 4:
            # 样条插值平滑
            try:
                # 创建更密集的x轴点
                x_dense = np.linspace(min(x_data), max(x_data), len(x_data) * 3)
                # 使用样条插值
                spline = make_interp_spline(x_data, y_data, k=min(3, len(y_data)-1))
                y_dense = spline(x_dense)
                # 重新采样到原始长度
                indices = np.linspace(0, len(y_dense)-1, len(y_data), dtype=int)
                y_smooth = y_dense[indices].tolist()
            except Exception as e:
                print(f"样条插值失败，回退到移动平均: {e}")
                y_smooth = self._moving_average(y_data, window_size)
        else:
            # 移动平均平滑
            y_smooth = self._moving_average(y_data, window_size)
        
        return x_data, y_smooth
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """简单的移动平均平滑"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window_data = data[start:end]
            smoothed.append(sum(window_data) / len(window_data))
        
        return smoothed
    
    def _update_environment_resources(self, virtual_work: VirtualWork, 
                                   node_mapping: Dict[int, int],
                                   bandwidth_allocation: Dict[Tuple[int, int], float],
                                   topology: NetworkTopology) -> bool:
        """更新物理环境资源状态"""
        # 更新节点资源
        for virtual_node, physical_node in node_mapping.items():
            if virtual_node in virtual_work.node_requirements:
                req = virtual_work.node_requirements[virtual_node]
                success = topology.allocate_node_resources(physical_node, req['cpu'], req['memory'])
                if not success:
                    print(f"警告：物理节点 {physical_node} 资源不足，无法分配虚拟节点 {virtual_node} 的资源")
                    return False
        
        # 更新链路带宽
        for (from_node, to_node), bandwidth in bandwidth_allocation.items():
            if from_node in node_mapping and to_node in node_mapping:
                physical_from = node_mapping[from_node]
                physical_to = node_mapping[to_node]
                
                if physical_from != physical_to:
                    path = topology.get_shortest_path(physical_from, physical_to)
                    if path:
                        success = topology.allocate_bandwidth(path, bandwidth)
                        if not success:
                            print(f"警告：路径 {path} 上的带宽分配失败")
                            return False
        
        return True
    
    def _evaluate_scheduling(self, virtual_work: VirtualWork, 
                           node_mapping: Dict[int, int],
                           bandwidth_allocation: Dict[Tuple[int, int], float],
                           topology: NetworkTopology) -> Dict[str, float]:
        """评估调度结果"""
        # 计算负载均衡分数
        load_balancing_score = self._calculate_load_balancing_score(topology)
        
        # 计算带宽满足度分数
        bandwidth_satisfaction_score = self._calculate_bandwidth_satisfaction_score(
            virtual_work, node_mapping, bandwidth_allocation
        )
        
        return {
            'load_balancing': load_balancing_score,
            'bandwidth_satisfaction': bandwidth_satisfaction_score
        }
    
    def _calculate_load_balancing_score(self, topology: NetworkTopology) -> float:
        """计算负载均衡分数"""
        # CPU负载均衡
        cpu_utilizations = []
        memory_utilizations = []
        bandwidth_utilizations = []
        
        for node_id in range(topology.num_nodes):
            if node_id in topology.node_resources:
                available = topology.get_available_resources(node_id)
                total_cpu = topology.node_resources[node_id]['cpu']
                total_memory = topology.node_resources[node_id]['memory']
                
                cpu_utilizations.append(1 - available['cpu'] / total_cpu)
                memory_utilizations.append(1 - available['memory'] / total_memory)
        
        # 网络带宽负载均衡
        for node1 in range(topology.num_nodes):
            for node2 in range(node1 + 1, topology.num_nodes):
                link_key = (node1, node2)
                if link_key in topology.links:
                    link = topology.links[link_key]
                    if link['bandwidth'] > 0:
                        utilization = link['used_bandwidth'] / link['bandwidth']
                        bandwidth_utilizations.append(utilization)
        
        # 计算标准差（越小越好）
        cpu_balance = 1 - np.std(cpu_utilizations) if cpu_utilizations else 1.0
        memory_balance = 1 - np.std(memory_utilizations) if memory_utilizations else 1.0
        bandwidth_balance = 1 - np.std(bandwidth_utilizations) if bandwidth_utilizations else 1.0
        
        # 确保分数在0-1范围内
        cpu_balance = np.clip(cpu_balance, 0, 1)
        memory_balance = np.clip(memory_balance, 0, 1)
        bandwidth_balance = np.clip(bandwidth_balance, 0, 1)
        
        # 综合负载均衡分数
        load_balancing_score = (0.4 * cpu_balance + 0.4 * memory_balance + 0.2 * bandwidth_balance)
        
        return load_balancing_score
    
    def _calculate_bandwidth_satisfaction_score(self, virtual_work: VirtualWork,
                                             node_mapping: Dict[int, int],
                                             bandwidth_allocation: Dict[Tuple[int, int], float]) -> float:
        """计算带宽满足度分数"""
        if not virtual_work.link_requirements:
            return 1.0
        
        total_satisfaction = 0
        satisfied_links = 0
        
        for link_req in virtual_work.link_requirements:
            from_node = link_req['from']
            to_node = link_req['to']
            
            if from_node in node_mapping and to_node in node_mapping:
                # 如果映射到同一物理节点，带宽需求为0，给予满分
                if node_mapping[from_node] == node_mapping[to_node]:
                    total_satisfaction += 1.0
                    satisfied_links += 1
                    continue
                
                # 检查正向链路
                allocated_1_to_2 = bandwidth_allocation.get((from_node, to_node), 0)
                min_req_1_to_2 = link_req['min_bandwidth_1_to_2']
                max_req_1_to_2 = link_req['max_bandwidth_1_to_2']
                
                # 检查反向链路
                allocated_2_to_1 = bandwidth_allocation.get((to_node, from_node), 0)
                min_req_2_to_1 = link_req['min_bandwidth_2_to_1']
                max_req_2_to_1 = link_req['max_bandwidth_2_to_1']
                
                # 计算满足度
                satisfaction_1_to_2 = self._calculate_bandwidth_satisfaction(
                    allocated_1_to_2, min_req_1_to_2, max_req_1_to_2
                )
                satisfaction_2_to_1 = self._calculate_bandwidth_satisfaction(
                    allocated_2_to_1, min_req_2_to_1, max_req_2_to_1
                )
                
                # 取两个方向的平均满足度
                avg_satisfaction = (satisfaction_1_to_2 + satisfaction_2_to_1) / 2
                total_satisfaction += avg_satisfaction
                satisfied_links += 1
        
        if satisfied_links == 0:
            return 0.0
        
        return total_satisfaction / satisfied_links
    
    def load_ppo_model(self):
        """加载训练好的PPO模型"""
        if not PPO_AVAILABLE:
            raise ImportError("PPO模块不可用")
        
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
        self.model_config = self._detect_model_config(checkpoint)
        
        # 创建智能体
        self.ppo_agent = TwoStagePPOAgent(
            physical_node_dim=self.model_config['physical_node_dim'],
            virtual_node_dim=self.model_config['virtual_node_dim'],
            max_physical_nodes=self.model_config['num_physical_nodes'],
            max_virtual_nodes=self.model_config['max_virtual_nodes'],
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
        print(f"   物理节点数: {self.model_config['num_physical_nodes']}")
        print(f"   最大虚拟节点数: {self.model_config['max_virtual_nodes']}")
        
        return self.model_config
    
    def select_session(self):
        """选择训练会话"""
        print("🔍 查找可用的训练会话...")
        
        # 检查stats目录
        if not os.path.exists(self.stats_dir):
            print(f"❌ 统计目录不存在: {self.stats_dir}")
            return None
        
        # 查找所有会话目录
        sessions = []
        for item in os.listdir(self.stats_dir):
            session_path = os.path.join(self.stats_dir, item)
            if os.path.isdir(session_path):
                # 检查是否有配置文件
                config_file = os.path.join(session_path, f"session_config_{item}.json")
                if os.path.exists(config_file):
                    # 检查是否有对应的模型目录
                    model_path = os.path.join(self.model_dir, item)
                    if os.path.exists(model_path):
                        # 检查是否有模型文件
                        model_files = [f for f in os.listdir(model_path) if f.startswith(f"ppo_model_{item}_episode_") and f.endswith(".pth")]
                        if model_files:
                            sessions.append(item)
        
        if not sessions:
            print("❌ 未找到可用的训练会话")
            return None
        
        print(f"📁 找到 {len(sessions)} 个可用会话:")
        for i, session in enumerate(sessions):
            print(f"   {i+1}. {session}")
        
        # 让用户选择
        while True:
            try:
                choice = input(f"\n请选择会话 (1-{len(sessions)}) 或输入会话名称: ").strip()
                
                # 如果输入的是数字
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(sessions):
                        selected_session = sessions[idx]
                        break
                    else:
                        print(f"❌ 无效选择，请输入 1-{len(sessions)} 之间的数字")
                        continue
                
                # 如果输入的是会话名称
                if choice in sessions:
                    selected_session = choice
                    break
                else:
                    print(f"❌ 未找到会话: {choice}")
                    print(f"可用会话: {', '.join(sessions)}")
                    continue
                    
            except KeyboardInterrupt:
                print("\n❌ 用户取消选择")
                return None
            except Exception as e:
                print(f"❌ 输入错误: {e}")
                continue
        
        print(f"✅ 选择会话: {selected_session}")
        return selected_session
    
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
        
        config = {
            'num_physical_nodes': num_physical_nodes,
            'physical_node_dim': physical_node_dim,
            'virtual_node_dim': virtual_node_dim,
            'max_virtual_nodes': max_virtual_nodes
        }
        
        print(f"📊 检测到的配置: {config}")
        return config
    
    def _run_ppo_scheduling(self, virtual_work: VirtualWork, topology: NetworkTopology) -> Tuple[Dict[int, int], Dict[Tuple[int, int], float]]:
        """运行PPO算法进行调度"""
        if not PPO_AVAILABLE or self.ppo_agent is None:
            raise RuntimeError("PPO智能体不可用")
        
        # 创建环境状态
        physical_state = self._create_physical_state(topology)
        virtual_state = self._create_virtual_state(virtual_work)
        
        # 执行PPO推理
        with torch.no_grad():
            # 调整状态形状
            adapted_physical_state = self._adapt_physical_state_for_ppo(physical_state)
            adapted_virtual_state = self._adapt_virtual_state_for_ppo(virtual_state)
            
            # 创建PPO期望的状态格式
            ppo_state = {
                'physical_features': adapted_physical_state['features'],
                'physical_edge_index': adapted_physical_state['edge_index'],
                'physical_edge_attr': adapted_physical_state['edge_attr'],
                'virtual_features': adapted_virtual_state['features'],
                'virtual_edge_index': adapted_virtual_state['edge_index'],
                'virtual_edge_attr': adapted_virtual_state['edge_attr']
            }
            
            # 执行PPO推理
            mapping_action, bandwidth_action, _, _, _, _, _, _ = self.ppo_agent.select_actions(
                ppo_state, temperature=0.1
            )
            
            # 验证动作
            mapping_action = self._validate_mapping_action(mapping_action, virtual_work.num_nodes, topology.num_nodes)
            bandwidth_action = self._validate_bandwidth_action(bandwidth_action, len(virtual_work.link_requirements))
        
        # 转换为网络调度器格式
        node_mapping = {}
        bandwidth_allocation = {}
        
        # 处理节点映射
        for i, physical_node in enumerate(mapping_action):
            if i < virtual_work.num_nodes:
                node_mapping[i] = physical_node
        
        # 处理带宽分配
        for i, bandwidth_level in enumerate(bandwidth_action):
            if i < len(virtual_work.link_requirements):
                link_req = virtual_work.link_requirements[i]
                from_node = link_req['from']
                to_node = link_req['to']
                
                # 将带宽等级转换为实际带宽值
                bandwidth_value = self._bandwidth_level_to_value(bandwidth_level, link_req)
                
                if from_node in node_mapping and to_node in node_mapping:
                    bandwidth_allocation[(from_node, to_node)] = bandwidth_value
                    bandwidth_allocation[(to_node, from_node)] = bandwidth_value
        
        return node_mapping, bandwidth_allocation
    
    def _create_physical_state(self, topology: NetworkTopology):
        """创建物理状态表示"""
        # 创建物理节点特征矩阵
        physical_features = []
        for node_id in range(topology.num_nodes):
            if node_id in topology.node_resources:
                resources = topology.node_resources[node_id]
                # 归一化资源特征
                cpu_usage = 1 - (resources['cpu'] - resources['used_cpu']) / resources['cpu']
                memory_usage = 1 - (resources['memory'] - resources['used_memory']) / resources['memory']
                physical_features.append([cpu_usage, memory_usage, 0.0])  # 添加一个占位符特征
            else:
                physical_features.append([0.0, 0.0, 0.0])
        
        # 创建物理边特征矩阵
        physical_edges = []
        physical_edge_attr = []
        for node1 in range(topology.num_nodes):
            for node2 in range(node1 + 1, topology.num_nodes):
                link_key = (node1, node2)
                if link_key in topology.links:
                    link = topology.links[link_key]
                    bandwidth_usage = link['used_bandwidth'] / link['bandwidth'] if link['bandwidth'] > 0 else 0
                    physical_edges.append([int(node1), int(node2)])
                    physical_edge_attr.append([bandwidth_usage])
        
        return {
            'features': torch.tensor(physical_features, dtype=torch.float32),
            'edge_index': torch.tensor(physical_edges, dtype=torch.long).t() if physical_edges else torch.empty((2, 0), dtype=torch.long),
            'edge_attr': torch.tensor(physical_edge_attr, dtype=torch.float32) if physical_edge_attr else torch.empty((0, 1), dtype=torch.float32)
        }
    
    def _create_virtual_state(self, virtual_work: VirtualWork):
        """创建虚拟工作状态表示"""
        # 创建虚拟节点特征矩阵
        virtual_features = []
        for node_id in range(virtual_work.num_nodes):
            if node_id in virtual_work.node_requirements:
                req = virtual_work.node_requirements[node_id]
                # 归一化需求特征
                cpu_req = req['cpu'] / 100.0  # 假设最大CPU为100
                memory_req = req['memory'] / 200.0  # 假设最大内存为200
                virtual_features.append([cpu_req, memory_req, 0.0])  # 添加一个占位符特征
            else:
                virtual_features.append([0.0, 0.0, 0.0])
        
        # 创建虚拟边特征矩阵
        virtual_edges = []
        virtual_edge_attr = []
        for link_req in virtual_work.link_requirements:
            from_node = link_req['from']
            to_node = link_req['to']
            # 归一化带宽需求 - 约束管理器期望 [min_bandwidth, max_bandwidth] 格式
            min_bandwidth = link_req['min_bandwidth_1_to_2'] / 100.0  # 假设最大带宽为100
            max_bandwidth = link_req['max_bandwidth_1_to_2'] / 100.0  # 假设最大带宽为100
            virtual_edges.append([int(from_node), int(to_node)])
            virtual_edge_attr.append([min_bandwidth, max_bandwidth])
        
        return {
            'features': torch.tensor(virtual_features, dtype=torch.float32),
            'edge_index': torch.tensor(virtual_edges, dtype=torch.long).t() if virtual_edges else torch.empty((2, 0), dtype=torch.long),
            'edge_attr': torch.tensor(virtual_edge_attr, dtype=torch.float32) if virtual_edge_attr else torch.empty((0, 1), dtype=torch.float32)
        }
    
    def _adapt_physical_state_for_ppo(self, physical_state):
        """调整物理状态以适应PPO模型"""
        # 确保特征维度匹配
        features = physical_state['features']
        if features.shape[1] < self.model_config['physical_node_dim']:
            # 填充到所需维度
            padding = torch.zeros(features.shape[0], self.model_config['physical_node_dim'] - features.shape[1])
            features = torch.cat([features, padding], dim=1)
        elif features.shape[1] > self.model_config['physical_node_dim']:
            # 截断到所需维度
            features = features[:, :self.model_config['physical_node_dim']]
        
        return {
            'features': features,
            'edge_index': physical_state['edge_index'],
            'edge_attr': physical_state['edge_attr']
        }
    
    def _adapt_virtual_state_for_ppo(self, virtual_state):
        """调整虚拟状态以适应PPO模型"""
        # 确保特征维度匹配
        features = virtual_state['features']
        if features.shape[1] < self.model_config['virtual_node_dim']:
            # 填充到所需维度
            padding = torch.zeros(features.shape[0], self.model_config['virtual_node_dim'] - features.shape[1])
            features = torch.cat([features, padding], dim=1)
        elif features.shape[1] > self.model_config['virtual_node_dim']:
            # 截断到所需维度
            features = features[:, :self.model_config['virtual_node_dim']]
        
        return {
            'features': features,
            'edge_index': virtual_state['edge_index'],
            'edge_attr': virtual_state['edge_attr']
        }
    
    def _validate_mapping_action(self, mapping_action, num_virtual_nodes, num_physical_nodes):
        """验证映射动作"""
        if len(mapping_action) < num_virtual_nodes:
            # 如果动作长度不足，用随机值填充
            padding = np.random.randint(0, num_physical_nodes, num_virtual_nodes - len(mapping_action))
            mapping_action = np.concatenate([mapping_action, padding])
        elif len(mapping_action) > num_virtual_nodes:
            # 如果动作长度过长，截断
            mapping_action = mapping_action[:num_virtual_nodes]
        
        # 确保所有值都在有效范围内
        mapping_action = np.clip(mapping_action, 0, num_physical_nodes - 1)
        
        return mapping_action
    
    def _validate_bandwidth_action(self, bandwidth_action, num_links):
        """验证带宽动作"""
        if len(bandwidth_action) < num_links:
            # 如果动作长度不足，用随机值填充
            padding = np.random.randint(0, self.bandwidth_levels, num_links - len(bandwidth_action))
            bandwidth_action = np.concatenate([bandwidth_action, padding])
        elif len(bandwidth_action) > num_links:
            # 如果动作长度过长，截断
            bandwidth_action = bandwidth_action[:num_links]
        
        # 确保所有值都在有效范围内
        bandwidth_action = np.clip(bandwidth_action, 0, self.bandwidth_levels - 1)
        
        return bandwidth_action
    
    def _bandwidth_level_to_value(self, bandwidth_level, link_req):
        """将带宽等级转换为实际带宽值"""
        # 将带宽等级映射到最小和最大需求之间的值
        min_bw = link_req['min_bandwidth_1_to_2']
        max_bw = link_req['max_bandwidth_1_to_2']
        
        if max_bw > min_bw:
            # 线性插值
            ratio = bandwidth_level / (self.bandwidth_levels - 1)
            bandwidth_value = min_bw + ratio * (max_bw - min_bw)
        else:
            bandwidth_value = min_bw
        
        return bandwidth_value
    
    def _calculate_bandwidth_satisfaction(self, allocated: float, min_req: float, max_req: float) -> float:
        """计算单个链路的带宽满足度"""
        if min_req <= allocated <= max_req:
            if max_req > min_req:
                return (allocated - min_req) / (max_req - min_req)
            else:
                return 1.0
        elif allocated > max_req:
            return 1.0  # 超过最大需求也是好的
        else:
            return 0.0  # 低于最小需求
    
    def _save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"test_results_{timestamp}.json")
        
        results = {
            'timestamp': timestamp,
            'test_results': self.test_results,
            'statistics': self.stats,
            'seed': self.seed
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 测试结果已保存到: {results_file}")
    
    def _generate_report(self):
        """生成测试报告"""
        print(f"\n📊 测试报告")
        print(f"=" * 50)
        print(f"🔒 随机种子: {self.seed}")  # 显示种子信息
        # 统计信息
        print(f"📈 统计信息:")
        print(f"   - 总虚拟工作数: {self.stats['total_works']}")
        print(f"   - 随机算法成功: {self.stats['successful_random']}")
        print(f"   - 随机算法失败: {self.stats['failed_random']}")
        
        # 如果PPO可用，添加PPO统计信息
        if PPO_AVAILABLE and self.session_name and 'successful_ppo' in self.stats:
            print(f"   - PPO算法成功: {self.stats['successful_ppo']}")
            print(f"   - PPO算法失败: {self.stats['failed_ppo']}")
        
        # 性能对比
        print(f"\n🏆 性能对比:")
        
        # 负载均衡
        random_load_mean = np.mean(self.test_results['random']['load_balancing'])
        random_load_std = np.std(self.test_results['random']['load_balancing'])
        
        print(f"   📊 负载均衡分数:")
        print(f"      - 随机算法: {random_load_mean:.4f} ± {random_load_std:.4f}")
        
        # 如果PPO可用，添加PPO负载均衡对比
        if PPO_AVAILABLE and self.session_name and 'ppo' in self.test_results:
            ppo_load_mean = np.mean(self.test_results['ppo']['load_balancing'])
            ppo_load_std = np.std(self.test_results['ppo']['load_balancing'])
            print(f"      - PPO算法: {ppo_load_mean:.4f} ± {ppo_load_std:.4f}")
            
            if ppo_load_mean > random_load_mean:
                improvement = ((ppo_load_mean - random_load_mean) / random_load_mean) * 100
                print(f"      - PPO提升: +{improvement:.2f}%")
            else:
                degradation = ((random_load_mean - ppo_load_mean) / ppo_load_mean) * 100
                print(f"      - PPO下降: -{degradation:.2f}%")
        
        # 带宽满足度
        random_bw_mean = np.mean(self.test_results['random']['bandwidth_satisfaction'])
        random_bw_std = np.std(self.test_results['random']['bandwidth_satisfaction'])
        
        print(f"   🌐 带宽满足度分数:")
        print(f"      - 随机算法: {random_bw_mean:.4f} ± {random_bw_std:.4f}")
        
        # 如果PPO可用，添加PPO带宽满足度对比
        if PPO_AVAILABLE and self.session_name and 'ppo' in self.test_results:
            ppo_bw_mean = np.mean(self.test_results['ppo']['bandwidth_satisfaction'])
            ppo_bw_std = np.std(self.test_results['ppo']['bandwidth_satisfaction'])
            print(f"      - PPO算法: {ppo_bw_mean:.4f} ± {ppo_bw_std:.4f}")
            
            if ppo_bw_mean > random_bw_mean:
                improvement = ((ppo_bw_mean - random_bw_mean) / random_bw_mean) * 100
                print(f"      - PPO提升: +{improvement:.2f}%")
            else:
                degradation = ((random_load_mean - ppo_bw_mean) / ppo_bw_mean) * 100
                print(f"      - PPO下降: -{degradation:.2f}%")
        
        # 生成图表
        self._plot_results()
    
    def _plot_results(self):
        """生成结果图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确定要绘制的算法
        algorithms_to_plot = ['random']
        algorithm_labels = ['Random']
        
        # 如果PPO可用且有结果，添加到图表中
        if self.session_name and 'ppo' in self.test_results and len(self.test_results['ppo']['load_balancing']) > 0:
            algorithms_to_plot.append('ppo')
            algorithm_labels.append('PPO')
            print(f"📊 图表绘制: 添加PPO算法，共{len(algorithms_to_plot)}个算法")
            print(f"   PPO数据长度: 负载均衡={len(self.test_results['ppo']['load_balancing'])}, 带宽满足度={len(self.test_results['ppo']['bandwidth_satisfaction'])}")
        else:
            print(f"📊 图表绘制: PPO不可用，共{len(algorithms_to_plot)}个算法")
            print(f"   session_name: {self.session_name}")
            print(f"   'ppo' in test_results: {'ppo' in self.test_results}")
            if 'ppo' in self.test_results:
                print(f"   PPO数据长度: 负载均衡={len(self.test_results['ppo']['load_balancing'])}, 带宽满足度={len(self.test_results['ppo']['bandwidth_satisfaction'])}")
        
        # 创建图表 - 2行2列布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 第一行：原有的箱线图对比
        # 负载均衡对比
        load_balancing_data = [self.test_results[algo]['load_balancing'] for algo in algorithms_to_plot]
        ax1.boxplot(load_balancing_data, labels=algorithm_labels)
        ax1.set_title('Load Balancing Score Comparison')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        
        # 带宽满足度对比
        bandwidth_data = [self.test_results[algo]['bandwidth_satisfaction'] for algo in algorithms_to_plot]
        ax2.boxplot(bandwidth_data, labels=algorithm_labels)
        ax2.set_title('Bandwidth Satisfaction Score Comparison')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        
        # 第二行：新增的虚拟工作数量vs性能图表
        # 虚拟工作数量vs负载均衡分数
        work_counts = list(range(1, len(self.test_results['random']['load_balancing']) + 1))
        
        # print(f"📊 绘制折线图: 工作数量范围 = {work_counts}")
        for i, algo in enumerate(algorithms_to_plot):
            if algo in self.test_results:
                data = self.test_results[algo]['load_balancing']
                # print(f"   {algo}: 数据长度={len(data)}, 数据={data}")
                
                # 平滑曲线数据
                x_smooth, y_smooth = self._smooth_curve(work_counts, data, method=self.smooth_method, window_size=self.smooth_window)
                
                # 绘制平滑曲线和原始数据点
                ax3.plot(x_smooth, y_smooth, 
                        linewidth=3, alpha=0.8, label=f"{algorithm_labels[i]} (smoothed)")
                ax3.scatter(work_counts, data, 
                           marker='o', s=30, alpha=0.6, label=f"{algorithm_labels[i]} (original)")
        
        ax3.set_title('Load Balancing Score vs Virtual Work Count (Smoothed)')
        ax3.set_xlabel('Virtual Work Count')
        ax3.set_ylabel('Load Balancing Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 虚拟工作数量vs带宽满足度分数
        for i, algo in enumerate(algorithms_to_plot):
            if algo in self.test_results:
                data = self.test_results[algo]['bandwidth_satisfaction']
                # print(f"   {algo}: 数据长度={len(data)}, 数据={data}")
                
                # 平滑曲线数据
                x_smooth, y_smooth = self._smooth_curve(work_counts, data, method=self.smooth_method, window_size=self.smooth_window)
                
                # 绘制平滑曲线和原始数据点
                ax4.plot(x_smooth, y_smooth, 
                        linewidth=3, alpha=0.8, label=f"{algorithm_labels[i]} (smoothed)")
                ax4.scatter(work_counts, data, 
                           marker='s', s=30, alpha=0.6, label=f"{algorithm_labels[i]} (original)")
        
        ax4.set_title('Bandwidth Satisfaction Score vs Virtual Work Count (Smoothed)')
        ax4.set_xlabel('Virtual Work Count')
        ax4.set_ylabel('Bandwidth Satisfaction Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(self.results_dir, f"results_plot_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 结果图表已保存到: {plot_file}")
        
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='物理环境测试器')
    # 新增seed参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    # PPO模型参数
    parser.add_argument('--session_name', type=str, default=None,
                       help='PPO训练会话名称 (如果提供，将加载并测试PPO模型)')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式选择PPO训练会话')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录 (默认: models)')
    parser.add_argument('--stats_dir', type=str, default='stats',
                       help='统计信息目录 (默认: stats)')
    
    # 物理环境参数
    parser.add_argument('--num_physical_nodes', type=int, default=10,
                       help='物理节点数量 (默认: 10)')
    parser.add_argument('--num_virtual_works', type=int, default=10,
                       help='虚拟工作数量 (默认: 5)')
    
    # 物理资源范围
    parser.add_argument('--physical_cpu_min', type=int, default=50,
                       help='物理节点CPU最小值 (默认: 50)')
    parser.add_argument('--physical_cpu_max', type=int, default=100,
                       help='物理节点CPU最大值 (默认: 200)')
    parser.add_argument('--physical_memory_min', type=int, default=50,
                       help='物理节点内存最小值 (默认: 100)')
    parser.add_argument('--physical_memory_max', type=int, default=100,
                       help='物理节点内存最大值 (默认: 400)')
    parser.add_argument('--physical_bandwidth_min', type=int, default=50,
                       help='物理链路带宽最小值 (默认: 100)')
    parser.add_argument('--physical_bandwidth_max', type=int, default=100,
                       help='物理链路带宽最大值 (默认: 1000)')
    
    # 虚拟资源范围
    parser.add_argument('--virtual_cpu_min', type=int, default=8,
                       help='虚拟节点CPU最小值 (默认: 10)')
    parser.add_argument('--virtual_cpu_max', type=int, default=10,
                       help='虚拟节点CPU最大值 (默认: 50)')
    parser.add_argument('--virtual_memory_min', type=int, default=8,
                       help='虚拟节点内存最小值 (默认: 20)')
    parser.add_argument('--virtual_memory_max', type=int, default=10,
                       help='虚拟节点内存最大值 (默认: 100)')
    parser.add_argument('--virtual_bandwidth_min', type=int, default=5,
                       help='虚拟链路带宽最小值 (默认: 10)')
    parser.add_argument('--virtual_bandwidth_max', type=int, default=15,
                       help='虚拟链路带宽最大值 (默认: 200)')
    
    # 网络参数
    parser.add_argument('--physical_connectivity_prob', type=float, default=0.9,
                       help='物理网络连接概率 (默认: 0.3)')
    parser.add_argument('--virtual_connectivity_prob', type=float, default=0.7,
                       help='虚拟网络连接概率 (默认: 0.4)')
    
    # 虚拟节点范围
    parser.add_argument('--virtual_nodes_min', type=int, default=3,
                       help='虚拟节点数量最小值 (默认: 3)')
    parser.add_argument('--virtual_nodes_max', type=int, default=8,
                       help='虚拟节点数量最大值 (默认: 8)')
    
    # 测试参数
    parser.add_argument('--num_trials', type=int, default=30,
                       help='测试次数 (默认: 10)')
    parser.add_argument('--results_dir', type=str, default='physical_test_results',
                       help='结果保存目录 (默认: physical_test_results)')
    parser.add_argument('--smooth_method', type=str, default='gaussian',
                       choices=['gaussian', 'spline', 'moving_average'],
                       help='曲线平滑方法 (默认: gaussian)')
    parser.add_argument('--smooth_window', type=int, default=3,
                       help='移动平均窗口大小 (默认: 3)')
    
    args = parser.parse_args()
    
    # 处理会话名称
    session_name = args.session_name
    if args.interactive and not session_name:
        print("🔍 交互式选择PPO训练会话...")
        # 这里先创建测试器，然后在run_test中处理会话选择
        session_name = None
    
    # 创建测试器
    tester = PhysicalEnvironmentTester(
        session_name=session_name,
        model_dir=args.model_dir,
        stats_dir=args.stats_dir,
        results_dir=args.results_dir,
        num_trials=args.num_trials,
        smooth_method=args.smooth_method,
        smooth_window=args.smooth_window,
        seed=args.seed
    )
    
    # 运行测试
    tester.run_test(
        num_physical_nodes=args.num_physical_nodes,
        num_virtual_works=args.num_virtual_works,
        physical_cpu_range=(args.physical_cpu_min, args.physical_cpu_max),
        physical_memory_range=(args.physical_memory_min, args.physical_memory_max),
        physical_bandwidth_range=(args.physical_bandwidth_min, args.physical_bandwidth_max),
        virtual_cpu_range=(args.virtual_cpu_min, args.virtual_cpu_max),
        virtual_memory_range=(args.virtual_memory_min, args.virtual_memory_max),
        virtual_bandwidth_range=(args.virtual_bandwidth_min, args.virtual_bandwidth_max),
        physical_connectivity_prob=args.physical_connectivity_prob,
        virtual_connectivity_prob=args.virtual_connectivity_prob,
        virtual_nodes_range=(args.virtual_nodes_min, args.virtual_nodes_max)
    )

if __name__ == "__main__":
    main()
