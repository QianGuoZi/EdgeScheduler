#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始问题定义的奖励函数实现
严格遵循define_problem.md中的数学模型
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from network_scheduler import NetworkScheduler, VirtualWork

class OriginalRewardCalculator:
    """
    基于原始数学模型的奖励计算器
    实现负载均衡度L和带宽满足度D_BW的计算
    """
    
    def __init__(self, 
                 w1_cpu: float = 0.4,
                 w2_memory: float = 0.4, 
                 w3_bandwidth: float = 0.2,
                 gamma1_load_balance: float = 0.7,
                 gamma2_bandwidth_satisfaction: float = 0.3):
        """
        初始化奖励计算器
        
        Args:
            w1_cpu: CPU负载均衡权重
            w2_memory: 内存负载均衡权重
            w3_bandwidth: 带宽负载均衡权重
            gamma1_load_balance: 负载均衡在总目标中的权重
            gamma2_bandwidth_satisfaction: 带宽满足度在总目标中的权重
        """
        # 负载均衡权重
        self.w1 = w1_cpu
        self.w2 = w2_memory
        self.w3 = w3_bandwidth
        assert abs(self.w1 + self.w2 + self.w3 - 1.0) < 1e-6, "w1 + w2 + w3 must equal 1"
        
        # 全局目标权重
        self.gamma1 = gamma1_load_balance
        self.gamma2 = gamma2_bandwidth_satisfaction
        assert abs(self.gamma1 + self.gamma2 - 1.0) < 1e-6, "gamma1 + gamma2 must equal 1"
    
    def calculate_utilization(self, scheduler: NetworkScheduler) -> Dict:
        """
        计算资源利用率 U^CPU, U^RAM, U^BW
        
        Returns:
            包含各类资源利用率的字典
        """
        utilizations = {
            'cpu': [],
            'memory': [],
            'bandwidth': []
        }
        
        # 计算每个物理节点的CPU和内存利用率
        for p in range(len(scheduler.topology.node_resources)):
            node_resource = scheduler.topology.node_resources[p]
            total_cpu = node_resource['cpu']
            total_memory = node_resource['memory']
            
            # 计算已使用的资源
            used_cpu = total_cpu - scheduler.topology.get_available_resources(p)['cpu']
            used_memory = total_memory - scheduler.topology.get_available_resources(p)['memory']
            
            # 利用率 = 已使用 / 总量
            u_cpu = used_cpu / total_cpu if total_cpu > 0 else 0
            u_memory = used_memory / total_memory if total_memory > 0 else 0
            
            utilizations['cpu'].append(u_cpu)
            utilizations['memory'].append(u_memory)
        
        # 计算每条物理链路的带宽利用率
        for link_key, link_info in scheduler.topology.links.items():
            total_bw = link_info['bandwidth']
            used_bw = link_info['used_bandwidth']
            
            u_bw = used_bw / total_bw if total_bw > 0 else 0
            utilizations['bandwidth'].append(u_bw)
        
        return utilizations
    
    def calculate_load_balance(self, utilizations: Dict) -> Tuple[float, Dict]:
        """
        计算负载均衡度 L
        L = w1 * L^CPU + w2 * L^RAM + w3 * L^BW
        其中L^X是X资源利用率的标准差
        
        Returns:
            (L, components): 总负载均衡度和各分量
        """
        components = {}
        
        # CPU负载均衡度 (标准差)
        cpu_utils = utilizations['cpu']
        if len(cpu_utils) > 1:
            l_cpu = np.std(cpu_utils)
        else:
            l_cpu = 0.0
        components['L_CPU'] = l_cpu
        
        # 内存负载均衡度 (标准差)
        mem_utils = utilizations['memory']
        if len(mem_utils) > 1:
            l_memory = np.std(mem_utils)
        else:
            l_memory = 0.0
        components['L_RAM'] = l_memory
        
        # 带宽负载均衡度 (标准差)
        bw_utils = utilizations['bandwidth']
        if len(bw_utils) > 1:
            l_bandwidth = np.std(bw_utils)
        else:
            l_bandwidth = 0.0
        components['L_BW'] = l_bandwidth
        
        # 综合负载均衡度
        L = self.w1 * l_cpu + self.w2 * l_memory + self.w3 * l_bandwidth
        
        return L, components
    
    def calculate_bandwidth_satisfaction(self, 
                                        scheduler: NetworkScheduler,
                                        virtual_work: VirtualWork) -> Tuple[float, List[float]]:
        """
        计算带宽满足度 D_BW
        D_BW = (1/|V|) * Σ δ(v)
        
        Returns:
            (D_BW, individual_satisfactions): 总满足度和各链路满足度
        """
        satisfactions = []
        
        for link_req in virtual_work.link_requirements:
            from_node = link_req['from']
            to_node = link_req['to']
            min_bw = link_req['min_bandwidth_1_to_2']
            max_bw = link_req['max_bandwidth_1_to_2']
            
            # 获取实际分配的带宽
            allocated_bw = scheduler.bandwidth_allocation.get((from_node, to_node), 0)
            
            # 计算满足度 δ(v)
            if max_bw == min_bw:
                # 如果最大最小相等，满足度为1（如果满足）或0（如果不满足）
                delta_v = 1.0 if allocated_bw >= min_bw else 0.0
            else:
                # 否则，满足度是线性插值
                if allocated_bw < min_bw:
                    delta_v = 0.0
                elif allocated_bw > max_bw:
                    delta_v = 1.0
                else:
                    delta_v = (allocated_bw - min_bw) / (max_bw - min_bw)
            
            satisfactions.append(delta_v)
        
        # 平均满足度
        D_BW = np.mean(satisfactions) if satisfactions else 0.0
        
        return D_BW, satisfactions
    
    def calculate_reward(self, 
                        scheduler: NetworkScheduler,
                        virtual_work: VirtualWork) -> Dict:
        """
        计算最终奖励值
        目标函数: Minimize γ1*L - γ2*D_BW
        转换为奖励: Maximize -γ1*L + γ2*D_BW
        
        Returns:
            包含奖励值和所有组件的字典
        """
        # 检查是否有映射
        if len(scheduler.scheduled_nodes) == 0:
            return {
                'total_reward': -1.0,
                'mapping_success': False,
                'L': 1.0,
                'D_BW': 0.0,
                'components': {}
            }
        
        # 检查映射完整性
        total_virtual_nodes = len(virtual_work.node_requirements)
        mapped_nodes = len(scheduler.scheduled_nodes)
        mapping_success = (mapped_nodes == total_virtual_nodes)
        
        if not mapping_success:
            # 部分映射惩罚
            penalty = -1.0 * (1.0 - mapped_nodes / total_virtual_nodes)
            return {
                'total_reward': penalty,
                'mapping_success': False,
                'mapped_ratio': mapped_nodes / total_virtual_nodes,
                'L': 1.0,
                'D_BW': 0.0,
                'components': {}
            }
        
        # 计算资源利用率
        utilizations = self.calculate_utilization(scheduler)
        
        # 计算负载均衡度 L
        L, load_balance_components = self.calculate_load_balance(utilizations)
        
        # 计算带宽满足度 D_BW
        D_BW, link_satisfactions = self.calculate_bandwidth_satisfaction(scheduler, virtual_work)
        
        # 计算最终奖励
        # 原始目标: Minimize γ1*L - γ2*D_BW
        # 转换为奖励: Maximize -γ1*L + γ2*D_BW
        # 归一化到[-1, 1]范围
        raw_reward = -self.gamma1 * L + self.gamma2 * D_BW
        
        # L的范围大约是[0, 1]，D_BW的范围是[0, 1]
        # raw_reward的范围大约是[-γ1, γ2]
        # 归一化
        min_reward = -self.gamma1
        max_reward = self.gamma2
        normalized_reward = 2.0 * (raw_reward - min_reward) / (max_reward - min_reward) - 1.0
        normalized_reward = np.clip(normalized_reward, -1.0, 1.0)
        
        return {
            'total_reward': normalized_reward,
            'raw_reward': raw_reward,
            'mapping_success': True,
            'L': L,
            'D_BW': D_BW,
            'components': {
                'load_balance': load_balance_components,
                'utilizations': {
                    'cpu_mean': np.mean(utilizations['cpu']),
                    'memory_mean': np.mean(utilizations['memory']),
                    'bandwidth_mean': np.mean(utilizations['bandwidth']) if utilizations['bandwidth'] else 0,
                },
                'link_satisfactions': link_satisfactions,
                'gamma1': self.gamma1,
                'gamma2': self.gamma2,
            }
        }

def integrate_with_network_scheduler(scheduler: NetworkScheduler):
    """
    将原始奖励函数集成到NetworkScheduler中
    """
    calculator = OriginalRewardCalculator()
    
    def calculate_original_reward(self, virtual_work: VirtualWork) -> float:
        """使用原始问题定义的奖励函数"""
        result = calculator.calculate_reward(self, virtual_work)
        
        # 打印详细信息用于调试
        if result['mapping_success']:
            print(f"📊 原始奖励计算:")
            print(f"   L (负载均衡度): {result['L']:.4f}")
            print(f"   D_BW (带宽满足度): {result['D_BW']:.4f}")
            print(f"   原始奖励: {result['raw_reward']:.4f}")
            print(f"   归一化奖励: {result['total_reward']:.4f}")
        
        return result['total_reward']
    
    def get_original_reward_components(self, virtual_work: VirtualWork) -> Dict:
        """获取原始奖励的详细组件"""
        return calculator.calculate_reward(self, virtual_work)
    
    # 添加方法到scheduler
    scheduler.calculate_original_reward = lambda vw: calculate_original_reward(scheduler, vw)
    scheduler.get_original_reward_components = lambda vw: get_original_reward_components(scheduler, vw)
    
    return scheduler

if __name__ == "__main__":
    print("原始奖励函数测试")
    print("-" * 50)
    
    # 创建测试计算器
    calc = OriginalRewardCalculator()
    print(f"权重配置:")
    print(f"  负载均衡权重: w1={calc.w1}, w2={calc.w2}, w3={calc.w3}")
    print(f"  全局权重: γ1={calc.gamma1}, γ2={calc.gamma2}")