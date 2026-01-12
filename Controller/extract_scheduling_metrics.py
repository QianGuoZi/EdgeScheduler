#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从调度结果中提取指标的工具
用于从test.py的输出、日志或API响应中提取负载均衡度和带宽满足度
"""

import os
import json
import re
import sys
from typing import Dict, Optional

# 添加路径以便导入相关模块
CONTROLLER_DIR = '/home/qianguo/Edge-Scheduler/Controller'
sys.path.insert(0, CONTROLLER_DIR)

try:
    from base.algorithm.PPO_my.original_reward import OriginalRewardCalculator
    from base.algorithm.PPO_my.network_scheduler import NetworkScheduler
    from base.algorithm.PPO_my.virtual_work import VirtualWork
except ImportError as e:
    print(f"警告: 无法导入相关模块: {e}")
    print("将使用备用方法提取指标")


class MetricsExtractor:
    """指标提取器"""
    
    def __init__(self):
        self.reward_calculator = None
        try:
            self.reward_calculator = OriginalRewardCalculator()
        except:
            pass
    
    def extract_from_log_file(self, log_file_path: str) -> Optional[Dict]:
        """从日志文件中提取指标"""
        if not os.path.exists(log_file_path):
            return None
        
        metrics = {
            "load_balance_degree": None,
            "bandwidth_satisfaction": None,
            "composite_score": None
        }
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 尝试从日志中提取指标（根据实际日志格式调整）
            # 示例：如果日志中有 "L=0.1234" 和 "D_BW=0.5678"
            l_match = re.search(r'L[=:]\s*([0-9.]+)', content)
            dbw_match = re.search(r'D_BW[=:]\s*([0-9.]+)', content)
            
            if l_match:
                metrics["load_balance_degree"] = float(l_match.group(1))
            if dbw_match:
                metrics["bandwidth_satisfaction"] = float(dbw_match.group(1))
            
            # 计算综合指标
            if metrics["load_balance_degree"] is not None and metrics["bandwidth_satisfaction"] is not None:
                metrics["composite_score"] = self._calculate_composite_score(
                    metrics["load_balance_degree"],
                    metrics["bandwidth_satisfaction"]
                )
        
        except Exception as e:
            print(f"从日志文件提取指标失败: {e}")
        
        return metrics if any(v is not None for v in metrics.values()) else None
    
    def extract_from_api_response(self, response_data: Dict) -> Optional[Dict]:
        """从API响应中提取指标"""
        metrics = {
            "load_balance_degree": None,
            "bandwidth_satisfaction": None,
            "composite_score": None
        }
        
        try:
            # 根据实际API响应格式调整
            if "load_balance_degree" in response_data:
                metrics["load_balance_degree"] = float(response_data["load_balance_degree"])
            if "bandwidth_satisfaction" in response_data:
                metrics["bandwidth_satisfaction"] = float(response_data["bandwidth_satisfaction"])
            if "L" in response_data:
                metrics["load_balance_degree"] = float(response_data["L"])
            if "D_BW" in response_data:
                metrics["bandwidth_satisfaction"] = float(response_data["D_BW"])
            
            # 计算综合指标
            if metrics["load_balance_degree"] is not None and metrics["bandwidth_satisfaction"] is not None:
                metrics["composite_score"] = self._calculate_composite_score(
                    metrics["load_balance_degree"],
                    metrics["bandwidth_satisfaction"]
                )
        
        except Exception as e:
            print(f"从API响应提取指标失败: {e}")
        
        return metrics if any(v is not None for v in metrics.values()) else None
    
    def calculate_from_scheduler(self, scheduler: NetworkScheduler, 
                                 virtual_work: VirtualWork) -> Dict:
        """直接从调度器计算指标"""
        metrics = {
            "load_balance_degree": None,
            "bandwidth_satisfaction": None,
            "composite_score": None
        }
        
        try:
            if self.reward_calculator is None:
                print("警告: 无法使用奖励计算器，尝试从调度器获取")
                if hasattr(scheduler, 'get_original_reward_components'):
                    components = scheduler.get_original_reward_components(virtual_work)
                    metrics["load_balance_degree"] = components.get('L', None)
                    metrics["bandwidth_satisfaction"] = components.get('D_BW', None)
            else:
                # 使用奖励计算器
                reward_result = self.reward_calculator.calculate_reward(scheduler, virtual_work)
                metrics["load_balance_degree"] = reward_result.get('L', None)
                metrics["bandwidth_satisfaction"] = reward_result.get('D_BW', None)
            
            # 计算综合指标
            if metrics["load_balance_degree"] is not None and metrics["bandwidth_satisfaction"] is not None:
                metrics["composite_score"] = self._calculate_composite_score(
                    metrics["load_balance_degree"],
                    metrics["bandwidth_satisfaction"]
                )
        
        except Exception as e:
            print(f"从调度器计算指标失败: {e}")
        
        return metrics
    
    def _calculate_composite_score(self, l: float, d_bw: float,
                                   l_weight: float = 0.5, dbw_weight: float = 0.5) -> float:
        """计算综合指标"""
        import math
        if math.isnan(l) or math.isnan(d_bw):
            return float('nan')
        # L越小越好，D_BW越大越好
        # 转换为统一的方向：越大越好
        normalized_l = max(0, 1 - l)  # 假设L在[0,1]范围内
        return l_weight * normalized_l + dbw_weight * d_bw


def main():
    """命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从调度结果中提取指标')
    parser.add_argument('--log', type=str, help='日志文件路径')
    parser.add_argument('--json', type=str, help='JSON响应文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    extractor = MetricsExtractor()
    metrics = None
    
    if args.log:
        metrics = extractor.extract_from_log_file(args.log)
    elif args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
        metrics = extractor.extract_from_api_response(data)
    else:
        print("请指定 --log 或 --json 参数")
        return
    
    if metrics:
        print("提取的指标:")
        print(f"  负载均衡度 (L): {metrics.get('load_balance_degree')}")
        print(f"  带宽满足度 (D_BW): {metrics.get('bandwidth_satisfaction')}")
        print(f"  综合指标: {metrics.get('composite_score')}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n指标已保存到: {args.output}")
    else:
        print("未能提取到指标")


if __name__ == '__main__':
    main()
