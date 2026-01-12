#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动记录单次实验数据的工具
当自动化脚本无法正常工作时，可以使用此脚本手动记录实验数据
"""

import os
import json
import csv
from datetime import datetime
from typing import Optional

EXPERIMENT_RESULTS_DIR = '/home/qianguo/Edge-Scheduler/Controller/experiment_results'
EXPERIMENT_CSV_FILE = os.path.join(EXPERIMENT_RESULTS_DIR, 'experiment_results.csv')
EXPERIMENT_JSON_FILE = os.path.join(EXPERIMENT_RESULTS_DIR, 'experiment_results.json')


def load_existing_results() -> list:
    """加载已有的实验结果"""
    if os.path.exists(EXPERIMENT_JSON_FILE):
        with open(EXPERIMENT_JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_results(results: list):
    """保存实验结果"""
    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)
    
    # 保存为JSON
    with open(EXPERIMENT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存为CSV
    if results:
        with open(EXPERIMENT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['experiment_id', 'dataset_id', 'timestamp', 'status',
                         'load_balance_degree', 'bandwidth_satisfaction', 'composite_score', 'error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    
    print(f"\n✓ 结果已保存到:")
    print(f"  - {EXPERIMENT_JSON_FILE}")
    print(f"  - {EXPERIMENT_CSV_FILE}")


def calculate_composite_score(l: float, d_bw: float, 
                             l_weight: float = 0.5, dbw_weight: float = 0.5) -> float:
    """计算联合参数（综合指标）"""
    import math
    if math.isnan(l) or math.isnan(d_bw):
        return float('nan')
    # L越小越好，D_BW越大越好
    # 转换为统一的方向：越大越好
    normalized_l = max(0, 1 - l)  # 假设L在[0,1]范围内
    return l_weight * normalized_l + dbw_weight * d_bw


def record_experiment(dataset_id: int, 
                     load_balance_degree: Optional[float] = None,
                     bandwidth_satisfaction: Optional[float] = None,
                     status: str = "success",
                     error: Optional[str] = None):
    """记录单次实验数据"""
    
    # 加载已有结果
    results = load_existing_results()
    
    # 计算实验ID
    experiment_id = len(results) + 1
    
    # 计算综合指标
    composite_score = None
    if load_balance_degree is not None and bandwidth_satisfaction is not None:
        composite_score = calculate_composite_score(load_balance_degree, bandwidth_satisfaction)
    
    # 创建实验记录
    experiment_result = {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "load_balance_degree": load_balance_degree,
        "bandwidth_satisfaction": bandwidth_satisfaction,
        "composite_score": composite_score,
        "error": error
    }
    
    # 添加到结果列表
    results.append(experiment_result)
    
    # 保存结果
    save_results(results)
    
    # 显示结果
    print(f"\n实验记录已添加:")
    print(f"  实验ID: {experiment_id}")
    print(f"  数据集ID: {dataset_id}")
    print(f"  负载均衡度 (L): {load_balance_degree}")
    print(f"  带宽满足度 (D_BW): {bandwidth_satisfaction}")
    print(f"  综合指标: {composite_score}")
    print(f"  状态: {status}")
    if error:
        print(f"  错误: {error}")


def main():
    """交互式记录实验数据"""
    import argparse
    
    parser = argparse.ArgumentParser(description='手动记录实验数据')
    parser.add_argument('--dataset', type=int, required=True, help='数据集ID')
    parser.add_argument('--L', type=float, help='负载均衡度 (L)')
    parser.add_argument('--D_BW', type=float, help='带宽满足度 (D_BW)')
    parser.add_argument('--status', type=str, default='success', choices=['success', 'failed'],
                       help='实验状态')
    parser.add_argument('--error', type=str, help='错误信息（如果实验失败）')
    
    args = parser.parse_args()
    
    record_experiment(
        dataset_id=args.dataset,
        load_balance_degree=args.L,
        bandwidth_satisfaction=args.D_BW,
        status=args.status,
        error=args.error
    )


if __name__ == '__main__':
    main()
