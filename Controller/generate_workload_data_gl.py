#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成负载数据和任务请求数据
- 负载数据：10个物理节点，10%-50%资源使用率
- 任务请求数据：5个任务节点，CPU 10-50份，RAM 10-50G，带宽需求
"""

import json
import random
import os

# 物理节点配置
PHYSICAL_NODES = [f'emulator-{i}' for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
TOTAL_CPU_SHARES = 100  # 4个CPU，每个粒度0.04，共100份
TOTAL_RAM = 256  # GB
TOTAL_BW = 1000  # mbps

# 任务节点配置
TASK_NODES = ['p1', 'n1', 'n2', 'n3', 'n4']
CPU_MIN = 10  # 份
CPU_MAX = 50  # 份
RAM_MIN = 10  # GB
RAM_MAX = 50  # GB
BW_MIN_RANGE = (10, 50)  # mbps
BW_MAX_RANGE = (50, 100)  # mbps

def generate_workload_data():
    """
    生成负载数据
    要求：
    - 所有资源使用率在0.1-0.5范围内随机生成
    - 所有节点都有CPU和RAM负载
    - 物理节点间全连接拓扑（非双向对称）
    - 70%的链接有负载（单向连接）
    """
    workloads = {}
    links = {}
    
    # 为每个物理节点生成负载（所有节点都有负载）
    for node in PHYSICAL_NODES:
        # CPU使用量：在0.1-0.5范围内随机生成
        cpu_utilization_rate = random.uniform(0.1, 0.5)
        cpu_used_shares = int(TOTAL_CPU_SHARES * cpu_utilization_rate)
        cpu_used_shares = max(1, min(cpu_used_shares, TOTAL_CPU_SHARES - 10))  # 保留至少10份可用
        cpu_value = cpu_used_shares * 0.04  # 转换为CPU核心数
        
        # RAM使用量：在0.1-0.5范围内随机生成
        ram_utilization_rate = random.uniform(0.1, 0.5)
        ram_used = int(TOTAL_RAM * ram_utilization_rate)
        ram_used = max(1, min(ram_used, TOTAL_RAM - 20))  # 保留至少20G可用
        
        # 所有节点都有负载
        workloads[node] = {
            "cpu": round(cpu_value, 2),
            "ram": ram_used,
            "unit": "G",
            "image": "stress:latest",
            "enabled": True
        }
    
    # 生成全连接的物理节点拓扑（非双向对称，单向连接）
    # 创建所有可能的单向连接（10个节点，每个节点到其他9个节点，共90个可能的单向连接）
    all_directed_links = []
    for node1 in PHYSICAL_NODES:
        for node2 in PHYSICAL_NODES:
            if node1 != node2:  # 排除自己到自己的连接
                all_directed_links.append((node1, node2))
    
    # 随机选择70%的单向连接分配带宽负载（非双向对称）
    num_active_links = round(len(all_directed_links) * 0.7)  # 90 * 0.7 = 63个连接
    active_links = random.sample(all_directed_links, num_active_links)
    
    # 为每个活跃连接分配带宽（单向，非双向对称）
    for source, dest in active_links:
        # 带宽使用量：在0.1-0.5范围内随机生成
        bw_utilization_rate = random.uniform(0.1, 0.5)
        bw_used = int(TOTAL_BW * bw_utilization_rate)
        bw_used = max(10, min(bw_used, TOTAL_BW - 50))  # 保留至少50mbps可用
        
        if source not in links:
            links[source] = []
        links[source].append({
            "dest": dest,
            "bw": f"{bw_used}mbps"
        })
    
    return {
        "description": "模拟各emulator上已有负载的配置文件 - 利用率0.1-0.5随机（全连接拓扑，非双向对称）",
        "workloads": workloads,
        "links": links
    }

def generate_task_request_data():
    """
    生成任务请求数据
    要求：
    - 任务节点间连接符合gossip learning（双向对称）
    - 构建连通的全连接或近似全连接拓扑
    """
    nodes = {}
    links = {}
    
    # 生成5个任务节点的CPU和RAM需求
    for node in TASK_NODES:
        cpu_shares = random.randint(CPU_MIN, CPU_MAX)
        ram = random.randint(RAM_MIN, RAM_MAX)
        nodes[node] = {
            "cpu": cpu_shares,
            "ram": ram
        }
    
    # 生成任务节点之间的带宽需求（符合gossip learning，双向对称）
    # 创建所有可能的连接对（C(5,2) = 10个连接对）
    all_pairs = []
    for i, node1 in enumerate(TASK_NODES):
        for node2 in TASK_NODES[i+1:]:
            all_pairs.append((node1, node2))
    
    # 为了符合gossip learning，选择70%-100%的连接对（确保连通性和冗余）
    # 这样可以形成一个强连通的拓扑
    num_active_pairs = random.randint(round(len(all_pairs) * 0.7), len(all_pairs))
    active_pairs = random.sample(all_pairs, num_active_pairs)
    
    # 为每个活跃连接对分配带宽（双向对称，符合gossip learning）
    for node1, node2 in active_pairs:
        # 生成双向对称的带宽需求
        # 正向连接
        bw_min_1 = random.randint(*BW_MIN_RANGE)
        bw_max_1 = random.randint(max(bw_min_1, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
        
        # 反向连接（双向对称，但带宽需求可以略有不同）
        bw_min_2 = random.randint(*BW_MIN_RANGE)
        bw_max_2 = random.randint(max(bw_min_2, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
        
        # 正向链接
        if node1 not in links:
            links[node1] = []
        links[node1].append({
            "dest": node2,
            "bw_min": f"{bw_min_1}mbps",
            "bw_max": f"{bw_max_1}mbps"
        })
        
        # 反向链接（gossip learning要求双向对称）
        if node2 not in links:
            links[node2] = []
        links[node2].append({
            "dest": node1,
            "bw_min": f"{bw_min_2}mbps",
            "bw_max": f"{bw_max_2}mbps"
        })
    
    return {
        "nodes": nodes,
        "links": links
    }

def generate_all_datasets():
    """
    生成5组数据集
    每个数据集的资源利用率都在0.1-0.5范围内随机生成
    """
    datasets = []
    
    # 生成5组数据集
    for i in range(1, 6):
        # 设置随机种子以确保可重现性（但每组数据不同）
        random.seed(i * 1000)
        
        workload_data = generate_workload_data()
        task_data = generate_task_request_data()
        
        datasets.append({
            "dataset_id": i,
            "utilization_rate": "random(0.1-0.5)",  # 标记为随机利用率
            "workload_config": workload_data,
            "task_links_config": task_data
        })
        
        print(f"✓ 生成数据集 {i}: 利用率0.1-0.5随机")
    
    return datasets

def save_datasets(datasets, output_dir='workload_datasets'):
    """
    保存数据集到文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        
        # 保存workload配置
        workload_file = os.path.join(output_dir, f'workload_config_{dataset_id}.json')
        with open(workload_file, 'w', encoding='utf-8') as f:
            json.dump(dataset["workload_config"], f, indent=2, ensure_ascii=False)
        
        # 保存task links配置
        task_file = os.path.join(output_dir, f'task_links_config_{dataset_id}.json')
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(dataset["task_links_config"], f, indent=2, ensure_ascii=False)
    
    # 保存数据集索引
    index_file = os.path.join(output_dir, 'datasets_index.json')
    index_data = {
        "description": "负载数据和任务请求数据集索引",
        "total_datasets": len(datasets),
        "datasets": [
            {
                "dataset_id": d["dataset_id"],
                "utilization_rate": d["utilization_rate"],
                "workload_file": f'workload_config_{d["dataset_id"]}.json',
                "task_links_file": f'task_links_config_{d["dataset_id"]}.json'
            }
            for d in datasets
        ]
    }
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 所有数据集已保存到 {output_dir}/ 目录")
    print(f"✓ 索引文件: {index_file}")

if __name__ == '__main__':
    print("开始生成负载数据和任务请求数据...")
    print("=" * 60)
    
    datasets = generate_all_datasets()
    save_datasets(datasets)
    
    print("\n生成完成！")
    print("\n数据集概览:")
    for dataset in datasets:
        print(f"  数据集 {dataset['dataset_id']}: 利用率0.1-0.5随机")
        print(f"    - 负载节点数: {len(dataset['workload_config']['workloads'])}")
        print(f"    - 任务节点数: {len(dataset['task_links_config']['nodes'])}")
