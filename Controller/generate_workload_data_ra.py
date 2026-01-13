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
    - 任务节点间连接符合ra任务（Ring Allreduce）的特点
    - 构建环形拓扑结构，包含中心节点和环形连接
    - 链接可以是单向的，不一定是双向对称的
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
    
    # ra任务特点：环形拓扑 + 中心节点
    # 1. 确定中心节点（通常是p1或n1）
    center_node = TASK_NODES[0]  # p1作为中心节点
    worker_nodes = TASK_NODES[1:]  # n1, n2, n3, n4作为工作节点
    
    # 2. 构建环形连接（工作节点之间形成环）
    # 每个工作节点连接到下一个节点，形成环形拓扑
    for i, node in enumerate(worker_nodes):
        next_node = worker_nodes[(i + 1) % len(worker_nodes)]  # 环形下一个节点
        
        # 生成带宽需求（单向或双向）
        bw_min_forward = random.randint(*BW_MIN_RANGE)
        bw_max_forward = random.randint(max(bw_min_forward, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
        
        # 环形连接：正向连接（当前节点到下一个节点）
        if node not in links:
            links[node] = []
        links[node].append({
            "dest": next_node,
            "bw_min": f"{bw_min_forward}mbps",
            "bw_max": f"{bw_max_forward}mbps"
        })
        
        # 50%概率添加反向连接（使环形双向）
        if random.random() < 0.5:
            bw_min_backward = random.randint(*BW_MIN_RANGE)
            bw_max_backward = random.randint(max(bw_min_backward, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
            
            if next_node not in links:
                links[next_node] = []
            links[next_node].append({
                "dest": node,
                "bw_min": f"{bw_min_backward}mbps",
                "bw_max": f"{bw_max_backward}mbps"
            })
    
    # 3. 中心节点连接到所有或大部分工作节点
    # 中心节点连接到70%-100%的工作节点
    num_center_connections = random.randint(
        round(len(worker_nodes) * 0.7), 
        len(worker_nodes)
    )
    center_targets = random.sample(worker_nodes, num_center_connections)
    
    for target in center_targets:
        # 中心节点到工作节点
        bw_min_center = random.randint(*BW_MIN_RANGE)
        bw_max_center = random.randint(max(bw_min_center, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
        
        if center_node not in links:
            links[center_node] = []
        links[center_node].append({
            "dest": target,
            "bw_min": f"{bw_min_center}mbps",
            "bw_max": f"{bw_max_center}mbps"
        })
        
        # 60%概率添加工作节点到中心节点的反向连接
        if random.random() < 0.6:
            bw_min_back = random.randint(*BW_MIN_RANGE)
            bw_max_back = random.randint(max(bw_min_back, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
            
            if target not in links:
                links[target] = []
            links[target].append({
                "dest": center_node,
                "bw_min": f"{bw_min_back}mbps",
                "bw_max": f"{bw_max_back}mbps"
            })
    
    # 4. 工作节点之间的额外连接（稀疏连接，不是全连接）
    # 随机添加一些工作节点之间的直接连接，但保持稀疏性
    # 最多添加30%的额外连接对
    all_worker_pairs = []
    for i, node1 in enumerate(worker_nodes):
        for node2 in worker_nodes[i+1:]:
            all_worker_pairs.append((node1, node2))
    
    # 随机选择最多30%的额外连接对
    num_extra_connections = random.randint(0, round(len(all_worker_pairs) * 0.3))
    extra_pairs = random.sample(all_worker_pairs, num_extra_connections)
    
    for node1, node2 in extra_pairs:
        # 检查是否已存在连接（避免重复）
        existing_link = False
        if node1 in links:
            existing_link = any(link["dest"] == node2 for link in links[node1])
        if node2 in links:
            existing_link = existing_link or any(link["dest"] == node1 for link in links.get(node2, []))
        
        if not existing_link:
            # 单向连接（ra任务不要求完全双向对称）
            if random.random() < 0.7:  # 70%概率单向连接
                direction = random.choice([(node1, node2), (node2, node1)])
                source, dest = direction
                
                bw_min = random.randint(*BW_MIN_RANGE)
                bw_max = random.randint(max(bw_min, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
                
                if source not in links:
                    links[source] = []
                links[source].append({
                    "dest": dest,
                    "bw_min": f"{bw_min}mbps",
                    "bw_max": f"{bw_max}mbps"
                })
            else:  # 30%概率双向连接
                bw_min_1 = random.randint(*BW_MIN_RANGE)
                bw_max_1 = random.randint(max(bw_min_1, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
                
                bw_min_2 = random.randint(*BW_MIN_RANGE)
                bw_max_2 = random.randint(max(bw_min_2, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])
                
                if node1 not in links:
                    links[node1] = []
                links[node1].append({
                    "dest": node2,
                    "bw_min": f"{bw_min_1}mbps",
                    "bw_max": f"{bw_max_1}mbps"
                })
                
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
        random.seed(i * 1001)
        
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
