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
    - 任务节点间连接符合 el 任务（树状 / 分层聚合 Edge Learning）的特点
    - 形成以 p1 为顶层、n1 为中间聚合节点、n2~n4 为叶子节点的分层拓扑
    - 父子节点之间优先双向连接，叶子之间可有少量额外连接
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

    # el 任务特点：分层 / 树状拓扑
    # 角色划分：
    # - p1：顶层聚合节点（top / trainer）
    # - n1：中间聚合节点（aggregator）
    # - n2, n3, n4：叶子节点（worker）
    top_node = TASK_NODES[0]      # p1
    mid_node = TASK_NODES[1]      # n1
    leaf_nodes = TASK_NODES[2:]   # n2, n3, n4

    def add_link(src, dest, bw_min=None, bw_max=None):
        """辅助函数：添加一条带宽范围的单向连接"""
        if bw_min is None:
            bw_min = random.randint(*BW_MIN_RANGE)
        if bw_max is None:
            bw_max = random.randint(max(bw_min, BW_MAX_RANGE[0]), BW_MAX_RANGE[1])

        if src not in links:
            links[src] = []
        links[src].append({
            "dest": dest,
            "bw_min": f"{bw_min}mbps",
            "bw_max": f"{bw_max}mbps"
        })

    # 1. 顶层节点 p1 与中间节点 n1 双向强连接
    add_link(top_node, mid_node)
    add_link(mid_node, top_node)

    # 2. 中间节点 n1 与所有叶子节点双向连接（形成星型 / 树的主干）
    for leaf in leaf_nodes:
        add_link(mid_node, leaf)
        add_link(leaf, mid_node)

    # 3. 叶子之间适度互联（少量横向边），增强鲁棒性但保持“树状为主”
    leaf_pairs = []
    for i, n1_leaf in enumerate(leaf_nodes):
        for n2_leaf in leaf_nodes[i + 1:]:
            leaf_pairs.append((n1_leaf, n2_leaf))

    # 最多为叶子对的 50% 添加横向连接
    if leaf_pairs:
        num_extra = random.randint(0, max(1, round(len(leaf_pairs) * 0.5)))
        extra_pairs = random.sample(leaf_pairs, num_extra)

        for node1, node2 in extra_pairs:
            # 70% 概率双向，30% 概率单向
            if random.random() < 0.7:
                add_link(node1, node2)
                add_link(node2, node1)
            else:
                src, dest = random.choice([(node1, node2), (node2, node1)])
                add_link(src, dest)

    # 4. 顶层节点 p1 与部分叶子可有直接连接（类似 Task_el/links_range 中 p1-n4）
    for leaf in leaf_nodes:
        if random.random() < 0.5:  # 50% 概率添加顶层到叶子的直连
            add_link(top_node, leaf)
            # 30% 概率再加反向链接 leaf -> p1
            if random.random() < 0.3:
                add_link(leaf, top_node)

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
        random.seed(i * 1002)
        
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
