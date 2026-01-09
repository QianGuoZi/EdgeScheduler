#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选择并更新负载配置和任务请求配置文件的脚本
"""

import json
import os
import sys
import shutil
from datetime import datetime

# 配置文件路径
WORKLOAD_CONFIG_PATH = '/home/qianguo/Edge-Scheduler/Controller/workload_config.json'
TASK_LINKS_CONFIG_PATH = '/home/qianguo/Edge-Scheduler/Controller/task_links/1/links_range.json'

# 数据集目录
DATASETS_DIR = 'workload_datasets'
INDEX_FILE = os.path.join(DATASETS_DIR, 'datasets_index.json')

def load_datasets_index():
    """
    加载数据集索引
    """
    if not os.path.exists(INDEX_FILE):
        print(f"❌ 错误: 数据集索引文件不存在: {INDEX_FILE}")
        print("   请先运行 generate_workload_data.py 生成数据集")
        sys.exit(1)
    
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_available_datasets():
    """
    列出可用的数据集
    """
    index = load_datasets_index()
    print("\n可用的数据集:")
    print("=" * 60)
    for dataset in index["datasets"]:
        dataset_id = dataset["dataset_id"]
        utilization = dataset.get("utilization_rate", "random(0.1-0.5)")
        if isinstance(utilization, (int, float)):
            print(f"  数据集 {dataset_id}: 利用率 {utilization*100:.0f}%")
        else:
            print(f"  数据集 {dataset_id}: 利用率{utilization}")
    print("=" * 60)
    return index

def backup_configs():
    """
    备份当前配置文件
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join('config_backups', timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    backups = {}
    
    if os.path.exists(WORKLOAD_CONFIG_PATH):
        backup_path = os.path.join(backup_dir, 'workload_config.json')
        shutil.copy2(WORKLOAD_CONFIG_PATH, backup_path)
        backups['workload'] = backup_path
        print(f"✓ 已备份 workload_config.json -> {backup_path}")
    
    if os.path.exists(TASK_LINKS_CONFIG_PATH):
        # 确保目标目录存在
        os.makedirs(os.path.dirname(TASK_LINKS_CONFIG_PATH), exist_ok=True)
        backup_path = os.path.join(backup_dir, 'links_range.json')
        shutil.copy2(TASK_LINKS_CONFIG_PATH, backup_path)
        backups['task_links'] = backup_path
        print(f"✓ 已备份 links_range.json -> {backup_path}")
    
    return backup_dir, backups

def update_configs(dataset_id):
    """
    使用指定数据集更新配置文件
    """
    index = load_datasets_index()
    
    # 查找指定的数据集
    dataset_info = None
    for d in index["datasets"]:
        if d["dataset_id"] == dataset_id:
            dataset_info = d
            break
    
    if not dataset_info:
        print(f"❌ 错误: 找不到数据集 {dataset_id}")
        sys.exit(1)
    
    # 备份当前配置
    print("\n备份当前配置文件...")
    backup_dir, backups = backup_configs()
    
    # 加载数据集
    workload_file = os.path.join(DATASETS_DIR, dataset_info["workload_file"])
    task_links_file = os.path.join(DATASETS_DIR, dataset_info["task_links_file"])
    
    if not os.path.exists(workload_file):
        print(f"❌ 错误: 找不到文件 {workload_file}")
        sys.exit(1)
    
    if not os.path.exists(task_links_file):
        print(f"❌ 错误: 找不到文件 {task_links_file}")
        sys.exit(1)
    
    # 读取数据集
    with open(workload_file, 'r', encoding='utf-8') as f:
        workload_data = json.load(f)
    
    with open(task_links_file, 'r', encoding='utf-8') as f:
        task_links_data = json.load(f)
    
    # 更新配置文件
    print(f"\n更新配置文件...")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(WORKLOAD_CONFIG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TASK_LINKS_CONFIG_PATH), exist_ok=True)
    
    # 写入workload_config.json
    with open(WORKLOAD_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(workload_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 已更新 {WORKLOAD_CONFIG_PATH}")
    
    # 写入task_links/1/links_range.json
    with open(TASK_LINKS_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(task_links_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 已更新 {TASK_LINKS_CONFIG_PATH}")
    
    utilization_str = dataset_info.get('utilization_rate', 'random(0.1-0.5)')
    if isinstance(utilization_str, (int, float)):
        utilization_display = f"{utilization_str*100:.0f}%"
    else:
        utilization_display = utilization_str
    print(f"\n✓ 成功使用数据集 {dataset_id} (利用率 {utilization_display}) 更新配置")
    print(f"✓ 备份位置: {backup_dir}")

def main():
    """
    主函数
    """
    if len(sys.argv) < 2:
        # 显示可用数据集列表
        list_available_datasets()
        print("\n用法:")
        print(f"  python {sys.argv[0]} <dataset_id>")
        print("\n示例:")
        print(f"  python {sys.argv[0]} 1  # 使用数据集1（利用率0.1-0.5随机）")
        print(f"  python {sys.argv[0]} 3  # 使用数据集3（利用率0.1-0.5随机）")
        sys.exit(0)
    
    try:
        dataset_id = int(sys.argv[1])
    except ValueError:
        print(f"❌ 错误: 无效的数据集ID: {sys.argv[1]}")
        print("   数据集ID必须是1-5之间的整数")
        sys.exit(1)
    
    # 检查数据集ID范围
    index = load_datasets_index()
    if dataset_id < 1 or dataset_id > index["total_datasets"]:
        print(f"❌ 错误: 数据集ID必须在1-{index['total_datasets']}之间")
        sys.exit(1)
    
    update_configs(dataset_id)

if __name__ == '__main__':
    main()
