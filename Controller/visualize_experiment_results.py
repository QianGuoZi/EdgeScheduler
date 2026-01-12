#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果可视化脚本
用于绘制实验数据的图表
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

EXPERIMENT_RESULTS_DIR = '/home/qianguo/Edge-Scheduler/Controller/experiment_results'
EXPERIMENT_CSV_FILE = os.path.join(EXPERIMENT_RESULTS_DIR, 'experiment_results.csv')
OUTPUT_DIR = os.path.join(EXPERIMENT_RESULTS_DIR, 'plots')


def load_data():
    """加载实验数据"""
    if not os.path.exists(EXPERIMENT_CSV_FILE):
        print(f"❌ 数据文件不存在: {EXPERIMENT_CSV_FILE}")
        return None
    
    df = pd.read_csv(EXPERIMENT_CSV_FILE)
    print(f"✓ 已加载 {len(df)} 条实验记录")
    return df


def plot_load_balance_degree(df, output_dir):
    """绘制负载均衡度图表"""
    plt.figure(figsize=(12, 6))
    
    # 按数据集分组
    for dataset_id in sorted(df['dataset_id'].unique()):
        dataset_data = df[df['dataset_id'] == dataset_id]
        plt.plot(dataset_data['experiment_id'], dataset_data['load_balance_degree'], 
                'o-', label=f'数据集 {dataset_id}', linewidth=2, markersize=6)
    
    plt.xlabel('实验ID', fontsize=12)
    plt.ylabel('负载均衡度 (L)', fontsize=12)
    plt.title('负载均衡度变化趋势', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'load_balance_degree.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_file}")


def plot_bandwidth_satisfaction(df, output_dir):
    """绘制带宽满足度图表"""
    plt.figure(figsize=(12, 6))
    
    # 按数据集分组
    for dataset_id in sorted(df['dataset_id'].unique()):
        dataset_data = df[df['dataset_id'] == dataset_id]
        plt.plot(dataset_data['experiment_id'], dataset_data['bandwidth_satisfaction'], 
                's-', label=f'数据集 {dataset_id}', linewidth=2, markersize=6)
    
    plt.xlabel('实验ID', fontsize=12)
    plt.ylabel('带宽满足度 (D_BW)', fontsize=12)
    plt.title('带宽满足度变化趋势', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'bandwidth_satisfaction.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_file}")


def plot_composite_score(df, output_dir):
    """绘制综合指标图表"""
    plt.figure(figsize=(12, 6))
    
    # 按数据集分组
    for dataset_id in sorted(df['dataset_id'].unique()):
        dataset_data = df[df['dataset_id'] == dataset_id]
        plt.plot(dataset_data['experiment_id'], dataset_data['composite_score'], 
                '^-', label=f'数据集 {dataset_id}', linewidth=2, markersize=6)
    
    plt.xlabel('实验ID', fontsize=12)
    plt.ylabel('综合指标', fontsize=12)
    plt.title('综合指标变化趋势', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'composite_score.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_file}")


def plot_comparison(df, output_dir):
    """绘制三个指标的对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 负载均衡度
    for dataset_id in sorted(df['dataset_id'].unique()):
        dataset_data = df[df['dataset_id'] == dataset_id]
        axes[0].plot(dataset_data['experiment_id'], dataset_data['load_balance_degree'], 
                     'o-', label=f'数据集 {dataset_id}', linewidth=2, markersize=5)
    axes[0].set_ylabel('负载均衡度 (L)', fontsize=11)
    axes[0].set_title('负载均衡度', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # 带宽满足度
    for dataset_id in sorted(df['dataset_id'].unique()):
        dataset_data = df[df['dataset_id'] == dataset_id]
        axes[1].plot(dataset_data['experiment_id'], dataset_data['bandwidth_satisfaction'], 
                     's-', label=f'数据集 {dataset_id}', linewidth=2, markersize=5)
    axes[1].set_ylabel('带宽满足度 (D_BW)', fontsize=11)
    axes[1].set_title('带宽满足度', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # 综合指标
    for dataset_id in sorted(df['dataset_id'].unique()):
        dataset_data = df[df['dataset_id'] == dataset_id]
        axes[2].plot(dataset_data['experiment_id'], dataset_data['composite_score'], 
                     '^-', label=f'数据集 {dataset_id}', linewidth=2, markersize=5)
    axes[2].set_xlabel('实验ID', fontsize=11)
    axes[2].set_ylabel('综合指标', fontsize=11)
    axes[2].set_title('综合指标', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_file}")


def plot_statistics_by_dataset(df, output_dir):
    """按数据集统计并绘制"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 按数据集分组统计
    stats = df.groupby('dataset_id').agg({
        'load_balance_degree': ['mean', 'std'],
        'bandwidth_satisfaction': ['mean', 'std'],
        'composite_score': ['mean', 'std']
    }).reset_index()
    
    datasets = stats['dataset_id'].values
    
    # 负载均衡度
    means = stats[('load_balance_degree', 'mean')].values
    stds = stats[('load_balance_degree', 'std')].values
    axes[0].bar(datasets, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0].set_xlabel('数据集ID', fontsize=11)
    axes[0].set_ylabel('负载均衡度 (L)', fontsize=11)
    axes[0].set_title('各数据集负载均衡度统计', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 带宽满足度
    means = stats[('bandwidth_satisfaction', 'mean')].values
    stds = stats[('bandwidth_satisfaction', 'std')].values
    axes[1].bar(datasets, means, yerr=stds, capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    axes[1].set_xlabel('数据集ID', fontsize=11)
    axes[1].set_ylabel('带宽满足度 (D_BW)', fontsize=11)
    axes[1].set_title('各数据集带宽满足度统计', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 综合指标
    means = stats[('composite_score', 'mean')].values
    stds = stats[('composite_score', 'std')].values
    axes[2].bar(datasets, means, yerr=stds, capsize=5, alpha=0.7, color='salmon', edgecolor='darkred')
    axes[2].set_xlabel('数据集ID', fontsize=11)
    axes[2].set_ylabel('综合指标', fontsize=11)
    axes[2].set_title('各数据集综合指标统计', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'statistics_by_dataset.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实验结果可视化')
    parser.add_argument('--all', action='store_true', help='生成所有图表')
    parser.add_argument('--load-balance', action='store_true', help='绘制负载均衡度')
    parser.add_argument('--bandwidth', action='store_true', help='绘制带宽满足度')
    parser.add_argument('--composite', action='store_true', help='绘制综合指标')
    parser.add_argument('--comparison', action='store_true', help='绘制对比图')
    parser.add_argument('--statistics', action='store_true', help='绘制统计图')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 如果没有指定任何选项，默认生成所有图表
    if not any([args.load_balance, args.bandwidth, args.composite, 
                args.comparison, args.statistics]):
        args.all = True
    
    if args.all or args.load_balance:
        plot_load_balance_degree(df, OUTPUT_DIR)
    
    if args.all or args.bandwidth:
        plot_bandwidth_satisfaction(df, OUTPUT_DIR)
    
    if args.all or args.composite:
        plot_composite_score(df, OUTPUT_DIR)
    
    if args.all or args.comparison:
        plot_comparison(df, OUTPUT_DIR)
    
    if args.all or args.statistics:
        plot_statistics_by_dataset(df, OUTPUT_DIR)
    
    print(f"\n✓ 所有图表已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
