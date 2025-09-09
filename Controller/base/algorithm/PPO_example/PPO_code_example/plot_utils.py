#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘图工具模块，提供中文字体支持和通用绘图功能
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

def setup_chinese_font():
    """设置中文字体支持"""
    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei',           # Windows 黑体
        'Microsoft YaHei',  # Windows 微软雅黑
        'PingFang SC',      # macOS 苹方
        'Hiragino Sans GB', # macOS 冬青黑体
        'WenQuanYi Micro Hei', # Linux 文泉驿微米黑
        'DejaVu Sans',      # 通用字体
        'Arial Unicode MS', # 通用字体
        'sans-serif'        # 默认字体
    ]
    
    # 检查系统可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 选择第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        print(f"✅ 使用字体: {selected_font}")
    else:
        print("⚠️ 未找到合适的中文字体，使用默认字体")
    
    # 修复负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

def create_entropy_plot(training_stats, save_path=None, figsize=(15, 10)):
    """
    创建策略熵监控图表
    
    Args:
        training_stats: 训练统计数据字典
        save_path: 保存路径，如果为None则不保存
        figsize: 图表大小
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 检查是否有熵数据
    if not training_stats.get('mapping_actor_entropies'):
        print("❌ 没有找到策略熵数据")
        return
    
    # 1. 策略熵曲线
    axes[0, 0].plot(training_stats['mapping_actor_entropies'], 
                   label='Mapping Actor Entropy', alpha=0.7, color='blue')
    axes[0, 0].plot(training_stats['bandwidth_actor_entropies'], 
                   label='Bandwidth Actor Entropy', alpha=0.7, color='red')
    axes[0, 0].plot(training_stats['total_entropies'], 
                   label='Total Entropy', linewidth=2, color='green')
    axes[0, 0].set_title('Policy Entropy Monitoring', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Entropy Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 策略熵移动平均
    window_size = min(50, len(training_stats['total_entropies']) // 2)
    if window_size > 0:
        mapping_entropy_ma = []
        bandwidth_entropy_ma = []
        total_entropy_ma = []
        
        for i in range(len(training_stats['mapping_actor_entropies'])):
            start = max(0, i - window_size + 1)
            mapping_entropy_ma.append(np.mean(training_stats['mapping_actor_entropies'][start:i+1]))
            bandwidth_entropy_ma.append(np.mean(training_stats['bandwidth_actor_entropies'][start:i+1]))
            total_entropy_ma.append(np.mean(training_stats['total_entropies'][start:i+1]))
        
        axes[0, 1].plot(mapping_entropy_ma, label=f'Mapping Actor Entropy (MA-{window_size})', 
                       alpha=0.7, color='blue')
        axes[0, 1].plot(bandwidth_entropy_ma, label=f'Bandwidth Actor Entropy (MA-{window_size})', 
                       alpha=0.7, color='red')
        axes[0, 1].plot(total_entropy_ma, label=f'Total Entropy (MA-{window_size})', 
                       linewidth=2, color='green')
        axes[0, 1].set_title('Policy Entropy Moving Average', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Entropy Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 奖励曲线
    if training_stats.get('episode_rewards'):
        axes[1, 0].plot(training_stats['episode_rewards'], color='purple', alpha=0.7)
        axes[1, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 约束违反率
    if training_stats.get('constraint_violations'):
        violation_rates = [1 if v > 0 else 0 for v in training_stats['constraint_violations']]
        window_size = min(50, len(violation_rates) // 2)
        if window_size > 0:
            moving_avg = []
            for i in range(len(violation_rates)):
                start = max(0, i - window_size + 1)
                moving_avg.append(np.mean(violation_rates[start:i+1]))
            
            axes[1, 1].plot(moving_avg, color='orange', linewidth=2)
            axes[1, 1].set_title('Constraint Violation Rate (Moving Average)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Violation Rate')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 熵监控图表已保存: {save_path}")
    
    return fig

def create_comprehensive_training_plot(training_stats, save_path=None, figsize=(20, 15)):
    """
    创建综合训练监控图表
    
    Args:
        training_stats: 训练统计数据字典
        save_path: 保存路径
        figsize: 图表大小
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建3x3的子图（增加一行来显示奖励组件）
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    
    # 1. 奖励曲线
    if training_stats.get('episode_rewards'):
        axes[0, 0].plot(training_stats['episode_rewards'], color='purple', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 约束违反率
    if training_stats.get('constraint_violations'):
        violation_rates = [1 if v > 0 else 0 for v in training_stats['constraint_violations']]
        window_size = min(50, len(violation_rates) // 2)
        if window_size > 0:
            moving_avg = []
            for i in range(len(violation_rates)):
                start = max(0, i - window_size + 1)
                moving_avg.append(np.mean(violation_rates[start:i+1]))
            
            axes[0, 1].plot(moving_avg, color='orange', linewidth=2)
            axes[0, 1].set_title('Constraint Violation Rate (Moving Average)', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Violation Rate')
            axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 奖励组件分析
    if training_stats.get('load_balance_rewards') and training_stats.get('bandwidth_satisfaction_rewards'):
        axes[0, 2].plot(training_stats['load_balance_rewards'], 
                       label='Load Balance Reward', alpha=0.7, color='blue')
        axes[0, 2].plot(training_stats['bandwidth_satisfaction_rewards'], 
                       label='Bandwidth Satisfaction Reward', alpha=0.7, color='red')
        if training_stats.get('total_rewards'):
            axes[0, 2].plot(training_stats['total_rewards'], 
                           label='Total Reward', linewidth=2, color='green')
        axes[0, 2].set_title('Reward Components Analysis', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Reward Value')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 平均奖励（移动平均）
    if training_stats.get('episode_rewards'):
        window_size = min(50, len(training_stats['episode_rewards']) // 2)
        if window_size > 0:
            reward_moving_avg = []
            for i in range(len(training_stats['episode_rewards'])):
                start = max(0, i - window_size + 1)
                reward_moving_avg.append(np.mean(training_stats['episode_rewards'][start:i+1]))
            
            axes[1, 0].plot(reward_moving_avg, color='green', linewidth=2)
            axes[1, 0].set_title('Average Reward (Moving Average)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Average Reward')
            axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 约束违反数量
    if training_stats.get('constraint_violations'):
        axes[1, 1].plot(training_stats['constraint_violations'], color='red', alpha=0.7)
        axes[1, 1].set_title('Constraint Violations', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Number of Violations')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 奖励组件移动平均
    if training_stats.get('load_balance_rewards') and training_stats.get('bandwidth_satisfaction_rewards'):
        window_size = min(50, len(training_stats['load_balance_rewards']) // 2)
        if window_size > 0:
            load_balance_ma = []
            bandwidth_satisfaction_ma = []
            
            for i in range(len(training_stats['load_balance_rewards'])):
                start = max(0, i - window_size + 1)
                load_balance_ma.append(np.mean(training_stats['load_balance_rewards'][start:i+1]))
                bandwidth_satisfaction_ma.append(np.mean(training_stats['bandwidth_satisfaction_rewards'][start:i+1]))
            
            axes[1, 2].plot(load_balance_ma, 
                           label=f'Load Balance (MA-{window_size})', alpha=0.7, color='blue')
            axes[1, 2].plot(bandwidth_satisfaction_ma, 
                           label=f'Bandwidth Satisfaction (MA-{window_size})', alpha=0.7, color='red')
            axes[1, 2].set_title('Reward Components Moving Average', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Reward Value')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
    
    # 7. 策略熵曲线
    if training_stats.get('mapping_actor_entropies'):
        axes[2, 0].plot(training_stats['mapping_actor_entropies'], 
                       label='Mapping Actor Entropy', alpha=0.7, color='blue')
        axes[2, 0].plot(training_stats['bandwidth_actor_entropies'], 
                       label='Bandwidth Actor Entropy', alpha=0.7, color='red')
        axes[2, 0].plot(training_stats['total_entropies'], 
                       label='Total Entropy', linewidth=2, color='green')
        axes[2, 0].set_title('Policy Entropy Monitoring', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Entropy Value')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # 8. 温度曲线
    if training_stats.get('temperatures'):
        axes[2, 1].plot(training_stats['temperatures'], label='Temperature', 
                       color='purple', linewidth=2)
        axes[2, 1].set_title('Temperature Schedule', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Temperature')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    # 9. 奖励组件相关性分析
    if training_stats.get('load_balance_rewards') and training_stats.get('bandwidth_satisfaction_rewards'):
        # 计算相关性
        load_balance = np.array(training_stats['load_balance_rewards'])
        bandwidth_satisfaction = np.array(training_stats['bandwidth_satisfaction_rewards'])
        
        # 计算移动窗口相关性
        window_size = min(100, len(load_balance) // 4)
        if window_size > 0:
            correlations = []
            for i in range(window_size, len(load_balance)):
                corr = np.corrcoef(load_balance[i-window_size:i], 
                                 bandwidth_satisfaction[i-window_size:i])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            if correlations:
                axes[2, 2].plot(range(window_size, len(load_balance)), correlations, 
                               color='brown', linewidth=2)
                axes[2, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[2, 2].set_title('Reward Components Correlation (Moving Window)', fontsize=14, fontweight='bold')
                axes[2, 2].set_xlabel('Episode')
                axes[2, 2].set_ylabel('Correlation Coefficient')
                axes[2, 2].grid(True, alpha=0.3)
                axes[2, 2].set_ylim(-1, 1)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 综合训练图表已保存: {save_path}")
    
    return fig

def create_reward_components_plot(training_stats, save_path=None, figsize=(20, 12)):
    """
    创建专门的奖励组件分析图表
    
    Args:
        training_stats: 训练统计数据字典
        save_path: 保存路径
        figsize: 图表大小
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 检查是否有奖励组件数据
    if not training_stats.get('load_balance_rewards') or not training_stats.get('bandwidth_satisfaction_rewards'):
        print("❌ 没有找到奖励组件数据")
        return None
    
    # 创建2x3的子图
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 第一行：奖励组件曲线
    # 1. 负载均衡奖励
    axes[0, 0].plot(training_stats['load_balance_rewards'], color='blue', alpha=0.7)
    axes[0, 0].set_title('Load Balance Reward', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 带宽满足度奖励
    axes[0, 1].plot(training_stats['bandwidth_satisfaction_rewards'], color='red', alpha=0.7)
    axes[0, 1].set_title('Bandwidth Satisfaction Reward', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 总奖励（如果有的话）
    if training_stats.get('total_rewards'):
        axes[0, 2].plot(training_stats['total_rewards'], color='green', alpha=0.7)
        axes[0, 2].set_title('Total Reward', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Reward Value')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 第二行：移动平均和相关性分析
    # 4. 奖励组件移动平均
    window_size = min(50, len(training_stats['load_balance_rewards']) // 2)
    if window_size > 0:
        load_balance_ma = []
        bandwidth_satisfaction_ma = []
        
        for i in range(len(training_stats['load_balance_rewards'])):
            start = max(0, i - window_size + 1)
            load_balance_ma.append(np.mean(training_stats['load_balance_rewards'][start:i+1]))
            bandwidth_satisfaction_ma.append(np.mean(training_stats['bandwidth_satisfaction_rewards'][start:i+1]))
        
        axes[1, 0].plot(load_balance_ma, 
                       label=f'Load Balance (MA-{window_size})', color='blue', linewidth=2)
        axes[1, 0].plot(bandwidth_satisfaction_ma, 
                       label=f'Bandwidth Satisfaction (MA-{window_size})', color='red', linewidth=2)
        axes[1, 0].set_title('Reward Components Moving Average', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 奖励组件分布对比
    if len(training_stats['load_balance_rewards']) > 10:
        load_balance = np.array(training_stats['load_balance_rewards'])
        bandwidth_satisfaction = np.array(training_stats['bandwidth_satisfaction_rewards'])
        
        # 计算统计信息
        axes[1, 1].hist(load_balance, bins=20, alpha=0.7, color='blue', 
                        label=f'Load Balance\nμ={np.mean(load_balance):.3f}\nσ={np.std(load_balance):.3f}')
        axes[1, 1].hist(bandwidth_satisfaction, bins=20, alpha=0.7, color='red',
                        label=f'Bandwidth Satisfaction\nμ={np.mean(bandwidth_satisfaction):.3f}\nσ={np.std(bandwidth_satisfaction):.3f}')
        axes[1, 1].set_title('Reward Components Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Reward Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 奖励组件相关性热图
    if len(training_stats['load_balance_rewards']) > 20:
        # 计算不同窗口大小的相关性
        window_sizes = [10, 25, 50]
        correlations = []
        window_labels = []
        
        for ws in window_sizes:
            if len(training_stats['load_balance_rewards']) >= ws:
                corr = np.corrcoef(training_stats['load_balance_rewards'][-ws:], 
                                 training_stats['bandwidth_satisfaction_rewards'][-ws:])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
                window_labels.append(f'Last {ws}')
        
        if correlations:
            # 创建相关性矩阵
            corr_matrix = np.array([[1, correlations[0]], [correlations[0], 1]])
            
            im = axes[1, 2].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 2].set_xticks([0, 1])
            axes[1, 2].set_yticks([0, 1])
            axes[1, 2].set_xticklabels(['Load Balance', 'Bandwidth Satisfaction'])
            axes[1, 2].set_yticklabels(['Load Balance', 'Bandwidth Satisfaction'])
            axes[1, 2].set_title('Reward Components Correlation Matrix', fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for i in range(2):
                for j in range(2):
                    text = axes[1, 2].text(j, i, f'{corr_matrix[i, j]:.3f}',
                                          ha="center", va="center", color="black", fontweight='bold')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        # 修改保存路径，添加奖励组件标识
        base_path = save_path.replace('.png', '')
        reward_plot_path = f"{base_path}_reward_components.png"
        plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 奖励组件分析图表已保存: {reward_plot_path}")
    
    return fig
