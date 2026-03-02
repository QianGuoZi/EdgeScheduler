#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立绘图脚本：两PPO模型对比结果可视化（中文）"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def load_results(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def configure_chinese_font(font_path: str = None):
    simsun_path = '/usr/share/fonts/myfonts/simsun.ttc'
    times_path = '/usr/share/fonts/myfonts/times.ttf'

    # 将字体文件添加到 matplotlib 的字体管理器（确保被识别）
    for path in [simsun_path, times_path]:
        if os.path.exists(path):
            fm.fontManager.addfont(path)

    # 获取字体名称
    simsun_prop = fm.FontProperties(fname=simsun_path)
    simsun_name = simsun_prop.get_name()
    times_prop = fm.FontProperties(fname=times_path)
    times_name = times_prop.get_name()

    # 直接设置字体搜索顺序：先 Times New Roman，再 SimSun
    plt.rcParams['font.family'] = [times_name, simsun_name]
    plt.rcParams['axes.unicode_minus'] = False

    print(f'英文主字体: {times_name}')
    print(f'中文主字体: {simsun_name}')
    print(f'字体顺序: {plt.rcParams["font.family"]}')


def configure_plot_sizes(base_font_size: int = 20):
    """统一设置图中文字大小。"""
    title_size = base_font_size + 2
    label_size = base_font_size
    tick_size = max(base_font_size - 1, 10)
    legend_size = max(base_font_size - 1, 10)

    plt.rcParams['font.size'] = base_font_size
    plt.rcParams['axes.titlesize'] = title_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['legend.fontsize'] = legend_size


def _safe_get(d, *keys, default=float('nan')):
    try:
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return default


def _plot_with_nan(ax, x, y, label, marker, color):
    valid = [i for i, v in enumerate(y) if not np.isnan(v)]
    if not valid:
        return
    x_v = [x[i] for i in valid]
    y_v = [y[i] for i in valid]
    ax.plot(x_v, y_v, label=label, marker=marker, color=color)


def plot_from_results(results: dict, out_dir: str = 'test_results') -> list:
    os.makedirs(out_dir, exist_ok=True)

    colors = ['#2E86AB', '#A23B72']

    virtual_nodes_range = results['test_config']['virtual_nodes_range']
    detailed = results['detailed_results']

    model1_name = results['model1']['name']
    model2_name = results['model2']['name']

    label_map = {
        model1_name: 'CES_PPO',
        model2_name: 'CES_PPO（无启发式奖励）'
    }

    def display_label(name: str) -> str:
        return label_map.get(name, name)

    def collect(model_key: str, metric_key: str):
        means = []
        stds = []
        for n in virtual_nodes_range:
            entry = detailed.get(n) or detailed.get(str(n))
            if entry is None:
                means.append(np.nan)
                stds.append(np.nan)
                continue
            means.append(_safe_get(entry, model_key, metric_key, 'mean', default=np.nan))
            stds.append(_safe_get(entry, model_key, metric_key, 'std', default=np.nan))
        return means, stds

    m1_l, m1_l_stds = collect('model1', 'load_balance')
    m2_l, m2_l_stds = collect('model2', 'load_balance')
    m1_dbw, m1_dbw_stds = collect('model1', 'bandwidth_satisfaction')
    m2_dbw, m2_dbw_stds = collect('model2', 'bandwidth_satisfaction')

    def composite(L, D):
        vals = []
        for l, d in zip(L, D):
            if np.isnan(l) or np.isnan(d):
                vals.append(np.nan)
            else:
                vals.append(0.5 * (1.0 - l) + 0.5 * d)
        return vals

    m1_comp = composite(m1_l, m1_dbw)
    m2_comp = composite(m2_l, m2_dbw)

    saved = []

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    _plot_with_nan(ax1, virtual_nodes_range, m1_l, display_label(model1_name), 'o', colors[0])
    _plot_with_nan(ax1, virtual_nodes_range, m2_l, display_label(model2_name), 'o', colors[1])
    ax1.set_xlabel('任务数量')
    ax1.set_ylabel('L')
    ax1.set_title('负载均衡度（L）')
    ax1.set_xticks(virtual_nodes_range)
    ax1.grid(True, alpha=0.3)

    _plot_with_nan(ax2, virtual_nodes_range, m1_dbw, display_label(model1_name), 'o', colors[0])
    _plot_with_nan(ax2, virtual_nodes_range, m2_dbw, display_label(model2_name), 'o', colors[1])
    ax2.set_xlabel('任务数量')
    ax2.set_ylabel('D_BW')
    ax2.set_title('带宽满足度（D_BW）')
    ax2.set_xticks(virtual_nodes_range)
    ax2.grid(True, alpha=0.3)

    _plot_with_nan(ax3, virtual_nodes_range, m1_comp, display_label(model1_name), 'o', colors[0])
    _plot_with_nan(ax3, virtual_nodes_range, m2_comp, display_label(model2_name), 'o', colors[1])
    ax3.set_xlabel('任务数量')
    ax3.set_ylabel('综合得分')
    ax3.set_title('综合指标（0.5*(1-L)+0.5*D_BW）')
    ax3.set_xticks(virtual_nodes_range)
    ax3.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out2 = os.path.join(out_dir, 'two_ppo_lb_bw_cn.png')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    saved.append(out2)

    return saved


def main():
    parser = argparse.ArgumentParser(description='双 PPO 对比结果绘图（中文）')
    parser.add_argument('json', help='two_ppo_comparison_*.json 路径')
    parser.add_argument('--out', '-o', default='test_results', help='输出目录')
    parser.add_argument('--font-path', default=None, help='可选：中文字体文件路径（.ttf/.ttc）')
    parser.add_argument('--font-size', type=int, default=24, help='基础字体大小（标题会在此基础上更大）')
    args = parser.parse_args()

    configure_chinese_font(args.font_path)
    configure_plot_sizes(args.font_size)
    results = load_results(args.json)
    paths = plot_from_results(results, args.out)
    for p in paths:
        print('已保存:', p)


if __name__ == '__main__':
    main()
#  python plot_two_ppo_comparison_zh.py test_results/two_ppo_comparison_new_heuristic_20250827_164927_vs_exp_20250826_162200_20250829_160239.json