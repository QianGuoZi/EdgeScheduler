#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立绘图脚本：两PPO模型对比结果可视化（与plot_five风格一致）"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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

    # style: keep consistent with plot_five_algorithms_load.py
    colors = ['#2E86AB', '#A23B72']

    virtual_nodes_range = results['test_config']['virtual_nodes_range']
    detailed = results['detailed_results']

    model1_name = results['model1']['name']
    model2_name = results['model2']['name']

    # 可在这里定义曲线标签（留空则使用模型名称）
    label_map = {
        model1_name: 'CES_PPO',
        model2_name: 'CES_PPO_without_heuristic'
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

    # rewards + success
    m1_rewards, m1_reward_stds = collect('model1', 'reward')
    m2_rewards, m2_reward_stds = collect('model2', 'reward')

    m1_sr = [_safe_get(detailed.get(n) or detailed.get(str(n)), 'model1', 'success_rate', default=np.nan)
             for n in virtual_nodes_range]
    m2_sr = [_safe_get(detailed.get(n) or detailed.get(str(n)), 'model2', 'success_rate', default=np.nan)
             for n in virtual_nodes_range]
    m1_sr = [s * 100 if not np.isnan(s) else np.nan for s in m1_sr]
    m2_sr = [s * 100 if not np.isnan(s) else np.nan for s in m2_sr]

    # L / D_BW / composite
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

    def composite_std(Ls, Ds):
        vals = []
        for l, d in zip(Ls, Ds):
            if np.isnan(l) or np.isnan(d):
                vals.append(np.nan)
            else:
                vals.append(0.5 * np.sqrt(l ** 2 + d ** 2))
        return vals

    m1_comp = composite(m1_l, m1_dbw)
    m2_comp = composite(m2_l, m2_dbw)
    m1_comp_std = composite_std(m1_l_stds, m1_dbw_stds)
    m2_comp_std = composite_std(m2_l_stds, m2_dbw_stds)

    saved = []

    # Figure: L / D_BW / composite
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    _plot_with_nan(ax1, virtual_nodes_range, m1_l,
                   display_label(model1_name), 'o', colors[0])
    _plot_with_nan(ax1, virtual_nodes_range, m2_l,
                   display_label(model2_name), 'o', colors[1])
    ax1.set_xlabel('Number of Tasks')
    ax1.set_ylabel('L')
    ax1.set_title('Load Balance (L)')
    ax1.set_xticks(virtual_nodes_range)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    _plot_with_nan(ax2, virtual_nodes_range, m1_dbw,
                   display_label(model1_name), 'o', colors[0])
    _plot_with_nan(ax2, virtual_nodes_range, m2_dbw,
                   display_label(model2_name), 'o', colors[1])
    ax2.set_xlabel('Number of Tasks')
    ax2.set_ylabel('D_BW')
    ax2.set_title('Bandwidth Satisfaction (D_BW)')
    ax2.set_xticks(virtual_nodes_range)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    _plot_with_nan(ax3, virtual_nodes_range, m1_comp,
                   display_label(model1_name), 'o', colors[0])
    _plot_with_nan(ax3, virtual_nodes_range, m2_comp,
                   display_label(model2_name), 'o', colors[1])
    ax3.set_xlabel('Number of Tasks')
    ax3.set_ylabel('Composite')
    ax3.set_title('Composite Metric (0.5*(1-L)+0.5*D_BW)')
    ax3.set_xticks(virtual_nodes_range)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(out_dir, 'two_ppo_lb_bw.png')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    saved.append(out2)

    return saved


def main():
    parser = argparse.ArgumentParser(description='Plot two PPO comparison results (same style as plot_five).')
    parser.add_argument('json', help='path to two_ppo_comparison_*.json')
    parser.add_argument('--out', '-o', default='test_results', help='output directory')
    args = parser.parse_args()

    results = load_results(args.json)
    paths = plot_from_results(results, args.out)
    for p in paths:
        print('Saved:', p)


if __name__ == '__main__':
    main()
