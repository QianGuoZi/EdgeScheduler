#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_get(d, *keys, default=float('nan')):
    try:
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return default


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


def plot_from_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    load_range = results.get('load_range') or results.get('loads')
    if load_range is None:
        raise RuntimeError('在结果中找不到 load_range')
    detailed = results['detailed_results']

    def collect_for_alg(alg_name, metric):
        means = []
        stds = []
        for load in load_range:
            entry = detailed.get(load) or detailed.get(str(load))
            if entry is None:
                means.append(np.nan)
                stds.append(np.nan)
                continue
            alg = entry.get(alg_name, {})
            means.append(safe_get(alg, metric, 'mean', default=np.nan))
            stds.append(safe_get(alg, metric, 'std', default=np.nan))
        return np.array(means, dtype=float), np.array(stds, dtype=float)

    algs = ['ppo', 'ppo_balance', 'flexitask', 'smart', 'random']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#16A085', '#7B2CBF']

    display_names = {
        'ppo': 'CES_PPO',
        'ppo_balance': 'PPO_Balance',
        'flexitask': 'FlexiTask',
        'smart': 'Smart',
        'random': 'Random'
    }

    L_vals = {}
    D_vals = {}
    L_stds = {}
    D_stds = {}
    for a in algs:
        L_vals[a], L_stds[a] = collect_for_alg(a, 'load_balance')
        D_vals[a], D_stds[a] = collect_for_alg(a, 'bandwidth_satisfaction')

    def composite(L, D):
        return 0.5 * (1.0 - L) + 0.5 * D

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    for i, a in enumerate(algs):
        ax1.plot(load_range, L_vals[a], label=display_names.get(a, a.upper()), color=colors[i], marker='o')
        ax2.plot(load_range, D_vals[a], label=display_names.get(a, a.upper()), color=colors[i], marker='o')
        comp = composite(L_vals[a], D_vals[a])
        ax3.plot(load_range, comp, label=display_names.get(a, a.upper()), color=colors[i], marker='o')

    ax1.set_title('负载均衡度（L）')
    ax1.set_xlabel('负载占比')
    ax1.set_ylabel('L')
    ax1.grid(True)

    ax2.set_title('带宽满足度（D_BW）')
    ax2.set_xlabel('负载占比')
    ax2.set_ylabel('D_BW')
    ax2.grid(True)

    ax3.set_title('综合指标（0.5*(1-L)+0.5*D_BW）')
    ax3.set_xlabel('负载占比')
    ax3.set_ylabel('综合得分')
    ax3.grid(True)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.01))

    try:
        min_load = float(np.nanmin(load_range))
        max_load = float(np.nanmax(load_range))
        xt = np.arange(min_load, max_load + 1e-9, 0.1)
        xt = np.round(xt, 2)
        ax1.set_xticks(xt)
        ax2.set_xticks(xt)
        ax3.set_xticks(xt)
    except Exception:
        pass

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out_path = os.path.join(out_dir, 'five_algorithms_load_cn.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print('已保存:', out_path)


def main():
    p = argparse.ArgumentParser(description='五算法负载对比绘图（中文）')
    p.add_argument('json', help='结果 JSON 路径')
    p.add_argument('--out', '-o', default='test_results', help='输出目录')
    p.add_argument('--font-path', default=None, help='可选：中文字体文件路径（.ttf/.ttc）')
    p.add_argument('--font-size', type=int, default=24, help='基础字体大小（标题会在此基础上更大）')
    args = p.parse_args()

    configure_chinese_font(args.font_path)
    configure_plot_sizes(args.font_size)
    results = load_results(args.json)
    plot_from_results(results, args.out)


if __name__ == '__main__':
    main()
# python3 plot_five_algorithms_load_zh.py test_results/five_algorithms_load_test_20260126_145528.json