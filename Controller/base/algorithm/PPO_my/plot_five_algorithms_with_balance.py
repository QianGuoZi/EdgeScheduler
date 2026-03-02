#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

def plot_from_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    loads = results.get('load_range') or results.get('load_range_values')
    if loads is None:
        # try to infer from detailed_results keys
        loads = list(results.get('detailed_results', {}).keys())
    detailed = results['detailed_results']

    algs = ['ppo', 'ppo_balance', 'flexitask', 'smart', 'random']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#16A085', '#7B2CBF']

    display_names = {
        'ppo': 'CES_PPO',
        'ppo_balance': 'Balance_PPO',
        'flexitask': 'FlexiTask',
        'smart': 'Smart',
        'random': 'Random'
    }

    def collect(alg, metric):
        means = []
        stds = []
        for load in loads:
            entry = detailed.get(load) or detailed.get(str(load))
            if entry is None:
                means.append(np.nan); stds.append(np.nan); continue
            alg_entry = entry.get(alg, {})
            means.append(safe_get(alg_entry, metric, 'mean', default=np.nan))
            stds.append(safe_get(alg_entry, metric, 'std', default=np.nan))
        return np.array(means, dtype=float), np.array(stds, dtype=float)

    L_vals = {}
    D_vals = {}
    L_stds = {}
    D_stds = {}
    for a in algs:
        L_vals[a], L_stds[a] = collect(a, 'load_balance')
        D_vals[a], D_stds[a] = collect(a, 'bandwidth_satisfaction')

    def composite(L, D):
        return 0.5 * (1.0 - L) + 0.5 * D

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))
    for i, a in enumerate(algs):
        # error bars disabled: use simple line plot
        ax1.plot(loads, L_vals[a], label=display_names.get(a, a.upper()), color=colors[i], marker='o')
        ax2.plot(loads, D_vals[a], label=display_names.get(a, a.upper()), color=colors[i], marker='o')
        comp = composite(L_vals[a], D_vals[a])
        # comp_std = 0.5 * np.sqrt(np.nan_to_num(L_stds[a])**2 + np.nan_to_num(D_stds[a])**2)
        ax3.plot(loads, comp, label=display_names.get(a, a.upper()), color=colors[i], marker='o')

    ax1.set_title('Load Balance (L)'); ax1.set_xlabel('Number of Tasks'); ax1.set_ylabel('L'); ax1.grid(True); ax1.legend()
    ax2.set_title('Bandwidth Satisfaction (D_BW)'); ax2.set_xlabel('Number of Tasks'); ax2.set_ylabel('D_BW'); ax2.grid(True); ax2.legend()
    ax3.set_title('Composite Metric (0.5*(1-L)+0.5*D_BW)'); ax3.set_xlabel('Number of Tasks'); ax3.set_ylabel('Composite'); ax3.grid(True); ax3.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'five_algorithms_with_balance_plot.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print('Saved:', out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('json', help='path to results json')
    p.add_argument('--out', '-o', default='test_results', help='output directory')
    args = p.parse_args()
    results = load_results(args.json)
    plot_from_results(results, args.out)

if __name__ == '__main__':
    main()
