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
    load_range = results.get('load_range') or results.get('loads')
    if load_range is None:
        raise RuntimeError('cannot find load_range in results')
    detailed = results['detailed_results']

    def collect_for_alg(alg_name, metric):
        means = []
        stds = []
        for load in load_range:
            entry = detailed.get(load) or detailed.get(str(load))
            if entry is None:
                means.append(np.nan); stds.append(np.nan); continue
            alg = entry.get(alg_name, {})
            means.append(safe_get(alg, metric, 'mean', default=np.nan))
            stds.append(safe_get(alg, metric, 'std', default=np.nan))
        return np.array(means, dtype=float), np.array(stds, dtype=float)

    algs = ['ppo', 'ppo_balance', 'flexitask', 'smart', 'random']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#16A085', '#7B2CBF']

    display_names = {
        'ppo': 'CES_PPO',
        'ppo_balance': 'Balance_PPO',
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

    ts = ''
    try:
        ts = results.get('base_seed', '')
    except Exception:
        ts = ''

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))
    for i, a in enumerate(algs):
        # error bars disabled: use simple line plot
        ax1.plot(load_range, L_vals[a], label=display_names.get(a, a.upper()), color=colors[i], marker='o')
        ax2.plot(load_range, D_vals[a], label=display_names.get(a, a.upper()), color=colors[i], marker='o')
        comp = composite(L_vals[a], D_vals[a])
        # comp_std = 0.5 * np.sqrt(np.nan_to_num(L_stds[a])**2 + np.nan_to_num(D_stds[a])**2)
        ax3.plot(load_range, comp, label=display_names.get(a, a.upper()), color=colors[i], marker='o')

    ax1.set_title('Load Balance (L) — Lower is better')
    ax1.set_xlabel('Load'); ax1.set_ylabel('L'); ax1.grid(True); ax1.legend()
    ax2.set_title('Bandwidth Satisfaction (D_BW) — Higher is better')
    ax2.set_xlabel('Load'); ax2.set_ylabel('D_BW'); ax2.grid(True); ax2.legend()
    ax3.set_title('Composite Metric (0.5*(1-L)+0.5*D_BW)')
    ax3.set_xlabel('Load'); ax3.set_ylabel('Composite'); ax3.grid(True); ax3.legend()

    # set x ticks with interval 0.1
    try:
        min_load = float(np.nanmin(load_range))
        max_load = float(np.nanmax(load_range))
        xt = np.arange(min_load, max_load + 1e-9, 0.1)
        xt = np.round(xt, 2)
        ax1.set_xticks(xt)
        ax2.set_xticks(xt)
        ax3.set_xticks(xt)
    except Exception:
        # fallback: keep default ticks
        pass

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'five_algorithms_load.png')
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
