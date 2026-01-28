import csv
import glob
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_history_csv(path):
    times = []
    avg_wait = []
    avg_cpu = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row.get('time', 0)))
            avg_wait.append(float(row.get('avg_waiting', 0)))
            avg_cpu.append(float(row.get('avg_cpu_util', 0)))
    return np.array(times), np.array(avg_wait), np.array(avg_cpu)


def smooth(x, window_len: int = 5):
    """简单移动平均平滑：在长度不足或 window_len<=1 时返回原序列。"""
    if window_len <= 1:
        return x
    if x is None:
        return x
    x = np.asarray(x)
    n = len(x)
    if n < window_len or n == 0:
        return x
    # 将输入转为浮点数组，并用线性插值填补 NaN（若存在）
    x = x.astype(float)
    if np.isnan(x).all():
        return x
    if np.isnan(x).any():
        idx = np.arange(x.size)
        valid = ~np.isnan(x)
        x[np.isnan(x)] = np.interp(idx[np.isnan(x)], idx[valid], x[valid])

    # 使用 edge 填充边界以避免卷积时的零填充伪影
    pad = max(0, int(window_len) // 2)
    padded = np.pad(x, pad_width=pad, mode='edge')
    window = np.ones(window_len) / window_len
    # 'valid' 在 padded 上产生与原序列等长的结果
    y = np.convolve(padded, window, mode='valid')
    return y


def load_history_csv_ext(path):
    times = []
    avg_wait = []
    avg_cpu = []
    avg_bw_sat = []
    load_balance = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row.get('time', 0)))
            avg_wait.append(float(row.get('avg_waiting', 0)))
            avg_cpu.append(float(row.get('avg_cpu_util', 0)))
            avg_bw_sat.append(float(row.get('avg_bw_satisfaction', 0)))
            load_balance.append(float(row.get('load_balance', 0)))
    return np.array(times), np.array(avg_wait), np.array(avg_cpu), np.array(avg_bw_sat), np.array(load_balance)


def plot_line_avg_wait(results_dir, out='results/avg_wait_line.png', window_len: int = 5):
    files = glob.glob(os.path.join(results_dir, '*_run*.csv'))
    sched_runs = {}
    for f in files:
        name = Path(f).stem
        sched = name.split('_run')[0]
        times, waits, _ = load_history_csv(f)
        sched_runs.setdefault(sched, []).append((times, waits))

    plt.figure(figsize=(10, 6))
    for sched, runs in sched_runs.items():
        # align by integer times up to max
        max_t = int(max(times.max() for times, _ in runs))
        arr = []
        for times, waits in runs:
            # create vector length max_t+1
            vec = np.zeros(max_t + 1)
            for t, w in zip(times.astype(int), waits):
                if t <= max_t:
                    vec[t] = w
            arr.append(vec)
        mean = np.mean(arr, axis=0)
        plt.plot(smooth(mean, window_len=window_len), label=sched)
    plt.xlabel('Time')
    plt.ylabel('Avg waiting')
    plt.legend()
    plt.title('Average Waiting Over Time (mean across runs)')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()

def plot_weighted_performance(results_dir, out_dir=None, window_len: int = 5, weights=None):
    """绘制加权综合性能曲线：对三个指标按时间点在调度器间归一化后按权重合成。

    组合项（越大越好）:
      - avg_bw_satisfaction (higher is better)
      - load_balance (lower is better -> invert)
      - avg_waiting (lower is better -> invert)

    weights: dict 或 None，格式 {'bw':0.5,'load':0.3,'wait':0.2}
    """
    if out_dir is None:
        out_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(out_dir, exist_ok=True)

    if weights is None:
        weights = {'bw': 0.5, 'load': 0.3, 'wait': 0.2}

    files = glob.glob(os.path.join(results_dir, '*_run*.csv'))
    sched_runs = {}
    for f in files:
        name = Path(f).stem
        sched = name.split('_run')[0]
        times, waits, cpus, bw_sats, lbs = load_history_csv_ext(f)
        sched_runs.setdefault(sched, []).append((times, waits, bw_sats, lbs))

    if not sched_runs:
        print('No run CSV files found in', results_dir)
        return

    # 计算每个 scheduler 在每个时间点的平均值向量
    sched_metrics = {}
    max_t = 0
    for sched, runs in sched_runs.items():
        if runs:
            max_t = max(max_t, max((r[0].max() if len(r[0])>0 else 0) for r in runs))
    max_t = int(max_t) if max_t > 0 else 1

    for sched, runs in sched_runs.items():
        bw_arrs = []
        load_arrs = []
        wait_arrs = []
        for times, waits, bw_sats, lbs in runs:
            vec_bw = np.zeros(max_t + 1)
            vec_load = np.zeros(max_t + 1)
            vec_wait = np.zeros(max_t + 1)
            for t, w, b, l in zip(times.astype(int), waits, bw_sats, lbs):
                if t <= max_t:
                    vec_bw[t] = b
                    vec_load[t] = l
                    vec_wait[t] = w
            bw_arrs.append(vec_bw)
            load_arrs.append(vec_load)
            wait_arrs.append(vec_wait)
        mean_bw = np.mean(bw_arrs, axis=0) if bw_arrs else np.zeros(max_t + 1)
        mean_load = np.mean(load_arrs, axis=0) if load_arrs else np.zeros(max_t + 1)
        mean_wait = np.mean(wait_arrs, axis=0) if wait_arrs else np.zeros(max_t + 1)
        sched_metrics[sched] = {'bw': mean_bw, 'load': mean_load, 'wait': mean_wait}

    scheds = list(sched_metrics.keys())
    comp_series = {s: np.zeros(max_t + 1) for s in scheds}
    for t in range(max_t + 1):
        bw_vals = np.array([sched_metrics[s]['bw'][t] for s in scheds])
        load_vals = np.array([sched_metrics[s]['load'][t] for s in scheds])
        wait_vals = np.array([sched_metrics[s]['wait'][t] for s in scheds])

        def norm_high(vals):
            mn = vals.min()
            mx = vals.max()
            if mx <= mn:
                return np.zeros_like(vals)
            return (vals - mn) / (mx - mn)

        def norm_low(vals):
            mn = vals.min()
            mx = vals.max()
            if mx <= mn:
                return np.zeros_like(vals)
            return (mx - vals) / (mx - mn)

        bw_n = norm_high(bw_vals)
        load_n = norm_low(load_vals)
        wait_n = norm_low(wait_vals)

        for i, s in enumerate(scheds):
            comp = weights.get('bw', 0.5) * bw_n[i] + weights.get('load', 0.3) * (1-load_n[i]) + weights.get('wait', 0.2) * (1-wait_n[i])
            comp_series[s][t] = comp

    plt.figure(figsize=(10, 6))
    for s in scheds:
        plt.plot(smooth(comp_series[s], window_len=window_len), label=s)
    plt.xlabel('Time')
    plt.ylabel('Weighted Composite Performance')
    plt.legend()
    plt.title('Weighted Composite Performance Over Time')
    out_path = os.path.join(out_dir, 'weighted_composite_line.png')
    plt.savefig(out_path)
    plt.close()


def plot_three_metrics_over_time(results_dir, out_dir=None, window_lens=None):
    """绘制三个关注指标随时间的折线图：负载均衡度、平均带宽满足度、平均等待时间（按调度器取平均）。"""
    if out_dir is None:
        out_dir = os.path.join(results_dir, 'results_plots')
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(results_dir, '*_run*.csv'))
    sched_runs = {}
    for f in files:
        name = Path(f).stem
        sched = name.split('_run')[0]
        times, waits, cpus, bw_sats, lbs = load_history_csv_ext(f)
        sched_runs.setdefault(sched, []).append((times, waits, bw_sats, lbs))

    # For each metric, build mean across runs per scheduler
    # 支持通过 window_lens 指定每个 metric 的平滑窗口大小，接受 dict 或 None
    if window_lens is None:
        window_lens = {'load_balance': 50, 'avg_bw_satisfaction': 10, 'avg_waiting': 5}

    for metric_index, (metric_name, extractor) in enumerate([
        ('load_balance', lambda t,w,b,l: l),
        ('avg_bw_satisfaction', lambda t,w,b,l: b),
        ('avg_waiting', lambda t,w,b,l: w),
    ]):
        plt.figure(figsize=(10, 6))
        for sched, runs in sched_runs.items():
            max_t = int(max((r[0].max() if len(r[0])>0 else 0) for r in runs))
            arr = []
            for times, waits, bw_sats, lbs in runs:
                vec = np.zeros(max_t + 1)
                for t, w, b, l in zip(times.astype(int), waits, bw_sats, lbs):
                    if t <= max_t:
                        vec[t] = extractor(times, w, b, l)
                arr.append(vec)
            if arr:
                mean = np.mean(arr, axis=0)
                wl = int(window_lens.get(metric_name, 5)) if isinstance(window_lens, dict) else int(window_lens)
                plt.plot(smooth(mean, window_len=wl), label=sched)
        plt.xlabel('Time')
        plt.ylabel(metric_name)
        plt.legend()
        plt.title(f'{metric_name} Over Time (mean across runs)')
        out_path = os.path.join(out_dir, f'{metric_name}_line.png')
        plt.savefig(out_path)
        plt.close()


def plot_box_summary(summary_csv, out='results/box_summary.png'):
    # read summary and boxplot avg_waiting per scheduler
    data = {}
    with open(summary_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sched = row['scheduler']
            data.setdefault(sched, []).append(float(row.get('avg_waiting', 0)))

    labels = list(data.keys())
    values = [data[k] for k in labels]
    plt.figure(figsize=(10, 6))
    plt.boxplot(values, labels=labels)
    plt.ylabel('Avg waiting (final)')
    plt.title('Algorithm Comparison: Avg Waiting Distribution')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_heatmap_cpu_over_time(results_dir, out='results/heatmap_cpu.png'):
    files = glob.glob(os.path.join(results_dir, '*_run*.csv'))
    sched_runs = {}
    for f in files:
        name = Path(f).stem
        sched = name.split('_run')[0]
        times, _, cpus = load_history_csv(f)
        sched_runs.setdefault(sched, []).append((times.astype(int), cpus))

    # build matrix sched x time averaged across runs
    scheds = list(sched_runs.keys())
    max_t = 0
    for runs in sched_runs.values():
        for times, _ in runs:
            if len(times) > 0:
                max_t = max(max_t, times.max())
    max_t = int(max_t) if max_t>0 else 1
    mat = np.zeros((len(scheds), max_t + 1))
    for i, sched in enumerate(scheds):
        runs = sched_runs[sched]
        arr = []
        for times, cpus in runs:
            vec = np.zeros(max_t + 1)
            for t, c in zip(times, cpus):
                t = int(t)
                if t <= max_t:
                    vec[t] = c
            arr.append(vec)
        if arr:
            mat[i, :] = np.mean(arr, axis=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(mat, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Avg CPU Utilization')
    plt.yticks(range(len(scheds)), scheds)
    plt.xlabel('Time')
    plt.title('CPU Utilization Over Time (avg across runs)')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot simulation results with optional smoothing')
    parser.add_argument('--results-dir', default=str(Path(__file__).resolve().parent / 'results'))
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--window-line-avg-wait', type=int, default=5, help='smoothing window for avg wait line plot')
    parser.add_argument('--window-load-balance', type=int, default=50, help='smoothing window for load_balance')
    parser.add_argument('--window-avg-bw', type=int, default=10, help='smoothing window for avg_bw_satisfaction')
    parser.add_argument('--window-avg-waiting', type=int, default=5, help='smoothing window for avg_waiting')
    parser.add_argument('--window-weighted', type=int, default=30, help='smoothing window for weighted composite performance')
    args = parser.parse_args()

    results_dir = args.results_dir
    # 生成 line avg wait（可单独平滑）
    plot_line_avg_wait(results_dir, out=os.path.join(results_dir, 'avg_wait_line.png'), window_len=args.window_line_avg_wait)

    # 生成三个 metrics 的折线图，分别传入各自的平滑窗口
    wl = {
        'load_balance': args.window_load_balance,
        'avg_bw_satisfaction': args.window_avg_bw,
        'avg_waiting': args.window_avg_waiting,
    }
    out_dir = args.out_dir or os.path.join(results_dir, 'metrics')
    plot_three_metrics_over_time(results_dir, out_dir=out_dir, window_lens=wl)
    # 生成加权综合性能图（默认权重 bw:0.5, load:0.3, wait:0.2 ）
    plot_weighted_performance(results_dir, out_dir=out_dir, window_len=args.window_weighted, weights={'bw':0.5,'load':0.3,'wait':0.2})
