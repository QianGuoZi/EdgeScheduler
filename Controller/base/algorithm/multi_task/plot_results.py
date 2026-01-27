import csv
import glob
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np


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


def plot_line_avg_wait(results_dir, out='results/avg_wait_line.png'):
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
        plt.plot(mean, label=sched)
    plt.xlabel('Time')
    plt.ylabel('Avg waiting')
    plt.legend()
    plt.title('Average Waiting Over Time (mean across runs)')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_three_metrics_over_time(results_dir, out_dir=None):
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
                plt.plot(mean, label=sched)
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
    results_dir = str(Path(__file__).resolve().parent / 'results')
    # 仅生成 metrics 目录下的三项指标折线图，避免生成 box_summary 与 heatmap_cpu
    plot_three_metrics_over_time(results_dir, out_dir=os.path.join(results_dir, 'metrics'))
