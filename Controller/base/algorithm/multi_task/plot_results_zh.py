import csv
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import argparse


def configure_chinese_font():
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

    x = x.astype(float)
    if np.isnan(x).all():
        return x
    if np.isnan(x).any():
        idx = np.arange(x.size)
        valid = ~np.isnan(x)
        x[np.isnan(x)] = np.interp(idx[np.isnan(x)], idx[valid], x[valid])

    pad = max(0, int(window_len) // 2)
    padded = np.pad(x, pad_width=pad, mode='edge')
    window = np.ones(window_len) / window_len
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


def _order_scheds(scheds, label_map=None):
    if not isinstance(label_map, dict):
        return list(scheds)
    ordered = []
    for key in label_map.keys():
        if key in scheds and key not in ordered:
            ordered.append(key)
    for s in scheds:
        if s not in ordered:
            ordered.append(s)
    return ordered


def plot_line_avg_wait(results_dir, out='results/avg_wait_line_zh.png', window_len: int = 5, label_map=None, colors=None):
    files = glob.glob(os.path.join(results_dir, '*_run*.csv'))
    sched_runs = {}
    for f in files:
        name = Path(f).stem
        sched = name.split('_run')[0]
        times, waits, _ = load_history_csv(f)
        sched_runs.setdefault(sched, []).append((times, waits))

    plt.figure(figsize=(10, 6))
    if colors is None:
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#16A085', '#7B2CBF']
    ordered_scheds = _order_scheds(sched_runs.keys(), label_map=label_map)
    for i, sched in enumerate(ordered_scheds):
        runs = sched_runs[sched]
        max_t = int(max(times.max() for times, _ in runs))
        arr = []
        for times, waits in runs:
            vec = np.zeros(max_t + 1)
            for t, w in zip(times.astype(int), waits):
                if t <= max_t:
                    vec[t] = w
            arr.append(vec)
        mean = np.mean(arr, axis=0)
        label = label_map.get(sched, sched) if isinstance(label_map, dict) else sched
        plt.plot(smooth(mean, window_len=window_len), label=label, color=colors[i % len(colors)])
    plt.xlabel('时间')
    plt.ylabel('平均等待时间')
    plt.legend()
    plt.title('平均等待时间随时间变化')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_weighted_performance(results_dir, out_dir=None, window_len: int = 5, weights=None, label_map=None, colors=None):
    """绘制加权综合性能曲线：对三个指标按时间点在调度器间归一化后按权重合成。"""
    if out_dir is None:
        out_dir = os.path.join(results_dir, '指标图')
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
        print('未在目录中找到运行结果 CSV 文件：', results_dir)
        return

    sched_metrics = {}
    max_t = 0
    for sched, runs in sched_runs.items():
        if runs:
            max_t = max(max_t, max((r[0].max() if len(r[0]) > 0 else 0) for r in runs))
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

    scheds = _order_scheds(sched_metrics.keys(), label_map=label_map)
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
            comp = (
                weights.get('bw', 0.5) * bw_n[i]
                + weights.get('load', 0.3) * load_n[i]
                + weights.get('wait', 0.2) * wait_n[i]
            )
            comp_series[s][t] = comp

    plt.figure(figsize=(10, 6))
    if colors is None:
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#16A085', '#7B2CBF']
    for i, s in enumerate(scheds):
        label = label_map.get(s, s) if isinstance(label_map, dict) else s
        plt.plot(smooth(comp_series[s], window_len=window_len), label=label, color=colors[i % len(colors)])
    plt.xlabel('时间')
    plt.ylabel('加权综合性能')
    plt.legend()
    plt.title('加权综合性能随时间变化')
    out_path = os.path.join(out_dir, 'weighted_composite_line_zh.png')
    plt.savefig(out_path)
    plt.close()


def plot_three_metrics_over_time(results_dir, out_dir=None, window_lens=None, label_map=None, colors=None):
    """绘制三个关注指标随时间的折线图：负载均衡度、平均带宽满足度、平均等待时间。"""
    if out_dir is None:
        out_dir = os.path.join(results_dir, '指标图')
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(results_dir, '*_run*.csv'))
    sched_runs = {}
    for f in files:
        name = Path(f).stem
        sched = name.split('_run')[0]
        times, waits, cpus, bw_sats, lbs = load_history_csv_ext(f)
        sched_runs.setdefault(sched, []).append((times, waits, bw_sats, lbs))

    if window_lens is None:
        window_lens = {'load_balance': 50, 'avg_bw_satisfaction': 10, 'avg_waiting': 5}

    metric_info = [
        ('load_balance', '负载均衡度', lambda t, w, b, l: l),
        ('avg_bw_satisfaction', '平均带宽满足度', lambda t, w, b, l: b),
        ('avg_waiting', '平均等待时间', lambda t, w, b, l: w),
    ]

    for metric_name, metric_cn, extractor in metric_info:
        plt.figure(figsize=(10, 6))
        if colors is None:
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#16A085', '#7B2CBF']
        ordered_scheds = _order_scheds(sched_runs.keys(), label_map=label_map)
        for i, sched in enumerate(ordered_scheds):
            runs = sched_runs[sched]
            max_t = int(max((r[0].max() if len(r[0]) > 0 else 0) for r in runs))
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
                label = label_map.get(sched, sched) if isinstance(label_map, dict) else sched
                plt.plot(smooth(mean, window_len=wl), label=label, color=colors[i % len(colors)])
        plt.xlabel('时间')
        plt.ylabel(metric_cn)
        plt.legend()
        plt.title(f'{metric_cn}随时间变化')
        out_path = os.path.join(out_dir, f'{metric_name}_line_zh.png')
        plt.savefig(out_path)
        plt.close()


def plot_box_summary(summary_csv, out='results/box_summary_zh.png'):
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
    plt.ylabel('平均等待时间（最终值）')
    plt.title('算法对比：平均等待时间分布')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_heatmap_cpu_over_time(results_dir, out='results/heatmap_cpu_zh.png'):
    files = glob.glob(os.path.join(results_dir, '*_run*.csv'))
    sched_runs = {}
    for f in files:
        name = Path(f).stem
        sched = name.split('_run')[0]
        times, _, cpus = load_history_csv(f)
        sched_runs.setdefault(sched, []).append((times.astype(int), cpus))

    scheds = list(sched_runs.keys())
    max_t = 0
    for runs in sched_runs.values():
        for times, _ in runs:
            if len(times) > 0:
                max_t = max(max_t, times.max())
    max_t = int(max_t) if max_t > 0 else 1
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
    plt.colorbar(label='平均 CPU 利用率')
    plt.yticks(range(len(scheds)), scheds)
    plt.xlabel('时间')
    plt.title('CPU 利用率随时间变化（跨运行取均值）')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制仿真结果图（支持平滑与中文标签）')
    parser.add_argument('--results-dir', default=str(Path(__file__).resolve().parent / 'results'))
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--font-size', type=int, default=18, help='基础字体大小（标题会在此基础上更大）')
    parser.add_argument('--window-line-avg-wait', type=int, default=5, help='平均等待时间折线图的平滑窗口')
    parser.add_argument('--window-load-balance', type=int, default=50, help='负载均衡度曲线的平滑窗口')
    parser.add_argument('--window-avg-bw', type=int, default=5, help='平均带宽满足度曲线的平滑窗口')
    parser.add_argument('--window-avg-waiting', type=int, default=5, help='平均等待时间曲线的平滑窗口')
    parser.add_argument('--window-weighted', type=int, default=30, help='加权综合性能曲线的平滑窗口')
    args = parser.parse_args()

    configure_chinese_font()
    configure_plot_sizes(base_font_size=args.font_size)

    label_map = {
        'ces_scheduler': 'CES-Multi-Jobs',
        'swts_scheduler': 'SWTS',
        'adaevo_scheduler': 'AdaEvo',
        'sjf_scheduler': 'SJF',
        'fifo_scheduler': 'FIFO'
    }
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#16A085', '#7B2CBF']

    results_dir = args.results_dir

    wl = {
        'load_balance': args.window_load_balance,
        'avg_bw_satisfaction': args.window_avg_bw,
        'avg_waiting': args.window_avg_waiting,
    }
    out_dir = args.out_dir or os.path.join(results_dir, 'metrics_zh')
    plot_three_metrics_over_time(results_dir, out_dir=out_dir, window_lens=wl, label_map=label_map, colors=colors)
    plot_weighted_performance(
        results_dir,
        out_dir=out_dir,
        window_len=args.window_weighted,
        weights={'bw': 0.5, 'load': 0.3, 'wait': 0.2},
        label_map=label_map,
        colors=colors
    )
