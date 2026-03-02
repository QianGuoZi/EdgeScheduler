import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd


TASK_TYPES = ["ra", "el", "gl"]
METHODS = ["ppo", "ppo_balance", "heuristic", "smart", "random"]
METRICS = ["load_balance_degree", "bandwidth_satisfaction", "composite_score"]


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


def configure_plot_sizes(base_font_size: int = 14):
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



def find_latest_csv(base_dir, task, method):
    """
    在类似 experiment_results_el/heuristic_20260113_1620/experiment_results.csv 结构中，
    找到某个 task 下某个 method 最近一次实验的 CSV 文件。
    """
    task_dir = base_dir / f"experiment_results_{task}"
    if not task_dir.is_dir():
        return None

    candidates = []
    method_prefix = method + "_"
    for d in task_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(method_prefix):
            if method == "ppo" and d.name.startswith("ppo_balance_"):
                continue
            candidates.append(d)
    if not candidates:
        return None

    latest_dir = sorted(candidates)[-1]
    csv_path = latest_dir / "experiment_results.csv"
    return csv_path if csv_path.is_file() else None


def collect_means(base_dir: Path):
    """
    收集每个 task、每个 method 在三种指标上的实验均值。
    返回结构: metrics_means[metric][method][task] = mean_value
    """
    metrics_means: dict[str, dict[str, dict[str, float]]] = {
        metric: {method: {} for method in METHODS} for metric in METRICS
    }

    for task in TASK_TYPES:
        for method in METHODS:
            csv_path = find_latest_csv(base_dir, task, method)
            if csv_path is None:
                print(f"[警告] 找不到 CSV: task={task}, method={method}")
                for metric in METRICS:
                    metrics_means[metric][method][task] = float("nan")
                continue

            if method in ("ppo", "ppo_balance"):
                print(f"[信息] 使用 CSV: task={task}, method={method}, path={csv_path}")

            df = pd.read_csv(csv_path)

            if "status" in df.columns:
                df = df[df["status"] == "success"]

            for metric in METRICS:
                if metric in df.columns:
                    metrics_means[metric][method][task] = df[metric].mean()
                else:
                    print(f"[警告] {csv_path} 中没有指标列: {metric}")
                    metrics_means[metric][method][task] = float("nan")

    return metrics_means


def print_method_means(metrics_means, methods=None):
    """
    在终端打印指定方法在各任务、各指标上的均值，默认只打印 ppo 和 ppo_balance。
    """
    if methods is None:
        methods = ["ppo", "ppo_balance"]

    print("\n=== 方法均值统计 ===")
    for method in methods:
        if method not in metrics_means[METRICS[0]]:
            continue
        print(f"\n--- {method} ---")
        for task in TASK_TYPES:
            vals = []
            for metric in METRICS:
                val = metrics_means[metric][method].get(task, float("nan"))
                vals.append(f"{metric}={val:.6f}" if val == val else f"{metric}=NaN")
            print(f"task={task}: " + ", ".join(vals))


def plot_results(metrics_means, save_path=None):
    """
    画出三个子图：
    - 负载均衡度
    - 带宽满足度
    - 综合指标

    横坐标: ra / el / gl
    纵坐标: 指标均值
    每个子图 5 条折线: ppo / ppo_balance / heuristic / smart / random
    """
    x_labels = TASK_TYPES
    x = range(len(x_labels))

    fig = plt.figure(figsize=(12, 4))
    legend_handles = None
    legend_labels = None

    titles = {
        "load_balance_degree": "负载均衡度（L）",
        "bandwidth_satisfaction": "带宽满足度（D_BW）",
        "composite_score": "综合指标（0.5*(1-L)+0.5*D_BW）",
    }

    ylabels = {
        "load_balance_degree": "L",
        "bandwidth_satisfaction": "D_BW",
        "composite_score": "综合得分",
    }

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#16A085", "#7B2CBF"]
    display_names = {
        "ppo": "CES_PPO",
        "ppo_balance": "PPO_Balance",
        "heuristic": "FlexiTask",
        "smart": "Smart",
        "random": "Random",
    }

    styles = {
        "ppo": {"marker": "o", "color": colors[0]},
        "ppo_balance": {"marker": "o", "color": colors[1], "linewidth": 1.0},
        "heuristic": {"marker": "o", "color": colors[2]},
        "smart": {"marker": "o", "color": colors[3]},
        "random": {"marker": "o", "color": colors[4]},
    }

    for idx, metric in enumerate(METRICS, start=1):
        plt.subplot(1, 3, idx)

        for method in METHODS:
            y = [metrics_means[metric][method].get(task, float("nan")) for task in TASK_TYPES]
            plt.plot(
                x,
                y,
                label=display_names.get(method, method),
                **styles.get(method, {}),
            )

        plt.xticks(x, x_labels)
        plt.xlabel("任务类型")
        plt.ylabel(ylabels.get(metric, metric))
        plt.title(titles.get(metric, metric))
        plt.grid(True, linestyle="--", alpha=0.4)
        if idx == 1:
            legend_handles, legend_labels = plt.gca().get_legend_handles_labels()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()


def main(font_path: str = None, font_size: int = 22):
    configure_chinese_font(font_path)
    configure_plot_sizes(base_font_size=font_size)
    base_dir = Path(__file__).resolve().parent
    metrics_means = collect_means(base_dir)

    print_method_means(metrics_means, methods=["ppo", "ppo_balance"])

    output_path = base_dir / "experiment_results_plot_zh.png"
    plot_results(metrics_means, save_path=output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实验结果绘图（中文）")
    parser.add_argument("--font-path", default=None, help="可选：手动指定中文字体文件路径（.ttf/.ttc）")
    parser.add_argument("--font-size", type=int, default=14, help="基础字体大小（标题会在此基础上略大）")
    args = parser.parse_args()

    main(font_path=args.font_path, font_size=args.font_size)
