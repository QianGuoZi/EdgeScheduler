import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_TYPES = ["ra", "el", "gl"]
METHODS = ["ppo", "ppo_balance", "heuristic", "smart", "random"]
METRICS = ["load_balance_degree", "bandwidth_satisfaction", "composite_score"]


def find_latest_csv(base_dir, task, method):
    """
    在类似 experiment_results_el/heuristic_20260113_1620/experiment_results.csv 结构中，
    找到某个 task 下某个 method 最近一次实验的 CSV 文件。
    """
    task_dir = base_dir / f"experiment_results_{task}"
    if not task_dir.is_dir():
        return None

    # 目录名一般是 like: method_20260113_1620
    # 注意：要严格匹配目录名以 method + "_" 开头，
    # 避免 "ppo" 错误匹配到 "ppo_balance_..." 这种目录。
    candidates = []
    method_prefix = method + "_"
    for d in task_dir.iterdir():
        if not d.is_dir():
            continue
        # 检查目录名是否以 method + "_" 开头
        if d.name.startswith(method_prefix):
            # 特殊处理：如果 method 是 "ppo"，要排除 "ppo_balance_" 开头的目录
            if method == "ppo" and d.name.startswith("ppo_balance_"):
                continue
            candidates.append(d)
    if not candidates:
        return None

    # 取按目录名排序后的最后一个，认为是最新一次实验
    latest_dir = sorted(candidates)[-1]
    csv_path = latest_dir / "experiment_results.csv"
    return csv_path if csv_path.is_file() else None


def collect_means(base_dir: Path):
    """
    收集每个 task、每个 method 在三种指标上的 5 次实验均值。
    返回结构: metrics_means[metric][method][task] = mean_value
    """
    metrics_means: dict[str, dict[str, dict[str, float]]] = {
        metric: {method: {} for method in METHODS} for metric in METRICS
    }

    for task in TASK_TYPES:
        for method in METHODS:
            csv_path = find_latest_csv(base_dir, task, method)
            if csv_path is None:
                print(f"[WARN] 找不到 CSV: task={task}, method={method}")
                for metric in METRICS:
                    metrics_means[metric][method][task] = float("nan")
                continue

            # 调试打印：确认每个方法在每个 task 下具体使用了哪个 CSV
            if method in ("ppo", "ppo_balance"):
                print(f"[INFO] 使用 CSV: task={task}, method={method}, path={csv_path}")

            df = pd.read_csv(csv_path)

            # 只统计 status == success 的记录（如果有该列）
            if "status" in df.columns:
                df = df[df["status"] == "success"]

            for metric in METRICS:
                if metric in df.columns:
                    metrics_means[metric][method][task] = df[metric].mean()
                else:
                    print(f"[WARN] {csv_path} 中没有指标列: {metric}")
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
    - load_balance_degree
    - bandwidth_satisfaction
    - composite_score

    横坐标: ra / el / gl
    纵坐标: 指标均值
    每个子图 5 条折线: ppo / ppo_balance / heuristic / smart / random
    """
    x_labels = TASK_TYPES
    x = range(len(x_labels))

    plt.figure(figsize=(12, 4))

    titles = {
        "load_balance_degree": "Load Balance Degree",
        "bandwidth_satisfaction": "Bandwidth Satisfaction",
        "composite_score": "Composite Score",
    }

    # 使用简单配色方案：整体用默认颜色，只适度弱化 ppo_balance 的视觉效果
    styles = {
        "ppo": {"marker": "o"},
        # ppo_balance：用稍浅的灰色和略细的线条，让它相对不那么显眼
        "ppo_balance": {"marker": "s", "color": "0.6", "linewidth": 1.0},
        "heuristic": {"marker": "D"},
        "smart": {"marker": "^"},
        "random": {"marker": "v"},
    }

    for idx, metric in enumerate(METRICS, start=1):
        plt.subplot(1, 3, idx)

        for method in METHODS:
            y = [metrics_means[metric][method].get(task, float("nan")) for task in TASK_TYPES]
            plt.plot(
                x,
                y,
                label=method,
                **styles.get(method, {}),
            )

        plt.xticks(x, x_labels)
        plt.xlabel("Task Type")
        plt.ylabel(metric)
        plt.title(titles.get(metric, metric))
        plt.grid(True, linestyle="--", alpha=0.4)
        if idx == 1:
            plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()


def main():
    base_dir = Path(__file__).resolve().parent
    metrics_means = collect_means(base_dir)

    # 打印 ppo 和 ppo_balance 的均值，便于检查
    print_method_means(metrics_means, methods=["ppo", "ppo_balance"])

    # 默认保存为当前目录下的 PNG，也可以改成直接 show()
    output_path = base_dir / "experiment_results_plot.png"
    plot_results(metrics_means, save_path=output_path)


if __name__ == "__main__":
    main()

