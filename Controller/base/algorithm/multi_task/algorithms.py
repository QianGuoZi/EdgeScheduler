from typing import Optional
from Controller.base.algorithm.multi_task.models import Queue
import random
from typing import Callable
import math


def make_fifo_scheduler() -> Callable[[Queue, float], Optional[int]]:
    def fifo(queue: Queue, now: float) -> Optional[int]:
        return 0 if len(queue) > 0 else None
    fifo.__name__ = "fifo_scheduler"
    return fifo


def make_sjf_scheduler() -> Callable[[Queue, float], Optional[int]]:
    def sjf(queue: Queue, now: float) -> Optional[int]:
        if len(queue) == 0:
            return None
        best_idx = 0
        best_len = queue.peek(0).length
        for i, j in enumerate(queue.jobs):
            if j.length < best_len:
                best_len = j.length
                best_idx = i
        return best_idx
    sjf.__name__ = "sjf_scheduler"
    return sjf


def make_random_scheduler() -> Callable[[Queue, float], Optional[int]]:
    def rnd(queue: Queue, now: float) -> Optional[int]:
        if len(queue) == 0:
            return None
        return random.randrange(len(queue))
    rnd.__name__ = "random_scheduler"
    return rnd


def make_priority_scheduler() -> Callable[[Queue, float], Optional[int]]:
    def pri(queue: Queue, now: float) -> Optional[int]:
        if len(queue) == 0:
            return None
        best_idx = 0
        best_p = queue.peek(0).priority
        for i, j in enumerate(queue.jobs):
            if j.priority > best_p:
                best_p = j.priority
                best_idx = i
        return best_idx
    pri.__name__ = "priority_scheduler"
    return pri


def make_ces_scheduler(network, max_load_factor: float = 1.0, theta: float = 5.0) -> Callable[[Queue, float], Optional[int]]:
    """返回一个闭包调度器，使用文档中的滚动窗口与评分逻辑。"""
    def ces(queue: Queue, now: float) -> Optional[int]:
        if len(queue) == 0:
            return None
        # 窗口大小：简单按节点数与因子限制
        W = min(int(max_load_factor * len(network.nodes)), len(queue))
        # 选前W作候选
        C = queue.jobs[:W]
        scores = []
        for j in C:
            # 资源适配度：以节点空闲资源最大值为分母
            max_cpu_free = max((n.cpu_free for n in network.nodes.values()), default=1.0)
            max_ram_free = max((n.ram_free for n in network.nodes.values()), default=1.0)
            max_bw_free = max((l.bw_free for l in network.links.values()), default=1.0)
            match_cpu = min(1.0, max_cpu_free / max(1e-6, j.cpu))
            match_ram = min(1.0, max_ram_free / max(1e-6, j.ram))
            match_bw = min(1.0, max_bw_free / max(1e-6, j.bw_max if j.bw_max>0 else 1.0))
            S_res = (match_cpu + match_ram + match_bw) / 3.0
            S_wait = 1 - (2.718281828 ** (-(j.wait_time(now) / theta)))
            S_total = 0.6 * S_res + 0.3 * (j.priority / 5.0) + 0.1 * S_wait
            scores.append(S_total)

        # 选择分数最高的作业索引（相对于 queue）
        best_local = int(max(range(len(C)), key=lambda i: scores[i]))
        return best_local

    ces.__name__ = "ces_scheduler"
    return ces


def make_swts_scheduler(network) -> Callable[[Queue, float], Optional[int]]:
    """简化实现的 SWTS 风格调度器。

    该实现提取 SWTS 的若干启发式指标构造作业得分：
    - CPL 近似（基于 run_time）
    - TPS 近似（基于 task_count）
    - DLS 近似（基于带宽需求）
    - 资源匹配（基于节点空闲资源）

    返回值为队列中得分最高的作业索引。
    """
    def swts(queue: Queue, now: float) -> Optional[int]:
        if len(queue) == 0:
            return None
        # 计算网络总体指标用于归一化
        total_cpu = sum((n.cpu_total for n in network.nodes.values())) if len(network.nodes) > 0 else 1.0
        max_cpu_free = max((n.cpu_free for n in network.nodes.values()), default=1.0)

        scores = []
        for j in queue.jobs:
            # CPL 近似：run_time 越短越优
            cpl = 1.0 / (1.0 + float(j.run_time))
            # TPS 近似：任务数越多并行潜力越高（相对网络规模归一化）
            tps = min(1.0, float(j.task_count) / max(1.0, total_cpu / 10.0))
            # DLS 近似：带宽需求越小越有利于局部性
            bw = getattr(j, 'bw_max', getattr(j, 'bw', 0.0))
            dls = 1.0 - (bw / (bw + 1.0))
            # 资源匹配：空闲 CPU 对作业需求的比率
            match_cpu = min(1.0, max_cpu_free / max(1e-6, getattr(j, 'cpu', 1.0)))

            # 权重组合（可根据需要调整）
            score = 0.35 * cpl + 0.25 * tps + 0.2 * dls + 0.2 * match_cpu
            # 小范围加入优先级影响
            score += 0.05 * (getattr(j, 'priority', 1) / 5.0)
            scores.append(score)

        best_local = int(max(range(len(scores)), key=lambda i: scores[i]))
        return best_local

    swts.__name__ = "swts_scheduler"
    return swts


def make_adaevo_scheduler(network, tau: float = 50.0, K: int = 3) -> Callable[[Queue, float], Optional[int]]:
    """简化的 AdaEvo 风格调度器实现。

    该调度器按作业紧急程度分组并在高紧急度组内使用启发式评分选择作业。
    返回值为队列中得分最高的作业索引。
    """
    def adaevo(queue: Queue, now: float) -> Optional[int]:
        if len(queue) == 0:
            return None

        # 计算等待时间和优先级的紧急程度
        waits = [j.wait_time(now) for j in queue.jobs]
        prios = [getattr(j, 'priority', 1) for j in queue.jobs]
        minp = min(prios) if len(prios) > 0 else 1
        maxp = max(prios) if len(prios) > 0 else 1
        p_range = max(1e-6, (maxp - minp))

        urgencies = []
        for j, p in zip(queue.jobs, prios):
            w = (p - minp) / p_range
            u = w * (1.0 + math.log(1.0 + j.wait_time(now) / tau))
            urgencies.append(u)

        # 选出属于最高紧急度的前一组（取按紧急度排序后的前 M 个）
        n = len(queue)
        group_size = max(1, n // K)
        idx_sorted = sorted(range(n), key=lambda i: urgencies[i], reverse=True)
        candidates_idx = idx_sorted[:group_size]

        # 评估候选作业的评分（资源匹配 + 作业属性）
        # 计算网络资源归一化量
        max_cpu_free = max((n.cpu_free for n in network.nodes.values()), default=1.0)
        max_ram_free = max((n.ram_free for n in network.nodes.values()), default=1.0)
        max_bw_free = max((l.bw_free for l in network.links.values()), default=1.0)

        scores = []
        for i in candidates_idx:
            j = queue.jobs[i]
            match_cpu = min(1.0, max_cpu_free / max(1e-6, getattr(j, 'cpu', 1.0)))
            match_ram = min(1.0, max_ram_free / max(1e-6, getattr(j, 'ram', 1.0)))
            bw = getattr(j, 'bw_max', getattr(j, 'bw', 0.0))
            match_bw = min(1.0, max_bw_free / max(1e-6, bw if bw > 0 else 1.0))
            S_res = (match_cpu + match_ram + match_bw) / 3.0

            # 作业内部指标：短运行优先 + 任务并行潜力 + 优先级
            cpl = 1.0 / (1.0 + float(getattr(j, 'run_time', 1.0)))
            tps = min(1.0, float(getattr(j, 'task_count', 1)) / max(1.0, len(network.nodes) / 10.0))
            pri = getattr(j, 'priority', 1) / 5.0

            score = 0.5 * S_res + 0.25 * cpl + 0.15 * tps + 0.1 * pri
            # 将紧急度作为加权因子，偏向更急迫的作业
            score *= (1.0 + urgencies[i])
            scores.append(score)

        # 选择分数最高的候选
        best_local_idx = int(max(range(len(scores)), key=lambda ii: scores[ii]))
        return candidates_idx[best_local_idx]

    adaevo.__name__ = "adaevo_scheduler"
    return adaevo


def fifo_scheduler(queue: Queue, now: float) -> Optional[int]:
    """FIFO：选择队列头部作业。"""
    return 0 if len(queue) > 0 else None


def sjf_scheduler(queue: Queue, now: float) -> Optional[int]:
    """SJF：选择最短总长度作业的索引（扫描实现）。"""
    if len(queue) == 0:
        return None
    best_idx = 0
    best_len = queue.peek(0).length
    for i, j in enumerate(queue.jobs):
        if j.length < best_len:
            best_len = j.length
            best_idx = i
    return best_idx
