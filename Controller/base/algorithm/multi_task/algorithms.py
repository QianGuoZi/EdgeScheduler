from typing import Optional
from Controller.base.algorithm.multi_task.models import Queue
import random
from typing import Callable


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
