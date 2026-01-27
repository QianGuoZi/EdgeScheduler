from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import uuid


@dataclass
class Job:
    """作业模型：包含到达时间、资源需求、优先级等，用于仿真与调度决策。"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    arrival: float = 0.0
    # 运行时长：统一使用 `run_time` 表示资源占用时间（单位：仿真时间步）
    run_time: float = 1.0
    cpu: float = 1.0
    ram: float = 1.0
    bw_min: float = 0.0
    bw_max: float = 0.0
    priority: int = 1
    topology: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    task_count: int = 1

    # 运行时状态（仿真用）
    remaining: float = field(init=False)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    skip_count: int = 0
    bw_alloc: float = 0.0
    assigned_nodes: List[str] = field(default_factory=list)
    # 存储已分配的物理路径及对应的带宽：[(path_links, bw), ...]
    allocated_paths: List[Tuple[List[Tuple[str, str]], float]] = field(default_factory=list)

    def __post_init__(self):
        # 初始化剩余运行时间（remaining）为 run_time
        self.remaining = float(self.run_time)

    # 兼容属性：保留 length 名称以兼容旧代码，映射到 run_time
    @property
    def length(self) -> float:
        return float(self.run_time)

    @length.setter
    def length(self, v: float):
        self.run_time = float(v)

    def is_finished(self) -> bool:
        return self.remaining <= 0.0

    def wait_time(self, now: float) -> float:
        return max(0.0, now - self.arrival)


@dataclass
class Queue:
    """简单队列封装，便于调试和替换调度策略。"""
    jobs: List[Job] = field(default_factory=list)

    def push(self, job: Job):
        self.jobs.append(job)

    def pop(self, idx: int = 0) -> Job:
        return self.jobs.pop(idx)

    def remove(self, job: Job):
        self.jobs.remove(job)

    def peek(self, idx: int = 0) -> Optional[Job]:
        return self.jobs[idx] if idx < len(self.jobs) else None

    def sort_by_sjf(self):
        self.jobs.sort(key=lambda j: j.length)

    def is_empty(self) -> bool:
        return len(self.jobs) == 0

    def __len__(self) -> int:
        return len(self.jobs)
