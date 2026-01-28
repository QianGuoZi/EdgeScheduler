"""仿真参数集中配置文件（便于修改实验设置）

将所有与实验相关的参数集中在本文件中，`run_sim.py` 会加载并使用这些参数。
可直接编辑本文件以调整网络规模、作业到达分布、调度器选择等。
"""

# 全局随机种子
# 全局随机种子：用于保证实验可重复（影响网络拓扑、作业生成等的随机性）
SEED = 42

# 物理网络参数
# 物理网络参数：用于生成随机物理拓扑
NETWORK = {
    'NUM_NODES': 10,          # 物理节点数量（整数）
    'CONN_PROB': 0.3,         # 任意两个节点之间存在链路的概率，范围 0.0-1.0
    'CPU_RANGE': (50, 100),   # 每个物理节点 CPU 总量的取值范围（浮点或整数）
    'RAM_RANGE': (50, 100),   # 每个物理节点内存总量的取值范围
    'BW_RANGE': (100, 1000),  # 链路带宽范围（每条链路的总带宽）
}

# 作业生成参数（用于 generate_jobs_poisson）
JOB_GEN = {
    # 到达过程参数
    'LAMBDA_RATE': 3.0,       # 到达率 λ，用于指数间隔（泊松过程）的参数，越大到达越频繁
    'T_MAX': 200.0,            # 作业生成的时间上限（只生成 arrival < T_MAX 的作业）
    # 作业资源需求范围
    'CPU_RANGE': (10, 50),    # 单个作业总体 CPU 需求范围（会按任务数均摊到每个任务）
    'RAM_RANGE': (10, 50),    # 单个作业总体内存需求范围
    # bw_range 使用元组 (min, max_extra)；实际 bw_max = bw_min + uniform(0, BW_EXTRA_MAX)
    'BW_RANGE': (10, 100),    # 带宽最小值和用于生成额外上限的范围参数
    'TASK_RANGE': (3, 8),     # 作业内部任务（虚拟节点）数量范围
    'PRIORITY_RANGE': (1, 5), # 作业优先级整数范围（较小或较大含义由调度器实现决定）
    # 运行时长范围：用于生成作业的 `run_time`（资源占用时间，单位与仿真时间一致）
    'RUN_TIME_RANGE': (50.0, 100.0),
}

# 仿真运行参数
SIM = {
    'DT': 1.0,    # 时间步长（仿真时间的最小推进单位）
    'T_MAX': 400.0, # 单次仿真最大时间（仿真在达到该时间后停止）
}

# 试验控制
RUNS_PER_SCHEDULER = 5   # 每个调度器重复运行的次数（用于统计均值与方差）
RESULTS_DIR = 'results'   # 存放仿真输出（history CSV、图像、summary 等）的目录

# 选择要运行的调度器（按名称）。可选值: 'ces', 'fifo', 'sjf', 'priority', 'swts'
# 将 SWTS 算法加入可选调度器列表以便进行对比实验
SCHEDULER_NAMES = ['ces','swts', 'fifo', 'sjf', 'AdaEvo']
# 可用调度器说明：
#  - 'ces'：基于综合评估分数的调度器（需 CES_PARAMS 支持）
#  - 'fifo'：先进先出
#  - 'sjf'：短作业优先（Shortest Job First）
#  - 'priority'：按作业优先级

# CES 调度器参数（若启用 'ces'）
CES_PARAMS = {
    # MAX_LOAD_FACTOR: 调度时允许的最大负载因子，用于判断节点是否可接受更多任务
    'MAX_LOAD_FACTOR': 1.0,
    # THETA: CES 调度器内部用于权衡或阈值的参数（具体含义见调度器实现）
    'THETA': 5.0,
}

# 负载均衡度权重（用于 L = w1*L_CPU + w2*L_RAM + w3*L_BW）
LOAD_BALANCE_WEIGHTS = {
    # w1_cpu: CPU 负载平衡在组合度量中的权重（越大表示越重视 CPU 的均衡）
    'w1_cpu': 0.4,
    # w2_ram: 内存负载平衡权重
    'w2_ram': 0.4,
    # w3_bw: 带宽负载平衡权重
    'w3_bw': 0.2,
}

# AdaEvo 算法参数：可在这里调整 tau（等待时间刻度）和分组数 K
ADAEVO_PARAMS = {
    # TAU: 用于计算紧急程度中的等待时间归一化因子（默认与文档中的建议值相近）
    'TAU': 50.0,
    # K: 将等待队列划分为 K 组用于分组调度
    'K': 3,
}
