import importlib.util
import pathlib
import random

# 动态从文件路径加载模块，避免导入上层 Controller 包导致的副作用
BASE = pathlib.Path(__file__).resolve().parent
def load_module_from(path: str, name: str):
    file = str((BASE / path).resolve())
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

models = load_module_from('models.py', 'mt_models')
import sys
# 将已加载的 models 模块挂入 sys.modules，供其他文件按绝对路径导入时使用，避免触发 Controller 包的 __init__
sys.modules['Controller.base.algorithm.multi_task.models'] = models
sim_mod = load_module_from('simulator.py', 'mt_simulator')
# simulator 可能在加载时需要 models 在 sys.modules 中，因此先设置后加载
sys.modules['Controller.base.algorithm.multi_task.simulator'] = sim_mod
alg_mod = load_module_from('algorithms.py', 'mt_algorithms')
sys.modules['Controller.base.algorithm.multi_task.algorithms'] = alg_mod
# 加载集中配置
sim_config = load_module_from('sim_config.py', 'mt_config')
sys.modules['Controller.base.algorithm.multi_task.sim_config'] = sim_config

Job = models.Job
Simulator = sim_mod.Simulator
PhysicalNetwork = sim_mod.PhysicalNetwork
make_ces_scheduler = alg_mod.make_ces_scheduler
make_fifo_scheduler = alg_mod.make_fifo_scheduler
make_sjf_scheduler = alg_mod.make_sjf_scheduler
make_random_scheduler = alg_mod.make_random_scheduler
make_priority_scheduler = alg_mod.make_priority_scheduler
make_swts_scheduler = alg_mod.make_swts_scheduler
make_adaevo_scheduler = getattr(alg_mod, 'make_adaevo_scheduler', None)


def build_random_network(num_nodes=None, conn_prob=None, cpu_range=None, ram_range=None, bw_range=None):
    # 优先使用 sim_config 中的设置，参数可覆盖
    num_nodes = num_nodes if num_nodes is not None else sim_config.NETWORK['NUM_NODES']
    conn_prob = conn_prob if conn_prob is not None else sim_config.NETWORK['CONN_PROB']
    cpu_range = cpu_range if cpu_range is not None else sim_config.NETWORK['CPU_RANGE']
    ram_range = ram_range if ram_range is not None else sim_config.NETWORK['RAM_RANGE']
    bw_range = bw_range if bw_range is not None else sim_config.NETWORK['BW_RANGE']

    net = PhysicalNetwork()
    for i in range(num_nodes):
        nid = f"p{i}"
        cpu = random.uniform(*cpu_range)
        ram = random.uniform(*ram_range)
        net.add_node(nid, cpu, ram)
    nodes = list(net.nodes.keys())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if random.random() < conn_prob:
                bw = random.uniform(*bw_range)
                net.add_link(nodes[i], nodes[j], bw)
    return net


def generate_jobs_poisson(lambda_rate=None, t_max=None, cpu_range=None, ram_range=None, bw_range=None, task_range=None, priority_range=None):
    # 使用 sim_config 中的默认参数（可通过参数覆盖）
    lambda_rate = lambda_rate if lambda_rate is not None else sim_config.JOB_GEN['LAMBDA_RATE']
    t_max = t_max if t_max is not None else sim_config.JOB_GEN['T_MAX']
    cpu_range = cpu_range if cpu_range is not None else sim_config.JOB_GEN['CPU_RANGE']
    ram_range = ram_range if ram_range is not None else sim_config.JOB_GEN['RAM_RANGE']
    bw_range = bw_range if bw_range is not None else sim_config.JOB_GEN['BW_RANGE']
    task_range = task_range if task_range is not None else sim_config.JOB_GEN['TASK_RANGE']
    priority_range = priority_range if priority_range is not None else sim_config.JOB_GEN['PRIORITY_RANGE']

    jobs = []
    t = 0.0
    while t < t_max:
        # inter-arrival exponential
        inter = random.expovariate(lambda_rate)
        t += inter
        if t >= t_max:
            break
        # 作业运行时长由配置中的 RUN_TIME_RANGE 决定
        run_time_range = sim_config.JOB_GEN.get('RUN_TIME_RANGE', (1.0, 10.0))
        run_time = random.uniform(*run_time_range)
        cpu = random.uniform(*cpu_range)
        ram = random.uniform(*ram_range)
        bw_min = random.uniform(*bw_range)
        # bw_range 的第二个元素被当作额外上限范围（与原实现兼容）
        bw_max = bw_min + random.uniform(0, bw_range[1])
        priority = random.randint(*priority_range)
        task_count = random.randint(*task_range)
        # 创建 Job ，将 run_time 传入表示资源占用时长
        j = Job(arrival=t, run_time=run_time, cpu=cpu, ram=ram, bw_min=bw_min, bw_max=bw_max, priority=priority, task_count=task_count)
        # 指示优先使用 new_heuristic 进行单作业调度映射
        j.metadata['use_new_heuristic'] = True
        jobs.append(j)
    # sort by arrival
    jobs.sort(key=lambda j: j.arrival)
    return jobs


def run_experiment(scheduler, net: PhysicalNetwork, jobs, t_max=200, dt=1.0):
    sim = Simulator(jobs=jobs, network=net, scheduler=scheduler, dt=dt)
    sim.run_until(t_max)
    return sim


def main():
    # 使用集中配置
    random.seed(sim_config.SEED)
    net = build_random_network(num_nodes=sim_config.NETWORK['NUM_NODES'], conn_prob=sim_config.NETWORK['CONN_PROB'])
    jobs = generate_jobs_poisson(lambda_rate=sim_config.JOB_GEN['LAMBDA_RATE'], t_max=sim_config.JOB_GEN['T_MAX'])

    # 注意：每次仿真应使用独立的物理网络实例，避免不同调度器/运行之间共享被污染的网络状态。
    # 因此不在此处一次性创建所有 scheduler（若 scheduler 捕获 network 引用会导致问题），
    # 而是在每次运行时构建新网络并创建对应的 scheduler。

    import os, csv
    results_dir = BASE / sim_config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    summary_path = results_dir / 'summary.csv'
    # write header for summary
    with open(summary_path, 'w', newline='') as sf:
        writer = csv.DictWriter(sf, fieldnames=['scheduler', 'run', 'completed', 'avg_waiting', 'avg_bw_satisfaction'])
        writer.writeheader()

    runs_per_scheduler = sim_config.RUNS_PER_SCHEDULER
    for name in sim_config.SCHEDULER_NAMES:
        for run_idx in range(runs_per_scheduler):
            # 为当前运行构建全新的物理网络和调度器
            net = build_random_network(num_nodes=sim_config.NETWORK['NUM_NODES'], conn_prob=sim_config.NETWORK['CONN_PROB'])
            name_lower = str(name).lower()
            if name_lower == 'ces':
                scheduler = make_ces_scheduler(net, max_load_factor=sim_config.CES_PARAMS['MAX_LOAD_FACTOR'], theta=sim_config.CES_PARAMS.get('THETA', 5.0))
            elif name == 'fifo':
                scheduler = make_fifo_scheduler()
            elif name == 'sjf':
                scheduler = make_sjf_scheduler()
            elif name == 'priority':
                scheduler = make_priority_scheduler()
            elif name == 'swts':
                # SWTS 风格调度器需要网络引用以评估节点异构性
                scheduler = make_swts_scheduler(net)
            elif name == 'random':
                scheduler = make_random_scheduler()
            elif name_lower == 'adaevo' and make_adaevo_scheduler is not None:
                ada_params = getattr(sim_config, 'ADAEVO_PARAMS', {})
                tau = ada_params.get('TAU', 50.0)
                K = ada_params.get('K', 3)
                scheduler = make_adaevo_scheduler(net, tau=tau, K=K)
            else:
                # 未知调度器名，跳过
                continue

            jobs = generate_jobs_poisson(lambda_rate=sim_config.JOB_GEN['LAMBDA_RATE'], t_max=sim_config.JOB_GEN['T_MAX'])
            sim = run_experiment(scheduler, net, jobs, t_max=sim_config.SIM['T_MAX'], dt=sim_config.SIM['DT'])
            # save history CSV
            fname = f"{scheduler.__name__}_run{run_idx}.csv"
            sim.save_history_csv(str(results_dir / fname))
            m = sim.metrics()
            # append summary
            with open(summary_path, 'a', newline='') as sf:
                writer = csv.DictWriter(sf, fieldnames=['scheduler', 'run', 'completed', 'avg_waiting', 'avg_bw_satisfaction'])
                writer.writerow({'scheduler': scheduler.__name__, 'run': run_idx, 'completed': m['completed'], 'avg_waiting': m['avg_waiting'], 'avg_bw_satisfaction': m.get('avg_bw_satisfaction', 0.0)})
            print(f"Scheduler: {scheduler.__name__} run {run_idx} -> Completed={m['completed']}, Avg waiting={m['avg_waiting']:.2f}, Avg BW sat={m.get('avg_bw_satisfaction',0.0):.2f}")


if __name__ == "__main__":
    main()
