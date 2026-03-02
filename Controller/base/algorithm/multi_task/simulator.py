from typing import List, Callable, Dict, Tuple, Optional
import random
from collections import deque
from Controller.base.algorithm.multi_task.models import Job, Queue


class PhysicalNode:
    def __init__(self, nid: str, cpu: float, ram: float):
        self.id = nid
        self.cpu_total = cpu
        self.ram_total = ram
        self.cpu_free = cpu
        self.ram_free = ram

    def allocate(self, cpu: float, ram: float) -> bool:
        if self.cpu_free >= cpu and self.ram_free >= ram:
            self.cpu_free -= cpu
            self.ram_free -= ram
            return True
        return False

    def release(self, cpu: float, ram: float):
        self.cpu_free += cpu
        self.ram_free += ram
        if self.cpu_free > self.cpu_total:
            self.cpu_free = self.cpu_total
        if self.ram_free > self.ram_total:
            self.ram_free = self.ram_total


class PhysicalLink:
    def __init__(self, a: str, b: str, bw: float):
        self.a = a
        self.b = b
        self.bw_total = bw
        self.bw_free = bw

    def allocate(self, bw: float) -> bool:
        if self.bw_free >= bw:
            self.bw_free -= bw
            return True
        return False

    def release(self, bw: float):
        self.bw_free += bw
        if self.bw_free > self.bw_total:
            self.bw_free = self.bw_total


class PhysicalNetwork:
    def __init__(self):
        self.nodes: Dict[str, PhysicalNode] = {}
        self.links: Dict[Tuple[str, str], PhysicalLink] = {}
        self.adj: Dict[str, List[str]] = {}

    def add_node(self, nid: str, cpu: float, ram: float):
        self.nodes[nid] = PhysicalNode(nid, cpu, ram)
        self.adj.setdefault(nid, [])

    def add_link(self, a: str, b: str, bw: float):
        link = PhysicalLink(a, b, bw)
        self.links[(a, b)] = link
        self.links[(b, a)] = PhysicalLink(b, a, bw)
        self.adj.setdefault(a, []).append(b)
        self.adj.setdefault(b, []).append(a)

    def shortest_path(self, src: str, dst: str) -> Optional[List[str]]:
        # BFS shortest by hops
        if src == dst:
            return [src]
        q = deque([[src]])
        seen = {src}
        while q:
            path = q.popleft()
            node = path[-1]
            for nb in self.adj.get(node, []):
                if nb in seen:
                    continue
                if nb == dst:
                    return path + [nb]
                seen.add(nb)
                q.append(path + [nb])
        return None

    def path_links(self, path: List[str]) -> List[Tuple[str, str]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def can_allocate_path(self, path: List[str], bw: float) -> bool:
        for a, b in self.path_links(path):
            link = self.links.get((a, b))
            if link is None or link.bw_free < bw:
                return False
        return True

    def allocate_path(self, path: List[str], bw: float):
        for a, b in self.path_links(path):
            self.links[(a, b)].allocate(bw)

    def release_path(self, path: List[str], bw: float):
        for a, b in self.path_links(path):
            self.links[(a, b)].release(bw)


class Simulator:
    """仿真器支持物理网络、资源分配、作业映射与调度器闭包（可捕获网络）。"""

    def __init__(self, jobs: List[Job], network: PhysicalNetwork, scheduler: Callable[[Queue, float], Optional[int]], dt: float = 1.0):
        self.now = 0.0
        self.dt = dt
        self.all_jobs = sorted(jobs, key=lambda j: j.arrival)
        self.wait_q = Queue()
        self.run_q = Queue()
        self.completed: List[Job] = []
        self.scheduler = scheduler
        self._job_idx = 0
        self.network = network
        self.history: List[Dict] = []

    def _release_arrivals(self):
        while self._job_idx < len(self.all_jobs) and self.all_jobs[self._job_idx].arrival <= self.now:
            self.wait_q.push(self.all_jobs[self._job_idx])
            self._job_idx += 1

    def try_map_job(self, job: Job) -> bool:
        # 如果作业要求使用 new_heuristic，则先尝试启发式映射
        if job.metadata.get('use_new_heuristic'):
            try:
                ok = self.try_map_job_with_new_heuristic(job)
                if ok:
                    return True
            except Exception as e:
                # 若启发式调用失败，记录并回退到贪心
                print(f"⚠️ new_heuristic mapping failed: {e}, fallback to greedy")

        # 贪心：为每个任务分配一个有足够资源的节点
        assigned = []
        nodes = list(self.network.nodes.keys())[:]
        random.shuffle(nodes)
        for _ in range(job.task_count):
            placed = False
            for nid in nodes:
                node = self.network.nodes[nid]
                # 尝试分配均匀拆分CPU/RAM需求
                cpu_need = job.cpu / job.task_count
                ram_need = job.ram / job.task_count
                if node.allocate(cpu_need, ram_need):
                    assigned.append((nid, cpu_need, ram_need))
                    placed = True
                    break
            if not placed:
                # 回滚已分配
                for nid, c, r in assigned:
                    self.network.nodes[nid].release(c, r)
                return False

        # 简化带宽需求：在两个被分配节点间寻找一条满足 bw_min 的路径，并分配 bw_min
        bw_req = max(job.bw_min, 0.0)
        if bw_req > 0 and len(assigned) >= 2:
            a = assigned[0][0]
            b = assigned[1][0]
            path = self.network.shortest_path(a, b)
            if path is None or not self.network.can_allocate_path(path, bw_req):
                # 回滚 node 分配
                for nid, c, r in assigned:
                    self.network.nodes[nid].release(c, r)
                return False
            else:
                path_links = self.network.path_links(path)
                # 只有当物理路径包含实际链路时，才分配物理带宽并记入 job.bw_alloc
                if path_links:
                        self.network.allocate_path(path, bw_req)
                        job.bw_alloc += bw_req
                        job.allocated_paths.append((path_links, bw_req))

        # 记录 assigned nodes id
        job.assigned_nodes = [nid for nid, _, _ in assigned]
        return True

    def try_map_job_with_new_heuristic(self, job: Job) -> bool:
        """尝试使用 new_heuristic 环境与启发式代理为单个作业完成映射。

        若环境或依赖不可用则抛出异常，调用者可回退到其他策略。
        返回 True/False 表示映射是否成功。
        """
        import sys, os
        # 支持多个候选路径（包含用户提供的绝对路径）
        candidates = []
        # user-known absolute path (from user)
        candidates.append('/home/qianguo/Edge-Scheduler/Controller/base/algorithm/PPO_my')
        # relative attempts from this file
        candidates.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'algorithm', 'PPO_my')))
        candidates.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'PPO_my')))
        candidates.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'PPO_my')))

        found = None
        for cand in candidates:
            if os.path.isdir(cand):
                found = cand
                break
        if found is None:
            tried = ', '.join(candidates)
            raise ImportError(f"PPO_my directory not found: tried {tried}")

        sys.path.append(found)
        # print(f"🔎 using PPO_my path: {found}")

        try:
            from network_scheduler import VirtualWork
            from new_heuristic_environment import NewHeuristicEnvironment
            # 先尝试从 PPO 加载 SimpleSequentialAgent 并运行一次 greedy episode
            try:
                import torch
                from sequential_agent import SimpleSequentialAgent
            except Exception:
                SimpleSequentialAgent = None

            from heuristic_algorithm import create_heuristic_agent, run_heuristic_episode
        except Exception as e:
            raise ImportError(f"failed to import PPO_my modules: {e}")

        # 构造 VirtualWork
        # topology in our simple model: create VirtualWork with task_count nodes
        num_vnodes = job.task_count
        vw = VirtualWork(num_vnodes)
        # 设置节点需求（假设均摊）
        cpu_each = int(max(1, job.cpu / max(1, job.task_count)))
        ram_each = int(max(1, job.ram / max(1, job.task_count)))
        for idx in range(num_vnodes):
            vw.set_node_requirement(idx, cpu_each, ram_each)

        # 设置虚拟链路需求：简单全连（可根据 job.topology 扩展）
        # use bw_min for link min
        for i in range(num_vnodes):
            for j in range(i + 1, num_vnodes):
                min_bw = int(job.bw_min)
                max_bw = int(job.bw_max) if job.bw_max and job.bw_max > job.bw_min else int(job.bw_min)
                vw.add_link_requirement(i, j, min_bandwidth_1_to_2=min_bw, max_bandwidth_1_to_2=max_bw,
                                        min_bandwidth_2_to_1=min_bw, max_bandwidth_2_to_1=max_bw)

        # 创建环境
        cfg = {
            'seed': 42,
            'virtual_nodes_range': (num_vnodes, num_vnodes),
            'max_virtual_nodes': max(num_vnodes, 8),
            'curriculum_enabled': False,
            'num_physical_nodes': len(self.network.nodes),
            'bandwidth_levels': 10,
            'use_external_virtual_work': True,
            'external_virtual_work': vw
        }

        env = NewHeuristicEnvironment(**cfg)
        # 优先尝试使用 PPO agent（若可用且存在 checkpoint）执行一次 greedy episode
        total_reward = 0.0
        success = False
        episode_info = {}

        if SimpleSequentialAgent is not None:
            try:
                # 寻找可能的 checkpoint 文件（优先 final.pt / best_model.pth）
                ckpt_candidate = None
                ckpt_root = os.path.join(found, 'checkpoints')
                if os.path.isdir(ckpt_root):
                    # 遍历子目录寻找模型文件
                    candidates = []
                    for root, dirs, files in os.walk(ckpt_root):
                        for fn in files:
                            if fn.endswith('.pt') or fn.endswith('.pth'):
                                candidates.append(os.path.join(root, fn))
                    if candidates:
                        # 优先选择包含 'final' 或 'best' 的文件
                        pref = [c for c in candidates if ('final' in os.path.basename(c).lower() or 'best' in os.path.basename(c).lower())]
                        if pref:
                            ckpt_candidate = pref[0]
                        else:
                            # 否则按修改时间选最新
                            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            ckpt_candidate = candidates[0]

                agent = None
                if ckpt_candidate is not None:
                    try:
                        ckpt = torch.load(ckpt_candidate, map_location='cpu')
                        # 尝试从 ckpt 中读取配置以构造 agent
                        agent_cfg = None
                        if isinstance(ckpt, dict) and 'config' in ckpt:
                            agent_cfg = ckpt['config'].get('agent_config', None)
                        # fallback defaults
                        agent_args = {
                            'max_physical_nodes': max(8, len(self.network.nodes)),
                            'max_virtual_nodes': max(4, num_vnodes),
                            'bandwidth_levels': cfg.get('bandwidth_levels', 10),
                        }
                        if agent_cfg and isinstance(agent_cfg, dict):
                            agent_args.update({k: agent_cfg.get(k, v) for k, v in agent_args.items()})

                        agent = SimpleSequentialAgent(**agent_args)
                        # 尝试不同常见的 state_dict key
                        state_dict = None
                        if isinstance(ckpt, dict):
                            for key in ('agent_state_dict', 'model_state_dict', 'state_dict', 'agent'):
                                if key in ckpt:
                                    state_dict = ckpt[key]
                                    break
                        if state_dict is None and isinstance(ckpt, dict):
                            # 有可能 checkpoint 就是 state_dict
                            state_dict = ckpt
                        if state_dict is not None:
                            try:
                                agent.load_state_dict(state_dict)
                            except Exception:
                                # 有时state_dict里层级不同，尝试直接使用
                                agent.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
                        agent.eval()

                        # 运行一个 greedy episode
                        state = env.reset()
                        done = False
                        steps = 0
                        max_steps = 50
                        total_reward = 0.0
                        while not done and steps < max_steps:
                            # 以贪心方式选择动作：直接选 logits 的 argmax（分别处理映射和带宽阶段）
                            with torch.no_grad():
                                s_dev = agent._move_state_to_device(state)
                                logits, _, _ = agent.forward(s_dev)
                                logits = logits.detach().cpu().numpy()
                            if state.get('mapping_phase', True):
                                num_actions = int(state.get('num_physical_nodes', len(self.network.nodes)))
                                act = int(logits[:num_actions].argmax()) if num_actions>0 else int(np.argmax(logits))
                            else:
                                num_levels = agent.bandwidth_levels
                                act = int(logits[:num_levels].argmax()) if num_levels>0 else int(np.argmax(logits))
                            state, reward, done, info = env.step(int(act))
                            total_reward += float(reward)
                            steps += 1

                        success = bool(info.get('success', False) or getattr(env, 'success_history', False) or total_reward>0)
                        # 尝试从 env 或 info 中提取映射与带宽
                        episode_info = {}
                        if hasattr(env, 'network_scheduler') and env.network_scheduler is not None:
                            ns_obj = env.network_scheduler
                            if hasattr(ns_obj, 'final_mapping'):
                                episode_info['final_mapping'] = getattr(ns_obj, 'final_mapping')
                            if hasattr(ns_obj, 'final_bandwidth'):
                                episode_info['final_bandwidth'] = getattr(ns_obj, 'final_bandwidth')
                        # 也尝试从 env.partial_mapping / env.partial_bandwidth
                        if 'final_mapping' not in episode_info and hasattr(env, 'partial_mapping'):
                            episode_info['final_mapping'] = getattr(env, 'partial_mapping')
                        if 'final_bandwidth' not in episode_info and 'final_bandwidth' in info:
                            episode_info['final_bandwidth'] = info.get('final_bandwidth')
                        episode_info['steps'] = steps
                        episode_info['success'] = success
                        episode_info['total_reward'] = total_reward
                    except Exception as e:
                        # 若任何步骤失败，清理并回退到启发式
                        print(f"⚠️ PPO agent run failed: {e}")
                        agent = None
                else:
                    agent = None
            except Exception as e:
                agent = None

        # 如果 PPO 路径失败或不可用，则回退到原有的启发式运行
        if 'total_reward' not in locals() or agent is None:
            agent = create_heuristic_agent('flexitask')
            total_reward, success, episode_info = run_heuristic_episode(env, agent)

        # 从 env 或 episode_info 中尝试提取映射结果
        mapping = None
        bw_alloc = {}
        if isinstance(episode_info, dict) and 'env' in episode_info:
            env_obj = episode_info['env']
        else:
            env_obj = env

        ns = getattr(env_obj, 'network_scheduler', None)
        if ns is not None:
            # 尝试常见属性
            if hasattr(ns, 'mapping'):
                mapping = getattr(ns, 'mapping')
            elif hasattr(ns, 'final_mapping'):
                mapping = getattr(ns, 'final_mapping')
            elif hasattr(ns, 'partial_mapping'):
                mapping = getattr(ns, 'partial_mapping')
            # 带宽信息
            if hasattr(ns, 'bandwidth_allocation'):
                bw_alloc = getattr(ns, 'bandwidth_allocation')

        # 若 ns 未提供 mapping，尝试从 episode_info 中提取 final_mapping / final_bandwidth
        final_bw_list = None
        if mapping is None and isinstance(episode_info, dict):
            if 'final_mapping' in episode_info and episode_info.get('final_mapping') is not None:
                fm = episode_info.get('final_mapping')
                # final_mapping 可能为 list 或 dict
                if isinstance(fm, list):
                    try:
                        mapping = {i: fm[i] for i in range(len(fm))}
                    except Exception:
                        mapping = None
                elif isinstance(fm, dict):
                    mapping = fm
            # 带宽列表通常与虚拟链路一一对应
            if 'final_bandwidth' in episode_info and episode_info.get('final_bandwidth') is not None:
                fb = episode_info.get('final_bandwidth')
                if isinstance(fb, list):
                    final_bw_list = fb

        # mapping 期望为 dict: vnode_idx -> physical_node_idx
        if mapping is None:
            # 不能解析映射，视为失败 —— 增加调试输出以便调查
            try:
                print("--- new_heuristic debug start ---")
                print("episode_info type:", type(episode_info))
                if isinstance(episode_info, dict):
                    print("episode_info keys:", list(episode_info.keys()))
                print("env_obj type:", type(env_obj))
                try:
                    env_attrs = [a for a in dir(env_obj) if not a.startswith('_')]
                    print('env_obj attrs sample:', env_attrs[:50])
                except Exception as _e:
                    print('failed to list env_obj attrs:', _e)
                print('ns type:', type(ns))
                try:
                    if ns is not None:
                        ns_attrs = [a for a in dir(ns) if not a.startswith('_')]
                        print('ns attrs sample:', ns_attrs[:50])
                    else:
                        print('ns is None')
                except Exception as _e:
                    print('failed to list ns attrs:', _e)
                # 打印 episode_info 的前部分，防止日志过长
                try:
                    s = str(episode_info)
                    print('episode_info (truncated):', s[:2000])
                except Exception:
                    pass
                print("--- new_heuristic debug end ---")
            except Exception:
                pass
            raise RuntimeError('heuristic did not produce a mapping')

        # 将映射转换为物理节点 id 列表，并尝试分配资源
        assigned = []
        for vnode_idx, p_idx in mapping.items():
            # p_idx 可能为整数索引，尝试从 network.nodes 顺序映射
            try:
                p_idx_int = int(p_idx)
                node_keys = list(self.network.nodes.keys())
                if p_idx_int < 0 or p_idx_int >= len(node_keys):
                    raise IndexError
                nid = node_keys[p_idx_int]
            except Exception:
                # 如果直接是节点名
                nid = str(p_idx)

            node = self.network.nodes.get(nid)
            if node is None:
                # 未找到物理节点，失败
                # 回滚已分配
                for nid2, c, r in assigned:
                    self.network.nodes[nid2].release(c, r)
                return False
            cpu_need = job.cpu / max(1, job.task_count)
            ram_need = job.ram / max(1, job.task_count)
            if not node.allocate(cpu_need, ram_need):
                for nid2, c, r in assigned:
                    self.network.nodes[nid2].release(c, r)
                return False
            assigned.append((nid, cpu_need, ram_need))

        # 处理带宽：尝试为每个虚拟链接分配 bw_min 沿 shortest path
        idx_link = 0
        for i in range(num_vnodes):
            for j in range(i + 1, num_vnodes):
                a = mapping.get(i)
                b = mapping.get(j)
                if a is None or b is None:
                    continue
                # resolve node ids
                try:
                    a_idx = int(a)
                    b_idx = int(b)
                    node_keys = list(self.network.nodes.keys())
                    a_id = node_keys[a_idx]
                    b_id = node_keys[b_idx]
                except Exception:
                    a_id = str(a)
                    b_id = str(b)
                path = self.network.shortest_path(a_id, b_id)
                if path is None:
                    # 回滚
                    for nid2, c, r in assigned:
                        self.network.nodes[nid2].release(c, r)
                    return False
                # 如果启发式提供了 final_bandwidth 列表，则使用对应条目的带宽，否则退回到 job.bw_min
                if 'final_bw_list' in locals() and final_bw_list is not None and idx_link < len(final_bw_list):
                    try:
                        bw_req = float(final_bw_list[idx_link])
                    except Exception:
                        bw_req = job.bw_min
                else:
                    bw_req = job.bw_min
                if not self.network.can_allocate_path(path, bw_req):
                    for nid2, c, r in assigned:
                        self.network.nodes[nid2].release(c, r)
                    return False
                path_links = self.network.path_links(path)
                # 只有在存在物理链路时才进行分配与累加
                if path_links:
                    self.network.allocate_path(path, bw_req)
                    job.allocated_paths.append((path_links, bw_req))
                    job.bw_alloc += bw_req
                idx_link += 1

        job.assigned_nodes = [nid for nid, _, _ in assigned]
        return True

    def release_job_resources(self, job: Job):
        # 释放节点资源（假设均分）
        if job.assigned_nodes:
            cpu_each = job.cpu / max(1, job.task_count)
            ram_each = job.ram / max(1, job.task_count)
            for nid in job.assigned_nodes:
                if nid in self.network.nodes:
                    self.network.nodes[nid].release(cpu_each, ram_each)
        # 释放带宽
        # allocated_paths may store either old format (list of tuples)
        # or new format ( (path_links, bw) ), so handle both for compatibility.
        for item in job.allocated_paths:
            if not item:
                continue
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
                path_links, bw = item
            else:
                # old format: item is path_links, and bw unknown -> fallback to job.bw_alloc
                path_links = item
                bw = job.bw_alloc
            # convert path_links list of tuples back to path list
            try:
                path = [path_links[0][0]] + [p[1] for p in path_links]
            except Exception:
                continue
            self.network.release_path(path, bw)

    def step(self):
        self._release_arrivals()

        # 调度决策：scheduler 返回要调度的索引或 None
        if not self.wait_q.is_empty():
            pick = self.scheduler(self.wait_q, self.now)
            if pick is not None and 0 <= pick < len(self.wait_q):
                job = self.wait_q.peek(pick)
                # 尝试映射并分配资源
                success = self.try_map_job(job)
                if success:
                    job = self.wait_q.pop(pick)
                    if job.start_time is None:
                        job.start_time = self.now
                    job.skip_count = 0
                    self.run_q.push(job)
                    # 仅在调度成功后打印相关信息
                    try:
                        print(f"[t={self.now:.1f}] 调度成功: job id={job.id}, task_count={job.task_count}, assigned_nodes={job.assigned_nodes}, bw_alloc={job.bw_alloc}")
                    except Exception:
                        pass
                else:
                    # 失败则增加 skip_count
                    job.skip_count += 1

        # 推进所有正在运行的作业（支持并行执行）
        if not self.run_q.is_empty():
            finished_jobs = []
            # 遍历运行队列的副本以安全修改队列
            for job in list(self.run_q.jobs):
                job.remaining -= self.dt
                if job.is_finished():
                    job.finish_time = self.now + self.dt
                    finished_jobs.append(job)
            # 移除并处理已完成作业
            for fj in finished_jobs:
                try:
                    self.run_q.remove(fj)
                except Exception:
                    # 若队列已变动且无法移除，忽略
                    pass
                self.release_job_resources(fj)
                self.completed.append(fj)

        self.now += self.dt

        # 记录历史指标
        completed_count = len(self.completed)
        # 平均等待时间（仅已完成作业）- 保留原有统计以兼容历史逻辑
        waiting_completed = [((j.start_time - j.arrival) if j.start_time is not None else 0.0) for j in self.completed]
        avg_wait_completed = (sum(waiting_completed) / len(waiting_completed)) if waiting_completed else 0.0

        # 新增：对已到达但未必完成的作业统计截止时间到作业开始的等待时间
        # 对每个已到达作业：若已开始，等待 = start_time - arrival；若未开始，等待 = now - arrival
        arrived_jobs = [j for j in self.all_jobs if j.arrival <= self.now]
        waiting_arrived = []
        for j in arrived_jobs:
            if j.start_time is not None:
                waiting_arrived.append(j.start_time - j.arrival)
            else:
                waiting_arrived.append(self.now - j.arrival)
        avg_wait_arrived = (sum(waiting_arrived) / len(waiting_arrived)) if waiting_arrived else 0.0
        # 将历史中的 'avg_waiting' 字段更新为包含未完成作业截止至当前的平均等待（新含义）
        avg_wait = avg_wait_arrived
        bw_satisfaction = []
        for j in self.completed:
            if j.bw_max > 0:
                # 如果所有虚拟节点映射到同一物理节点，则视为带宽满足（无需占用物理链路）
                if getattr(j, 'assigned_nodes', None) and len(set(j.assigned_nodes)) <= 1:
                    bw_satisfaction.append(1.0)
                else:
                    bw_satisfaction.append(min(1.0, j.bw_alloc / j.bw_max))
        avg_bw = (sum(bw_satisfaction) / len(bw_satisfaction)) if bw_satisfaction else 0.0

        # 节点平均 CPU 利用 (used / total)
        cpu_used = 0.0
        cpu_total = 0.0
        node_cpu_utils = []
        node_ram_utils = []
        for n in self.network.nodes.values():
            used_cpu = (n.cpu_total - n.cpu_free)
            used_ram = (n.ram_total - n.ram_free)
            cpu_total += n.cpu_total
            cpu_used += used_cpu
            node_cpu_utils.append((used_cpu / n.cpu_total) if n.cpu_total > 0 else 0.0)
            node_ram_utils.append((used_ram / n.ram_total) if n.ram_total > 0 else 0.0)
        avg_cpu_util = (cpu_used / cpu_total) if cpu_total > 0 else 0.0

        # 链路平均带宽利用
        bw_used = 0.0
        bw_total = 0.0
        seen = set()
        link_bw_utils = []
        for (a, b), l in self.network.links.items():
            # links stored bidirectionally; count each undirected once
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            bw_used += (l.bw_total - l.bw_free)
            bw_total += l.bw_total
            link_bw_utils.append(((l.bw_total - l.bw_free) / l.bw_total) if l.bw_total>0 else 0.0)
        avg_bw_util = (bw_used / bw_total) if bw_total > 0 else 0.0

        # 负载均衡度按三类资源的标准差计算（population std）
        def _std(vals):
            if not vals:
                return 0.0
            m = sum(vals) / len(vals)
            return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

        L_cpu = _std(node_cpu_utils)
        L_ram = _std(node_ram_utils)
        L_bw = _std(link_bw_utils)
        try:
            from Controller.base.algorithm.multi_task.sim_config import LOAD_BALANCE_WEIGHTS as _w
            w = _w
        except Exception:
            w = {'w1_cpu': 0.4, 'w2_ram': 0.4, 'w3_bw': 0.2}
        load_balance = w.get('w1_cpu', 0.4) * L_cpu + w.get('w2_ram', 0.4) * L_ram + w.get('w3_bw', 0.2) * L_bw

        self.history.append({
            'time': self.now,
            'completed': completed_count,
            # 新增两个字段：
            #  - avg_waiting: 对所有已到达作业（包含未开始）按当前时间计算的平均等待
            #  - avg_waiting_completed: 保留原先仅对已完成作业的平均等待
            'avg_waiting': avg_wait,
            'avg_waiting_completed': avg_wait_completed,
            'avg_bw_satisfaction': avg_bw,
            'avg_cpu_util': avg_cpu_util,
            'avg_bw_util': avg_bw_util,
            'L_cpu': L_cpu,
            'L_ram': L_ram,
            'L_bw': L_bw,
            'load_balance': load_balance,
        })

    def save_history_csv(self, path: str):
        import csv, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self.history:
            return
        keys = ['time', 'completed', 'avg_waiting', 'avg_waiting_completed', 'avg_bw_satisfaction', 'avg_cpu_util', 'avg_bw_util',
            'L_cpu', 'L_ram', 'L_bw', 'load_balance']
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def run_until(self, t_max: float):
        while self.now < t_max and (self._job_idx < len(self.all_jobs) or len(self.wait_q) > 0 or len(self.run_q) > 0):
            self.step()

    def metrics(self) -> Dict:
        makespan = max((j.finish_time for j in self.completed), default=0.0)
        turnaround = [((j.finish_time - j.arrival) if j.finish_time is not None else 0.0) for j in self.completed]
        waiting = [((j.start_time - j.arrival) if j.start_time is not None else 0.0) for j in self.completed] 
        bw_satisfaction = []
        for j in self.completed:
            if j.bw_max > 0:
                if getattr(j, 'assigned_nodes', None) and len(set(j.assigned_nodes)) <= 1:
                    bw_satisfaction.append(1.0)
                else:
                    bw_satisfaction.append(min(1.0, j.bw_alloc / j.bw_max))
        # New average waiting time calculations
        arrived_jobs = [j for j in self.all_jobs if j.arrival <= self.now]
        waiting_arrived = []
        for j in arrived_jobs:
            if j.start_time is not None:
                waiting_arrived.append(j.start_time - j.arrival)
            else:
                waiting_arrived.append(self.now - j.arrival)
        avg_wait_arrived = (sum(waiting_arrived) / len(waiting_arrived)) if waiting_arrived else 0.0
        waiting_completed = [((j.start_time - j.arrival) if j.start_time is not None else 0.0) for j in self.completed]
        avg_wait_completed = (sum(waiting_completed) / len(waiting_completed)) if waiting_completed else 0.0
        return {
            "now": self.now,
            "completed": len(self.completed),
            "makespan": makespan,
            "avg_turnaround": (sum(turnaround) / len(turnaround)) if turnaround else 0.0,
            "avg_waiting": avg_wait_arrived,
            "avg_waiting_completed": avg_wait_completed,
            "avg_bw_satisfaction": (sum(bw_satisfaction) / len(bw_satisfaction)) if bw_satisfaction else 0.0,
        }