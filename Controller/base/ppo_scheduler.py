"""
PPO调度器模块
封装PPO模型加载、调度决策等功能
"""
import os
import sys
from typing import Dict, Tuple
import torch
import numpy as np

# PPO相关导入
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithm/PPO_my'))
from sequential_agent import SimpleSequentialAgent
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler
from original_reward import OriginalRewardCalculator


dirName = '/home/qianguo/Edge-Scheduler/Controller'


class PPOScheduler:
    """PPO调度器，封装PPO模型的加载和调度功能"""
    
    def __init__(self, controller):
        self.controller = controller
        self.reward_calculator = OriginalRewardCalculator()
        self.ppo_agent = None
        self.ppo_env_config = {}
        self.ppo_agent_config = {}
        
        # 节点映射相关
        self.node_id_mapping = {}  # 物理节点名字到索引
        self.idx_node_mapping = {}  # 索引到物理节点名字
        self.virtual_node_mapping = {}  # 虚拟节点名字到索引
        self.idx_virtual_mapping = {}  # 索引到虚拟节点名字
        self.virtual_node_names = {}  # 索引到原始节点名（不带taskId前缀）
        self.current_task_id = None
        
        self._init_ppo_agent()
    
    def _init_ppo_agent(self):
        """初始化PPO智能体"""
        try:
            model_path = os.path.join(dirName, 'base/algorithm/PPO_my/checkpoints/new_heuristic_20250827_164927/final.pt')
            if os.path.exists(model_path):
                self.ppo_agent, self.ppo_env_config, self.ppo_agent_config = self._load_ppo_agent_and_configs(model_path)
                print(f"✅ PPO模型加载成功: {model_path}")
            else:
                print(f"⚠️  PPO模型文件不存在: {model_path}")
                self.ppo_agent = None
        except Exception as e:
            print(f"❌ PPO模型加载失败: {e}")
            self.ppo_agent = None
    
    def _load_ppo_agent_and_configs(self, ckpt_path: str):
        """加载PPO Agent与配置"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔄 加载PPO检查点: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        if 'config' in checkpoint:
            env_config = checkpoint['config'].get('env_config', {})
            agent_config = checkpoint['config'].get('agent_config', {})
            
            agent = SimpleSequentialAgent(**agent_config)
            state = checkpoint.get('agent_state_dict', None)
            if state is None:
                raise KeyError("Checkpoint missing 'agent_state_dict'.")
            agent.load_state_dict(state)
        else:
            print("⚠️  使用旧格式checkpoint，采用默认配置")
            env_config = {}
            agent_config = {
                'max_physical_nodes': 10,
                'max_virtual_nodes': 8,
                'bandwidth_levels': 10,
                'hidden_dim': 128,
                'lr': 3e-4
            }
            
            agent = SimpleSequentialAgent(**agent_config)
            
            if 'agent_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['agent_state_dict'])
            elif 'model_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['state_dict'])
            else:
                agent.load_state_dict(checkpoint)
        
        agent.eval()
        print("✅ PPO Agent与配置加载完成")
        return agent, env_config, agent_config
    
    def is_available(self) -> bool:
        """检查PPO模型是否可用"""
        return self.ppo_agent is not None
    
    def schedule(self, taskId: int, links_data: Dict) -> Tuple[Dict, Dict]:
        """
        使用PPO算法进行调度
        
        Args:
            taskId: 任务ID
            links_data: 链路数据（从links_range.json读取）
        
        Returns:
            allocation: 节点分配结果
            bandwidth_allocation: 带宽分配结果
        """
        print(f"🤖 使用PPO算法调度任务 {taskId}")
        
        self.current_task_id = taskId
        
        # 创建网络拓扑
        topology = self._create_network_topology()
        
        # 创建虚拟工作
        virtual_work = self._create_virtual_work(taskId, links_data)
        
        # 使用PPO代理进行调度决策
        allocation, episode_info, bandwidth_allocation = self._run_ppo_scheduling_with_metrics(virtual_work)
        
        return allocation, bandwidth_allocation
    
    def _create_network_topology(self) -> NetworkTopology:
        """根据当前的物理资源创建网络拓扑"""
        num_nodes = len(self.controller.emulator)
        topology = NetworkTopology(num_nodes)
        
        node_idx = 0
        self.node_id_mapping = {}
        self.idx_node_mapping = {}
        
        for emulator in self.controller.emulator.values():
            used_cpu = emulator.cpuPreMap
            used_memory = emulator.ramPreMap
            
            topology.set_node_resources(
                node_idx, 
                cpu=emulator.cpu, 
                memory=emulator.ram,
                used_cpu=used_cpu,
                used_memory=used_memory
            )
            
            self.node_id_mapping[emulator.nameW] = node_idx
            self.idx_node_mapping[node_idx] = emulator.nameW
            node_idx += 1
        
        for emu1, emu2, bw, used_bw in self.controller.iter_bandwidth():
            if emu1 in self.node_id_mapping and emu2 in self.node_id_mapping:
                idx1 = self.node_id_mapping[emu1]
                idx2 = self.node_id_mapping[emu2]
                
                topology.add_link(
                    idx1, idx2, 
                    bandwidth_1_to_2=bw,
                    bandwidth_2_to_1=bw,
                    used_bandwidth_1_to_2=used_bw,
                    used_bandwidth_2_to_1=used_bw
                )
        
        return topology
    
    def _create_virtual_work(self, taskId: int, links_data: Dict) -> VirtualWork:
        """创建虚拟工作对象"""
        import random
        
        virtual_nodes = list(links_data.keys())
        num_virtual_nodes = len(virtual_nodes)
        virtual_work = VirtualWork(num_virtual_nodes)
        
        self.virtual_node_mapping = {}
        self.idx_virtual_mapping = {}
        self.virtual_node_names = {}
        
        for idx, node in enumerate(virtual_nodes):
            node_name = str(taskId) + '_' + node
            self.virtual_node_mapping[node_name] = idx
            self.idx_virtual_mapping[idx] = node_name
            self.virtual_node_names[idx] = node
            
            # 使用随机生成的CPU和RAM值
            cpu_demand = random.randint(1, 5)
            ram_demand = random.randint(1, 5)
            virtual_work.set_node_requirement(idx, cpu_demand, ram_demand)
        
        # 设置虚拟链路需求
        for node, connections in links_data.items():
            src_idx = self.virtual_node_mapping[str(taskId) + '_' + node]
            
            for dest in connections:
                dest_node = str(taskId) + '_' + dest['dest']
                if dest_node in self.virtual_node_mapping:
                    dest_idx = self.virtual_node_mapping[dest_node]
                    
                    # 解析带宽范围
                    if 'bw_min' in dest and 'bw_max' in dest:
                        min_bw = int(dest['bw_min'].replace('mbps', ''))
                        max_bw = int(dest['bw_max'].replace('mbps', ''))
                    elif 'bw' in dest and 'bw_max' in dest:
                        min_bw = int(dest['bw'].replace('mbps', ''))
                        max_bw = int(dest['bw_max'].replace('mbps', ''))
                    else:
                        bw = int(dest.get('bw', '10mbps').replace('mbps', ''))
                        min_bw = bw
                        max_bw = bw
                    
                    virtual_work.add_link_requirement(
                        src_idx, dest_idx,
                        min_bandwidth_1_to_2=min_bw,
                        max_bandwidth_1_to_2=max_bw,
                        min_bandwidth_2_to_1=min_bw,
                        max_bandwidth_2_to_1=max_bw
                    )
        
        return virtual_work
    
    def _make_ppo_env_with_virtual_work(self, virtual_work: VirtualWork, seed: int = 42, env_type: str = "NewHeuristic"):
        """基于虚拟工作和环境配置创建PPO环境"""
        try:
            from sequential_environment import SequentialNetworkSchedulerEnvironment
            from lightweight_heuristic_integration_environment import LightweightHeuristicEnvironment
            from new_heuristic_environment import NewHeuristicEnvironment
        except ImportError:
            try:
                sys.path.append(os.path.join(dirName, 'base/algorithm/PPO_my'))
                from sequential_environment import SequentialNetworkSchedulerEnvironment
                from lightweight_heuristic_integration_environment import LightweightHeuristicEnvironment
                from new_heuristic_environment import NewHeuristicEnvironment
            except ImportError as e:
                print(f"⚠️  无法导入PPO环境类: {e}")
                raise ImportError("PPO环境类导入失败，请检查路径配置")
        
        num_virtual_nodes = len(virtual_work.node_requirements)
        
        cfg = dict(self.ppo_env_config) if hasattr(self, 'ppo_env_config') else {}
        cfg.update({
            'seed': seed,
            'virtual_nodes_range': (num_virtual_nodes, num_virtual_nodes),
            'max_virtual_nodes': max(num_virtual_nodes, cfg.get('max_virtual_nodes', 8)),
            'curriculum_enabled': False,
            'num_physical_nodes': len(self.controller.emulator),
            'bandwidth_levels': cfg.get('bandwidth_levels', 10)
        })
        
        if env_type == "Sequential":
            env = SequentialNetworkSchedulerEnvironment(**cfg)
        elif env_type == "Lightweight":
            env = LightweightHeuristicEnvironment(**cfg)
        elif env_type == "NewHeuristic":
            env = NewHeuristicEnvironment(**cfg)
        else:
            raise ValueError(f"不支持的环境类型: {env_type}")
        
        return env
    
    def _select_action_greedy_with_fallback(self, env, agent: SimpleSequentialAgent, state: Dict) -> int:
        """PPO贪心策略选择动作，带回退机制"""
        agent.eval()
        with torch.no_grad():
            move_state = agent._move_state_to_device(state)
            logits, _, _ = agent.forward(move_state)
            logits = logits.detach().cpu().numpy()

        if state.get('mapping_phase', True):
            num_actions = state.get('num_physical_nodes', env.num_physical_nodes if hasattr(env, 'num_physical_nodes') else 10)
            candidates = np.argsort(-logits[:num_actions])
            for a in candidates:
                if hasattr(env, '_validate_mapping_action'):
                    ok, _ = env._validate_mapping_action(state.get('current_virtual_node', 0), int(a))
                    if ok:
                        return int(a)
            return int(candidates[0]) if len(candidates) > 0 else 0
        else:
            num_actions = getattr(env, 'bandwidth_levels', 4)
            candidates = np.argsort(-logits[:num_actions])
            for a in candidates:
                if hasattr(env, '_validate_bandwidth_action'):
                    ok, _ = env._validate_bandwidth_action(state.get('current_link_index', 0), int(a))
                    if ok:
                        return int(a)
            return int(candidates[0]) if len(candidates) > 0 else 0
    
    def _integrate_original_reward_after_reset(self, env):
        """在环境reset后集成原始奖励计算器"""
        if hasattr(env, 'network_scheduler') and env.network_scheduler is not None:
            try:
                from original_reward import integrate_with_network_scheduler
                integrate_with_network_scheduler(env.network_scheduler)
                return True
            except ImportError:
                try:
                    sys.path.append(os.path.join(dirName, 'base/algorithm/PPO_my'))
                    from original_reward import integrate_with_network_scheduler
                    integrate_with_network_scheduler(env.network_scheduler)
                    return True
                except Exception as e:
                    print(f"Warning: Failed to integrate original reward calculator: {e}")
                    return False
            except Exception as e:
                print(f"Warning: Failed to integrate original reward calculator: {e}")
                return False
        return False
    
    def _run_ppo_scheduling_with_metrics(self, virtual_work: VirtualWork) -> Tuple[Dict, Dict, Dict]:
        """使用PPO代理进行调度决策，返回allocation、episode_info和bandwidth_allocation"""
        print("🤖 开始PPO环境-代理交互调度...")
        
        try:
            env_type = "NewHeuristic"
            env = self._make_ppo_env_with_virtual_work(virtual_work, seed=42, env_type=env_type)
            
            total_reward, success, episode_info = self._run_ppo_episode(
                env, self.ppo_agent, 
                temperature=0.05,
                max_steps=50,
                greedy=True
            )
            
            L_val = episode_info.get('load_balance_degree', 0.0)
            D_BW_val = episode_info.get('bandwidth_satisfaction', 0.0)
            print(f"📊 PPO调度结果: 奖励={total_reward:.3f}, 成功={success}, 步数={episode_info['steps']}")
            print(f"   L={L_val if isinstance(L_val, (int, float)) else 0.0:.4f}, D_BW={D_BW_val if isinstance(D_BW_val, (int, float)) else 0.0:.4f}")
            
            # 从环境中提取分配结果和带宽分配
            allocation = self._extract_allocation_from_env(env, episode_info)
            bandwidth_allocation = self._extract_bandwidth_allocation_from_env(env)
            
            return allocation, episode_info, bandwidth_allocation
            
        except Exception as e:
            print(f"❌ PPO环境调度失败: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}, {}
    
    def _run_ppo_episode(self, env, agent: SimpleSequentialAgent, 
                        temperature: float = 0.1, max_steps: int = 50, 
                        greedy: bool = True) -> Tuple[float, bool, Dict]:
        """运行PPO算法的Episode"""
        state = env.reset()
        
        self._integrate_original_reward_after_reset(env)
        
        done = False
        total_reward = 0.0
        steps = 0
        
        episode_info = {
            'num_virtual_nodes': state.get('num_virtual_nodes', state.get('num_tasks', 0)),
            'num_virtual_links': state.get('num_virtual_links', 0),
            'algorithm': 'PPO',
            'steps': 0
        }

        while not done and steps < max_steps:
            if greedy:
                action = self._select_action_greedy_with_fallback(env, agent, state)
            else:
                action, _, _ = agent.select_action(state, temperature)

            state, reward, done, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1

        # 成功判定
        if hasattr(env, 'partial_mapping'):
            all_nodes_mapped = all(node != -1 for node in (env.partial_mapping or []))
            success = bool(all_nodes_mapped and total_reward > 0)
        else:
            success = bool(done and total_reward > 0)
        
        # 计算L和D_BW指标
        load_balance_degree = 0.0
        bandwidth_satisfaction = 0.0
        
        if (hasattr(env, 'network_scheduler') and env.network_scheduler is not None and 
            hasattr(env.network_scheduler, 'get_original_reward_components')):
            try:
                if hasattr(env, 'virtual_work_obj') and env.virtual_work_obj is not None:
                    components = env.network_scheduler.get_original_reward_components(env.virtual_work_obj)
                    load_balance_degree = components.get('L', 0.0)
                    bandwidth_satisfaction = components.get('D_BW', 0.0)
                    print(f"   从环境获取指标: L={load_balance_degree:.4f}, D_BW={bandwidth_satisfaction:.4f}")
            except Exception as e:
                print(f"⚠️  从环境获取指标失败: {e}")
        
        if load_balance_degree == 0.0 and bandwidth_satisfaction == 0.0:
            try:
                if hasattr(env, 'network_scheduler') and hasattr(env, 'virtual_work_obj'):
                    reward_result = self.reward_calculator.calculate_reward(
                        env.network_scheduler, 
                        env.virtual_work_obj
                    )
                    load_balance_degree = reward_result.get('L', 0.0)
                    bandwidth_satisfaction = reward_result.get('D_BW', 0.0)
                    print(f"   使用备用计算器获取指标: L={load_balance_degree:.4f}, D_BW={bandwidth_satisfaction:.4f}")
            except Exception as e:
                print(f"⚠️  备用指标计算也失败: {e}")
        
        episode_info.update({
            'steps': steps,
            'success': success,
            'total_reward': total_reward,
            'load_balance_degree': load_balance_degree,
            'bandwidth_satisfaction': bandwidth_satisfaction,
            'env': env
        })
        
        return total_reward, success, episode_info
    
    def _extract_allocation_from_env(self, env, episode_info: Dict) -> Dict:
        """从PPO环境中提取分配结果"""
        allocation = {}
        
        try:
            if hasattr(env, 'network_scheduler') and env.network_scheduler is not None:
                scheduler = env.network_scheduler
                
                if hasattr(scheduler, 'node_mapping'):
                    for virtual_idx, physical_idx in scheduler.node_mapping.items():
                        if physical_idx != -1:
                            virtual_node_name = self.idx_virtual_mapping.get(virtual_idx, f"{self.current_task_id}_{virtual_idx}")
                            physical_node_name = self.idx_node_mapping.get(physical_idx, f"node_{physical_idx}")
                            
                            if hasattr(env, 'virtual_work_obj') and env.virtual_work_obj:
                                virtual_req = env.virtual_work_obj.node_requirements.get(virtual_idx, {})
                                cpu_req = virtual_req.get('cpu', 2)
                                ram_req = virtual_req.get('memory', 5)
                            else:
                                cpu_req = 2
                                ram_req = 5
                            
                            allocation[virtual_node_name] = {
                                'emulator': physical_node_name,
                                'cpu': cpu_req,
                                'ram': ram_req
                            }
                
                print(f"✅ 从PPO环境提取到 {len(allocation)} 个节点分配")
                
            else:
                print("⚠️  PPO环境中没有找到调度器，无法提取分配结果")
                
        except Exception as e:
            print(f"❌ 从PPO环境提取分配结果失败: {e}")
        
        return allocation
    
    def _extract_bandwidth_allocation_from_env(self, env) -> Dict:
        """从PPO环境中提取带宽分配结果"""
        bandwidth_allocation = {}
        
        try:
            if hasattr(env, 'network_scheduler') and env.network_scheduler is not None:
                scheduler = env.network_scheduler
                
                if hasattr(scheduler, 'bandwidth_allocation'):
                    for (v_from, v_to), bw in scheduler.bandwidth_allocation.items():
                        src_name = self.virtual_node_names.get(v_from, f"node_{v_from}")
                        dst_name = self.virtual_node_names.get(v_to, f"node_{v_to}")
                        bandwidth_allocation[(src_name, dst_name)] = bw
                        print(f"   带宽分配: {src_name} -> {dst_name}: {bw} mbps")
                
                print(f"✅ 从PPO环境提取到 {len(bandwidth_allocation)} 条带宽分配")
                
        except Exception as e:
            print(f"❌ 从PPO环境提取带宽分配失败: {e}")
        
        return bandwidth_allocation

