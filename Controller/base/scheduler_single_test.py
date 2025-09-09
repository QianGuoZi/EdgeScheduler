import os
import random  # 添加这行
from typing import Dict, Tuple
from flask import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import numpy as np

from .algorithm.GA import NodeMappingGA
from .algorithm.Rand import NodeMappingRandom
# PPO相关导入
import sys
sys.path.append('./algorithm/PPO_my')
from sequential_agent import SimpleSequentialAgent
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler
from original_reward import OriginalRewardCalculator


dirName = '/home/qianguo/Edge-Scheduler/Controller'
class SingleTestScheduler(object):
    def __init__(self, controller):
        self.controller = controller
        # 创建记录文件夹
        self.log_dir = os.path.join(dirName, 'scheduling_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file = None
        self.node_count = 0
        
        # PPO相关初始化
        self.reward_calculator = OriginalRewardCalculator() 
        self.metrics_log_file = None  # 用于记录L和D_BW指标
        self._init_ppo_agent()
        
    def _init_ppo_agent(self):
        """初始化PPO智能体"""
        try:
            # 加载训练好的PPO模型
            model_path = os.path.join(dirName, 'base/algorithm/PPO_my/checkpoints/new_heuristic_20250827_164927/final.pt')
            if os.path.exists(model_path):
                self.ppo_agent, self.ppo_env_config, self.ppo_agent_config = self._load_ppo_agent_and_configs(model_path)
                print(f"✅ PPO模型加载成功: {model_path}")
            else:
                print(f"⚠️  PPO模型文件不存在: {model_path}")
                self.ppo_agent = None
                self.ppo_env_config = {}
                self.ppo_agent_config = {}
        except Exception as e:
            print(f"❌ PPO模型加载失败: {e}")
            self.ppo_agent = None
            self.ppo_env_config = {}
            self.ppo_agent_config = {}
    
    def _load_ppo_agent_and_configs(self, ckpt_path: str):
        """加载PPO Agent与配置"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔄 加载PPO检查点: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # 兼容不同的checkpoint格式
        if 'config' in checkpoint:
            # 新格式：包含完整配置信息
            env_config = checkpoint['config'].get('env_config', {})
            agent_config = checkpoint['config'].get('agent_config', {})
            
            # 创建Agent并载入权重
            agent = SimpleSequentialAgent(**agent_config)
            state = checkpoint.get('agent_state_dict', None)
            if state is None:
                raise KeyError("Checkpoint missing 'agent_state_dict'.")
            agent.load_state_dict(state)
        else:
            # 旧格式：使用默认配置
            print("⚠️  使用旧格式checkpoint，采用默认配置")
            env_config = {}
            agent_config = {
                'max_physical_nodes': 10,
                'max_virtual_nodes': 8,
                'bandwidth_levels': 10,
                'hidden_dim': 128,
                'lr': 3e-4
            }
            
            # 创建智能体
            agent = SimpleSequentialAgent(**agent_config)
            
            # 检查checkpoint的结构并加载权重
            if 'agent_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['agent_state_dict'])
            elif 'model_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['state_dict'])
            else:
                # 直接加载权重字典
                agent.load_state_dict(checkpoint)
        
        agent.eval()
        print("✅ PPO Agent与配置加载完成")
        return agent, env_config, agent_config
    
    def _select_action_greedy_with_fallback(self, env, agent: SimpleSequentialAgent, state: Dict) -> int:
        """PPO贪心策略选择动作，带回退机制"""
        agent.eval()
        with torch.no_grad():
            # 将状态移动到设备
            move_state = agent._move_state_to_device(state)
            logits, _, _ = agent.forward(move_state)
            logits = logits.detach().cpu().numpy()

        if state.get('mapping_phase', True):
            num_actions = state.get('num_physical_nodes', env.num_physical_nodes if hasattr(env, 'num_physical_nodes') else 10)
            candidates = np.argsort(-logits[:num_actions])  # 降序
            # 逐个尝试候选动作，选择第一个有效动作
            for a in candidates:
                if hasattr(env, '_validate_mapping_action'):
                    ok, _ = env._validate_mapping_action(state.get('current_virtual_node', 0), int(a))
                    if ok:
                        return int(a)
            # 若均无效，返回得分最高者（让环境给出惩罚）
            return int(candidates[0]) if len(candidates) > 0 else 0
        else:
            num_actions = getattr(env, 'bandwidth_levels', 4)
            candidates = np.argsort(-logits[:num_actions])
            # 带宽阶段也尝试从高到低选第一个有效
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
                # 尝试相对路径导入
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
    
    def _make_ppo_env_with_virtual_work(self, virtual_work: VirtualWork, seed: int = 42, env_type: str = "Sequential"):
        """
        基于虚拟工作和环境配置创建PPO环境
        
        Args:
            virtual_work: VirtualWork对象，包含虚拟节点和链路需求
            seed: 随机种子
            env_type: 环境类型 ("Sequential", "Lightweight", "NewHeuristic")
        
        Returns:
            相应类型的环境实例
        """
        try:
            from sequential_environment import SequentialNetworkSchedulerEnvironment
            from lightweight_heuristic_integration_environment import LightweightHeuristicEnvironment
            from new_heuristic_environment import NewHeuristicEnvironment
        except ImportError:
            # 如果直接导入失败，尝试相对路径
            try:
                sys.path.append(os.path.join(dirName, 'base/algorithm/PPO_my'))
                from sequential_environment import SequentialNetworkSchedulerEnvironment
                from lightweight_heuristic_integration_environment import LightweightHeuristicEnvironment
                from new_heuristic_environment import NewHeuristicEnvironment
            except ImportError as e:
                print(f"⚠️  无法导入PPO环境类: {e}")
                raise ImportError("PPO环境类导入失败，请检查路径配置")
        
        # 获取虚拟节点数量
        num_virtual_nodes = len(virtual_work.node_requirements)
        
        # 基础环境配置
        cfg = dict(self.ppo_env_config) if hasattr(self, 'ppo_env_config') else {}
        cfg.update({
            'seed': seed,
            'virtual_nodes_range': (num_virtual_nodes, num_virtual_nodes),
            'max_virtual_nodes': max(num_virtual_nodes, cfg.get('max_virtual_nodes', 8)),
            'curriculum_enabled': False,  # 禁用课程学习以确保稳定性
            'num_physical_nodes': len(self.controller.emulator),
            'bandwidth_levels': cfg.get('bandwidth_levels', 10)
        })
        
        # 根据环境类型创建相应的环境
        if env_type == "Sequential":
            env = SequentialNetworkSchedulerEnvironment(**cfg)
        elif env_type == "Lightweight":
            env = LightweightHeuristicEnvironment(**cfg)
        elif env_type == "NewHeuristic":
            env = NewHeuristicEnvironment(**cfg)
        else:
            raise ValueError(f"不支持的环境类型: {env_type}. 支持的类型: Sequential, Lightweight, NewHeuristic")
        
        return env
        
    def record_load(self, node_count: int, allocation: Dict):
        """记录当前负载情况"""
        if self.current_log_file is None:
            # 使用时间戳创建新的日志文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_log_file = os.path.join(self.log_dir, f'load_log_{timestamp}.csv')
            # 创建表头
            with open(self.current_log_file, 'w') as f:
                f.write('nodes,cpu_load,ram_load,bw_load\n')
                
        # 初始化指标记录文件
        if self.metrics_log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.metrics_log_file = os.path.join(self.log_dir, f'metrics_log_{timestamp}.csv')
            with open(self.metrics_log_file, 'w') as f:
                f.write('nodes,algorithm,L_load_balance,D_BW_satisfaction,mapping_success,total_reward\n')
        
        # 计算总负载
        total_cpu_load = 0
        total_ram_load = 0
        total_bw_load = 0
        total_cpu_capacity = 0
        total_ram_capacity = 0
        
        # 计算 CPU 和 RAM 负载
        for emulator in self.controller.emulator.values():
            total_cpu_load += emulator.cpuPreMap
            total_ram_load += emulator.ramPreMap
            total_cpu_capacity += emulator.cpu
            total_ram_capacity += emulator.ram
            
        # 计算带宽负载
        total_bw_capacity = 0
        total_bw_used = 0
        for emu1, emu2, bw, used_bw in self.controller.iter_bandwidth():
            total_bw_capacity += bw
            total_bw_used += used_bw
            
        # 计算负载率
        cpu_load_ratio = total_cpu_load / total_cpu_capacity if total_cpu_capacity > 0 else 0
        ram_load_ratio = total_ram_load / total_ram_capacity if total_ram_capacity > 0 else 0
        bw_load_ratio = total_bw_used / total_bw_capacity if total_bw_capacity > 0 else 0
        
        # 记录到文件
        with open(self.current_log_file, 'a') as f:
            f.write(f'{node_count},{cpu_load_ratio},{ram_load_ratio},{bw_load_ratio}\n')
    
    def record_metrics(self, node_count: int, algorithm: str, 
                      scheduler: NetworkScheduler = None, 
                      virtual_work: VirtualWork = None,
                      L: float = None, D_BW: float = None, 
                      mapping_success: bool = None, total_reward: float = None):
        """记录调度指标L和D_BW"""
        if self.metrics_log_file is None:
            return
            
        # 如果提供了scheduler和virtual_work，计算指标
        if scheduler and virtual_work and L is None and D_BW is None:
            try:
                reward_result = self.reward_calculator.calculate_reward(scheduler, virtual_work)
                L = reward_result.get('L', 0.0)
                D_BW = reward_result.get('D_BW', 0.0)
                mapping_success = reward_result.get('mapping_success', False)
                total_reward = reward_result.get('total_reward', 0.0)
            except Exception as e:
                print(f"⚠️  指标计算失败: {e}")
                L = 0.0
                D_BW = 0.0
                mapping_success = False
                total_reward = 0.0
        
        # 使用默认值填充缺失的指标
        L = L if L is not None else 0.0
        D_BW = D_BW if D_BW is not None else 0.0
        mapping_success = mapping_success if mapping_success is not None else False
        total_reward = total_reward if total_reward is not None else 0.0
        
        # 记录到文件
        with open(self.metrics_log_file, 'a') as f:
            f.write(f'{node_count},{algorithm},{L:.4f},{D_BW:.4f},{mapping_success},{total_reward:.4f}\n')
            
    def plot_load_history(self):
        """生成负载历史图表"""
        if not self.current_log_file or not os.path.exists(self.current_log_file):
            print("没有找到负载记录文件")
            return
            
        # 读取数据
        df = pd.read_csv(self.current_log_file)
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(df['nodes'], df['cpu_load'], 'r-', label='CPU Load')
        plt.plot(df['nodes'], df['ram_load'], 'b-', label='RAM Load')
        plt.plot(df['nodes'], df['bw_load'], 'g-', label='Bandwidth Load')
        
        plt.xlabel('Number of Nodes')
        plt.ylabel('Load Ratio')
        plt.title('Resource Load History')
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        plot_file = self.current_log_file.replace('.csv', '.png')
        plt.savefig(plot_file)
        plt.close()
        
        print(f"负载历史图表已保存到: {plot_file}")
    
    def plot_metrics_history(self):
        """生成指标历史图表（L和D_BW）"""
        if not self.metrics_log_file or not os.path.exists(self.metrics_log_file):
            print("没有找到指标记录文件")
            return
            
        try:
            # 读取数据
            df = pd.read_csv(self.metrics_log_file)
            
            if len(df) == 0:
                print("指标记录文件为空")
                return
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 按算法分组绘图
            algorithms = df['algorithm'].unique()
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, alg in enumerate(algorithms):
                alg_data = df[df['algorithm'] == alg]
                color = colors[i % len(colors)]
                
                # 负载均衡度 L
                ax1.plot(alg_data['nodes'], alg_data['L_load_balance'], 
                        color=color, marker='o', label=f'{alg} L', linewidth=2)
                
                # 带宽满足度 D_BW
                ax2.plot(alg_data['nodes'], alg_data['D_BW_satisfaction'], 
                        color=color, marker='s', label=f'{alg} D_BW', linewidth=2)
                
                # 成功率
                success_rate = alg_data['mapping_success'].rolling(window=5, min_periods=1).mean()
                ax3.plot(alg_data['nodes'], success_rate, 
                        color=color, marker='^', label=f'{alg} Success', linewidth=2)
                
                # 总奖励
                ax4.plot(alg_data['nodes'], alg_data['total_reward'], 
                        color=color, marker='d', label=f'{alg} Reward', linewidth=2)
            
            # 设置图表标题和标签
            ax1.set_title('负载均衡度 L 历史', fontsize=14, fontweight='bold')
            ax1.set_xlabel('节点数量')
            ax1.set_ylabel('L 值')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.set_title('带宽满足度 D_BW 历史', fontsize=14, fontweight='bold')
            ax2.set_xlabel('节点数量')
            ax2.set_ylabel('D_BW 值')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            ax3.set_title('映射成功率历史', fontsize=14, fontweight='bold')
            ax3.set_xlabel('节点数量')
            ax3.set_ylabel('成功率')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            ax4.set_title('总奖励历史', fontsize=14, fontweight='bold')
            ax4.set_xlabel('节点数量')
            ax4.set_ylabel('奖励值')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = self.metrics_log_file.replace('.csv', '_metrics.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 指标历史图表已保存到: {plot_file}")
            
        except Exception as e:
            print(f"❌ 生成指标图表失败: {e}")
    
    def generate_summary_report(self):
        """生成调度总结报告"""
        print("\n" + "="*60)
        print("               📊 调度性能总结报告")
        print("="*60)
        
        # 生成所有图表
        self.plot_load_history()
        self.plot_metrics_history()
        
        # 输出文件信息
        if self.current_log_file:
            print(f"📄 负载日志: {self.current_log_file}")
        if self.metrics_log_file:
            print(f"📄 指标日志: {self.metrics_log_file}")
        
        print(f"📊 总调度节点数: {self.node_count}")
        
        # 读取指标统计
        if self.metrics_log_file and os.path.exists(self.metrics_log_file):
            try:
                df = pd.read_csv(self.metrics_log_file)
                if len(df) > 0:
                    print("\n📈 指标统计:")
                    for alg in df['algorithm'].unique():
                        alg_data = df[df['algorithm'] == alg]
                        avg_L = alg_data['L_load_balance'].mean()
                        avg_D_BW = alg_data['D_BW_satisfaction'].mean()
                        success_rate = alg_data['mapping_success'].mean()
                        avg_reward = alg_data['total_reward'].mean()
                        
                        print(f"  {alg} 算法:")
                        print(f"    平均负载均衡度 L: {avg_L:.4f}")
                        print(f"    平均带宽满足度 D_BW: {avg_D_BW:.4f}")
                        print(f"    映射成功率: {success_rate:.2%}")
                        print(f"    平均奖励: {avg_reward:.4f}")
            except Exception as e:
                print(f"⚠️  指标统计读取失败: {e}")
        
        print("\n✅ 调度总结完成!")
        print("="*60)

    def resource_schedule(self, taskId: int) -> Dict:
        """
        需要访问Controller获取目前的资源，还有现有的需求，然后根据调度算法提供
        优先使用PPO算法，如果不可用则使用GA算法
        """
        # self.testbed.emulater
        with open(os.path.join(dirName, 'task_links', str(taskId),'links_range.json'), 'r') as file:
            links_data = json.load(file)
        
                # 尝试使用PPO算法
        if self.ppo_agent is not None:
            try:
                return self._schedule_with_ppo(taskId, links_data)
            except Exception as e:
                print(f"❌ PPO调度失败: {e}")

    
    def _schedule_with_ppo(self, taskId: int, links_data: Dict) -> Dict:
        """使用PPO算法进行调度"""
        print(f"🤖 使用PPO算法调度任务 {taskId}")
        
        # 存储当前任务ID，供后续方法使用
        self.current_task_id = taskId
        
        # 创建网络拓扑
        topology = self._create_network_topology()
        
        # 创建虚拟工作
        virtual_work = self._create_virtual_work(taskId, links_data)
        
        # 创建调度器
        scheduler = NetworkScheduler(topology)
        scheduler.add_virtual_work(virtual_work)
        
        # 使用PPO代理进行调度决策
        allocation = self._run_ppo_scheduling(scheduler, virtual_work)
        
        # 记录指标
        self.record_metrics(
            node_count=self.node_count + len(allocation),
            algorithm='PPO',
            scheduler=scheduler,
            virtual_work=virtual_work
        )
        
        return allocation
    
    def _create_network_topology(self) -> NetworkTopology:
        """根据当前的物理资源创建网络拓扑"""
        # 获取物理节点数量
        num_nodes = len(self.controller.emulator)
        topology = NetworkTopology(num_nodes)
        
        # 设置节点资源
        node_idx = 0
        self.node_id_mapping = {}  # 物理节点名字到索引的映射
        self.idx_node_mapping = {}  # 索引到物理节点名字的映射
        
        for emulator in self.controller.emulator.values():
            # 计算已使用的资源
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
        
        # 设置网络链路
        for emu1, emu2, bw, used_bw in self.controller.iter_bandwidth():
            if emu1 in self.node_id_mapping and emu2 in self.node_id_mapping:
                idx1 = self.node_id_mapping[emu1]
                idx2 = self.node_id_mapping[emu2]
                
                # 假设双向链路带宽相等
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
        virtual_nodes = list(links_data.keys())
        num_virtual_nodes = len(virtual_nodes)
        virtual_work = VirtualWork(num_virtual_nodes)
        
        # 创建虚拟节点索引映射
        self.virtual_node_mapping = {}  # 虚拟节点名字到索引
        self.idx_virtual_mapping = {}   # 索引到虚拟节点名字
        
        for idx, node in enumerate(virtual_nodes):
            node_name = str(taskId) + '_' + node
            self.virtual_node_mapping[node_name] = idx
            self.idx_virtual_mapping[idx] = node_name
            
            # 设置资源需求（使用随机或固定值）
            cpu_demand = 2  # 固定值，与原有逻辑保持一致
            ram_demand = 5
            virtual_work.set_node_requirement(idx, cpu_demand, ram_demand)
        
        # 设置虚拟链路需求
        for node, connections in links_data.items():
            src_idx = self.virtual_node_mapping[str(taskId) + '_' + node]
            
            for dest in connections:
                dest_node = str(taskId) + '_' + dest['dest']
                if dest_node in self.virtual_node_mapping:
                    dest_idx = self.virtual_node_mapping[dest_node]
                    
                    # 解析带宽范围 - 支持新的JSON格式
                    if 'bw_min' in dest and 'bw_max' in dest:
                        # 新格式：使用bw_min和bw_max
                        min_bw = int(dest['bw_min'].replace('mbps', ''))
                        max_bw = int(dest['bw_max'].replace('mbps', ''))
                    elif 'bw' in dest:
                        # 兼容旧格式：使用单一bw值
                        if 'bw_max' in dest:
                            # 如果有bw_max，bw作为最小值
                            min_bw = int(dest['bw'].replace('mbps', ''))
                            max_bw = int(dest['bw_max'].replace('mbps', ''))
                        else:
                            # 只有bw，最小最大相等
                            bw = int(dest['bw'].replace('mbps', ''))
                            min_bw = bw
                            max_bw = bw
                    else:
                        # 默认值
                        print(f"⚠️  链路 {node} -> {dest['dest']} 缺少带宽信息，使用默认值")
                        min_bw = 10
                        max_bw = 20
                    
                    # 设置带宽需求范围（支持不对称带宽）
                    virtual_work.add_link_requirement(
                        src_idx, dest_idx,
                        min_bandwidth_1_to_2=min_bw,
                        max_bandwidth_1_to_2=max_bw,
                        min_bandwidth_2_to_1=min_bw,  # 假设双向带宽相同
                        max_bandwidth_2_to_1=max_bw
                    )
        
        return virtual_work
    
    def _run_ppo_scheduling(self, scheduler: NetworkScheduler, virtual_work: VirtualWork) -> Dict:
        """使用PPO代理进行调度决策，采用环境-代理交互模式"""
        print("🤖 开始PPO环境-代理交互调度...")
        
        # 创建PPO环境
        try:
            env_type = "NewHeuristic"  # 默认使用NewHeuristic环境，可以根据需要修改
            env = self._make_ppo_env_with_virtual_work(virtual_work, seed=42, env_type=env_type)
            
            # 运行PPO Episode
            total_reward, success, episode_info = self._run_ppo_episode(
                env, self.ppo_agent, 
                temperature=0.05,  # 低温度，更贪心
                max_steps=50,
                greedy=True
            )
            
            print(f"📊 PPO调度结果: 奖励={total_reward:.3f}, 成功={success}, 步数={episode_info['steps']}")
            
            # 从环境中提取分配结果
            allocation = self._extract_allocation_from_env(env, episode_info)
            
            return allocation
            
        except Exception as e:
            print(f"❌ PPO环境调度失败，回退到启发式方法: {e}")
            # 回退到原有的启发式方法
            return self._run_heuristic_scheduling_fallback(scheduler, virtual_work)
    
    def _run_ppo_episode(self, env, agent: SimpleSequentialAgent, 
                        temperature: float = 0.1, max_steps: int = 50, 
                        greedy: bool = True) -> Tuple[float, bool, Dict]:
        """运行PPO算法的Episode"""
        state = env.reset()
        
        # 在reset后集成原始奖励计算器
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
        load_balance_degree = float('nan')
        bandwidth_satisfaction = float('nan')
        
        if (hasattr(env, 'network_scheduler') and env.network_scheduler is not None and 
            hasattr(env.network_scheduler, 'get_original_reward_components')):
            try:
                if hasattr(env, 'virtual_work_obj') and env.virtual_work_obj is not None:
                    components = env.network_scheduler.get_original_reward_components(env.virtual_work_obj)
                    load_balance_degree = components.get('L', float('nan'))
                    bandwidth_satisfaction = components.get('D_BW', float('nan'))
            except Exception as e:
                print(f"Warning: Failed to calculate L and D_BW for PPO: {e}")
        
        episode_info.update({
            'steps': steps,
            'success': success,
            'total_reward': total_reward,
            'load_balance_degree': load_balance_degree,
            'bandwidth_satisfaction': bandwidth_satisfaction,
            'env': env  # 保存环境引用以便后续提取分配结果
        })
        
        return total_reward, success, episode_info
    
    def _extract_allocation_from_env(self, env, episode_info: Dict) -> Dict:
        """从PPO环境中提取分配结果"""
        allocation = {}
        
        try:
            # 检查环境是否有调度结果
            if hasattr(env, 'network_scheduler') and env.network_scheduler is not None:
                scheduler = env.network_scheduler
                
                # 提取节点映射
                if hasattr(scheduler, 'node_mapping'):
                    for virtual_idx, physical_idx in scheduler.node_mapping.items():
                        if physical_idx != -1:  # 已分配
                            # 构建虚拟节点名称
                            virtual_node_name = f"{self.current_task_id}_{virtual_idx}"
                            physical_node_name = self.idx_node_mapping.get(physical_idx, f"node_{physical_idx}")
                            
                            # 获取资源需求
                            if hasattr(env, 'virtual_work_obj') and env.virtual_work_obj:
                                virtual_req = env.virtual_work_obj.node_requirements.get(virtual_idx, {})
                                cpu_req = virtual_req.get('cpu', 2)
                                ram_req = virtual_req.get('memory', 5)
                            else:
                                cpu_req = 2  # 默认值
                                ram_req = 5
                            
                            allocation[virtual_node_name] = {
                                'emulator': physical_node_name,
                                'cpu': cpu_req,
                                'ram': ram_req
                            }
                            self.node_count += 1
                
                print(f"✅ 从PPO环境提取到 {len(allocation)} 个节点分配")
                
            else:
                print("⚠️  PPO环境中没有找到调度器，无法提取分配结果")
                
        except Exception as e:
            print(f"❌ 从PPO环境提取分配结果失败: {e}")
        
        return allocation
    
    def _run_heuristic_scheduling_fallback(self, scheduler: NetworkScheduler, virtual_work: VirtualWork) -> Dict:
        """启发式调度回退方法"""
        print("🔄 使用启发式方法作为回退...")
        allocation = {}
        
        # 优先尝试映射所有虚拟节点
        virtual_nodes = list(virtual_work.node_requirements.keys())
        
        for v_idx in virtual_nodes:
            best_physical = self._find_best_physical_node(v_idx, virtual_work, scheduler)
            if best_physical is not None:
                # 执行映射
                if scheduler.schedule_node(v_idx, best_physical):
                    virtual_node_name = self.idx_virtual_mapping[v_idx]
                    physical_node_name = self.idx_node_mapping[best_physical]
                    
                    allocation[virtual_node_name] = {
                        'emulator': physical_node_name,
                        'cpu': virtual_work.node_requirements[v_idx]['cpu'],
                        'ram': virtual_work.node_requirements[v_idx]['memory']
                    }
                    self.node_count += 1
        
        # 分配虚拟链路带宽
        for link_req in virtual_work.link_requirements:
            from_idx = link_req['from']
            to_idx = link_req['to']
            min_bandwidth = link_req['min_bandwidth_1_to_2']
            max_bandwidth = link_req['max_bandwidth_1_to_2']
            
            if from_idx in scheduler.scheduled_nodes and to_idx in scheduler.scheduled_nodes:
                # 使用启发式策略选择带宽：尝试分配最大带宽，如果不够则使用最小带宽
                allocated_bandwidth = min_bandwidth
                
                # 检查是否可以分配更多带宽（简化的贪心策略）
                physical_from = scheduler.node_mapping[from_idx]
                physical_to = scheduler.node_mapping[to_idx]
                
                if physical_from != physical_to:
                    # 获取路径上的可用带宽
                    path = scheduler.topology.get_shortest_path(physical_from, physical_to)
                    if path and scheduler.topology.check_bandwidth_availability(path, max_bandwidth):
                        allocated_bandwidth = max_bandwidth
                    elif not scheduler.topology.check_bandwidth_availability(path, min_bandwidth):
                        print(f"⚠️  链路 {from_idx}->{to_idx} 带宽不足，跳过分配")
                        continue
                
                success = scheduler.allocate_bandwidth(from_idx, to_idx, allocated_bandwidth)
                if not success:
                    print(f"⚠️  链路 {from_idx}->{to_idx} 带宽分配失败")
        
        return allocation
    
    def _find_best_physical_node(self, virtual_idx: int, virtual_work: VirtualWork, scheduler: NetworkScheduler) -> int:
        """使用启发式方法找到最佳物理节点"""
        virtual_req = virtual_work.node_requirements[virtual_idx]
        best_node = None
        best_score = -1
        
        for p_idx in range(scheduler.topology.num_nodes):
            # 检查资源是否足够
            available = scheduler.topology.get_available_resources(p_idx)
            if available['cpu'] >= virtual_req['cpu'] and available['memory'] >= virtual_req['memory']:
                # 计算适合度分数（资源利用率 + 负载均衡）
                total_res = scheduler.topology.node_resources[p_idx]
                cpu_util = (total_res['cpu'] - available['cpu']) / total_res['cpu']
                mem_util = (total_res['memory'] - available['memory']) / total_res['memory']
                
                # 倾向于选择资源充裕但不过载的节点
                score = (available['cpu'] + available['memory']) * (1 - max(cpu_util, mem_util))
                
                if score > best_score:
                    best_score = score
                    best_node = p_idx
        
        return best_node