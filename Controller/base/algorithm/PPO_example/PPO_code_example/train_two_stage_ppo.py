#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from plot_utils import create_comprehensive_training_plot

# TensorBoard相关导入
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  TensorBoard未安装，将跳过TensorBoard功能")
    TENSORBOARD_AVAILABLE = False

from two_stage_actor_design import TwoStagePPOAgent
from two_stage_environment import TwoStageNetworkSchedulerEnvironment

class TwoStagePPOTrainer:
    """两阶段PPO训练器"""
    
    def __init__(self, 
                 max_physical_nodes: int = 10,
                 num_physical_nodes_range: Tuple[int, int] = (5, 10),
                 virtual_nodes_range: Tuple[int, int] = (3, 8),
                 bandwidth_levels: int = 10,
                 # 物理节点资源范围
                 physical_cpu_range: Tuple[int, int] = (50, 200),
                 physical_memory_range: Tuple[int, int] = (100, 400),
                 physical_bandwidth_range: Tuple[int, int] = (100, 1000),
                 # 虚拟节点资源范围
                 virtual_cpu_range: Tuple[int, int] = (10, 50),
                 virtual_memory_range: Tuple[int, int] = (20, 100),
                 virtual_bandwidth_range: Tuple[int, int] = (10, 200),
                 # 网络连接概率
                 physical_connectivity_prob: float = 0.3,
                 virtual_connectivity_prob: float = 0.4,
                 # 训练参数
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 # PPO训练参数
                 n_epochs: int = 4,  # 每次更新的训练轮数
                 # 经验回放参数
                 replay_buffer_type: str = "simple",  # "simple", "prioritized", "multistep"
                 replay_buffer_size: int = 2000,
                 batch_size: int = 64,
                 update_frequency: int = 1,
                 priority_alpha: float = 0.6,
                 priority_beta: float = 0.4,
                 n_steps: int = 256,
                 # 温度调度参数
                 initial_temperature: float = 2.0, # 初始温度，增加探索
                 final_temperature: float = 0.5,   # 最终温度，减少探索
                 temperature_decay: float = 0.9986, # 温度衰减率
                 # 随机种子参数
                 seed: int = None,  # 新增：随机种子
                 # TensorBoard参数
                 use_tensorboard: bool = True,  # 是否使用TensorBoard
                 tensorboard_log_dir: str = "runs",  # TensorBoard日志目录
                 # 文件管理
                 model_dir: str = "models",
                 stats_dir: str = "stats",
                 session_name: str = None):
        
        # 新增：设置随机种子
        if seed is not None:
            self._set_random_seed(seed)
            print(f"🌱 设置随机种子: {seed}")
        else:
            # 如果没有指定seed，生成一个随机种子
            import random
            import time
            seed = int(time.time() * 1000) % 1000000
            self._set_random_seed(seed)
            print(f"🌱 自动生成随机种子: {seed}")
        
        self.seed = seed
        
        self.max_physical_nodes = max_physical_nodes
        self.num_physical_nodes_range = num_physical_nodes_range
        self.virtual_nodes_range = virtual_nodes_range
        self.bandwidth_levels = bandwidth_levels
        
        # 资源范围
        self.physical_cpu_range = physical_cpu_range
        self.physical_memory_range = physical_memory_range
        self.physical_bandwidth_range = physical_bandwidth_range
        self.virtual_cpu_range = virtual_cpu_range
        self.virtual_memory_range = virtual_memory_range
        self.virtual_bandwidth_range = virtual_bandwidth_range
        
        # 连接概率
        self.physical_connectivity_prob = physical_connectivity_prob
        self.virtual_connectivity_prob = virtual_connectivity_prob
        
        # 训练参数
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # PPO训练参数
        self.n_epochs = n_epochs
        
        # 经验回放参数
        self.replay_buffer_type = replay_buffer_type
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.n_steps = n_steps
        
        # 温度调度参数
        self.initial_temperature = initial_temperature # 初始温度，增加探索
        self.final_temperature = final_temperature    # 最终温度，减少探索
        self.temperature_decay = temperature_decay  # 温度衰减率
        self.current_temperature = self.initial_temperature
        
        # 文件管理 - 创建唯一的会话文件夹
        self.session_name = session_name or self._generate_session_name()
        self.model_dir = os.path.join(model_dir, self.session_name)
        self.stats_dir = os.path.join(stats_dir, self.session_name)

        # TensorBoard参数
        self.use_tensorboard = use_tensorboard
        self.tensorboard_log_dir = os.path.join(tensorboard_log_dir, self.session_name)

        # 创建目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        
        print(f"📁 创建训练会话: {self.session_name}")
        print(f"   模型目录: {self.model_dir}")
        print(f"   统计目录: {self.stats_dir}")
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'mapping_actor_losses': [],
            'bandwidth_actor_losses': [],
            'critic_losses': [],
            'constraint_violations': [],
            'resource_utilization': [],
            'load_balancing': [],
            'bandwidth_satisfaction': [],
            'mapping_actor_entropies': [],
            'bandwidth_actor_entropies': [],
            'total_entropies': [],
            'temperatures': [],
            'load_balance_rewards': [],
            'bandwidth_satisfaction_rewards': [],
            'total_rewards': []
        }
        
        # 初始化智能体和环境
        self._initialize_agent_and_env()
        
        # 初始化TensorBoard
        self._initialize_tensorboard()
        
        # 保存会话配置信息
        self._save_session_config()
    
    def _set_random_seed(self, seed: int):
        """
        设置所有相关的随机种子
        
        Args:
            seed: 随机种子值
        """
        import random
        import numpy as np
        import torch
        
        # 设置Python内置random模块的种子
        random.seed(seed)
        
        # 设置numpy的随机种子
        np.random.seed(seed)
        
        # 设置PyTorch的随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        
        # 设置PyTorch的确定性模式（可能影响性能，但确保结果可重现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 设置环境变量（如果使用某些特定的随机数生成器）
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print(f"🔒 随机种子设置完成: {seed}")
        print(f"   - Python random: {seed}")
        print(f"   - NumPy: {seed}")
        print(f"   - PyTorch CPU: {seed}")
        print(f"   - PyTorch CUDA: {seed}")
        print(f"   - CUDNN确定性模式: 启用")
    
    def _generate_session_name(self):
        """生成唯一的会话名称"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        import random
        session_id = random.randint(1000, 9999)
        return f"session_{timestamp}_{session_id}"
    
    def _initialize_agent_and_env(self):
        """初始化智能体和环境"""
        # 随机选择节点数量
        self.num_physical_nodes = np.random.randint(*self.num_physical_nodes_range)
        # max_virtual_nodes 定义为区间的最大值，用于网络初始化
        self.max_virtual_nodes = self.virtual_nodes_range[1]
        
        # 创建环境
        self.env = TwoStageNetworkSchedulerEnvironment(
            num_physical_nodes=self.num_physical_nodes,
            max_virtual_nodes=self.max_virtual_nodes,
            bandwidth_levels=self.bandwidth_levels,
            physical_cpu_range=self.physical_cpu_range,
            physical_memory_range=self.physical_memory_range,
            physical_bandwidth_range=self.physical_bandwidth_range,
            virtual_cpu_range=self.virtual_cpu_range,
            virtual_memory_range=self.virtual_memory_range,
            virtual_bandwidth_range=self.virtual_bandwidth_range,
            physical_connectivity_prob=self.physical_connectivity_prob,
            virtual_connectivity_prob=self.virtual_connectivity_prob,
            virtual_nodes_range=self.virtual_nodes_range,  # 传递虚拟节点数范围
            seed=self.seed  # 新增：传递随机种子
        )
        
        # 获取状态维度
        state = self.env.reset()
        physical_node_dim = state['physical_features'].size(1)
        virtual_node_dim = state['virtual_features'].size(1)
        
        # 创建智能体
        self.agent = TwoStagePPOAgent(
            physical_node_dim=physical_node_dim,
            virtual_node_dim=virtual_node_dim,
            max_physical_nodes=self.max_physical_nodes,
            max_virtual_nodes=self.max_virtual_nodes,
            bandwidth_levels=self.bandwidth_levels,
            lr=self.lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_ratio=self.clip_ratio,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef,
            replay_buffer_type=self.replay_buffer_type,
            replay_buffer_size=self.replay_buffer_size,
            batch_size=self.batch_size,
            update_frequency=self.update_frequency,
            priority_alpha=self.priority_alpha,
            priority_beta=self.priority_beta,
            n_steps=self.n_steps,
            n_epochs=self.n_epochs,  # 添加n_epochs参数
            max_grad_norm=2.0,           # 增加整体梯度裁剪阈值 1.0
            task_gradient_clip=1.0,      # 增加任务梯度裁剪阈值 0.6
            head_gradient_clip=0.8,      # 增加注意力头梯度裁剪阈值 0.4
        )
        
        print(f"✅ 智能体和环境初始化完成")
        print(f"   物理节点数: {self.num_physical_nodes}")
        print(f"   虚拟节点数范围: {self.virtual_nodes_range}")
        print(f"   最大虚拟节点数(网络维度): {self.max_virtual_nodes}")
        print(f"   物理节点特征维度: {physical_node_dim}")
        print(f"   虚拟节点特征维度: {virtual_node_dim}")
        print(f"   每次更新训练轮数: {self.n_epochs}")
        print(f"   经验回放缓冲区类型: {self.replay_buffer_type}")
        print(f"   缓冲区大小: {self.replay_buffer_size}")
        print(f"   批量大小: {self.batch_size}")
    
    def train_episode(self):
        """训练一个episode"""
        # 重置环境
        print(f"重置环境")
        state = self.env.reset()
        
        # 获取动作
        print(f"获取动作")
        mapping_action, bandwidth_action, \
             mapping_log_prob, bandwidth_log_prob, value, \
                mapping_entropy, bandwidth_entropy, \
                    link_indices = self.agent.select_actions(state, temperature=self.current_temperature)
        
        # 执行动作
        print(f"执行动作")
        next_state, reward, done, info = self.env.step(mapping_action, bandwidth_action)
        
        # 计算单步TD误差用于优先级（PER）
        with torch.no_grad():
            if done:
                next_value = 0.0
            else:
                # 将next_state张量移动到agent设备
                device = self.agent.device
                pv = next_state['physical_features'].to(device)
                pei = next_state['physical_edges'].to(device)
                pea = next_state['physical_edge_features'].to(device)
                vv = next_state['virtual_features'].to(device)
                vei = next_state['virtual_edges'].to(device)
                vea = next_state['virtual_edge_features'].to(device)
                pv_gcn = next_state.get('physical_gcn_features', None)
                vv_gcn = next_state.get('virtual_gcn_features', None)
                if pv_gcn is not None:
                    pv_gcn = pv_gcn.to(device)
                if vv_gcn is not None:
                    vv_gcn = vv_gcn.to(device)
                next_value_tensor = self.agent.critic(
                    pv, pei, pea,
                    vv, vei, vea,
                    physical_gcn_features=pv_gcn,
                    virtual_gcn_features=vv_gcn
                )
                next_value = float(next_value_tensor.item())
        td_error = reward + (0.0 if done else self.gamma * next_value) - float(value)

        # 获取奖励组件（从环境信息中提取）
        load_balance_reward = info.get('load_balance_reward', 0.0)
        bandwidth_satisfaction_reward = info.get('bandwidth_satisfaction_reward', 0.0)
        total_reward = info.get('total_reward', 0.0)

        # 存储经验
        print(f"存储经验")
        self.agent.store_transition(
            state, mapping_action, bandwidth_action, 
            reward, value, mapping_log_prob, bandwidth_log_prob, done,
            td_error=td_error
        )
        
        # 注意：网络更新现在在store_transition中自动触发
        # 当经验回放缓冲区中的转移数量达到batch_size时，会自动调用_update_from_replay_buffer()
        
        # 记录统计信息
        # 确保constraint_violations是可迭代的
        constraint_violations = info['constraint_violations']
        if hasattr(constraint_violations, 'numpy'):  # 如果是torch tensor
            constraint_violations = constraint_violations.numpy()
        elif isinstance(constraint_violations, np.ndarray):  # 如果已经是numpy数组
            constraint_violations = constraint_violations.tolist()
        
        episode_stats = {
            'reward': reward,
            'length': 1,  # 两阶段环境一步完成
            'is_valid': info['is_valid'],
            'constraint_violations': len(constraint_violations),
            'mapping_result': info['mapping_result'],
            'bandwidth_result': info['bandwidth_result'],
            'mapping_entropy': mapping_entropy,
            'bandwidth_entropy': bandwidth_entropy,
            'total_entropy': mapping_entropy + bandwidth_entropy,
            # 新增：记录奖励组件
            'load_balance_reward': load_balance_reward,
            'bandwidth_satisfaction_reward': bandwidth_satisfaction_reward,
            'total_reward': total_reward
        }
        
        return episode_stats
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100, eval_interval: int = 50):
        """训练主循环"""
        print(f"🚀 开始两阶段PPO训练")
        print(f"   总episodes: {num_episodes}")
        print(f"   保存间隔: {save_interval}")
        print(f"   评估间隔: {eval_interval}")
        print(f"   每次更新训练轮数: {self.n_epochs}")
        print(f"   经验回放缓冲区类型: {self.replay_buffer_type}")
        print(f"   缓冲区大小: {self.replay_buffer_size}")
        print(f"   批量大小: {self.batch_size}")
        print(f"   更新频率: {self.update_frequency}")
        print(f"   优先级指数: {self.priority_alpha}")
        print(f"   重要性采样指数: {self.priority_beta}")
        print(f"   n步返回: {self.n_steps}")
        if self.use_tensorboard and TENSORBOARD_AVAILABLE:
            print(f"   TensorBoard: 启用 (日志目录: {self.tensorboard_log_dir})")
            print(f"   启动TensorBoard: tensorboard --logdir={self.tensorboard_log_dir}")
        else:
            print(f"   TensorBoard: 禁用")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="训练进度"):
            # 训练一个episode
            episode_stats = self.train_episode()
            
            # 记录统计信息
            self.training_stats['episode_rewards'].append(episode_stats['reward'])
            self.training_stats['episode_lengths'].append(episode_stats['length'])
            self.training_stats['constraint_violations'].append(episode_stats['constraint_violations'])
            self.training_stats['mapping_actor_entropies'].append(episode_stats['mapping_entropy'])
            self.training_stats['bandwidth_actor_entropies'].append(episode_stats['bandwidth_entropy'])
            self.training_stats['total_entropies'].append(episode_stats['total_entropy'])
            self.training_stats['temperatures'].append(self.current_temperature)
            self.training_stats['load_balance_rewards'].append(episode_stats['load_balance_reward'])
            self.training_stats['bandwidth_satisfaction_rewards'].append(episode_stats['bandwidth_satisfaction_reward'])
            self.training_stats['total_rewards'].append(episode_stats['total_reward'])
            
            # 记录到TensorBoard
            self._log_to_tensorboard(episode + 1, episode_stats)
            
            # 更新温度
            self.current_temperature = max(
                self.final_temperature,
                self.current_temperature * self.temperature_decay
            )
            
            # 定期评估
            if (episode + 1) % eval_interval == 0:
                self._evaluate_and_log(episode + 1)
            
            # 定期保存
            if (episode + 1) % save_interval == 0:
                self.save_model(episode + 1)
                self.save_training_stats(episode + 1)
        
        # 最终保存
        self.save_model(num_episodes)
        self.save_training_stats(num_episodes)
        
        # 关闭TensorBoard
        self._close_tensorboard()
        
        training_time = time.time() - start_time
        print(f"\n🎯 训练完成！")
        print(f"   总训练时间: {training_time:.2f}秒")
        print(f"   平均每episode时间: {training_time/num_episodes:.3f}秒")
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        # 分析策略熵
        self.analyze_entropy_stats()
    
    def _evaluate_and_log(self, episode):
        """评估并记录日志"""
        # 计算最近episodes的平均奖励
        recent_rewards = self.training_stats['episode_rewards'][-50:]
        avg_reward = np.mean(recent_rewards)
        
        # 计算约束违反率
        recent_violations = self.training_stats['constraint_violations'][-50:]
        violation_rate = np.mean([1 if v > 0 else 0 for v in recent_violations])

        # 新增：计算奖励组件的平均值
        recent_load_balance = self.training_stats['load_balance_rewards'][-50:]
        recent_bandwidth_satisfaction = self.training_stats['bandwidth_satisfaction_rewards'][-50:]
        recent_total_reward = self.training_stats['total_rewards'][-50:]
        avg_load_balance = np.mean(recent_load_balance) if recent_load_balance else 0.0
        avg_bandwidth_satisfaction = np.mean(recent_bandwidth_satisfaction) if recent_bandwidth_satisfaction else 0.0
        avg_total_reward = np.mean(recent_total_reward) if recent_total_reward else 0.0
        
        # 获取策略熵摘要
        entropy_summary = self.get_entropy_summary()
        
        if entropy_summary:
            mapping_entropy_info = entropy_summary['mapping_entropy']
            bandwidth_entropy_info = entropy_summary['bandwidth_entropy']
            total_entropy_info = entropy_summary['total_entropy']
            
            # 添加趋势指示器
            trend_symbols = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}
            total_trend = trend_symbols.get(total_entropy_info['trend'], '➡️')
            
            print(f"Episode {episode:4d} | 平均奖励: {avg_reward:6.3f} | 约束违反率: {violation_rate:.2%} | 映射熵: {mapping_entropy_info['avg']:.3f} | 带宽熵: {bandwidth_entropy_info['avg']:.3f} | 总熵: {total_entropy_info['avg']:.3f} {total_trend} | 温度: {self.current_temperature:.3f} | 负载均衡奖励: {avg_load_balance:6.3f} | 带宽满意度奖励: {avg_bandwidth_satisfaction:6.3f} | 总奖励: {avg_total_reward:6.3f} | 训练轮数: {self.n_epochs}")
        else:
            print(f"Episode {episode:4d} | 平均奖励: {avg_reward:6.3f} | 约束违反率: {violation_rate:.2%} | 训练轮数: {self.n_epochs}")
    
    def save_model(self, episode):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f"ppo_model_{self.session_name}_episode_{episode}.pth")
        
        torch.save({
            'episode': episode,
            'mapping_actor_state_dict': self.agent.mapping_actor.state_dict(),
            'bandwidth_actor_state_dict': self.agent.bandwidth_actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'mapping_optimizer_state_dict': self.agent.mapping_optimizer.state_dict(),
            'bandwidth_optimizer_state_dict': self.agent.bandwidth_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'env_config': {
                'max_physical_nodes': self.max_physical_nodes,
                'max_virtual_nodes': self.max_virtual_nodes,
                'bandwidth_levels': self.bandwidth_levels,
                'physical_cpu_range': self.physical_cpu_range,
                'physical_memory_range': self.physical_memory_range,
                'physical_bandwidth_range': self.physical_bandwidth_range,
                'virtual_cpu_range': self.virtual_cpu_range,
                'virtual_memory_range': self.virtual_memory_range,
                'virtual_bandwidth_range': self.virtual_bandwidth_range,
                'physical_connectivity_prob': self.physical_connectivity_prob,
                'virtual_connectivity_prob': self.virtual_connectivity_prob
            }
        }, model_path)
        
        print(f"💾 模型已保存: {model_path}")
    
    def load_model(self, episode, session_name=None):
        """加载模型"""
        if session_name is None:
            session_name = self.session_name
        model_path = os.path.join(self.model_dir, f"ppo_model_{session_name}_episode_{episode}.pth")
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location=self.agent.device)
        
        self.agent.mapping_actor.load_state_dict(checkpoint['mapping_actor_state_dict'])
        self.agent.bandwidth_actor.load_state_dict(checkpoint['bandwidth_actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.mapping_optimizer.load_state_dict(checkpoint['mapping_optimizer_state_dict'])
        self.agent.bandwidth_optimizer.load_state_dict(checkpoint['bandwidth_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_stats = checkpoint['training_stats']
        
        print(f"📂 模型已加载: {model_path}")
        return True
    
    def save_training_stats(self, episode):
        """保存训练统计"""
        stats_path = os.path.join(self.stats_dir, f"training_stats_{self.session_name}_episode_{episode}.json")
        
        # 转换为可序列化的格式
        serializable_stats = {}
        for key, value in self.training_stats.items():
            if isinstance(value, list):
                serializable_stats[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
            else:
                serializable_stats[key] = value
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 训练统计已保存: {stats_path}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        try:
            # 使用新的绘图工具
            plot_path = os.path.join(self.stats_dir, f'training_curves_{self.session_name}.png')
            create_comprehensive_training_plot(self.training_stats, plot_path)
            
            # 新增：创建专门的奖励组件分析图表
            from plot_utils import create_reward_components_plot
            create_reward_components_plot(self.training_stats, plot_path)
            
        except Exception as e:
            print(f"⚠️ 绘制训练曲线时出错: {e}")

    def analyze_entropy_stats(self):
        """分析策略熵统计信息"""
        if not self.training_stats['mapping_actor_entropies']:
            print("❌ 没有可用的策略熵数据")
            return
        
        print(f"\n📊 策略熵分析报告")
        print("=" * 60)
        
        # 计算基本统计信息
        mapping_entropies = np.array(self.training_stats['mapping_actor_entropies'])
        bandwidth_entropies = np.array(self.training_stats['bandwidth_actor_entropies'])
        total_entropies = np.array(self.training_stats['total_entropies'])
        
        print(f"映射Actor熵统计:")
        print(f"  平均值: {np.mean(mapping_entropies):.4f}")
        print(f"  标准差: {np.std(mapping_entropies):.4f}")
        print(f"  最小值: {np.min(mapping_entropies):.4f}")
        print(f"  最大值: {np.max(mapping_entropies):.4f}")
        print(f"  中位数: {np.median(mapping_entropies):.4f}")
        
        print(f"\n带宽Actor熵统计:")
        print(f"  平均值: {np.mean(bandwidth_entropies):.4f}")
        print(f"  标准差: {np.std(bandwidth_entropies):.4f}")
        print(f"  最小值: {np.min(bandwidth_entropies):.4f}")
        print(f"  最大值: {np.max(bandwidth_entropies):.4f}")
        print(f"  中位数: {np.median(bandwidth_entropies):.4f}")
        
        print(f"\n总策略熵统计:")
        print(f"  平均值: {np.mean(total_entropies):.4f}")
        print(f"  标准差: {np.std(total_entropies):.4f}")
        print(f"  最小值: {np.min(total_entropies):.4f}")
        print(f"  最大值: {np.max(total_entropies):.4f}")
        print(f"  中位数: {np.median(total_entropies):.4f}")
        
        # 分析熵的变化趋势
        if len(total_entropies) > 100:
            early_entropy = np.mean(total_entropies[:50])
            late_entropy = np.mean(total_entropies[-50:])
            entropy_change = late_entropy - early_entropy
            
            print(f"\n熵变化趋势分析:")
            print(f"  早期平均熵 (前50个episode): {early_entropy:.4f}")
            print(f"  后期平均熵 (后50个episode): {late_entropy:.4f}")
            print(f"  熵变化: {entropy_change:.4f}")
            
            if entropy_change < -0.1:
                print(f"  📉 策略熵显著下降，可能表明策略正在收敛")
            elif entropy_change > 0.1:
                print(f"  📈 策略熵显著上升，可能表明策略正在探索")
            else:
                print(f"  ➡️ 策略熵相对稳定")
        
        # 保存熵分析报告
        entropy_report = {
            'mapping_actor_entropy': {
                'mean': float(np.mean(mapping_entropies)),
                'std': float(np.std(mapping_entropies)),
                'min': float(np.min(mapping_entropies)),
                'max': float(np.max(mapping_entropies)),
                'median': float(np.median(mapping_entropies))
            },
            'bandwidth_actor_entropy': {
                'mean': float(np.mean(bandwidth_entropies)),
                'std': float(np.std(bandwidth_entropies)),
                'min': float(np.min(bandwidth_entropies)),
                'max': float(np.max(bandwidth_entropies)),
                'median': float(np.median(bandwidth_entropies))
            },
            'total_entropy': {
                'mean': float(np.mean(total_entropies)),
                'std': float(np.std(total_entropies)),
                'min': float(np.min(total_entropies)),
                'max': float(np.max(total_entropies)),
                'median': float(np.median(total_entropies))
            }
        }
        
        if len(total_entropies) > 100:
            entropy_report['trend_analysis'] = {
                'early_entropy': float(early_entropy),
                'late_entropy': float(late_entropy),
                'entropy_change': float(entropy_change)
            }
        
        # 保存到文件
        entropy_report_path = os.path.join(self.stats_dir, f'entropy_analysis_{self.session_name}.json')
        with open(entropy_report_path, 'w', encoding='utf-8') as f:
            json.dump(entropy_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 熵分析报告已保存: {entropy_report_path}")

    def get_entropy_summary(self, window_size=50):
        """获取策略熵的实时摘要"""
        if not self.training_stats['mapping_actor_entropies']:
            return None
        
        recent_mapping_entropies = self.training_stats['mapping_actor_entropies'][-window_size:]
        recent_bandwidth_entropies = self.training_stats['bandwidth_actor_entropies'][-window_size:]
        recent_total_entropies = self.training_stats['total_entropies'][-window_size:]
        
        summary = {
            'mapping_entropy': {
                'current': recent_mapping_entropies[-1] if recent_mapping_entropies else 0.0,
                'avg': np.mean(recent_mapping_entropies),
                'trend': 'stable'
            },
            'bandwidth_entropy': {
                'current': recent_bandwidth_entropies[-1] if recent_bandwidth_entropies else 0.0,
                'avg': np.mean(recent_bandwidth_entropies),
                'trend': 'stable'
            },
            'total_entropy': {
                'current': recent_total_entropies[-1] if recent_total_entropies else 0.0,
                'avg': np.mean(recent_total_entropies),
                'trend': 'stable'
            }
        }
        
        # 计算趋势（如果数据足够）
        if len(recent_total_entropies) >= 10:
            early_half = np.mean(recent_total_entropies[:len(recent_total_entropies)//2])
            late_half = np.mean(recent_total_entropies[len(recent_total_entropies)//2:])
            
            if late_half - early_half > 0.05:
                summary['total_entropy']['trend'] = 'increasing'
            elif early_half - late_half > 0.05:
                summary['total_entropy']['trend'] = 'decreasing'
        
        return summary

    def _initialize_tensorboard(self):
        """初始化TensorBoard"""
        if not self.use_tensorboard or not TENSORBOARD_AVAILABLE:
            print("TensorBoard未启用或未安装，跳过TensorBoard初始化。")
            return
        
        # 确保日志目录存在
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        
        # 创建SummaryWriter
        self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        print(f"✅ TensorBoard已初始化，日志目录: {self.tensorboard_log_dir}")
    
    def _log_to_tensorboard(self, episode, episode_stats):
        """记录数据到TensorBoard"""
        if not self.use_tensorboard or not TENSORBOARD_AVAILABLE or not hasattr(self, 'writer'):
            return
        
        try:
            # 记录奖励相关指标
            self.writer.add_scalar('Training/Reward', episode_stats['reward'], episode)
            self.writer.add_scalar('Training/Episode_Length', episode_stats['length'], episode)
            
            # 记录约束违反情况
            self.writer.add_scalar('Training/Constraint_Violations', episode_stats['constraint_violations'], episode)
            self.writer.add_scalar('Training/Is_Valid', int(episode_stats['is_valid']), episode)
            
            # 记录熵值
            self.writer.add_scalar('Entropy/Mapping_Entropy', episode_stats['mapping_entropy'], episode)
            self.writer.add_scalar('Entropy/Bandwidth_Entropy', episode_stats['bandwidth_entropy'], episode)
            self.writer.add_scalar('Entropy/Total_Entropy', episode_stats['total_entropy'], episode)
            
            # 记录温度
            self.writer.add_scalar('Training/Temperature', self.current_temperature, episode)

            # 记录奖励组件
            self.writer.add_scalar('Training/Load_Balance_Reward', episode_stats['load_balance_reward'], episode)
            self.writer.add_scalar('Training/Bandwidth_Satisfaction_Reward', episode_stats['bandwidth_satisfaction_reward'], episode)
            self.writer.add_scalar('Training/Total_Reward', episode_stats['total_reward'], episode)
            
            # 记录映射和带宽分配结果
            if 'mapping_result' in episode_stats:
                # 确保mapping_result是numpy数组或列表
                mapping_result = episode_stats['mapping_result']
                if hasattr(mapping_result, 'numpy'):  # 如果是torch tensor
                    mapping_result = mapping_result.numpy()
                elif isinstance(mapping_result, np.ndarray):  # 如果已经是numpy数组
                    mapping_result = mapping_result.tolist()
                
                mapping_success = sum(1 for x in mapping_result if x >= 0)
                mapping_success_rate = mapping_success / len(mapping_result) if mapping_result else 0
                self.writer.add_scalar('Results/Mapping_Success_Rate', mapping_success_rate, episode)
            
            if 'bandwidth_result' in episode_stats:
                # 确保bandwidth_result是numpy数组或列表
                bandwidth_result = episode_stats['bandwidth_result']
                if hasattr(bandwidth_result, 'numpy'):  # 如果是torch tensor
                    bandwidth_result = bandwidth_result.numpy()
                elif isinstance(bandwidth_result, np.ndarray):  # 如果已经是numpy数组
                    bandwidth_result = bandwidth_result.tolist()
                
                bandwidth_satisfaction = sum(1 for x in bandwidth_result if x > 0)
                bandwidth_satisfaction_rate = bandwidth_satisfaction / len(bandwidth_result) if bandwidth_result else 0
                self.writer.add_scalar('Results/Bandwidth_Satisfaction_Rate', bandwidth_satisfaction_rate, episode)
            
            # 记录最近50个episode的平均奖励
            if len(self.training_stats['episode_rewards']) >= 50:
                recent_rewards = self.training_stats['episode_rewards'][-50:]
                avg_reward = np.mean(recent_rewards)
                self.writer.add_scalar('Training/Average_Reward_50', avg_reward, episode)
            
            # 记录最近50个episode的约束违反率
            if len(self.training_stats['constraint_violations']) >= 50:
                recent_violations = self.training_stats['constraint_violations'][-50:]
                violation_rate = np.mean([1 if v > 0 else 0 for v in recent_violations])
                self.writer.add_scalar('Training/Constraint_Violation_Rate_50', violation_rate, episode)
            
            # 记录学习率（如果可用）
            if hasattr(self.agent, 'mapping_optimizer'):
                for param_group in self.agent.mapping_optimizer.param_groups:
                    self.writer.add_scalar('Training/Learning_Rate', param_group['lr'], episode)
                    break
            
            # 记录网络参数统计（每100个episode记录一次）
            if episode % 100 == 0:
                self._log_network_parameters(episode)
            
            # 强制刷新
            self.writer.flush()
            
        except Exception as e:
            print(f"⚠️  TensorBoard记录失败: {e}")
    
    def _log_network_parameters(self, episode):
        """记录网络参数统计到TensorBoard"""
        if not self.use_tensorboard or not TENSORBOARD_AVAILABLE or not hasattr(self, 'writer'):
            return
        
        try:
            # 记录映射Actor参数统计
            for name, param in self.agent.mapping_actor.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Parameters/MappingActor/{name}', param.data, episode)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/MappingActor/{name}', param.grad, episode)
            
            # 记录带宽Actor参数统计
            for name, param in self.agent.bandwidth_actor.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Parameters/BandwidthActor/{name}', param.data, episode)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/BandwidthActor/{name}', param.grad, episode)
            
            # 记录Critic参数统计
            for name, param in self.agent.critic.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Parameters/Critic/{name}', param.data, episode)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/Critic/{name}', param.grad, episode)
            
        except Exception as e:
            print(f"⚠️  网络参数记录失败: {e}")
    
    def _close_tensorboard(self):
        """关闭TensorBoard"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            print("✅ TensorBoard已关闭")
    
    def _save_session_config(self):
        """保存会话配置信息"""
        config_path = os.path.join(self.stats_dir, f'session_config_{self.session_name}.json')
        
        config_data = {
            'timestamp': self.session_name.split('_')[-2], # 从会话名称提取时间戳
            'seed': self.seed,  # 新增：保存随机种子
            'num_physical_nodes_range': self.num_physical_nodes_range,
            'virtual_nodes_range': self.virtual_nodes_range,  # 虚拟节点数范围
            'bandwidth_levels': self.bandwidth_levels,
            'physical_cpu_range': self.physical_cpu_range,
            'physical_memory_range': self.physical_memory_range,
            'physical_bandwidth_range': self.physical_bandwidth_range,
            'virtual_cpu_range': self.virtual_cpu_range,
            'virtual_memory_range': self.virtual_memory_range,
            'virtual_bandwidth_range': self.virtual_bandwidth_range,
            'physical_connectivity_prob': self.physical_connectivity_prob,
            'virtual_connectivity_prob': self.virtual_connectivity_prob,
            'lr': self.lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'n_epochs': self.n_epochs,
            'replay_buffer_type': self.replay_buffer_type,
            'replay_buffer_size': self.replay_buffer_size,
            'batch_size': self.batch_size,
            'update_frequency': self.update_frequency,
            'priority_alpha': self.priority_alpha,
            'priority_beta': self.priority_beta,
            'n_steps': self.n_steps,
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'temperature_decay': self.temperature_decay,
            'use_tensorboard': self.use_tensorboard,
            'tensorboard_log_dir': self.tensorboard_log_dir
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 会话配置已保存: {config_path}")
    
    @staticmethod
    def list_sessions(base_model_dir="models", base_stats_dir="stats"):
        """列出所有可用的训练会话"""
        sessions = []
        
        if os.path.exists(base_model_dir):
            for session_name in os.listdir(base_model_dir):
                session_path = os.path.join(base_model_dir, session_name)
                if os.path.isdir(session_path):
                    # 检查是否有配置文件
                    config_path = os.path.join(base_stats_dir, session_name, f'session_config_{session_name}.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            sessions.append({
                                'name': session_name,
                                'timestamp': config.get('timestamp', 'Unknown'),
                                'config': config
                            })
                        except:
                            sessions.append({
                                'name': session_name,
                                'timestamp': 'Unknown',
                                'config': {}
                            })
        
        return sessions
    
    @staticmethod
    def print_sessions(base_model_dir="models", base_stats_dir="stats"):
        """打印所有可用的训练会话"""
        sessions = TwoStagePPOTrainer.list_sessions(base_model_dir, base_stats_dir)
        
        if not sessions:
            print("📁 没有找到任何训练会话")
            return
        
        print(f"📁 找到 {len(sessions)} 个训练会话:")
        print("=" * 80)
        
        for i, session in enumerate(sessions, 1):
            print(f"{i:2d}. {session['name']}")
            print(f"    时间: {session['timestamp']}")
            if session['config']:
                config = session['config']
                print(f"    随机种子: {config.get('seed', 'N/A')}")
                print(f"    物理节点范围: {config.get('num_physical_nodes_range', 'N/A')}")
                print(f"    虚拟节点范围: {config.get('virtual_nodes_range', 'N/A')}")
                print(f"    学习率: {config.get('lr', 'N/A')}")
            print()

def main():
    """主函数"""
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # 列出所有会话
        TwoStagePPOTrainer.print_sessions()
        return
    
    print("🎯 两阶段PPO网络调度器训练")
    print("=" * 60)
    
    # 设置随机种子（可选）
    seed = 42  # 你可以修改这个值来获得不同的随机结果

    # 创建训练器
    trainer = TwoStagePPOTrainer(
        max_physical_nodes = 10,
        num_physical_nodes_range=(5, 8),
        virtual_nodes_range=(5, 8),
        bandwidth_levels=10,
        physical_cpu_range=(50, 100),
        physical_memory_range=(50, 100),
        physical_bandwidth_range=(50, 100),
        virtual_cpu_range=(8, 10),
        virtual_memory_range=(8, 10),
        virtual_bandwidth_range=(5, 15),
        physical_connectivity_prob=0.9,
        virtual_connectivity_prob=0.7,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        n_epochs=8,  # 每次更新的训练轮数
        replay_buffer_type="simple",  # 经验回放缓冲区类型
        replay_buffer_size=256,  # 缓冲区大小
        batch_size=64,  # 批量大小
        n_steps=256,  # n步返回
        update_frequency=1,  # 更新频率
        priority_alpha=0.6,  # 优先级指数
        priority_beta=0.4,  # 重要性采样指数
        initial_temperature=2.0,  # 初始温度
        final_temperature=0.5,  # 最终温度
        temperature_decay=0.9986,  # 温度衰减率
        seed=seed,  # 新增：设置随机种子
        use_tensorboard=True,  # 启用TensorBoard
        tensorboard_log_dir="runs"  # TensorBoard日志目录
    )
    
    # 开始训练
    trainer.train(num_episodes=1000, save_interval=100, eval_interval=50)
    
    print(f"\n🎉 训练和测试完成！")
    print(f"📁 训练结果保存在: {trainer.stats_dir}")
    print(f"💾 模型保存在: {trainer.model_dir}")
    print(f"🌱 使用的随机种子: {seed}")
    print(f"📋 使用 'python train_two_stage_ppo.py --list' 查看所有训练会话")

if __name__ == "__main__":
    main() 