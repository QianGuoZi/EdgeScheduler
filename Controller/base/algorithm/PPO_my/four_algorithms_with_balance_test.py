#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# PPO_balance相关导入
import sys
sys.path.append('../PPO_mapping')
from balance_environment import PPOBalanceNetworkEnvironment
from balance_agent import BalanceAgent

# PPO和其他算法导入
sys.path.append('../PPO_my')
from sequential_environment import SequentialNetworkSchedulerEnvironment
from sequential_agent import SimpleSequentialAgent
from lightweight_heuristic_integration_environment import LightweightHeuristicEnvironment
from new_heuristic_environment import NewHeuristicEnvironment
from heuristic_algorithm import create_heuristic_agent, run_heuristic_episode
from original_reward import integrate_with_network_scheduler


def _set_global_seed(seed: int):
    """统一设置全局随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _list_checkpoint_directories(base_dir: str = "checkpoints") -> List[str]:
    """列出所有checkpoints文件夹"""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base checkpoints directory not found: {base_dir}")
    
    directories = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            directories.append(item_path)
    
    if not directories:
        raise FileNotFoundError(f"No subdirectories found in {base_dir}")
    
    # 按名称排序
    directories.sort()
    return directories


def _find_latest_checkpoint_in_dir(checkpoints_dir: str) -> str:
    """在指定目录中查找最新的 episode_*.pt 文件"""
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    # 查找所有匹配的episode文件
    episode_pattern = re.compile(r"^.*episode_(\d+)\.pt$")
    candidates = []
    
    for fname in os.listdir(checkpoints_dir):
        m = episode_pattern.match(fname)
        if m:
            candidates.append((int(m.group(1)), os.path.join(checkpoints_dir, fname)))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching '*episode_*.pt' found in {checkpoints_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_balance_checkpoint_directories() -> List[str]:
    """查找PPO_balance的结果目录"""
    balance_results_dir = "../PPO_mapping/balance_results"
    if not os.path.isdir(balance_results_dir):
        raise FileNotFoundError(f"Balance results directory not found: {balance_results_dir}")
    
    directories = []
    for item in os.listdir(balance_results_dir):
        item_path = os.path.join(balance_results_dir, item)
        if os.path.isdir(item_path) and item.startswith("balance_ppo_"):
            checkpoints_dir = os.path.join(item_path, "checkpoints")
            if os.path.isdir(checkpoints_dir):
                directories.append(item_path)
    
    if not directories:
        raise FileNotFoundError(f"No balance checkpoint directories found in {balance_results_dir}")
    
    directories.sort()
    return directories


def _find_latest_balance_checkpoint(balance_result_dir: str) -> str:
    """查找最新的balance模型文件"""
    checkpoints_dir = os.path.join(balance_result_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # 优先选择best_model.pth，其次是model_final.pth
    for model_file in ["best_model.pth", "model_final.pth"]:
        model_path = os.path.join(checkpoints_dir, model_file)
        if os.path.exists(model_path):
            return model_path
    
    # 如果没有，查找最新的model_ep_*.pth
    ep_pattern = re.compile(r"^model_ep_(\d+)\.pth$")
    candidates = []
    
    for fname in os.listdir(checkpoints_dir):
        m = ep_pattern.match(fname)
        if m:
            candidates.append((int(m.group(1)), os.path.join(checkpoints_dir, fname)))
    
    if not candidates:
        raise FileNotFoundError(f"No model files found in {checkpoints_dir}")
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _select_ppo_checkpoint() -> str:
    """交互式选择PPO checkpoint目录"""
    print("🔍 查找可用的PPO checkpoint目录...")
    
    try:
        ppo_checkpoints_dir = "checkpoints"
        directories = _list_checkpoint_directories(ppo_checkpoints_dir)
        
        print(f"\n📁 找到 {len(directories)} 个PPO checkpoint目录:")
        print("-" * 60)
        
        for i, dir_path in enumerate(directories, 1):
            dir_name = os.path.basename(dir_path)
            
            # 尝试获取目录中的文件信息
            try:
                files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
                file_count = len(files)
                
                # 查找最新的episode文件
                latest_file = None
                try:
                    latest_file = os.path.basename(_find_latest_checkpoint_in_dir(dir_path))
                except:
                    pass
                
                info = f"({file_count} .pt files"
                if latest_file:
                    info += f", latest: {latest_file}"
                info += ")"
                
            except Exception as e:
                info = "(无法读取目录信息)"
            
            print(f"  {i:2d}. {dir_name} {info}")
        
        print("-" * 60)
        
        while True:
            try:
                choice = input(f"请选择PPO checkpoint目录 (1-{len(directories)}) 或输入 'q' 退出: ").strip()
                
                if choice.lower() == 'q':
                    print("❌ 用户取消选择")
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(directories):
                    selected_dir = directories[choice_idx]
                    selected_name = os.path.basename(selected_dir)
                    print(f"✅ 选择了PPO目录: {selected_name}")
                    return _find_latest_checkpoint_in_dir(selected_dir)
                else:
                    print(f"❌ 无效选择，请输入 1-{len(directories)} 之间的数字")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n❌ 用户中断选择")
                return None
                
    except Exception as e:
        print(f"❌ 查找PPO checkpoint目录失败: {e}")
        return None


def _select_ppo_environment_type() -> str:
    """交互式选择PPO环境类型"""
    print("🏗️ 选择PPO训练环境类型...")
    
    environments = [
        ("Sequential", "SequentialNetworkSchedulerEnvironment", "基础的顺序网络调度环境"),
        ("Lightweight", "LightweightHeuristicEnvironment", "轻量级启发式环境（集成启发式奖励）"),
        ("NewHeuristic", "NewHeuristicEnvironment", "新启发式环境（改进的启发式奖励机制）")
    ]
    
    print(f"\n📁 可用的环境类型:")
    print("-" * 80)
    
    for i, (short_name, class_name, description) in enumerate(environments, 1):
        print(f"  {i:2d}. {short_name:<12} -> {description}")
        print(f"      类名: {class_name}")
    
    print("-" * 80)
    
    while True:
        try:
            choice = input(f"请选择环境类型 (1-{len(environments)}) 或输入 'q' 退出: ").strip()
            
            if choice.lower() == 'q':
                print("❌ 用户取消选择")
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(environments):
                selected_env = environments[choice_idx]
                print(f"✅ 选择了环境类型: {selected_env[0]} ({selected_env[1]})")
                return selected_env[0]
            else:
                print(f"❌ 无效选择，请输入 1-{len(environments)} 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n❌ 用户中断选择")
            return None


def _select_balance_checkpoint() -> Tuple[str, Dict]:
    """交互式选择PPO_balance checkpoint目录"""
    print("🔍 查找可用的PPO_balance checkpoint目录...")
    
    try:
        directories = _find_balance_checkpoint_directories()
        
        print(f"\n📁 找到 {len(directories)} 个PPO_balance checkpoint目录:")
        print("-" * 60)
        
        for i, dir_path in enumerate(directories, 1):
            dir_name = os.path.basename(dir_path)
            
            # 尝试获取训练摘要信息
            try:
                summary_path = os.path.join(dir_path, "training_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    
                    results = summary.get('results', {})
                    success_rate = results.get('final_success_rate', 0)
                    avg_reward = results.get('average_reward', 0)
                    total_episodes = results.get('total_episodes', 0)
                    
                    info = f"(Episodes: {total_episodes}, Success Rate: {success_rate:.2%}, Avg Reward: {avg_reward:.3f})"
                else:
                    info = "(无训练摘要)"
                
            except Exception as e:
                info = "(无法读取训练信息)"
            
            print(f"  {i:2d}. {dir_name} {info}")
        
        print("-" * 60)
        
        while True:
            try:
                choice = input(f"请选择PPO_balance目录 (1-{len(directories)}) 或输入 'q' 退出: ").strip()
                
                if choice.lower() == 'q':
                    print("❌ 用户取消选择")
                    return None, None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(directories):
                    selected_dir = directories[choice_idx]
                    selected_name = os.path.basename(selected_dir)
                    print(f"✅ 选择了PPO_balance目录: {selected_name}")
                    
                    # 读取配置信息
                    config = {}
                    summary_path = os.path.join(selected_dir, "training_summary.json")
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                        config = summary.get('config', {})
                    
                    model_path = _find_latest_balance_checkpoint(selected_dir)
                    return model_path, config
                else:
                    print(f"❌ 无效选择，请输入 1-{len(directories)} 之间的数字")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n❌ 用户中断选择")
                return None, None
                
    except Exception as e:
        print(f"❌ 查找PPO_balance checkpoint目录失败: {e}")
        return None, None


def _load_ppo_agent_and_configs(ckpt_path: str) -> Tuple[SimpleSequentialAgent, Dict, Dict]:
    """加载PPO Agent与配置。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔄 加载PPO检查点: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'config' not in checkpoint:
        raise KeyError("Checkpoint missing 'config' dict. 请使用 train_sequential_original.py 产生的检查点。")

    env_config = checkpoint['config'].get('env_config', {})
    agent_config = checkpoint['config'].get('agent_config', {})

    # 创建Agent并载入权重
    agent = SimpleSequentialAgent(**agent_config)
    state = checkpoint.get('agent_state_dict', None)
    if state is None:
        raise KeyError("Checkpoint missing 'agent_state_dict'.")
    agent.load_state_dict(state)
    agent.eval()

    print("✅ PPO Agent与配置加载完成")
    return agent, env_config, agent_config


def _load_balance_agent_and_configs(model_path: str, config: Dict) -> Tuple[BalanceAgent, Dict, Dict]:
    """加载PPO_balance Agent与配置"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔄 加载PPO_balance模型: {model_path}")
    
    env_config = config.get('env_config', {})
    agent_config = config.get('agent_config', {})
    
    # 创建Agent并载入权重
    agent = BalanceAgent(**agent_config)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查checkpoint格式并正确提取模型状态
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint  # 旧格式，直接是状态字典
    
    agent.load_state_dict(model_state_dict)
    agent.eval()
    
    print("✅ PPO_balance Agent与配置加载完成")
    return agent, env_config, agent_config


def _make_ppo_env_with_num_tasks(env_config: Dict, num_tasks: int, seed: int, env_type: str = "Sequential"):
    """
    基于给定配置、任务数量、种子和环境类型构建PPO环境。
    
    Args:
        env_config: 环境配置字典
        num_tasks: 任务数量
        seed: 随机种子
        env_type: 环境类型 ("Sequential", "Lightweight", "NewHeuristic")
    
    Returns:
        相应类型的环境实例
    """
    cfg = dict(env_config)
    cfg['seed'] = seed
    
    # 设置固定的虚拟节点数量（对于PPO环境）
    cfg['virtual_nodes_range'] = (num_tasks, num_tasks)
    cfg['max_virtual_nodes'] = max(num_tasks, cfg.get('max_virtual_nodes', 8))
    
    # 禁用课程学习以确保稳定性
    cfg['curriculum_enabled'] = False
    
    # 根据环境类型创建相应的环境
    if env_type == "Sequential":
        env = SequentialNetworkSchedulerEnvironment(**cfg)
        
    elif env_type == "Lightweight":
        # 轻量级启发式环境支持所有参数
        env = LightweightHeuristicEnvironment(**cfg)
        
    elif env_type == "NewHeuristic":
        # 新启发式环境支持所有参数
        env = NewHeuristicEnvironment(**cfg)
        
    else:
        raise ValueError(f"不支持的环境类型: {env_type}. 支持的类型: Sequential, Lightweight, NewHeuristic")
    
    return env


def _make_balance_env_with_num_tasks(env_config: Dict, num_tasks: int, seed: int) -> PPOBalanceNetworkEnvironment:
    """基于给定配置、任务数量和种子构建PPO_balance环境。"""
    cfg = dict(env_config)
    cfg['seed'] = seed
    
    # 设置固定的任务数量
    cfg['task_nodes_range'] = (num_tasks, num_tasks)
    cfg['max_task_nodes'] = max(num_tasks, cfg.get('max_task_nodes', 8))
    
    # 禁用课程学习以确保稳定性
    if 'curriculum_enabled' in cfg:
        cfg['curriculum_enabled'] = False
    
    env = PPOBalanceNetworkEnvironment(**cfg)
    return env


def _integrate_original_reward_after_reset(env):
    """在环境reset后集成原始奖励计算器"""
    if hasattr(env, 'network_scheduler') and env.network_scheduler is not None:
        try:
            integrate_with_network_scheduler(env.network_scheduler)
            return True
        except Exception as e:
            print(f"Warning: Failed to integrate original reward calculator: {e}")
            return False
    return False


def _select_action_greedy_with_fallback(env,
                                        agent: SimpleSequentialAgent,
                                        state: Dict) -> int:
    """PPO贪心策略选择动作，带回退机制。"""
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


def _select_balance_action_greedy(env: PPOBalanceNetworkEnvironment,
                                  agent: BalanceAgent,
                                  state: Dict) -> int:
    """PPO_balance贪心策略选择动作"""
    agent.eval()
    with torch.no_grad():
        # 从状态字典中提取tensor数据
        physical_resources = state.get('physical_resources', torch.zeros(env.num_physical_nodes, 2))
        task_requirements = state.get('task_requirements', torch.zeros(2))
        valid_actions = state.get('valid_actions', torch.ones(env.num_physical_nodes, dtype=torch.bool))
        
        action_logits, _, _ = agent.forward(physical_resources, task_requirements)
        action_probs = torch.softmax(action_logits, dim=0)
        action_probs = action_probs.detach().cpu().numpy()
    
    num_physical_nodes = env.num_physical_nodes
    candidates = np.argsort(-action_probs[:num_physical_nodes])  # 降序
    
    # 逐个尝试候选动作，选择第一个有效动作
    current_task_idx = state.get('current_task_index', 0)
    for a in candidates:
        # 检查是否为有效动作
        if valid_actions[a].item():
            return int(a)
    
    # 若均无效，返回得分最高者
    return int(candidates[0]) if len(candidates) > 0 else 0


def _run_ppo_episode(env,
                     agent: SimpleSequentialAgent,
                     temperature: float = 0.1,
                     max_steps: int = 50,
                     greedy: bool = True) -> Tuple[float, bool, Dict]:
    """运行PPO算法的Episode"""
    state = env.reset()
    
    # 在reset后集成原始奖励计算器
    _integrate_original_reward_after_reset(env)
    
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
            action = _select_action_greedy_with_fallback(env, agent, state)
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
        'bandwidth_satisfaction': bandwidth_satisfaction
    })
    
    return total_reward, success, episode_info


def _run_balance_episode(env: PPOBalanceNetworkEnvironment,
                         agent: BalanceAgent,
                         max_steps: int = 50) -> Tuple[float, bool, Dict]:
    """运行PPO_balance算法的Episode"""
    state = env.reset()
    
    # 尝试为PPO_balance环境也集成原始奖励计算器
    _integrate_original_reward_after_reset(env)
    
    done = False
    total_reward = 0.0
    steps = 0
    
    episode_info = {
        'num_virtual_nodes': state.get('num_tasks', 0),
        'num_virtual_links': 0,  # PPO_balance环境没有链路概念
        'algorithm': 'PPO_balance',
        'steps': 0
    }

    while not done and steps < max_steps:
        # 使用PPO_balance agent选择动作
        action = _select_balance_action_greedy(env, agent, state)
        state, reward, done, info = env.step(int(action))
        total_reward += float(reward)
        steps += 1

    # 成功判定 - 检查所有任务是否都分配成功
    success = bool(done and total_reward > 0)
    
    # 计算L和D_BW指标
    load_balance_degree = float('nan')
    bandwidth_satisfaction = float('nan')
    
    if (hasattr(env, 'network_scheduler') and env.network_scheduler is not None and 
        hasattr(env.network_scheduler, 'get_original_reward_components')):
        try:
            if hasattr(env, 'virtual_work') and env.virtual_work is not None:
                components = env.network_scheduler.get_original_reward_components(env.virtual_work)
                load_balance_degree = components.get('L', float('nan'))
                bandwidth_satisfaction = components.get('D_BW', float('nan'))
        except Exception as e:
            print(f"Warning: Failed to calculate L and D_BW for PPO_balance: {e}")
    elif hasattr(env, 'get_load_balance_metrics'):
        try:
            metrics = env.get_load_balance_metrics()
            load_balance_degree = metrics.get('load_balance_degree', float('nan'))
            # PPO_balance环境可能没有带宽满意度概念
            bandwidth_satisfaction = float('nan')
        except Exception as e:
            print(f"Warning: Failed to calculate load balance metrics for PPO_balance: {e}")
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward,
        'load_balance_degree': load_balance_degree,
        'bandwidth_satisfaction': bandwidth_satisfaction
    })
    
    return total_reward, success, episode_info


def _run_random_episode(env,
                        max_steps: int = 50) -> Tuple[float, bool, Dict]:
    """运行随机算法的Episode（适用于各种环境）"""
    state = env.reset()
    
    # 为Random算法也尝试集成原始奖励计算器
    _integrate_original_reward_after_reset(env)
    
    done = False
    total_reward = 0.0
    steps = 0
    
    episode_info = {
        'num_virtual_nodes': state.get('num_virtual_nodes', state.get('num_tasks', 0)),
        'num_virtual_links': state.get('num_virtual_links', 0),
        'algorithm': 'Random',
        'steps': 0
    }

    while not done and steps < max_steps:
        # 随机策略
        if state.get('mapping_phase', True):
            action = np.random.randint(0, state.get('num_physical_nodes', 10))
        else:
            if hasattr(env, 'bandwidth_levels'):
                action = np.random.randint(0, env.bandwidth_levels)
            else:
                # PPO_balance环境没有带宽等级概念
                action = 0

        state, reward, done, info = env.step(int(action))
        total_reward += float(reward)
        steps += 1

    # 成功判定
    if hasattr(env, 'partial_mapping'):
        # PPO环境
        all_nodes_mapped = all(node != -1 for node in (env.partial_mapping or []))
        success = bool(all_nodes_mapped and total_reward > 0)
    else:
        # PPO_balance环境 - 检查episode是否成功完成
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
            print(f"Warning: Failed to calculate L and D_BW for Random: {e}")
    elif hasattr(env, 'get_load_balance_metrics'):
        try:
            metrics = env.get_load_balance_metrics()
            load_balance_degree = metrics.get('load_balance_degree', float('nan'))
        except Exception as e:
            print(f"Warning: Failed to calculate load balance metrics for Random: {e}")
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward,
        'load_balance_degree': load_balance_degree,
        'bandwidth_satisfaction': bandwidth_satisfaction
    })
    
    return total_reward, success, episode_info


def evaluate_four_algorithms(ppo_agent: SimpleSequentialAgent,
                              ppo_env_config: Dict,
                              balance_agent: BalanceAgent,
                              balance_env_config: Dict,
                              task_range: List[int] = [3, 4, 5, 6],
                              episodes_per_task_count: int = 30,
                              base_seed: int = 42,
                              temperature: float = 0.05,
                              heuristic_type: str = "naive",
                              ppo_env_type: str = "Sequential") -> Dict:
    """评测四种算法在不同任务数量下的性能"""
    print(f"🧪 开始评测四种算法性能对比")
    print(f"🎯 算法类型: PPO ({ppo_env_type}), PPO_balance, Heuristic, Random")
    print(f"📊 任务数量范围: {task_range}")
    print(f"📋 每个任务数量测试 {episodes_per_task_count} 个Episode")
    
    _set_global_seed(base_seed)
    
    results = {
        'algorithms': ['PPO', 'PPO_balance', 'Heuristic', 'Random'],
        'task_range': task_range,
        'episodes_per_task_count': episodes_per_task_count,
        'base_seed': base_seed,
        'temperature': temperature,
        'heuristic_type': heuristic_type,
        'ppo_env_type': ppo_env_type,
        'detailed_results': {},
        'summary': {}
    }
    
    # 创建启发式算法代理
    heuristic_agent = create_heuristic_agent(heuristic_type)
    
    for num_tasks in task_range:
        print(f"\n🔍 测试 {num_tasks} 个任务...")
        
        # 存储四种算法的结果
        ppo_rewards, ppo_success, ppo_steps = [], [], []
        balance_rewards, balance_success, balance_steps = [], [], []
        heu_rewards, heu_success, heu_steps = [], [], []
        rnd_rewards, rnd_success, rnd_steps = [], [], []
        
        # 存储L和D_BW指标
        ppo_l_values, ppo_dbw_values = [], []
        balance_l_values, balance_dbw_values = [], []
        heu_l_values, heu_dbw_values = [], []
        rnd_l_values, rnd_dbw_values = [], []
        
        for ep in range(episodes_per_task_count):
            ep_seed = base_seed + ep + num_tasks * 1000
            
            # 创建四个相同配置的环境
            env_ppo = _make_ppo_env_with_num_tasks(ppo_env_config, num_tasks, ep_seed, ppo_env_type)
            env_balance = _make_balance_env_with_num_tasks(balance_env_config, num_tasks, ep_seed)
            env_heu = _make_ppo_env_with_num_tasks(ppo_env_config, num_tasks, ep_seed, ppo_env_type)
            env_rnd_ppo = _make_ppo_env_with_num_tasks(ppo_env_config, num_tasks, ep_seed, ppo_env_type)
            
            # 运行PPO
            r_ppo, s_ppo, info_ppo = _run_ppo_episode(env_ppo, ppo_agent, temperature=temperature)
            ppo_rewards.append(r_ppo)
            ppo_success.append(1.0 if s_ppo else 0.0)
            ppo_steps.append(info_ppo['steps'])
            ppo_l_values.append(info_ppo.get('load_balance_degree', float('nan')))
            ppo_dbw_values.append(info_ppo.get('bandwidth_satisfaction', float('nan')))
            
            # 运行PPO_balance
            r_balance, s_balance, info_balance = _run_balance_episode(env_balance, balance_agent)
            balance_rewards.append(r_balance)
            balance_success.append(1.0 if s_balance else 0.0)
            balance_steps.append(info_balance['steps'])
            balance_l_values.append(info_balance.get('load_balance_degree', float('nan')))
            balance_dbw_values.append(info_balance.get('bandwidth_satisfaction', float('nan')))
            
            # 运行启发式算法
            r_heu, s_heu, info_heu = run_heuristic_episode(env_heu, heuristic_agent)
            heu_rewards.append(r_heu)
            heu_success.append(1.0 if s_heu else 0.0)
            heu_steps.append(info_heu['steps'])
            heu_l_values.append(info_heu.get('load_balance_degree', float('nan')))
            heu_dbw_values.append(info_heu.get('bandwidth_satisfaction', float('nan')))
            
            # 运行随机算法
            r_rnd, s_rnd, info_rnd = _run_random_episode(env_rnd_ppo)
            rnd_rewards.append(r_rnd)
            rnd_success.append(1.0 if s_rnd else 0.0)
            rnd_steps.append(info_rnd['steps'])
            rnd_l_values.append(info_rnd.get('load_balance_degree', float('nan')))
            rnd_dbw_values.append(info_rnd.get('bandwidth_satisfaction', float('nan')))
            
            if (ep + 1) % 10 == 0:
                print(f"   完成 {ep + 1}/{episodes_per_task_count} episodes")
        
        # 统计结果
        def _stats(arr):
            return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
        
        def _stats_with_nan(arr):
            """计算包含NaN值的数组统计信息"""
            valid_arr = [x for x in arr if not np.isnan(x)]
            if len(valid_arr) == 0:
                return float('nan'), float('nan'), float('nan'), float('nan')
            return float(np.mean(valid_arr)), float(np.std(valid_arr)), float(np.min(valid_arr)), float(np.max(valid_arr))
        
        ppo_reward_mean, ppo_reward_std, ppo_reward_min, ppo_reward_max = _stats(ppo_rewards)
        balance_reward_mean, balance_reward_std, balance_reward_min, balance_reward_max = _stats(balance_rewards)
        heu_reward_mean, heu_reward_std, heu_reward_min, heu_reward_max = _stats(heu_rewards)
        rnd_reward_mean, rnd_reward_std, rnd_reward_min, rnd_reward_max = _stats(rnd_rewards)
        
        # L和D_BW统计
        ppo_l_mean, ppo_l_std, ppo_l_min, ppo_l_max = _stats_with_nan(ppo_l_values)
        balance_l_mean, balance_l_std, balance_l_min, balance_l_max = _stats_with_nan(balance_l_values)
        heu_l_mean, heu_l_std, heu_l_min, heu_l_max = _stats_with_nan(heu_l_values)
        rnd_l_mean, rnd_l_std, rnd_l_min, rnd_l_max = _stats_with_nan(rnd_l_values)
        
        ppo_dbw_mean, ppo_dbw_std, ppo_dbw_min, ppo_dbw_max = _stats_with_nan(ppo_dbw_values)
        balance_dbw_mean, balance_dbw_std, balance_dbw_min, balance_dbw_max = _stats_with_nan(balance_dbw_values)
        heu_dbw_mean, heu_dbw_std, heu_dbw_min, heu_dbw_max = _stats_with_nan(heu_dbw_values)
        rnd_dbw_mean, rnd_dbw_std, rnd_dbw_min, rnd_dbw_max = _stats_with_nan(rnd_dbw_values)
        
        ppo_sr = float(np.mean(ppo_success))
        balance_sr = float(np.mean(balance_success))
        heu_sr = float(np.mean(heu_success))
        rnd_sr = float(np.mean(rnd_success))
        
        ppo_avg_steps = float(np.mean(ppo_steps))
        balance_avg_steps = float(np.mean(balance_steps))
        heu_avg_steps = float(np.mean(heu_steps))
        rnd_avg_steps = float(np.mean(rnd_steps))
        
        node_result = {
            'num_tasks': num_tasks,
            'ppo': {
                'avg_reward': ppo_reward_mean,
                'std_reward': ppo_reward_std,
                'min_reward': ppo_reward_min,
                'max_reward': ppo_reward_max,
                'success_rate': ppo_sr,
                'avg_steps': ppo_avg_steps,
                'load_balance': {
                    'mean': ppo_l_mean,
                    'std': ppo_l_std,
                    'min': ppo_l_min,
                    'max': ppo_l_max
                },
                'bandwidth_satisfaction': {
                    'mean': ppo_dbw_mean,
                    'std': ppo_dbw_std,
                    'min': ppo_dbw_min,
                    'max': ppo_dbw_max
                }
            },
            'ppo_balance': {
                'avg_reward': balance_reward_mean,
                'std_reward': balance_reward_std,
                'min_reward': balance_reward_min,
                'max_reward': balance_reward_max,
                'success_rate': balance_sr,
                'avg_steps': balance_avg_steps,
                'load_balance': {
                    'mean': balance_l_mean,
                    'std': balance_l_std,
                    'min': balance_l_min,
                    'max': balance_l_max
                },
                'bandwidth_satisfaction': {
                    'mean': balance_dbw_mean,
                    'std': balance_dbw_std,
                    'min': balance_dbw_min,
                    'max': balance_dbw_max
                }
            },
            'heuristic': {
                'avg_reward': heu_reward_mean,
                'std_reward': heu_reward_std,
                'min_reward': heu_reward_min,
                'max_reward': heu_reward_max,
                'success_rate': heu_sr,
                'avg_steps': heu_avg_steps,
                'load_balance': {
                    'mean': heu_l_mean,
                    'std': heu_l_std,
                    'min': heu_l_min,
                    'max': heu_l_max
                },
                'bandwidth_satisfaction': {
                    'mean': heu_dbw_mean,
                    'std': heu_dbw_std,
                    'min': heu_dbw_min,
                    'max': heu_dbw_max
                }
            },
            'random': {
                'avg_reward': rnd_reward_mean,
                'std_reward': rnd_reward_std,
                'min_reward': rnd_reward_min,
                'max_reward': rnd_reward_max,
                'success_rate': rnd_sr,
                'avg_steps': rnd_avg_steps,
                'load_balance': {
                    'mean': rnd_l_mean,
                    'std': rnd_l_std,
                    'min': rnd_l_min,
                    'max': rnd_l_max
                },
                'bandwidth_satisfaction': {
                    'mean': rnd_dbw_mean,
                    'std': rnd_dbw_std,
                    'min': rnd_dbw_min,
                    'max': rnd_dbw_max
                }
            },
            'comparisons': {
                'ppo_vs_heuristic': {
                    'reward_improvement': ppo_reward_mean - heu_reward_mean,
                    'success_rate_improvement': ppo_sr - heu_sr
                },
                'ppo_balance_vs_heuristic': {
                    'reward_improvement': balance_reward_mean - heu_reward_mean,
                    'success_rate_improvement': balance_sr - heu_sr
                },
                'ppo_vs_ppo_balance': {
                    'reward_improvement': ppo_reward_mean - balance_reward_mean,
                    'success_rate_improvement': ppo_sr - balance_sr
                },
                'ppo_vs_random': {
                    'reward_improvement': ppo_reward_mean - rnd_reward_mean,
                    'success_rate_improvement': ppo_sr - rnd_sr
                },
                'ppo_balance_vs_random': {
                    'reward_improvement': balance_reward_mean - rnd_reward_mean,
                    'success_rate_improvement': balance_sr - rnd_sr
                }
            }
        }
        
        results['detailed_results'][num_tasks] = node_result
        
        print(f"✅ {num_tasks} 任务测试完成:")
        print(f"   PPO         -> 奖励: {ppo_reward_mean:.3f}±{ppo_reward_std:.3f}, 成功率: {ppo_sr:.1%}, 平均步数: {ppo_avg_steps:.1f}")
        print(f"                -> L: {ppo_l_mean:.4f}±{ppo_l_std:.4f}, D_BW: {ppo_dbw_mean:.4f}±{ppo_dbw_std:.4f}")
        print(f"   PPO_balance -> 奖励: {balance_reward_mean:.3f}±{balance_reward_std:.3f}, 成功率: {balance_sr:.1%}, 平均步数: {balance_avg_steps:.1f}")
        print(f"                -> L: {balance_l_mean:.4f}±{balance_l_std:.4f}, D_BW: {balance_dbw_mean:.4f}±{balance_dbw_std:.4f}")
        print(f"   Heuristic   -> 奖励: {heu_reward_mean:.3f}±{heu_reward_std:.3f}, 成功率: {heu_sr:.1%}, 平均步数: {heu_avg_steps:.1f}")
        print(f"                -> L: {heu_l_mean:.4f}±{heu_l_std:.4f}, D_BW: {heu_dbw_mean:.4f}±{heu_dbw_std:.4f}")
        print(f"   Random      -> 奖励: {rnd_reward_mean:.3f}±{rnd_reward_std:.3f}, 成功率: {rnd_sr:.1%}, 平均步数: {rnd_avg_steps:.1f}")
        print(f"                -> L: {rnd_l_mean:.4f}±{rnd_l_std:.4f}, D_BW: {rnd_dbw_mean:.4f}±{rnd_dbw_std:.4f}")
    
    # 生成总结
    results['summary'] = _generate_summary(results)
    
    return results


def _generate_summary(results: Dict) -> Dict:
    """生成测试结果的总结分析"""
    detailed = results['detailed_results']
    task_range = results['task_range']
    
    # 收集数据
    ppo_rewards = [detailed[n]['ppo']['avg_reward'] for n in task_range]
    balance_rewards = [detailed[n]['ppo_balance']['avg_reward'] for n in task_range]
    heu_rewards = [detailed[n]['heuristic']['avg_reward'] for n in task_range]
    rnd_rewards = [detailed[n]['random']['avg_reward'] for n in task_range]
    
    ppo_success_rates = [detailed[n]['ppo']['success_rate'] for n in task_range]
    balance_success_rates = [detailed[n]['ppo_balance']['success_rate'] for n in task_range]
    heu_success_rates = [detailed[n]['heuristic']['success_rate'] for n in task_range]
    rnd_success_rates = [detailed[n]['random']['success_rate'] for n in task_range]
    
    # 计算平均性能
    summary = {
        'overall_performance': {
            'ppo': {
                'avg_reward': float(np.mean(ppo_rewards)),
                'avg_success_rate': float(np.mean(ppo_success_rates))
            },
            'ppo_balance': {
                'avg_reward': float(np.mean(balance_rewards)),
                'avg_success_rate': float(np.mean(balance_success_rates))
            },
            'heuristic': {
                'avg_reward': float(np.mean(heu_rewards)),
                'avg_success_rate': float(np.mean(heu_success_rates))
            },
            'random': {
                'avg_reward': float(np.mean(rnd_rewards)),
                'avg_success_rate': float(np.mean(rnd_success_rates))
            }
        },
        'rankings': {
            'by_reward': [],
            'by_success_rate': []
        },
        'win_rates': {
            'ppo_vs_heuristic': 0,
            'ppo_balance_vs_heuristic': 0,
            'ppo_vs_ppo_balance': 0,
            'ppo_vs_random': 0,
            'ppo_balance_vs_random': 0
        }
    }
    
    # 计算胜率
    ppo_beats_heu = sum(1 for n in task_range if detailed[n]['ppo']['avg_reward'] > detailed[n]['heuristic']['avg_reward'])
    balance_beats_heu = sum(1 for n in task_range if detailed[n]['ppo_balance']['avg_reward'] > detailed[n]['heuristic']['avg_reward'])
    ppo_beats_balance = sum(1 for n in task_range if detailed[n]['ppo']['avg_reward'] > detailed[n]['ppo_balance']['avg_reward'])
    ppo_beats_rnd = sum(1 for n in task_range if detailed[n]['ppo']['avg_reward'] > detailed[n]['random']['avg_reward'])
    balance_beats_rnd = sum(1 for n in task_range if detailed[n]['ppo_balance']['avg_reward'] > detailed[n]['random']['avg_reward'])
    
    summary['win_rates']['ppo_vs_heuristic'] = ppo_beats_heu / len(task_range)
    summary['win_rates']['ppo_balance_vs_heuristic'] = balance_beats_heu / len(task_range)
    summary['win_rates']['ppo_vs_ppo_balance'] = ppo_beats_balance / len(task_range)
    summary['win_rates']['ppo_vs_random'] = ppo_beats_rnd / len(task_range)
    summary['win_rates']['ppo_balance_vs_random'] = balance_beats_rnd / len(task_range)
    
    # 排名
    avg_performance = [
        ('PPO', summary['overall_performance']['ppo']['avg_reward'], summary['overall_performance']['ppo']['avg_success_rate']),
        ('PPO_balance', summary['overall_performance']['ppo_balance']['avg_reward'], summary['overall_performance']['ppo_balance']['avg_success_rate']),
        ('Heuristic', summary['overall_performance']['heuristic']['avg_reward'], summary['overall_performance']['heuristic']['avg_success_rate']),
        ('Random', summary['overall_performance']['random']['avg_reward'], summary['overall_performance']['random']['avg_success_rate'])
    ]
    
    summary['rankings']['by_reward'] = sorted(avg_performance, key=lambda x: x[1], reverse=True)
    summary['rankings']['by_success_rate'] = sorted(avg_performance, key=lambda x: x[2], reverse=True)
    
    return summary


def _save_results(results: Dict, out_dir: str = 'test_results') -> str:
    """保存测试结果到JSON文件"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'four_algorithms_with_balance_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


def _create_visualization(results: Dict, out_dir: str = 'test_results') -> List[str]:
    """创建可视化图表"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    task_range = results['task_range']
    detailed = results['detailed_results']
    heuristic_type = results['heuristic_type'].title()
    ppo_env_type = results.get('ppo_env_type', 'Sequential')
    
    # 准备数据
    ppo_rewards = [detailed[n]['ppo']['avg_reward'] for n in task_range]
    balance_rewards = [detailed[n]['ppo_balance']['avg_reward'] for n in task_range]
    heu_rewards = [detailed[n]['heuristic']['avg_reward'] for n in task_range]
    rnd_rewards = [detailed[n]['random']['avg_reward'] for n in task_range]
    
    ppo_success_rates = [detailed[n]['ppo']['success_rate'] * 100 for n in task_range]
    balance_success_rates = [detailed[n]['ppo_balance']['success_rate'] * 100 for n in task_range]
    heu_success_rates = [detailed[n]['heuristic']['success_rate'] * 100 for n in task_range]
    rnd_success_rates = [detailed[n]['random']['success_rate'] * 100 for n in task_range]
    
    ppo_reward_stds = [detailed[n]['ppo']['std_reward'] for n in task_range]
    balance_reward_stds = [detailed[n]['ppo_balance']['std_reward'] for n in task_range]
    heu_reward_stds = [detailed[n]['heuristic']['std_reward'] for n in task_range]
    rnd_reward_stds = [detailed[n]['random']['std_reward'] for n in task_range]
    
    # 准备L和D_BW数据
    ppo_l_values = [detailed[n]['ppo']['load_balance']['mean'] for n in task_range]
    balance_l_values = [detailed[n]['ppo_balance']['load_balance']['mean'] for n in task_range]
    heu_l_values = [detailed[n]['heuristic']['load_balance']['mean'] for n in task_range]
    rnd_l_values = [detailed[n]['random']['load_balance']['mean'] for n in task_range]
    
    ppo_l_stds = [detailed[n]['ppo']['load_balance']['std'] for n in task_range]
    balance_l_stds = [detailed[n]['ppo_balance']['load_balance']['std'] for n in task_range]
    heu_l_stds = [detailed[n]['heuristic']['load_balance']['std'] for n in task_range]
    rnd_l_stds = [detailed[n]['random']['load_balance']['std'] for n in task_range]
    
    ppo_dbw_values = [detailed[n]['ppo']['bandwidth_satisfaction']['mean'] for n in task_range]
    balance_dbw_values = [detailed[n]['ppo_balance']['bandwidth_satisfaction']['mean'] for n in task_range]
    heu_dbw_values = [detailed[n]['heuristic']['bandwidth_satisfaction']['mean'] for n in task_range]
    rnd_dbw_values = [detailed[n]['random']['bandwidth_satisfaction']['mean'] for n in task_range]
    
    ppo_dbw_stds = [detailed[n]['ppo']['bandwidth_satisfaction']['std'] for n in task_range]
    balance_dbw_stds = [detailed[n]['ppo_balance']['bandwidth_satisfaction']['std'] for n in task_range]
    heu_dbw_stds = [detailed[n]['heuristic']['bandwidth_satisfaction']['std'] for n in task_range]
    rnd_dbw_stds = [detailed[n]['random']['bandwidth_satisfaction']['std'] for n in task_range]
    
    # 计算综合指标 = 0.5*L + 0.5*D_BW (注意：这里L越小越好，D_BW越大越好，所以我们调整为 0.5*(1-L) + 0.5*D_BW 来保持越大越好的一致性)
    def _calculate_composite_score(l_val, dbw_val, l_weight=0.5, dbw_weight=0.5):
        """计算综合指标，处理NaN值"""
        if np.isnan(l_val) or np.isnan(dbw_val):
            return float('nan')
        # 由于L越小越好，我们使用(1-L)来转换为越大越好
        # 假设L的正常范围在[0,1]，如果超出可能需要normalization
        normalized_l = max(0, 1 - l_val)  # 转换为越大越好
        return l_weight * normalized_l + dbw_weight * dbw_val
    
    ppo_composite_values = [_calculate_composite_score(ppo_l_values[i], ppo_dbw_values[i]) for i in range(len(task_range))]
    balance_composite_values = [_calculate_composite_score(balance_l_values[i], balance_dbw_values[i]) for i in range(len(task_range))]
    heu_composite_values = [_calculate_composite_score(heu_l_values[i], heu_dbw_values[i]) for i in range(len(task_range))]
    rnd_composite_values = [_calculate_composite_score(rnd_l_values[i], rnd_dbw_values[i]) for i in range(len(task_range))]
    
    # 计算综合指标的标准差（简化处理，使用L和D_BW标准差的加权组合）
    def _calculate_composite_std(l_std, dbw_std, l_weight=0.5, dbw_weight=0.5):
        """计算综合指标标准差的近似值"""
        if np.isnan(l_std) or np.isnan(dbw_std):
            return float('nan')
        return np.sqrt((l_weight * l_std) ** 2 + (dbw_weight * dbw_std) ** 2)
    
    ppo_composite_stds = [_calculate_composite_std(ppo_l_stds[i], ppo_dbw_stds[i]) for i in range(len(task_range))]
    balance_composite_stds = [_calculate_composite_std(balance_l_stds[i], balance_dbw_stds[i]) for i in range(len(task_range))]
    heu_composite_stds = [_calculate_composite_std(heu_l_stds[i], heu_dbw_stds[i]) for i in range(len(task_range))]
    rnd_composite_stds = [_calculate_composite_std(rnd_l_stds[i], rnd_dbw_stds[i]) for i in range(len(task_range))]
    
    saved_files = []
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#7B2CBF']  # PPO, PPO_balance, Heuristic, Random
    
    # 图1: 奖励和成功率对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 奖励曲线
    ax1.errorbar(task_range, ppo_rewards, yerr=ppo_reward_stds, 
                label=f'PPO ({ppo_env_type})', marker='o', linewidth=2, capsize=5, color=colors[0])
    ax1.errorbar(task_range, balance_rewards, yerr=balance_reward_stds,
                label='PPO_balance', marker='s', linewidth=2, capsize=5, color=colors[1])
    ax1.errorbar(task_range, heu_rewards, yerr=heu_reward_stds,
                label=f'{heuristic_type} Heuristic', marker='^', linewidth=2, capsize=5, color=colors[2])
    ax1.errorbar(task_range, rnd_rewards, yerr=rnd_reward_stds,
                label='Random', marker='d', linewidth=2, capsize=5, color=colors[3])
    ax1.set_xlabel('Number of Tasks')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward Comparison')
    ax1.set_xticks(task_range)  # 只显示整数刻度
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 成功率对比
    ax2.plot(task_range, ppo_success_rates, label=f'PPO ({ppo_env_type})', 
            marker='o', linewidth=2, color=colors[0])
    ax2.plot(task_range, balance_success_rates, label='PPO_balance', 
            marker='s', linewidth=2, color=colors[1])
    ax2.plot(task_range, heu_success_rates, label=f'{heuristic_type} Heuristic', 
            marker='^', linewidth=2, color=colors[2])
    ax2.plot(task_range, rnd_success_rates, label='Random', 
            marker='d', linewidth=2, color=colors[3])
    ax2.set_xlabel('Number of Tasks')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate Comparison')
    ax2.set_xticks(task_range)  # 只显示整数刻度
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_comparison_path = os.path.join(out_dir, f'four_algorithms_with_balance_performance_{ts}.png')
    plt.savefig(performance_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(performance_comparison_path)
    
    # 图2: 负载均衡度(L)和带宽满意度(D_BW)对比
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 负载均衡度L对比 (越小越好)
    def _plot_with_nan_handling(ax, x, y, yerr, label, marker, color):
        """处理包含NaN值的绘图"""
        valid_indices = [i for i, val in enumerate(y) if not np.isnan(val)]
        if valid_indices:
            x_valid = [x[i] for i in valid_indices]
            y_valid = [y[i] for i in valid_indices]
            yerr_valid = [yerr[i] for i in valid_indices] if yerr else None
            if yerr_valid:
                ax.errorbar(x_valid, y_valid, yerr=yerr_valid, 
                           label=label, marker=marker, linewidth=2, capsize=5, color=color)
            else:
                ax.plot(x_valid, y_valid, label=label, marker=marker, linewidth=2, color=color)
    
    _plot_with_nan_handling(ax1, task_range, ppo_l_values, ppo_l_stds, 
                           f'PPO ({ppo_env_type})', 'o', colors[0])
    _plot_with_nan_handling(ax1, task_range, balance_l_values, balance_l_stds,
                           'PPO_balance', 's', colors[1])
    _plot_with_nan_handling(ax1, task_range, heu_l_values, heu_l_stds,
                           f'{heuristic_type} Heuristic', '^', colors[2])
    _plot_with_nan_handling(ax1, task_range, rnd_l_values, rnd_l_stds,
                           'Random', 'd', colors[3])
    ax1.set_xlabel('Number of Tasks')
    ax1.set_ylabel('Load Balance Degree (L)')
    ax1.set_title('Load Balance Degree Comparison\n(Lower is Better)')
    ax1.set_xticks(task_range)  # 只显示整数刻度
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 带宽满意度D_BW对比 (越大越好)
    _plot_with_nan_handling(ax2, task_range, ppo_dbw_values, ppo_dbw_stds,
                           f'PPO ({ppo_env_type})', 'o', colors[0])
    _plot_with_nan_handling(ax2, task_range, balance_dbw_values, balance_dbw_stds,
                           'PPO_balance', 's', colors[1])
    _plot_with_nan_handling(ax2, task_range, heu_dbw_values, heu_dbw_stds,
                           f'{heuristic_type} Heuristic', '^', colors[2])
    _plot_with_nan_handling(ax2, task_range, rnd_dbw_values, rnd_dbw_stds,
                           'Random', 'd', colors[3])
    ax2.set_xlabel('Number of Tasks')
    ax2.set_ylabel('Bandwidth Satisfaction (D_BW)')
    ax2.set_title('Bandwidth Satisfaction Comparison\n(Higher is Better)')
    ax2.set_xticks(task_range)  # 只显示整数刻度
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 综合指标对比 (越大越好)
    _plot_with_nan_handling(ax3, task_range, ppo_composite_values, ppo_composite_stds,
                           f'PPO ({ppo_env_type})', 'o', colors[0])
    _plot_with_nan_handling(ax3, task_range, balance_composite_values, balance_composite_stds,
                           'PPO_balance', 's', colors[1])
    _plot_with_nan_handling(ax3, task_range, heu_composite_values, heu_composite_stds,
                           f'{heuristic_type} Heuristic', '^', colors[2])
    _plot_with_nan_handling(ax3, task_range, rnd_composite_values, rnd_composite_stds,
                           'Random', 'd', colors[3])
    ax3.set_xlabel('Number of Tasks')
    ax3.set_ylabel('Composite Score')
    ax3.set_title('Composite Score Comparison\n(0.5×(1-L) + 0.5×D_BW, Higher is Better)')
    ax3.set_xticks(task_range)  # 只显示整数刻度
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    lb_bw_comparison_path = os.path.join(out_dir, f'four_algorithms_with_balance_lb_bw_{ts}.png')
    plt.savefig(lb_bw_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(lb_bw_comparison_path)
    
    return saved_files


def print_detailed_results(results: Dict):
    """打印详细的测试结果"""
    print("\n" + "="*100)
    print("📊 四算法性能对比详细结果")
    print("="*100)
    
    # 测试配置
    heuristic_type = results['heuristic_type'].title()
    ppo_env_type = results.get('ppo_env_type', 'Sequential')
    print(f"🔧 测试配置:")
    print(f"   算法类型: PPO ({ppo_env_type}), PPO_balance, {heuristic_type} Heuristic, Random")
    print(f"   任务数量范围: {results['task_range']}")
    print(f"   每个数量测试: {results['episodes_per_task_count']} episodes")
    print(f"   基础种子: {results['base_seed']}")
    
    # 详细结果
    print(f"\n📋 详细结果:")
    print("-"*140)
    header = f"{'任务数':<6} {'PPO奖励':<10} {'PPO_bal奖励':<12} {'启发式奖励':<12} {'随机奖励':<10} {'PPO成功率':<10} {'PPO_bal成功率':<12} {'启发式成功率':<12} {'随机成功率':<10}"
    print(header)
    print("-"*140)
    
    for num_tasks in results['task_range']:
        detail = results['detailed_results'][num_tasks]
        ppo = detail['ppo']
        balance = detail['ppo_balance']
        heu = detail['heuristic']
        rnd = detail['random']
        
        row = (f"{num_tasks:<6} "
               f"{ppo['avg_reward']:<10.3f} "
               f"{balance['avg_reward']:<12.3f} "
               f"{heu['avg_reward']:<12.3f} "
               f"{rnd['avg_reward']:<10.3f} "
               f"{ppo['success_rate']:<10.1%} "
               f"{balance['success_rate']:<12.1%} "
               f"{heu['success_rate']:<12.1%} "
               f"{rnd['success_rate']:<10.1%}")
        print(row)
        
        # 添加L指标信息
        def _format_metric(val, std_val):
            if np.isnan(val) or np.isnan(std_val):
                return "N/A"
            return f"{val:.4f}±{std_val:.4f}"
        
        ppo_l_str = _format_metric(ppo['load_balance']['mean'], ppo['load_balance']['std'])
        balance_l_str = _format_metric(balance['load_balance']['mean'], balance['load_balance']['std'])
        heu_l_str = _format_metric(heu['load_balance']['mean'], heu['load_balance']['std'])
        rnd_l_str = _format_metric(rnd['load_balance']['mean'], rnd['load_balance']['std'])
        
        l_row = f"{'L值':<6} {ppo_l_str:<10} {balance_l_str:<12} {heu_l_str:<12} {rnd_l_str:<10}"
        
        print(f"       {l_row}")
        print()
    
    # 总结分析
    summary = results['summary']
    print(f"\n📈 总结分析:")
    print("-"*80)
    overall = summary['overall_performance']
    print(f"PPO平均性能         -> 奖励: {overall['ppo']['avg_reward']:.3f}, 成功率: {overall['ppo']['avg_success_rate']:.1%}")
    print(f"PPO_balance平均性能 -> 奖励: {overall['ppo_balance']['avg_reward']:.3f}, 成功率: {overall['ppo_balance']['avg_success_rate']:.1%}")
    print(f"{heuristic_type}启发式平均性能   -> 奖励: {overall['heuristic']['avg_reward']:.3f}, 成功率: {overall['heuristic']['avg_success_rate']:.1%}")
    print(f"Random平均性能      -> 奖励: {overall['random']['avg_reward']:.3f}, 成功率: {overall['random']['avg_success_rate']:.1%}")
    
    # 胜率统计
    win_rates = summary['win_rates']
    print(f"\n🏆 胜率统计:")
    print(f"PPO vs {heuristic_type}启发式: {win_rates['ppo_vs_heuristic']:.1%}")
    print(f"PPO_balance vs {heuristic_type}启发式: {win_rates['ppo_balance_vs_heuristic']:.1%}")
    print(f"PPO vs PPO_balance: {win_rates['ppo_vs_ppo_balance']:.1%}")
    print(f"PPO vs Random: {win_rates['ppo_vs_random']:.1%}")
    print(f"PPO_balance vs Random: {win_rates['ppo_balance_vs_random']:.1%}")
    
    # 排名
    print(f"\n🥇 算法排名:")
    print("按奖励排名:")
    for i, (name, reward, _) in enumerate(summary['rankings']['by_reward'], 1):
        print(f"   {i}. {name}: {reward:.3f}")
    print("按成功率排名:")
    for i, (name, _, success_rate) in enumerate(summary['rankings']['by_success_rate'], 1):
        print(f"   {i}. {name}: {success_rate:.1%}")


def main():
    print("🧪 四算法性能对比测试 (PPO vs PPO_balance vs Heuristic vs Random)")
    print("="*80)

    try:
        # 交互式选择checkpoint
        print("步骤1: 选择PPO模型")
        ppo_ckpt = _select_ppo_checkpoint()
        if ppo_ckpt is None:
            print("❌ 未选择PPO checkpoint，程序退出")
            return
        
        print("\n步骤2: 选择PPO环境类型")
        ppo_env_type = _select_ppo_environment_type()
        if ppo_env_type is None:
            print("❌ 未选择PPO环境类型，程序退出")
            return
        
        print("\n步骤3: 选择PPO_balance模型")
        balance_ckpt, balance_config = _select_balance_checkpoint()
        if balance_ckpt is None:
            print("❌ 未选择PPO_balance checkpoint，程序退出")
            return
        
        # 加载模型
        print(f"\n📂 加载模型...")
        ppo_agent, ppo_env_config, _ = _load_ppo_agent_and_configs(ppo_ckpt)
        balance_agent, balance_env_config, _ = _load_balance_agent_and_configs(balance_ckpt, balance_config)

        # 测试参数
        task_range = [3, 4, 5, 6, 7, 8]  # 测试3-8个任务
        episodes_per_task_count = 30  # 每个任务数量测试30次
        base_seed = 42
        temperature = 0.05  # 低温度，更贪心
        heuristic_type = "flexitask"  # 使用启发式算法 naive, moderate, smart, flexitask

        # 执行评测
        results = evaluate_four_algorithms(
            ppo_agent=ppo_agent,
            ppo_env_config=ppo_env_config,
            balance_agent=balance_agent,
            balance_env_config=balance_env_config,
            task_range=task_range,
            episodes_per_task_count=episodes_per_task_count,
            base_seed=base_seed,
            temperature=temperature,
            heuristic_type=heuristic_type,
            ppo_env_type=ppo_env_type
        )

        # 打印结果
        print_detailed_results(results)

        # 保存结果
        json_path = _save_results(results)
        print(f"\n💾 详细结果已保存到: {json_path}")

        # 创建可视化
        try:
            vis_files = _create_visualization(results)
            print(f"📊 可视化图表已保存:")
            for f in vis_files:
                print(f"   - {f}")
        except Exception as e:
            print(f"⚠️ 可视化创建失败: {e}")
            print("💡 提示: 确保安装了 matplotlib 和 seaborn")

        # 最终结论
        summary = results['summary']
        win_rates = summary['win_rates']
        
        print(f"\n🎯 最终结论:")
        print("-"*50)
        
        # 判断算法优劣
        rankings_by_reward = summary['rankings']['by_reward']
        best_algorithm = rankings_by_reward[0][0]
        
        if best_algorithm == "PPO":
            print("🥇 PPO算法表现最佳!")
        elif best_algorithm == "PPO_balance":
            print("🥇 PPO_balance算法表现最佳!")
        elif best_algorithm == "Heuristic":
            print("🥇 启发式算法表现最佳!")
        else:
            print("⚠️ 所有算法表现都需要改进")
        
        print(f"📊 关键对比:")
        print(f"   PPO vs PPO_balance: {win_rates['ppo_vs_ppo_balance']:.1%}")
        print(f"   PPO vs 启发式: {win_rates['ppo_vs_heuristic']:.1%}")
        print(f"   PPO_balance vs 启发式: {win_rates['ppo_balance_vs_heuristic']:.1%}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
