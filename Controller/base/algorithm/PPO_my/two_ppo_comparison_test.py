#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两个PPO算法模型对比测试
专门用于对比不同PPO模型（如原始PPO vs 轻量级PPO）的性能指标
"""

import os
import re
import json
import time
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# PPO相关导入
from sequential_environment import SequentialNetworkSchedulerEnvironment
from sequential_agent import SimpleSequentialAgent
from lightweight_heuristic_integration_environment import LightweightHeuristicEnvironment
from new_heuristic_environment import NewHeuristicEnvironment
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


def _select_ppo_environment_type(model_name: str = "PPO模型") -> Optional[str]:
    """交互式选择PPO环境类型"""
    print(f"🏗️ 为{model_name}选择训练环境类型...")
    
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
            choice = input(f"请选择{model_name}的环境类型 (1-{len(environments)}) 或输入 'q' 退出: ").strip()
            
            if choice.lower() == 'q':
                print("❌ 用户取消选择")
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(environments):
                selected_env = environments[choice_idx]
                print(f"✅ 为{model_name}选择了环境类型: {selected_env[0]} ({selected_env[1]})")
                return selected_env[0]
            else:
                print(f"❌ 无效选择，请输入 1-{len(environments)} 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n❌ 用户中断选择")
            return None


def _load_ppo_agent_and_configs(ckpt_path: str) -> Tuple[SimpleSequentialAgent, Dict, Dict, Dict]:
    """加载PPO Agent与配置，返回额外的实验信息。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔄 加载PPO检查点: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'config' not in checkpoint:
        raise KeyError("Checkpoint missing 'config' dict. 请使用 train_sequential_original.py 或 train_lightweight_ppo.py 产生的检查点。")

    env_config = checkpoint['config'].get('env_config', {})
    agent_config = checkpoint['config'].get('agent_config', {})
    experiment_info = checkpoint.get('experiment_info', {})

    # 创建Agent并载入权重
    agent = SimpleSequentialAgent(**agent_config)
    state = checkpoint.get('agent_state_dict', None)
    if state is None:
        raise KeyError("Checkpoint missing 'agent_state_dict'.")
    agent.load_state_dict(state)
    agent.eval()

    print("✅ PPO Agent与配置加载完成")
    return agent, env_config, agent_config, experiment_info


def _select_ppo_checkpoint(prompt: str = "请选择PPO checkpoint目录") -> Optional[str]:
    """交互式选择PPO checkpoint目录"""
    print(f"🔍 查找可用的PPO checkpoint目录...")
    
    try:
        ppo_checkpoints_dir = "checkpoints"
        directories = _list_checkpoint_directories(ppo_checkpoints_dir)
        
        print(f"\n📁 找到 {len(directories)} 个PPO checkpoint目录:")
        print("-" * 80)
        
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
                
                # 尝试获取算法类型信息
                config_file = os.path.join(dir_path, "experiment_config.json")
                algorithm_type = "Unknown"
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                        algorithm_type = config_data.get('algorithm_type', 'Unknown')
                    except:
                        pass
                
                info = f"({file_count} .pt files, {algorithm_type}"
                if latest_file:
                    info += f", latest: {latest_file}"
                info += ")"
                
            except Exception as e:
                info = "(无法读取目录信息)"
            
            print(f"  {i:2d}. {dir_name}")
            print(f"      {info}")
        
        print("-" * 80)
        
        while True:
            try:
                choice = input(f"{prompt} (1-{len(directories)}) 或输入 'q' 退出: ").strip()
                
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


def _make_ppo_env_with_virtual_nodes(env_config: Dict, num_virtual_nodes: int, seed: int, env_type: str = "Sequential"):
    """
    基于给定配置、虚拟节点数、种子和环境类型构建PPO环境。
    
    Args:
        env_config: 环境配置字典
        num_virtual_nodes: 虚拟节点数量
        seed: 随机种子
        env_type: 环境类型 ("Sequential", "Lightweight", "NewHeuristic")
    
    Returns:
        相应类型的环境实例
    """
    cfg = dict(env_config)
    cfg['seed'] = seed
    
    # 设置固定的虚拟节点数量
    cfg['virtual_nodes_range'] = (num_virtual_nodes, num_virtual_nodes)
    cfg['max_virtual_nodes'] = max(num_virtual_nodes, cfg.get('max_virtual_nodes', 8))
    
    # 禁用课程学习以确保稳定性
    cfg['curriculum_enabled'] = False
    
    # 根据环境类型创建相应的环境
    if env_type == "Sequential":
        # 过滤掉轻量级PPO特有的参数，这些参数原始环境不支持
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


def _integrate_original_reward_after_reset(env: SequentialNetworkSchedulerEnvironment):
    """在环境reset后集成原始奖励计算器"""
    if hasattr(env, 'network_scheduler') and env.network_scheduler is not None:
        try:
            integrate_with_network_scheduler(env.network_scheduler)
            return True
        except Exception as e:
            print(f"Warning: Failed to integrate original reward calculator: {e}")
            return False
    return False


def _select_action_greedy_with_fallback(env: SequentialNetworkSchedulerEnvironment,
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
        num_actions = state.get('num_physical_nodes', env.num_physical_nodes)
        candidates = np.argsort(-logits[:num_actions])  # 降序
        # 逐个尝试候选动作，选择第一个有效动作
        for a in candidates:
            ok, _ = env._validate_mapping_action(state.get('current_virtual_node', 0), int(a))
            if ok:
                return int(a)
        # 若均无效，返回得分最高者（让环境给出惩罚）
        return int(candidates[0]) if len(candidates) > 0 else 0
    else:
        num_actions = env.bandwidth_levels
        candidates = np.argsort(-logits[:num_actions])
        # 带宽阶段也尝试从高到低选第一个有效
        for a in candidates:
            ok, _ = env._validate_bandwidth_action(state.get('current_link_index', 0), int(a))
            if ok:
                return int(a)
        return int(candidates[0]) if len(candidates) > 0 else 0


def _run_ppo_episode(env: SequentialNetworkSchedulerEnvironment,
                     agent: SimpleSequentialAgent,
                     agent_name: str = "PPO",
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
        'num_virtual_nodes': state.get('num_virtual_nodes', 0),
        'num_virtual_links': state.get('num_virtual_links', 0),
        'algorithm': agent_name,
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
    all_nodes_mapped = all(node != -1 for node in (env.partial_mapping or []))
    success = bool(all_nodes_mapped and total_reward > 0)
    
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
            print(f"Warning: Failed to calculate L and D_BW for {agent_name}: {e}")
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward,
        'load_balance_degree': load_balance_degree,
        'bandwidth_satisfaction': bandwidth_satisfaction
    })
    
    return total_reward, success, episode_info


def compare_two_ppo_models(agent1: SimpleSequentialAgent,
                           env_config1: Dict,
                           agent1_info: Dict,
                           agent2: SimpleSequentialAgent,
                           env_config2: Dict,
                           agent2_info: Dict,
                           virtual_nodes_range: List[int] = [3, 4, 5, 6, 7, 8],
                           episodes_per_node_count: int = 30,
                           base_seed: int = 42,
                           temperature: float = 0.05,
                           env_type1: str = "Sequential",
                           env_type2: str = "Sequential") -> Dict:
    """对比两个PPO模型的性能"""
    
    # 确定模型名称
    agent1_name = agent1_info.get('experiment_name', 'PPO_Model_1')
    agent2_name = agent2_info.get('experiment_name', 'PPO_Model_2')
    
    # 识别算法类型
    def _get_algorithm_type(info, config):
        # 从实验信息中获取
        if 'algorithm_type' in info:
            return info['algorithm_type']
        # 从配置中推断
        if 'heuristic_reward_weight' in config:
            return 'lightweight_heuristic_ppo'
        return 'sequential_ppo'
    
    agent1_type = _get_algorithm_type(agent1_info, env_config1)
    agent2_type = _get_algorithm_type(agent2_info, env_config2)
    
    print(f"🧪 开始对比两个PPO模型性能")
    print(f"🎯 模型1: {agent1_name} ({agent1_type}) -> 环境: {env_type1}")
    print(f"🎯 模型2: {agent2_name} ({agent2_type}) -> 环境: {env_type2}")
    print(f"📊 虚拟节点范围: {virtual_nodes_range}")
    print(f"📋 每个节点数量测试 {episodes_per_node_count} 个Episode")
    
    _set_global_seed(base_seed)
    
    results = {
        'model1': {
            'name': agent1_name,
            'type': agent1_type,
            'env_type': env_type1,
            'experiment_info': agent1_info
        },
        'model2': {
            'name': agent2_name,
            'type': agent2_type,
            'env_type': env_type2,
            'experiment_info': agent2_info
        },
        'test_config': {
            'virtual_nodes_range': virtual_nodes_range,
            'episodes_per_node_count': episodes_per_node_count,
            'base_seed': base_seed,
            'temperature': temperature
        },
        'detailed_results': {},
        'summary': {}
    }
    
    for num_virtual_nodes in virtual_nodes_range:
        print(f"\n🔍 测试 {num_virtual_nodes} 个虚拟节点...")
        
        # 存储两个模型的结果
        model1_rewards, model1_success, model1_steps = [], [], []
        model2_rewards, model2_success, model2_steps = [], [], []
        
        # 存储L和D_BW指标
        model1_l_values, model1_dbw_values = [], []
        model2_l_values, model2_dbw_values = [], []
        
        # 存储详细episode信息
        model1_episodes, model2_episodes = [], []
        
        for ep in range(episodes_per_node_count):
            ep_seed = base_seed + ep + num_virtual_nodes * 1000
            
            # 创建两个相同配置的环境
            env1 = _make_ppo_env_with_virtual_nodes(env_config1, num_virtual_nodes, ep_seed, env_type1)
            env2 = _make_ppo_env_with_virtual_nodes(env_config2, num_virtual_nodes, ep_seed, env_type2)
            
            # 运行模型1
            r1, s1, info1 = _run_ppo_episode(env1, agent1, agent1_name, temperature=temperature)
            model1_rewards.append(r1)
            model1_success.append(1.0 if s1 else 0.0)
            model1_steps.append(info1['steps'])
            model1_l_values.append(info1.get('load_balance_degree', float('nan')))
            model1_dbw_values.append(info1.get('bandwidth_satisfaction', float('nan')))
            model1_episodes.append(info1)
            
            # 运行模型2
            r2, s2, info2 = _run_ppo_episode(env2, agent2, agent2_name, temperature=temperature)
            model2_rewards.append(r2)
            model2_success.append(1.0 if s2 else 0.0)
            model2_steps.append(info2['steps'])
            model2_l_values.append(info2.get('load_balance_degree', float('nan')))
            model2_dbw_values.append(info2.get('bandwidth_satisfaction', float('nan')))
            model2_episodes.append(info2)
            
            if (ep + 1) % 10 == 0:
                print(f"   完成 {ep + 1}/{episodes_per_node_count} episodes")
        
        # 统计结果
        def _stats(arr):
            return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
        
        def _stats_with_nan(arr):
            """计算包含NaN值的数组统计信息"""
            valid_arr = [x for x in arr if not np.isnan(x)]
            if len(valid_arr) == 0:
                return float('nan'), float('nan'), float('nan'), float('nan')
            return float(np.mean(valid_arr)), float(np.std(valid_arr)), float(np.min(valid_arr)), float(np.max(valid_arr))
        
        # 计算基础统计
        m1_reward_mean, m1_reward_std, m1_reward_min, m1_reward_max = _stats(model1_rewards)
        m2_reward_mean, m2_reward_std, m2_reward_min, m2_reward_max = _stats(model2_rewards)
        
        m1_l_mean, m1_l_std, m1_l_min, m1_l_max = _stats_with_nan(model1_l_values)
        m2_l_mean, m2_l_std, m2_l_min, m2_l_max = _stats_with_nan(model2_l_values)
        
        m1_dbw_mean, m1_dbw_std, m1_dbw_min, m1_dbw_max = _stats_with_nan(model1_dbw_values)
        m2_dbw_mean, m2_dbw_std, m2_dbw_min, m2_dbw_max = _stats_with_nan(model2_dbw_values)
        
        m1_sr = float(np.mean(model1_success))
        m2_sr = float(np.mean(model2_success))
        
        m1_avg_steps = float(np.mean(model1_steps))
        m2_avg_steps = float(np.mean(model2_steps))
        
        # 计算综合指标（与四算法测试保持一致）
        def _calculate_composite_score(l_val, dbw_val, l_weight=0.5, dbw_weight=0.5):
            """计算综合指标，处理NaN值"""
            if np.isnan(l_val) or np.isnan(dbw_val):
                return float('nan')
            normalized_l = max(0, 1 - l_val)
            return l_weight * normalized_l + dbw_weight * dbw_val
        
        def _calculate_composite_std(l_std, dbw_std, l_weight=0.5, dbw_weight=0.5):
            """计算综合指标标准差的近似值"""
            if np.isnan(l_std) or np.isnan(dbw_std):
                return float('nan')
            return np.sqrt((l_weight * l_std) ** 2 + (dbw_weight * dbw_std) ** 2)
        
        # 计算综合分数
        m1_composite_values = [_calculate_composite_score(model1_l_values[i], model1_dbw_values[i]) for i in range(len(model1_l_values))]
        m2_composite_values = [_calculate_composite_score(model2_l_values[i], model2_dbw_values[i]) for i in range(len(model2_l_values))]
        
        m1_composite_mean, m1_composite_std, m1_composite_min, m1_composite_max = _stats_with_nan(m1_composite_values)
        m2_composite_mean, m2_composite_std, m2_composite_min, m2_composite_max = _stats_with_nan(m2_composite_values)
        
        # 计算胜负关系
        reward_wins = sum(1 for r1, r2 in zip(model1_rewards, model2_rewards) if r1 > r2)
        reward_ties = sum(1 for r1, r2 in zip(model1_rewards, model2_rewards) if abs(r1 - r2) < 1e-6)
        success_wins = sum(1 for s1, s2 in zip(model1_success, model2_success) if s1 > s2)
        success_ties = sum(1 for s1, s2 in zip(model1_success, model2_success) if s1 == s2)
        
        # 统计分析
        from scipy import stats as scipy_stats
        
        # t检验比较奖励差异
        t_stat, p_value = scipy_stats.ttest_rel(model1_rewards, model2_rewards)
        
        # Wilcoxon符号秩检验（非参数）
        try:
            wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(model1_rewards, model2_rewards)
        except:
            wilcoxon_stat, wilcoxon_p = float('nan'), float('nan')
        
        node_result = {
            'num_virtual_nodes': num_virtual_nodes,
            'model1': {
                'reward': {
                    'mean': m1_reward_mean,
                    'std': m1_reward_std,
                    'min': m1_reward_min,
                    'max': m1_reward_max
                },
                'success_rate': m1_sr,
                'avg_steps': m1_avg_steps,
                'load_balance': {
                    'mean': m1_l_mean,
                    'std': m1_l_std,
                    'min': m1_l_min,
                    'max': m1_l_max
                },
                'bandwidth_satisfaction': {
                    'mean': m1_dbw_mean,
                    'std': m1_dbw_std,
                    'min': m1_dbw_min,
                    'max': m1_dbw_max
                },
                'composite_score': {
                    'mean': m1_composite_mean,
                    'std': m1_composite_std,
                    'min': m1_composite_min,
                    'max': m1_composite_max
                },
                'episodes': model1_episodes
            },
            'model2': {
                'reward': {
                    'mean': m2_reward_mean,
                    'std': m2_reward_std,
                    'min': m2_reward_min,
                    'max': m2_reward_max
                },
                'success_rate': m2_sr,
                'avg_steps': m2_avg_steps,
                'load_balance': {
                    'mean': m2_l_mean,
                    'std': m2_l_std,
                    'min': m2_l_min,
                    'max': m2_l_max
                },
                'bandwidth_satisfaction': {
                    'mean': m2_dbw_mean,
                    'std': m2_dbw_std,
                    'min': m2_dbw_min,
                    'max': m2_dbw_max
                },
                'composite_score': {
                    'mean': m2_composite_mean,
                    'std': m2_composite_std,
                    'min': m2_composite_min,
                    'max': m2_composite_max
                },
                'episodes': model2_episodes
            },
            'comparison': {
                'reward_difference': m1_reward_mean - m2_reward_mean,
                'success_rate_difference': m1_sr - m2_sr,
                'steps_difference': m1_avg_steps - m2_avg_steps,
                'model1_reward_wins': reward_wins,
                'model1_reward_win_rate': reward_wins / episodes_per_node_count,
                'reward_ties': reward_ties,
                'model1_success_wins': success_wins,
                'model1_success_win_rate': success_wins / episodes_per_node_count,
                'success_ties': success_ties,
                'statistical_tests': {
                    't_test': {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': float(p_value) < 0.05
                    },
                    'wilcoxon': {
                        'statistic': float(wilcoxon_stat),
                        'p_value': float(wilcoxon_p),
                        'significant': float(wilcoxon_p) < 0.05
                    }
                }
            }
        }
        
        results['detailed_results'][num_virtual_nodes] = node_result
        
        print(f"✅ {num_virtual_nodes} 虚拟节点测试完成:")
        print(f"   {agent1_name:15} -> 奖励: {m1_reward_mean:.3f}±{m1_reward_std:.3f}, 成功率: {m1_sr:.1%}, 平均步数: {m1_avg_steps:.1f}")
        print(f"                     -> L: {m1_l_mean:.4f}±{m1_l_std:.4f}, D_BW: {m1_dbw_mean:.4f}±{m1_dbw_std:.4f}")
        print(f"   {agent2_name:15} -> 奖励: {m2_reward_mean:.3f}±{m2_reward_std:.3f}, 成功率: {m2_sr:.1%}, 平均步数: {m2_avg_steps:.1f}")
        print(f"                     -> L: {m2_l_mean:.4f}±{m2_l_std:.4f}, D_BW: {m2_dbw_mean:.4f}±{m2_dbw_std:.4f}")
        print(f"   胜负关系: {agent1_name} 胜 {reward_wins}/{episodes_per_node_count} 轮 ({reward_wins/episodes_per_node_count:.1%})")
        if p_value < 0.05:
            print(f"   统计显著性: p={p_value:.4f} (显著差异)")
        else:
            print(f"   统计显著性: p={p_value:.4f} (无显著差异)")
    
    # 生成总结
    results['summary'] = _generate_two_model_summary(results)
    
    return results


def _generate_two_model_summary(results: Dict) -> Dict:
    """生成两模型对比的总结分析"""
    detailed = results['detailed_results']
    virtual_nodes_range = results['test_config']['virtual_nodes_range']
    
    model1_name = results['model1']['name']
    model2_name = results['model2']['name']
    
    # 收集数据
    model1_rewards = [detailed[n]['model1']['reward']['mean'] for n in virtual_nodes_range]
    model2_rewards = [detailed[n]['model2']['reward']['mean'] for n in virtual_nodes_range]
    
    model1_success_rates = [detailed[n]['model1']['success_rate'] for n in virtual_nodes_range]
    model2_success_rates = [detailed[n]['model2']['success_rate'] for n in virtual_nodes_range]
    
    # 计算总体胜率
    total_wins = sum(detailed[n]['comparison']['model1_reward_wins'] for n in virtual_nodes_range)
    total_episodes = len(virtual_nodes_range) * results['test_config']['episodes_per_node_count']
    overall_win_rate = total_wins / total_episodes
    
    # 计算显著差异的节点数
    significant_nodes = sum(1 for n in virtual_nodes_range 
                           if detailed[n]['comparison']['statistical_tests']['t_test']['significant'])
    
    # 计算平均性能差异
    avg_reward_diff = np.mean([detailed[n]['comparison']['reward_difference'] for n in virtual_nodes_range])
    avg_success_diff = np.mean([detailed[n]['comparison']['success_rate_difference'] for n in virtual_nodes_range])
    
    summary = {
        'overall_performance': {
            'model1': {
                'avg_reward': float(np.mean(model1_rewards)),
                'avg_success_rate': float(np.mean(model1_success_rates)),
                'name': model1_name
            },
            'model2': {
                'avg_reward': float(np.mean(model2_rewards)),
                'avg_success_rate': float(np.mean(model2_success_rates)),
                'name': model2_name
            }
        },
        'comparison_summary': {
            'model1_overall_win_rate': overall_win_rate,
            'model1_wins_by_node_count': {n: detailed[n]['comparison']['model1_reward_wins'] 
                                         for n in virtual_nodes_range},
            'significant_difference_nodes': significant_nodes,
            'total_test_nodes': len(virtual_nodes_range),
            'avg_reward_difference': avg_reward_diff,
            'avg_success_rate_difference': avg_success_diff,
            'performance_advantage': model1_name if avg_reward_diff > 0 else model2_name,
            'advantage_magnitude': abs(avg_reward_diff)
        },
        'statistical_analysis': {
            'consistent_winner': model1_name if overall_win_rate > 0.5 else model2_name,
            'win_rate_confidence': max(overall_win_rate, 1 - overall_win_rate),
            'significant_advantage': significant_nodes > len(virtual_nodes_range) / 2
        }
    }
    
    return summary


def _save_comparison_results(results: Dict, out_dir: str = 'test_results') -> str:
    """保存对比测试结果到JSON文件"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    model1_name = results['model1']['name'].replace('/', '_')
    model2_name = results['model2']['name'].replace('/', '_')
    
    path = os.path.join(out_dir, f'two_ppo_comparison_{model1_name}_vs_{model2_name}_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        # 移除episodes详情以减小文件大小
        results_copy = dict(results)
        for n in results_copy['detailed_results']:
            results_copy['detailed_results'][n]['model1'].pop('episodes', None)
            results_copy['detailed_results'][n]['model2'].pop('episodes', None)
        
        json.dump(results_copy, f, indent=2, ensure_ascii=False)
    return path


def _create_two_model_visualization(results: Dict, out_dir: str = 'test_results') -> List[str]:
    """创建两模型对比的可视化图表 - 与四算法测试风格保持一致"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    virtual_nodes_range = results['test_config']['virtual_nodes_range']
    detailed = results['detailed_results']
    
    model1_name = results['model1']['name']
    model2_name = results['model2']['name']
    model1_type = results['model1']['type']
    model2_type = results['model2']['type']
    model1_env_type = results['model1']['env_type']
    model2_env_type = results['model2']['env_type']
    
    # 准备数据
    model1_rewards = [detailed[n]['model1']['reward']['mean'] for n in virtual_nodes_range]
    model2_rewards = [detailed[n]['model2']['reward']['mean'] for n in virtual_nodes_range]
    model1_reward_stds = [detailed[n]['model1']['reward']['std'] for n in virtual_nodes_range]
    model2_reward_stds = [detailed[n]['model2']['reward']['std'] for n in virtual_nodes_range]
    
    model1_success_rates = [detailed[n]['model1']['success_rate'] * 100 for n in virtual_nodes_range]
    model2_success_rates = [detailed[n]['model2']['success_rate'] * 100 for n in virtual_nodes_range]
    
    model1_l_values = [detailed[n]['model1']['load_balance']['mean'] for n in virtual_nodes_range]
    model2_l_values = [detailed[n]['model2']['load_balance']['mean'] for n in virtual_nodes_range]
    model1_l_stds = [detailed[n]['model1']['load_balance']['std'] for n in virtual_nodes_range]
    model2_l_stds = [detailed[n]['model2']['load_balance']['std'] for n in virtual_nodes_range]
    
    model1_dbw_values = [detailed[n]['model1']['bandwidth_satisfaction']['mean'] for n in virtual_nodes_range]
    model2_dbw_values = [detailed[n]['model2']['bandwidth_satisfaction']['mean'] for n in virtual_nodes_range]
    model1_dbw_stds = [detailed[n]['model1']['bandwidth_satisfaction']['std'] for n in virtual_nodes_range]
    model2_dbw_stds = [detailed[n]['model2']['bandwidth_satisfaction']['std'] for n in virtual_nodes_range]
    
    # 计算综合指标 = 0.5*(1-L) + 0.5*D_BW（与四算法测试保持一致）
    def _calculate_composite_score(l_val, dbw_val, l_weight=0.5, dbw_weight=0.5):
        """计算综合指标，处理NaN值"""
        if np.isnan(l_val) or np.isnan(dbw_val):
            return float('nan')
        # 由于L越小越好，我们使用(1-L)来转换为越大越好
        # 假设L的正常范围在[0,1]，如果超出可能需要normalization
        normalized_l = max(0, 1 - l_val)  # 转换为越大越好
        return l_weight * normalized_l + dbw_weight * dbw_val
    
    model1_composite_values = [_calculate_composite_score(model1_l_values[i], model1_dbw_values[i]) for i in range(len(virtual_nodes_range))]
    model2_composite_values = [_calculate_composite_score(model2_l_values[i], model2_dbw_values[i]) for i in range(len(virtual_nodes_range))]
    
    # 计算综合指标的标准差（简化处理，使用L和D_BW标准差的加权组合）
    def _calculate_composite_std(l_std, dbw_std, l_weight=0.5, dbw_weight=0.5):
        """计算综合指标标准差的近似值"""
        if np.isnan(l_std) or np.isnan(dbw_std):
            return float('nan')
        return np.sqrt((l_weight * l_std) ** 2 + (dbw_weight * dbw_std) ** 2)
    
    model1_composite_stds = [_calculate_composite_std(model1_l_stds[i], model1_dbw_stds[i]) for i in range(len(virtual_nodes_range))]
    model2_composite_stds = [_calculate_composite_std(model2_l_stds[i], model2_dbw_stds[i]) for i in range(len(virtual_nodes_range))]
    
    saved_files = []
    
    # 设置图表样式 - 与四算法测试保持一致
    plt.style.use('seaborn-v0_8')
    colors = ['#2E86AB', '#A23B72']  # Model1, Model2（使用四算法的前两个颜色）
    
    # 图1: 奖励和成功率对比 - 采用四算法的(1,2)布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 奖励曲线
    ax1.errorbar(virtual_nodes_range, model1_rewards, yerr=model1_reward_stds, 
                label=f'{model1_name} ({model1_type}, {model1_env_type})', marker='o', linewidth=2, capsize=5, color=colors[0])
    ax1.errorbar(virtual_nodes_range, model2_rewards, yerr=model2_reward_stds,
                label=f'{model2_name} ({model2_type}, {model2_env_type})', marker='s', linewidth=2, capsize=5, color=colors[1])
    ax1.set_xlabel('Number of Tasks')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward Comparison')
    ax1.set_xticks(virtual_nodes_range)  # 只显示整数刻度
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 成功率对比
    ax2.plot(virtual_nodes_range, model1_success_rates, label=f'{model1_name} ({model1_type}, {model1_env_type})', 
            marker='o', linewidth=2, color=colors[0])
    ax2.plot(virtual_nodes_range, model2_success_rates, label=f'{model2_name} ({model2_type}, {model2_env_type})', 
            marker='s', linewidth=2, color=colors[1])
    ax2.set_xlabel('Number of Tasks')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate Comparison')
    ax2.set_xticks(virtual_nodes_range)  # 只显示整数刻度
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_comparison_path = os.path.join(out_dir, f'two_ppo_performance_{ts}.png')
    plt.savefig(performance_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(performance_comparison_path)
    
    # 图2: 负载均衡度(L)和带宽满意度(D_BW)对比 - 采用四算法的(1,3)布局
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 负载均衡度L对比 (越小越好) - 处理包含NaN值的绘图
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
    
    _plot_with_nan_handling(ax1, virtual_nodes_range, model1_l_values, model1_l_stds, 
                           f'{model1_name} ({model1_type}, {model1_env_type})', 'o', colors[0])
    _plot_with_nan_handling(ax1, virtual_nodes_range, model2_l_values, model2_l_stds,
                           f'{model2_name} ({model2_type}, {model2_env_type})', 's', colors[1])
    ax1.set_xlabel('Number of Tasks')
    ax1.set_ylabel('Load Balance Degree (L)')
    ax1.set_title('Load Balance Degree Comparison\n(Lower is Better)')
    ax1.set_xticks(virtual_nodes_range)  # 只显示整数刻度
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 带宽满意度D_BW对比 (越大越好)
    _plot_with_nan_handling(ax2, virtual_nodes_range, model1_dbw_values, model1_dbw_stds,
                           f'{model1_name} ({model1_type}, {model1_env_type})', 'o', colors[0])
    _plot_with_nan_handling(ax2, virtual_nodes_range, model2_dbw_values, model2_dbw_stds,
                           f'{model2_name} ({model2_type}, {model2_env_type})', 's', colors[1])
    ax2.set_xlabel('Number of Tasks')
    ax2.set_ylabel('Bandwidth Satisfaction (D_BW)')
    ax2.set_title('Bandwidth Satisfaction Comparison\n(Higher is Better)')
    ax2.set_xticks(virtual_nodes_range)  # 只显示整数刻度
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 综合指标对比 (越大越好) - 新增，与四算法保持一致
    _plot_with_nan_handling(ax3, virtual_nodes_range, model1_composite_values, model1_composite_stds,
                           f'{model1_name} ({model1_type}, {model1_env_type})', 'o', colors[0])
    _plot_with_nan_handling(ax3, virtual_nodes_range, model2_composite_values, model2_composite_stds,
                           f'{model2_name} ({model2_type}, {model2_env_type})', 's', colors[1])
    ax3.set_xlabel('Number of Tasks')
    ax3.set_ylabel('Composite Score')
    ax3.set_title('Composite Score Comparison\n(0.5×(1-L) + 0.5×D_BW, Higher is Better)')
    ax3.set_xticks(virtual_nodes_range)  # 只显示整数刻度
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    lb_bw_comparison_path = os.path.join(out_dir, f'two_ppo_lb_bw_{ts}.png')
    plt.savefig(lb_bw_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(lb_bw_comparison_path)
    
    return saved_files


def print_two_model_results(results: Dict):
    """打印两模型对比的详细结果"""
    model1_name = results['model1']['name']
    model2_name = results['model2']['name']
    model1_type = results['model1']['type']
    model2_type = results['model2']['type']
    model1_env_type = results['model1']['env_type']
    model2_env_type = results['model2']['env_type']
    
    print("\n" + "="*100)
    print("📊 两个PPO模型性能对比详细结果")
    print("="*100)
    
    # 模型信息
    print(f"🔧 模型信息:")
    print(f"   模型1: {model1_name} ({model1_type}) -> 环境: {model1_env_type}")
    print(f"   模型2: {model2_name} ({model2_type}) -> 环境: {model2_env_type}")
    print(f"   虚拟节点范围: {results['test_config']['virtual_nodes_range']}")
    print(f"   每个数量测试: {results['test_config']['episodes_per_node_count']} episodes")
    
    # 详细结果
    print(f"\n📋 详细结果:")
    print("-"*120)
    header = f"{'节点数':<6} {model1_name+'奖励':<15} {model2_name+'奖励':<15} {'奖励差异':<10} {model1_name+'胜率':<12} {'p值':<10} {'显著性':<8}"
    print(header)
    print("-"*120)
    
    for num_nodes in results['test_config']['virtual_nodes_range']:
        detail = results['detailed_results'][num_nodes]
        m1 = detail['model1']
        m2 = detail['model2']
        comp = detail['comparison']
        
        p_value = comp['statistical_tests']['t_test']['p_value']
        significant = "显著" if p_value < 0.05 else "不显著"
        
        row = (f"{num_nodes:<6} "
               f"{m1['reward']['mean']:<15.3f} "
               f"{m2['reward']['mean']:<15.3f} "
               f"{comp['reward_difference']:<10.3f} "
               f"{comp['model1_reward_win_rate']:<12.1%} "
               f"{p_value:<10.4f} "
               f"{significant:<8}")
        print(row)
        
        # 成功率和L、D_BW信息
        success_row = (f"{'成功率':<6} "
                      f"{m1['success_rate']:<15.1%} "
                      f"{m2['success_rate']:<15.1%} "
                      f"{comp['success_rate_difference']:<10.3f}")
        print(f"       {success_row}")
        
        def _format_metric(val, std_val):
            if np.isnan(val) or np.isnan(std_val):
                return "N/A"
            return f"{val:.4f}±{std_val:.4f}"
        
        m1_l_str = _format_metric(m1['load_balance']['mean'], m1['load_balance']['std'])
        m2_l_str = _format_metric(m2['load_balance']['mean'], m2['load_balance']['std'])
        m1_dbw_str = _format_metric(m1['bandwidth_satisfaction']['mean'], m1['bandwidth_satisfaction']['std'])
        m2_dbw_str = _format_metric(m2['bandwidth_satisfaction']['mean'], m2['bandwidth_satisfaction']['std'])
        
        m1_comp_str = _format_metric(m1['composite_score']['mean'], m1['composite_score']['std'])
        m2_comp_str = _format_metric(m2['composite_score']['mean'], m2['composite_score']['std'])
        
        l_row = f"{'L值':<6} {m1_l_str:<15} {m2_l_str:<15}"
        dbw_row = f"{'D_BW值':<6} {m1_dbw_str:<15} {m2_dbw_str:<15}"
        comp_row = f"{'综合分数':<6} {m1_comp_str:<15} {m2_comp_str:<15}"
        
        print(f"       {l_row}")
        print(f"       {dbw_row}")
        print(f"       {comp_row}")
        print()
    
    # 总结分析
    summary = results['summary']
    print(f"\n📈 总结分析:")
    print("-"*80)
    overall = summary['overall_performance']
    comp_summary = summary['comparison_summary']
    
    print(f"{model1_name}平均性能 -> 奖励: {overall['model1']['avg_reward']:.3f}, 成功率: {overall['model1']['avg_success_rate']:.1%}")
    print(f"{model2_name}平均性能 -> 奖励: {overall['model2']['avg_reward']:.3f}, 成功率: {overall['model2']['avg_success_rate']:.1%}")
    
    print(f"\n🏆 对比结果:")
    print(f"总体胜率: {model1_name} 胜 {comp_summary['model1_overall_win_rate']:.1%}")
    print(f"平均奖励差异: {comp_summary['avg_reward_difference']:+.3f}")
    print(f"平均成功率差异: {comp_summary['avg_success_rate_difference']:+.1%}")
    print(f"统计显著的节点数: {comp_summary['significant_difference_nodes']}/{comp_summary['total_test_nodes']}")
    print(f"性能优势: {comp_summary['performance_advantage']} (优势幅度: {comp_summary['advantage_magnitude']:.3f})")
    
    stat_analysis = summary['statistical_analysis']
    print(f"\n📊 统计分析:")
    print(f"一致性胜者: {stat_analysis['consistent_winner']}")
    print(f"胜率置信度: {stat_analysis['win_rate_confidence']:.1%}")
    print(f"是否有显著优势: {'是' if stat_analysis['significant_advantage'] else '否'}")


def main():
    print("🧪 两个PPO模型性能对比测试")
    print("="*60)

    try:
        # 交互式选择两个模型
        print("步骤1: 选择第一个PPO模型")
        ppo_ckpt1 = _select_ppo_checkpoint("请选择第一个PPO模型")
        if ppo_ckpt1 is None:
            print("❌ 未选择第一个PPO模型，程序退出")
            return
        
        print("\n步骤2: 选择第一个PPO模型的环境类型")
        env_type1 = _select_ppo_environment_type("第一个PPO模型")
        if env_type1 is None:
            print("❌ 未选择第一个PPO模型的环境类型，程序退出")
            return
        
        print("\n步骤3: 选择第二个PPO模型")
        ppo_ckpt2 = _select_ppo_checkpoint("请选择第二个PPO模型")
        if ppo_ckpt2 is None:
            print("❌ 未选择第二个PPO模型，程序退出")
            return
        
        print("\n步骤4: 选择第二个PPO模型的环境类型")
        env_type2 = _select_ppo_environment_type("第二个PPO模型")
        if env_type2 is None:
            print("❌ 未选择第二个PPO模型的环境类型，程序退出")
            return
        
        # 加载模型
        print(f"\n📂 加载模型...")
        agent1, env_config1, agent_config1, exp_info1 = _load_ppo_agent_and_configs(ppo_ckpt1)
        agent2, env_config2, agent_config2, exp_info2 = _load_ppo_agent_and_configs(ppo_ckpt2)

        # 测试参数
        virtual_nodes_range = [3, 4, 5, 6, 7, 8]  # 测试3-8个虚拟节点
        episodes_per_node_count = 30  # 每个节点数量测试30次
        base_seed = 42
        temperature = 0.05  # 低温度，更贪心

        # 执行对比评测
        results = compare_two_ppo_models(
            agent1=agent1,
            env_config1=env_config1,
            agent1_info=exp_info1,
            agent2=agent2,
            env_config2=env_config2,
            agent2_info=exp_info2,
            virtual_nodes_range=virtual_nodes_range,
            episodes_per_node_count=episodes_per_node_count,
            base_seed=base_seed,
            temperature=temperature,
            env_type1=env_type1,
            env_type2=env_type2
        )

        # 打印结果
        print_two_model_results(results)

        # 保存结果
        json_path = _save_comparison_results(results)
        print(f"\n💾 详细结果已保存到: {json_path}")

        # 创建可视化
        try:
            vis_files = _create_two_model_visualization(results)
            print(f"📊 可视化图表已保存:")
            for f in vis_files:
                print(f"   - {f}")
        except Exception as e:
            print(f"⚠️ 可视化创建失败: {e}")
            print("💡 提示: 确保安装了 matplotlib, seaborn 和 scipy")

        # 最终结论
        summary = results['summary']
        comp_summary = summary['comparison_summary']
        
        print(f"\n🎯 最终结论:")
        print("-"*50)
        
        if comp_summary['model1_overall_win_rate'] > 0.6:
            print(f"🥇 {results['model1']['name']} 明显优于 {results['model2']['name']}!")
        elif comp_summary['model1_overall_win_rate'] < 0.4:
            print(f"🥇 {results['model2']['name']} 明显优于 {results['model1']['name']}!")
        else:
            print(f"🤝 两个模型性能相当，无明显差异")
        
        print(f"📊 关键指标:")
        print(f"   总体胜率: {comp_summary['model1_overall_win_rate']:.1%}")
        print(f"   平均奖励差异: {comp_summary['avg_reward_difference']:+.3f}")
        print(f"   统计显著性: {comp_summary['significant_difference_nodes']}/{comp_summary['total_test_nodes']} 个节点数")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 需要安装scipy用于统计检验
    try:
        import scipy.stats
    except ImportError:
        print("⚠️ 警告: 未安装scipy库，统计检验功能将不可用")
        print("请运行: pip install scipy")
    
    main()
