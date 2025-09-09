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

from sequential_environment import SequentialNetworkSchedulerEnvironment
from sequential_agent import SimpleSequentialAgent
from heuristic_algorithm import create_heuristic_agent, run_heuristic_episode
from original_reward import integrate_with_network_scheduler


def _set_global_seed(seed: int):
    """统一设置全局随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_latest_checkpoint(checkpoints_dir: str) -> str:
    """在给定目录中查找最新的 original_sequential_ppo_episode_*.pt 文件。"""
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    pattern = re.compile(r"^original_sequential_ppo_episode_(\d+)\.pt$")
    candidates = []
    for fname in os.listdir(checkpoints_dir):
        m = pattern.match(fname)
        if m:
            candidates.append((int(m.group(1)), os.path.join(checkpoints_dir, fname)))

    if not candidates:
        raise FileNotFoundError("No checkpoint matching 'original_sequential_ppo_episode_*.pt' found in checkpoints/")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_agent_and_configs(ckpt_path: str) -> Tuple[SimpleSequentialAgent, Dict, Dict]:
    """加载Agent与配置。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔄 加载检查点: {ckpt_path}")
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

    print("✅ Agent与配置加载完成")
    return agent, env_config, agent_config


def _make_env_with_virtual_nodes(env_config: Dict, num_virtual_nodes: int, seed: int) -> SequentialNetworkSchedulerEnvironment:
    """基于给定配置、虚拟节点数和种子构建环境。"""
    cfg = dict(env_config)
    cfg['seed'] = seed
    
    # 设置固定的虚拟节点数量
    cfg['virtual_nodes_range'] = (num_virtual_nodes, num_virtual_nodes)
    cfg['max_virtual_nodes'] = max(num_virtual_nodes, cfg.get('max_virtual_nodes', 8))
    
    # 禁用课程学习以确保稳定性
    cfg['curriculum_enabled'] = False
    
    env = SequentialNetworkSchedulerEnvironment(**cfg)
    
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
    """贪心策略选择动作，带回退机制。"""
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
            print(f"Warning: Failed to calculate L and D_BW for PPO: {e}")
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward,
        'load_balance_degree': load_balance_degree,
        'bandwidth_satisfaction': bandwidth_satisfaction
    })
    
    return total_reward, success, episode_info


def _run_random_episode(env: SequentialNetworkSchedulerEnvironment,
                        max_steps: int = 50) -> Tuple[float, bool, Dict]:
    """运行随机算法的Episode"""
    state = env.reset()
    
    # 在reset后集成原始奖励计算器
    _integrate_original_reward_after_reset(env)
    
    done = False
    total_reward = 0.0
    steps = 0
    
    episode_info = {
        'num_virtual_nodes': state.get('num_virtual_nodes', 0),
        'num_virtual_links': state.get('num_virtual_links', 0),
        'algorithm': 'Random',
        'steps': 0
    }

    while not done and steps < max_steps:
        # 随机策略
        if state.get('mapping_phase', True):
            action = np.random.randint(0, state.get('num_physical_nodes', 1))
        else:
            action = np.random.randint(0, env.bandwidth_levels)

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
            print(f"Warning: Failed to calculate L and D_BW for Random: {e}")
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward,
        'load_balance_degree': load_balance_degree,
        'bandwidth_satisfaction': bandwidth_satisfaction
    })
    
    return total_reward, success, episode_info


def evaluate_three_algorithms(agent: SimpleSequentialAgent,
                              env_config: Dict,
                              virtual_nodes_range: List[int] = [3, 4, 5, 6, 7, 8],
                              episodes_per_node_count: int = 30,
                              base_seed: int = 42,
                              temperature: float = 0.05,
                              heuristic_type: str = "smart") -> Dict:
    """评测三种算法在不同虚拟节点数量下的性能"""
    print(f"🧪 开始评测三种算法性能对比")
    print(f"🎯 算法类型: PPO, Heuristic, Random")
    print(f"📊 虚拟节点范围: {virtual_nodes_range}")
    print(f"📋 每个节点数量测试 {episodes_per_node_count} 个Episode")
    
    _set_global_seed(base_seed)
    
    results = {
        'algorithms': ['PPO', 'Heuristic', 'Random'],
        'virtual_nodes_range': virtual_nodes_range,
        'episodes_per_node_count': episodes_per_node_count,
        'base_seed': base_seed,
        'temperature': temperature,
        'heuristic_type': heuristic_type,
        'detailed_results': {},
        'summary': {}
    }
    
    # 创建启发式算法代理
    heuristic_agent = create_heuristic_agent(heuristic_type)
    
    for num_virtual_nodes in virtual_nodes_range:
        print(f"\n🔍 测试 {num_virtual_nodes} 个虚拟节点...")
        
        # 存储三种算法的结果
        ppo_rewards, ppo_success, ppo_steps = [], [], []
        heu_rewards, heu_success, heu_steps = [], [], []
        rnd_rewards, rnd_success, rnd_steps = [], [], []
        
        # 存储L和D_BW指标
        ppo_l_values, ppo_dbw_values = [], []
        heu_l_values, heu_dbw_values = [], []
        rnd_l_values, rnd_dbw_values = [], []
        
        for ep in range(episodes_per_node_count):
            ep_seed = base_seed + ep + num_virtual_nodes * 1000
            
            # 创建三个相同的环境
            env_ppo = _make_env_with_virtual_nodes(env_config, num_virtual_nodes, ep_seed)
            env_heu = _make_env_with_virtual_nodes(env_config, num_virtual_nodes, ep_seed)
            env_rnd = _make_env_with_virtual_nodes(env_config, num_virtual_nodes, ep_seed)
            
            # 运行PPO
            r_ppo, s_ppo, info_ppo = _run_ppo_episode(env_ppo, agent, temperature=temperature)
            ppo_rewards.append(r_ppo)
            ppo_success.append(1.0 if s_ppo else 0.0)
            ppo_steps.append(info_ppo['steps'])
            ppo_l_values.append(info_ppo.get('load_balance_degree', float('nan')))
            ppo_dbw_values.append(info_ppo.get('bandwidth_satisfaction', float('nan')))
            
            # 运行启发式算法
            r_heu, s_heu, info_heu = run_heuristic_episode(env_heu, heuristic_agent)
            heu_rewards.append(r_heu)
            heu_success.append(1.0 if s_heu else 0.0)
            heu_steps.append(info_heu['steps'])
            heu_l_values.append(info_heu.get('load_balance_degree', float('nan')))
            heu_dbw_values.append(info_heu.get('bandwidth_satisfaction', float('nan')))
            
            # 运行随机算法
            r_rnd, s_rnd, info_rnd = _run_random_episode(env_rnd)
            rnd_rewards.append(r_rnd)
            rnd_success.append(1.0 if s_rnd else 0.0)
            rnd_steps.append(info_rnd['steps'])
            rnd_l_values.append(info_rnd.get('load_balance_degree', float('nan')))
            rnd_dbw_values.append(info_rnd.get('bandwidth_satisfaction', float('nan')))
            
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
        
        ppo_reward_mean, ppo_reward_std, ppo_reward_min, ppo_reward_max = _stats(ppo_rewards)
        heu_reward_mean, heu_reward_std, heu_reward_min, heu_reward_max = _stats(heu_rewards)
        rnd_reward_mean, rnd_reward_std, rnd_reward_min, rnd_reward_max = _stats(rnd_rewards)
        
        # L和D_BW统计
        ppo_l_mean, ppo_l_std, ppo_l_min, ppo_l_max = _stats_with_nan(ppo_l_values)
        heu_l_mean, heu_l_std, heu_l_min, heu_l_max = _stats_with_nan(heu_l_values)
        rnd_l_mean, rnd_l_std, rnd_l_min, rnd_l_max = _stats_with_nan(rnd_l_values)
        
        ppo_dbw_mean, ppo_dbw_std, ppo_dbw_min, ppo_dbw_max = _stats_with_nan(ppo_dbw_values)
        heu_dbw_mean, heu_dbw_std, heu_dbw_min, heu_dbw_max = _stats_with_nan(heu_dbw_values)
        rnd_dbw_mean, rnd_dbw_std, rnd_dbw_min, rnd_dbw_max = _stats_with_nan(rnd_dbw_values)
        
        ppo_sr = float(np.mean(ppo_success))
        heu_sr = float(np.mean(heu_success))
        rnd_sr = float(np.mean(rnd_success))
        
        ppo_avg_steps = float(np.mean(ppo_steps))
        heu_avg_steps = float(np.mean(heu_steps))
        rnd_avg_steps = float(np.mean(rnd_steps))
        
        node_result = {
            'num_virtual_nodes': num_virtual_nodes,
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
                'ppo_vs_random': {
                    'reward_improvement': ppo_reward_mean - rnd_reward_mean,
                    'success_rate_improvement': ppo_sr - rnd_sr
                },
                'heuristic_vs_random': {
                    'reward_improvement': heu_reward_mean - rnd_reward_mean,
                    'success_rate_improvement': heu_sr - rnd_sr
                }
            }
        }
        
        results['detailed_results'][num_virtual_nodes] = node_result
        
        print(f"✅ {num_virtual_nodes} 虚拟节点测试完成:")
        print(f"   PPO        -> 奖励: {ppo_reward_mean:.3f}±{ppo_reward_std:.3f}, 成功率: {ppo_sr:.1%}, 平均步数: {ppo_avg_steps:.1f}")
        print(f"              -> L: {ppo_l_mean:.4f}±{ppo_l_std:.4f}, D_BW: {ppo_dbw_mean:.4f}±{ppo_dbw_std:.4f}")
        print(f"   Heuristic  -> 奖励: {heu_reward_mean:.3f}±{heu_reward_std:.3f}, 成功率: {heu_sr:.1%}, 平均步数: {heu_avg_steps:.1f}")
        print(f"              -> L: {heu_l_mean:.4f}±{heu_l_std:.4f}, D_BW: {heu_dbw_mean:.4f}±{heu_dbw_std:.4f}")
        print(f"   Random     -> 奖励: {rnd_reward_mean:.3f}±{rnd_reward_std:.3f}, 成功率: {rnd_sr:.1%}, 平均步数: {rnd_avg_steps:.1f}")
        print(f"              -> L: {rnd_l_mean:.4f}±{rnd_l_std:.4f}, D_BW: {rnd_dbw_mean:.4f}±{rnd_dbw_std:.4f}")
    
    # 生成总结
    results['summary'] = _generate_summary(results)
    
    return results


def _generate_summary(results: Dict) -> Dict:
    """生成测试结果的总结分析"""
    detailed = results['detailed_results']
    virtual_nodes_range = results['virtual_nodes_range']
    
    # 收集数据
    ppo_rewards = [detailed[n]['ppo']['avg_reward'] for n in virtual_nodes_range]
    heu_rewards = [detailed[n]['heuristic']['avg_reward'] for n in virtual_nodes_range]
    rnd_rewards = [detailed[n]['random']['avg_reward'] for n in virtual_nodes_range]
    
    ppo_success_rates = [detailed[n]['ppo']['success_rate'] for n in virtual_nodes_range]
    heu_success_rates = [detailed[n]['heuristic']['success_rate'] for n in virtual_nodes_range]
    rnd_success_rates = [detailed[n]['random']['success_rate'] for n in virtual_nodes_range]
    
    # 计算平均性能
    summary = {
        'overall_performance': {
            'ppo': {
                'avg_reward': float(np.mean(ppo_rewards)),
                'avg_success_rate': float(np.mean(ppo_success_rates))
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
            'ppo_vs_random': 0,
            'heuristic_vs_random': 0
        }
    }
    
    # 计算胜率
    ppo_beats_heu = sum(1 for n in virtual_nodes_range if detailed[n]['ppo']['avg_reward'] > detailed[n]['heuristic']['avg_reward'])
    ppo_beats_rnd = sum(1 for n in virtual_nodes_range if detailed[n]['ppo']['avg_reward'] > detailed[n]['random']['avg_reward'])
    heu_beats_rnd = sum(1 for n in virtual_nodes_range if detailed[n]['heuristic']['avg_reward'] > detailed[n]['random']['avg_reward'])
    
    summary['win_rates']['ppo_vs_heuristic'] = ppo_beats_heu / len(virtual_nodes_range)
    summary['win_rates']['ppo_vs_random'] = ppo_beats_rnd / len(virtual_nodes_range)
    summary['win_rates']['heuristic_vs_random'] = heu_beats_rnd / len(virtual_nodes_range)
    
    # 排名
    avg_performance = [
        ('PPO', summary['overall_performance']['ppo']['avg_reward'], summary['overall_performance']['ppo']['avg_success_rate']),
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
    path = os.path.join(out_dir, f'three_algorithms_comparison_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


def _create_visualization(results: Dict, out_dir: str = 'test_results') -> List[str]:
    """创建可视化图表"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    virtual_nodes_range = results['virtual_nodes_range']
    detailed = results['detailed_results']
    heuristic_type = results['heuristic_type'].title()
    
    # 准备数据
    ppo_rewards = [detailed[n]['ppo']['avg_reward'] for n in virtual_nodes_range]
    heu_rewards = [detailed[n]['heuristic']['avg_reward'] for n in virtual_nodes_range]
    rnd_rewards = [detailed[n]['random']['avg_reward'] for n in virtual_nodes_range]
    
    ppo_success_rates = [detailed[n]['ppo']['success_rate'] * 100 for n in virtual_nodes_range]
    heu_success_rates = [detailed[n]['heuristic']['success_rate'] * 100 for n in virtual_nodes_range]
    rnd_success_rates = [detailed[n]['random']['success_rate'] * 100 for n in virtual_nodes_range]
    
    ppo_reward_stds = [detailed[n]['ppo']['std_reward'] for n in virtual_nodes_range]
    heu_reward_stds = [detailed[n]['heuristic']['std_reward'] for n in virtual_nodes_range]
    rnd_reward_stds = [detailed[n]['random']['std_reward'] for n in virtual_nodes_range]
    
    # 准备L和D_BW数据
    ppo_l_values = [detailed[n]['ppo']['load_balance']['mean'] for n in virtual_nodes_range]
    heu_l_values = [detailed[n]['heuristic']['load_balance']['mean'] for n in virtual_nodes_range]
    rnd_l_values = [detailed[n]['random']['load_balance']['mean'] for n in virtual_nodes_range]
    
    ppo_l_stds = [detailed[n]['ppo']['load_balance']['std'] for n in virtual_nodes_range]
    heu_l_stds = [detailed[n]['heuristic']['load_balance']['std'] for n in virtual_nodes_range]
    rnd_l_stds = [detailed[n]['random']['load_balance']['std'] for n in virtual_nodes_range]
    
    ppo_dbw_values = [detailed[n]['ppo']['bandwidth_satisfaction']['mean'] for n in virtual_nodes_range]
    heu_dbw_values = [detailed[n]['heuristic']['bandwidth_satisfaction']['mean'] for n in virtual_nodes_range]
    rnd_dbw_values = [detailed[n]['random']['bandwidth_satisfaction']['mean'] for n in virtual_nodes_range]
    
    ppo_dbw_stds = [detailed[n]['ppo']['bandwidth_satisfaction']['std'] for n in virtual_nodes_range]
    heu_dbw_stds = [detailed[n]['heuristic']['bandwidth_satisfaction']['std'] for n in virtual_nodes_range]
    rnd_dbw_stds = [detailed[n]['random']['bandwidth_satisfaction']['std'] for n in virtual_nodes_range]
    
    saved_files = []
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # PPO, Heuristic, Random
    
    # 图1: 奖励和成功率对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 奖励曲线
    ax1.errorbar(virtual_nodes_range, ppo_rewards, yerr=ppo_reward_stds, 
                label='PPO', marker='o', linewidth=2, capsize=5, color=colors[0])
    ax1.errorbar(virtual_nodes_range, heu_rewards, yerr=heu_reward_stds,
                label='Heuristic', marker='s', linewidth=2, capsize=5, color=colors[1])
    ax1.errorbar(virtual_nodes_range, rnd_rewards, yerr=rnd_reward_stds,
                label='Random', marker='^', linewidth=2, capsize=5, color=colors[2])
    ax1.set_xlabel('Virtual Nodes Number')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 成功率对比
    ax2.plot(virtual_nodes_range, ppo_success_rates, label='PPO', 
            marker='o', linewidth=2, color=colors[0])
    ax2.plot(virtual_nodes_range, heu_success_rates, label=f'{heuristic_type} Heuristic', 
            marker='s', linewidth=2, color=colors[1])
    ax2.plot(virtual_nodes_range, rnd_success_rates, label='Random', 
            marker='^', linewidth=2, color=colors[2])
    ax2.set_xlabel('Virtual Nodes Number')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_comparison_path = os.path.join(out_dir, f'three_algorithms_performance_{ts}.png')
    plt.savefig(performance_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(performance_comparison_path)
    
    # 图2: 负载均衡度(L)和带宽满意度(D_BW)对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    
    _plot_with_nan_handling(ax1, virtual_nodes_range, ppo_l_values, ppo_l_stds, 
                           'PPO', 'o', colors[0])
    _plot_with_nan_handling(ax1, virtual_nodes_range, heu_l_values, heu_l_stds,
                           f'{heuristic_type} Heuristic', 's', colors[1])
    _plot_with_nan_handling(ax1, virtual_nodes_range, rnd_l_values, rnd_l_stds,
                           'Random', '^', colors[2])
    ax1.set_xlabel('Virtual Nodes Number')
    ax1.set_ylabel('Load Balance Degree (L)')
    ax1.set_title('Load Balance Degree Comparison\n(Lower is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 带宽满意度D_BW对比 (越大越好)
    _plot_with_nan_handling(ax2, virtual_nodes_range, ppo_dbw_values, ppo_dbw_stds,
                           'PPO', 'o', colors[0])
    _plot_with_nan_handling(ax2, virtual_nodes_range, heu_dbw_values, heu_dbw_stds,
                           f'{heuristic_type} Heuristic', 's', colors[1])
    _plot_with_nan_handling(ax2, virtual_nodes_range, rnd_dbw_values, rnd_dbw_stds,
                           'Random', '^', colors[2])
    ax2.set_xlabel('Virtual Nodes Number')
    ax2.set_ylabel('Bandwidth Satisfaction (D_BW)')
    ax2.set_title('Bandwidth Satisfaction Comparison\n(Higher is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    lb_bw_comparison_path = os.path.join(out_dir, f'three_algorithms_lb_bw_{ts}.png')
    plt.savefig(lb_bw_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(lb_bw_comparison_path)
    
    return saved_files


def print_detailed_results(results: Dict):
    """打印详细的测试结果"""
    print("\n" + "="*90)
    print("📊 三算法性能对比详细结果")
    print("="*90)
    
    # 测试配置
    heuristic_type = results['heuristic_type'].title()
    print(f"🔧 测试配置:")
    print(f"   算法类型: PPO, {heuristic_type} Heuristic, Random")
    print(f"   虚拟节点范围: {results['virtual_nodes_range']}")
    print(f"   每个数量测试: {results['episodes_per_node_count']} episodes")
    print(f"   基础种子: {results['base_seed']}")
    
    # 详细结果
    print(f"\n📋 详细结果:")
    print("-"*120)
    header = f"{'节点数':<6} {'PPO奖励':<10} {'启发式奖励':<12} {'随机奖励':<10} {'PPO成功率':<10} {'启发式成功率':<12} {'随机成功率':<10}"
    print(header)
    print("-"*120)
    
    for num_nodes in results['virtual_nodes_range']:
        detail = results['detailed_results'][num_nodes]
        ppo = detail['ppo']
        heu = detail['heuristic']
        rnd = detail['random']
        
        row = (f"{num_nodes:<6} "
               f"{ppo['avg_reward']:<10.3f} "
               f"{heu['avg_reward']:<12.3f} "
               f"{rnd['avg_reward']:<10.3f} "
               f"{ppo['success_rate']:<10.1%} "
               f"{heu['success_rate']:<12.1%} "
               f"{rnd['success_rate']:<10.1%}")
        print(row)
        
        # 添加L和D_BW信息
        def _format_metric(val, std_val):
            if np.isnan(val) or np.isnan(std_val):
                return "N/A"
            return f"{val:.4f}±{std_val:.4f}"
        
        ppo_l_str = _format_metric(ppo['load_balance']['mean'], ppo['load_balance']['std'])
        heu_l_str = _format_metric(heu['load_balance']['mean'], heu['load_balance']['std'])
        rnd_l_str = _format_metric(rnd['load_balance']['mean'], rnd['load_balance']['std'])
        
        ppo_dbw_str = _format_metric(ppo['bandwidth_satisfaction']['mean'], ppo['bandwidth_satisfaction']['std'])
        heu_dbw_str = _format_metric(heu['bandwidth_satisfaction']['mean'], heu['bandwidth_satisfaction']['std'])
        rnd_dbw_str = _format_metric(rnd['bandwidth_satisfaction']['mean'], rnd['bandwidth_satisfaction']['std'])
        
        l_row = f"{'L值':<6} {ppo_l_str:<10} {heu_l_str:<12} {rnd_l_str:<10}"
        dbw_row = f"{'D_BW值':<6} {ppo_dbw_str:<10} {heu_dbw_str:<12} {rnd_dbw_str:<10}"
        
        print(f"       {l_row}")
        print(f"       {dbw_row}")
        print()
    
    # 总结分析
    summary = results['summary']
    print(f"\n📈 总结分析:")
    print("-"*60)
    overall = summary['overall_performance']
    print(f"PPO平均性能       -> 奖励: {overall['ppo']['avg_reward']:.3f}, 成功率: {overall['ppo']['avg_success_rate']:.1%}")
    print(f"{heuristic_type}启发式平均性能 -> 奖励: {overall['heuristic']['avg_reward']:.3f}, 成功率: {overall['heuristic']['avg_success_rate']:.1%}")
    print(f"Random平均性能    -> 奖励: {overall['random']['avg_reward']:.3f}, 成功率: {overall['random']['avg_success_rate']:.1%}")
    
    # 胜率统计
    win_rates = summary['win_rates']
    print(f"\n🏆 胜率统计:")
    print(f"PPO vs {heuristic_type}启发式: {win_rates['ppo_vs_heuristic']:.1%}")
    print(f"PPO vs Random: {win_rates['ppo_vs_random']:.1%}")
    print(f"{heuristic_type}启发式 vs Random: {win_rates['heuristic_vs_random']:.1%}")
    
    # 排名
    print(f"\n🥇 算法排名:")
    print("按奖励排名:")
    for i, (name, reward, _) in enumerate(summary['rankings']['by_reward'], 1):
        print(f"   {i}. {name}: {reward:.3f}")
    print("按成功率排名:")
    for i, (name, _, success_rate) in enumerate(summary['rankings']['by_success_rate'], 1):
        print(f"   {i}. {name}: {success_rate:.1%}")


def main():
    print("🧪 三算法性能对比测试 (PPO vs Heuristic vs Random)")
    print("="*70)

    try:
        # 加载PPO模型
        ckpt = _find_latest_checkpoint('checkpoints')
        agent, env_config, _ = _load_agent_and_configs(ckpt)

        # 测试参数
        virtual_nodes_range = [3, 4, 5, 6, 7, 8]  # 测试3-8个虚拟节点
        episodes_per_node_count = 30  # 每个节点数量测试30次
        base_seed = 42
        temperature = 0.05  # 低温度，更贪心
        heuristic_type = "naive"  # 使用天真启发式算法（笨拙版本）

        # 执行评测
        results = evaluate_three_algorithms(
            agent=agent,
            env_config=env_config,
            virtual_nodes_range=virtual_nodes_range,
            episodes_per_node_count=episodes_per_node_count,
            base_seed=base_seed,
            temperature=temperature,
            heuristic_type=heuristic_type
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
        print("-"*40)
        
        # 判断算法优劣
        rankings_by_reward = summary['rankings']['by_reward']
        best_algorithm = rankings_by_reward[0][0]
        
        if best_algorithm == "PPO":
            print("🥇 PPO算法表现最佳!")
            if win_rates['ppo_vs_heuristic'] >= 0.8:
                print("✅ PPO显著优于启发式算法")
            elif win_rates['ppo_vs_heuristic'] >= 0.6:
                print("⚡ PPO适度优于启发式算法")
            else:
                print("🤔 PPO与启发式算法性能接近")
        elif best_algorithm == "Heuristic":
            print("🥇 启发式算法表现最佳!")
            print("💡 启发式算法在这个问题上可能更适合")
        else:
            print("⚠️ 所有算法表现都需要改进")
        
        print(f"📊 PPO胜率: vs启发式={win_rates['ppo_vs_heuristic']:.1%}, vs随机={win_rates['ppo_vs_random']:.1%}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
