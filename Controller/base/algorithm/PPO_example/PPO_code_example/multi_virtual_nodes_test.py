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
    """加载Agent与配置。
    
    Returns:
        agent, env_config, agent_config
    """
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
    
    return SequentialNetworkSchedulerEnvironment(**cfg)


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


def _run_episode(env: SequentialNetworkSchedulerEnvironment,
                 agent: SimpleSequentialAgent = None,
                 temperature: float = 0.1,
                 max_steps: int = 50,
                 greedy: bool = True) -> Tuple[float, bool, Dict]:
    """运行单Episode，返回(总奖励, 是否成功, 详细信息)。"""
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    
    # 记录详细信息
    episode_info = {
        'num_virtual_nodes': state.get('num_virtual_nodes', 0),
        'num_virtual_links': state.get('num_virtual_links', 0),
        'mapping_rewards': [],
        'bandwidth_rewards': [],
        'steps': steps
    }

    while not done and steps < max_steps:
        if agent is not None:
            if greedy:
                action = _select_action_greedy_with_fallback(env, agent, state)
            else:
                action, _, _ = agent.select_action(state, temperature)
        else:
            # 随机策略
            if state.get('mapping_phase', True):
                action = np.random.randint(0, state.get('num_physical_nodes', 1))
            else:
                action = np.random.randint(0, env.bandwidth_levels)

        state, reward, done, info = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        
        # 记录不同阶段的奖励
        if info.get('phase') == 'mapping':
            episode_info['mapping_rewards'].append(float(reward))
        elif info.get('phase') == 'bandwidth':
            episode_info['bandwidth_rewards'].append(float(reward))

    # 成功判定：所有虚拟节点映射完成且总奖励为正
    all_nodes_mapped = all(node != -1 for node in (env.partial_mapping or []))
    success = bool(all_nodes_mapped and total_reward > 0)
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward,
        'final_mapping': env.partial_mapping.copy() if env.partial_mapping else [],
        'final_bandwidth': env.partial_bandwidth.copy() if env.partial_bandwidth else []
    })
    
    return total_reward, success, episode_info


def evaluate_virtual_nodes_range(agent: SimpleSequentialAgent,
                                 env_config: Dict,
                                 virtual_nodes_range: List[int] = [3, 4, 5, 6, 7, 8],
                                 episodes_per_node_count: int = 30,
                                 base_seed: int = 42,
                                 temperature: float = 0.05,
                                 greedy: bool = True) -> Dict:
    """评测不同虚拟节点数量下的性能。"""
    print(f"🧪 开始评测虚拟节点数量范围: {virtual_nodes_range}")
    print(f"📊 每个节点数量测试 {episodes_per_node_count} 个Episode")
    
    _set_global_seed(base_seed)
    
    results = {
        'virtual_nodes_range': virtual_nodes_range,
        'episodes_per_node_count': episodes_per_node_count,
        'base_seed': base_seed,
        'temperature': temperature,
        'greedy': greedy,
        'detailed_results': {},
        'summary': {}
    }
    
    for num_virtual_nodes in virtual_nodes_range:
        print(f"\n🔍 测试 {num_virtual_nodes} 个虚拟节点...")
        
        ppo_rewards, ppo_success = [], []
        rnd_rewards, rnd_success = [], []
        ppo_episodes_info = []
        rnd_episodes_info = []
        
        for ep in range(episodes_per_node_count):
            ep_seed = base_seed + ep + num_virtual_nodes * 1000  # 确保不同节点数的种子不同
            
            # PPO环境
            env_ppo = _make_env_with_virtual_nodes(env_config, num_virtual_nodes, ep_seed)
            # 随机环境（相同种子）
            env_rnd = _make_env_with_virtual_nodes(env_config, num_virtual_nodes, ep_seed)
            
            # 运行PPO
            r_ppo, s_ppo, info_ppo = _run_episode(env_ppo, agent=agent, temperature=temperature, greedy=greedy)
            ppo_rewards.append(r_ppo)
            ppo_success.append(1.0 if s_ppo else 0.0)
            ppo_episodes_info.append(info_ppo)
            
            # 运行随机
            r_rnd, s_rnd, info_rnd = _run_episode(env_rnd, agent=None, temperature=1.0)
            rnd_rewards.append(r_rnd)
            rnd_success.append(1.0 if s_rnd else 0.0)
            rnd_episodes_info.append(info_rnd)
            
            if (ep + 1) % 10 == 0:
                print(f"   完成 {ep + 1}/{episodes_per_node_count} episodes")
        
        # 统计结果
        def _stats(arr):
            return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
        
        ppo_reward_mean, ppo_reward_std, ppo_reward_min, ppo_reward_max = _stats(ppo_rewards)
        rnd_reward_mean, rnd_reward_std, rnd_reward_min, rnd_reward_max = _stats(rnd_rewards)
        ppo_sr = float(np.mean(ppo_success))
        rnd_sr = float(np.mean(rnd_success))
        
        # 计算平均步数和平均映射/带宽奖励
        ppo_avg_steps = np.mean([info['steps'] for info in ppo_episodes_info])
        rnd_avg_steps = np.mean([info['steps'] for info in rnd_episodes_info])
        
        ppo_avg_mapping_reward = np.mean([np.mean(info['mapping_rewards']) if info['mapping_rewards'] else 0 
                                         for info in ppo_episodes_info])
        rnd_avg_mapping_reward = np.mean([np.mean(info['mapping_rewards']) if info['mapping_rewards'] else 0 
                                         for info in rnd_episodes_info])
        
        node_result = {
            'num_virtual_nodes': num_virtual_nodes,
            'ppo': {
                'avg_reward': ppo_reward_mean,
                'std_reward': ppo_reward_std,
                'min_reward': ppo_reward_min,
                'max_reward': ppo_reward_max,
                'success_rate': ppo_sr,
                'avg_steps': float(ppo_avg_steps),
                'avg_mapping_reward': float(ppo_avg_mapping_reward)
            },
            'random': {
                'avg_reward': rnd_reward_mean,
                'std_reward': rnd_reward_std,
                'min_reward': rnd_reward_min,
                'max_reward': rnd_reward_max,
                'success_rate': rnd_sr,
                'avg_steps': float(rnd_avg_steps),
                'avg_mapping_reward': float(rnd_avg_mapping_reward)
            },
            'improvement': {
                'reward_improvement': ppo_reward_mean - rnd_reward_mean,
                'success_rate_improvement': ppo_sr - rnd_sr,
                'relative_reward_improvement': (ppo_reward_mean - rnd_reward_mean) / max(abs(rnd_reward_mean), 1e-6)
            }
        }
        
        results['detailed_results'][num_virtual_nodes] = node_result
        
        print(f"✅ {num_virtual_nodes} 虚拟节点测试完成:")
        print(f"   PPO    -> 奖励: {ppo_reward_mean:.3f}±{ppo_reward_std:.3f}, 成功率: {ppo_sr:.1%}")
        print(f"   Random -> 奖励: {rnd_reward_mean:.3f}±{rnd_reward_std:.3f}, 成功率: {rnd_sr:.1%}")
        print(f"   改善    -> 奖励: {node_result['improvement']['reward_improvement']:+.3f}, 成功率: {node_result['improvement']['success_rate_improvement']:+.1%}")
    
    # 生成总结
    results['summary'] = _generate_summary(results)
    
    return results


def _generate_summary(results: Dict) -> Dict:
    """生成测试结果的总结分析。"""
    detailed = results['detailed_results']
    virtual_nodes_range = results['virtual_nodes_range']
    
    # 收集数据用于分析
    ppo_rewards = [detailed[n]['ppo']['avg_reward'] for n in virtual_nodes_range]
    rnd_rewards = [detailed[n]['random']['avg_reward'] for n in virtual_nodes_range]
    ppo_success_rates = [detailed[n]['ppo']['success_rate'] for n in virtual_nodes_range]
    rnd_success_rates = [detailed[n]['random']['success_rate'] for n in virtual_nodes_range]
    reward_improvements = [detailed[n]['improvement']['reward_improvement'] for n in virtual_nodes_range]
    success_improvements = [detailed[n]['improvement']['success_rate_improvement'] for n in virtual_nodes_range]
    
    summary = {
        'overall_performance': {
            'ppo_avg_reward_across_all': float(np.mean(ppo_rewards)),
            'random_avg_reward_across_all': float(np.mean(rnd_rewards)),
            'ppo_avg_success_rate_across_all': float(np.mean(ppo_success_rates)),
            'random_avg_success_rate_across_all': float(np.mean(rnd_success_rates)),
            'avg_reward_improvement': float(np.mean(reward_improvements)),
            'avg_success_improvement': float(np.mean(success_improvements))
        },
        'best_performance': {
            'ppo_best_reward_node_count': virtual_nodes_range[np.argmax(ppo_rewards)],
            'ppo_best_reward_value': max(ppo_rewards),
            'ppo_best_success_node_count': virtual_nodes_range[np.argmax(ppo_success_rates)],
            'ppo_best_success_value': max(ppo_success_rates)
        },
        'worst_performance': {
            'ppo_worst_reward_node_count': virtual_nodes_range[np.argmin(ppo_rewards)],
            'ppo_worst_reward_value': min(ppo_rewards),
            'ppo_worst_success_node_count': virtual_nodes_range[np.argmin(ppo_success_rates)],
            'ppo_worst_success_value': min(ppo_success_rates)
        },
        'trends': {
            'ppo_reward_trend': 'increasing' if ppo_rewards[-1] > ppo_rewards[0] else 'decreasing',
            'ppo_success_trend': 'increasing' if ppo_success_rates[-1] > ppo_success_rates[0] else 'decreasing',
            'improvement_consistency': len([x for x in reward_improvements if x > 0]) / len(reward_improvements)
        }
    }
    
    return summary


def _save_results(results: Dict, out_dir: str = 'test_results') -> str:
    """保存测试结果到JSON文件。"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'multi_virtual_nodes_test_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


def _create_visualization(results: Dict, out_dir: str = 'test_results') -> List[str]:
    """创建可视化图表。"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    virtual_nodes_range = results['virtual_nodes_range']
    detailed = results['detailed_results']
    
    # 准备数据
    ppo_rewards = [detailed[n]['ppo']['avg_reward'] for n in virtual_nodes_range]
    rnd_rewards = [detailed[n]['random']['avg_reward'] for n in virtual_nodes_range]
    ppo_success_rates = [detailed[n]['ppo']['success_rate'] * 100 for n in virtual_nodes_range]
    rnd_success_rates = [detailed[n]['random']['success_rate'] * 100 for n in virtual_nodes_range]
    ppo_reward_stds = [detailed[n]['ppo']['std_reward'] for n in virtual_nodes_range]
    rnd_reward_stds = [detailed[n]['random']['std_reward'] for n in virtual_nodes_range]
    
    saved_files = []
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 图1: 奖励对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 奖励曲线
    ax1.errorbar(virtual_nodes_range, ppo_rewards, yerr=ppo_reward_stds, 
                label='PPO', marker='o', linewidth=2, capsize=5)
    ax1.errorbar(virtual_nodes_range, rnd_rewards, yerr=rnd_reward_stds,
                label='Random', marker='s', linewidth=2, capsize=5)
    ax1.set_xlabel('virtual nodes number n')
    ax1.set_ylabel('average reward')
    ax1.set_title('average reward comparison under different virtual nodes number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 成功率对比
    ax2.plot(virtual_nodes_range, ppo_success_rates, label='PPO', 
            marker='o', linewidth=2)
    ax2.plot(virtual_nodes_range, rnd_success_rates, label='Random', 
            marker='s', linewidth=2)
    ax2.set_xlabel('virtual nodes number n')
    ax2.set_ylabel('success rate (%)')
    ax2.set_title('success rate comparison under different virtual nodes number')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    reward_comparison_path = os.path.join(out_dir, f'reward_comparison_{ts}.png')
    plt.savefig(reward_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(reward_comparison_path)
    
    # 图2: 改善程度分析
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    reward_improvements = [detailed[n]['improvement']['reward_improvement'] for n in virtual_nodes_range]
    success_improvements = [detailed[n]['improvement']['success_rate_improvement'] * 100 for n in virtual_nodes_range]
    
    # 奖励改善
    colors = ['green' if x > 0 else 'red' for x in reward_improvements]
    ax1.bar(virtual_nodes_range, reward_improvements, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('virtual nodes number n')
    ax1.set_ylabel('reward improvement')
    ax1.set_title('reward improvement of PPO compared to random strategy')
    ax1.grid(True, alpha=0.3)
    
    # 成功率改善
    colors = ['green' if x > 0 else 'red' for x in success_improvements]
    ax2.bar(virtual_nodes_range, success_improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('virtual nodes number n')
    ax2.set_ylabel('success rate improvement (%)')
    ax2.set_title('success rate improvement of PPO compared to random strategy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    improvement_path = os.path.join(out_dir, f'improvement_analysis_{ts}.png')
    plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(improvement_path)
    
    return saved_files


def print_detailed_results(results: Dict):
    """打印详细的测试结果。"""
    print("\n" + "="*80)
    print("📊 多虚拟节点数量测试详细结果")
    print("="*80)
    
    # 测试配置
    print(f"🔧 测试配置:")
    print(f"   虚拟节点范围: {results['virtual_nodes_range']}")
    print(f"   每个数量测试: {results['episodes_per_node_count']} episodes")
    print(f"   基础种子: {results['base_seed']}")
    print(f"   温度参数: {results['temperature']}")
    print(f"   贪心策略: {results['greedy']}")
    
    # 详细结果
    print(f"\n📋 详细结果:")
    print("-"*80)
    header = f"{'节点数':<6} {'PPO奖励':<12} {'Random奖励':<12} {'PPO成功率':<10} {'Random成功率':<12} {'奖励改善':<10} {'成功率改善':<10}"
    print(header)
    print("-"*80)
    
    for num_nodes in results['virtual_nodes_range']:
        detail = results['detailed_results'][num_nodes]
        ppo = detail['ppo']
        rnd = detail['random']
        imp = detail['improvement']
        
        row = (f"{num_nodes:<6} "
               f"{ppo['avg_reward']:<12.3f} "
               f"{rnd['avg_reward']:<12.3f} "
               f"{ppo['success_rate']:<10.1%} "
               f"{rnd['success_rate']:<12.1%} "
               f"{imp['reward_improvement']:<10.3f} "
               f"{imp['success_rate_improvement']:<10.1%}")
        print(row)
    
    # 总结分析
    summary = results['summary']
    print(f"\n📈 总结分析:")
    print("-"*50)
    overall = summary['overall_performance']
    print(f"PPO平均性能    -> 奖励: {overall['ppo_avg_reward_across_all']:.3f}, 成功率: {overall['ppo_avg_success_rate_across_all']:.1%}")
    print(f"Random平均性能 -> 奖励: {overall['random_avg_reward_across_all']:.3f}, 成功率: {overall['random_avg_success_rate_across_all']:.1%}")
    print(f"平均改善       -> 奖励: {overall['avg_reward_improvement']:+.3f}, 成功率: {overall['avg_success_improvement']:+.1%}")
    
    best = summary['best_performance']
    worst = summary['worst_performance']
    print(f"\nPPO最佳表现   -> {best['ppo_best_reward_node_count']}节点(奖励{best['ppo_best_reward_value']:.3f}), {best['ppo_best_success_node_count']}节点(成功率{best['ppo_best_success_value']:.1%})")
    print(f"PPO最差表现   -> {worst['ppo_worst_reward_node_count']}节点(奖励{worst['ppo_worst_reward_value']:.3f}), {worst['ppo_worst_success_node_count']}节点(成功率{worst['ppo_worst_success_value']:.1%})")
    
    trends = summary['trends']
    print(f"\n趋势分析      -> 奖励趋势: {trends['ppo_reward_trend']}, 成功率趋势: {trends['ppo_success_trend']}")
    print(f"改善一致性    -> {trends['improvement_consistency']:.1%} 的情况下PPO优于Random")


def main():
    print("🧪 多虚拟节点数量下的PPO性能测试")
    print("="*60)

    try:
        # 加载模型
        ckpt = _find_latest_checkpoint('checkpoints')
        agent, env_config, _ = _load_agent_and_configs(ckpt)

        # 测试参数
        virtual_nodes_range = [3, 4, 5, 6, 7, 8]  # 测试3-8个虚拟节点
        episodes_per_node_count = 30  # 每个节点数量测试30次
        base_seed = 42
        temperature = 0.05  # 低温度，更贪心

        # 执行评测
        results = evaluate_virtual_nodes_range(
            agent=agent,
            env_config=env_config,
            virtual_nodes_range=virtual_nodes_range,
            episodes_per_node_count=episodes_per_node_count,
            base_seed=base_seed,
            temperature=temperature,
            greedy=True
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
            print("💡 提示: 确保安装了 matplotlib, seaborn 和 scikit-learn")

        # 最终结论
        summary = results['summary']
        overall = summary['overall_performance']
        
        print(f"\n🎯 最终结论:")
        print("-"*30)
        if overall['avg_reward_improvement'] > 0 and overall['avg_success_improvement'] > 0:
            print("✅ PPO在所有虚拟节点数量下平均表现均优于随机策略")
        elif overall['avg_reward_improvement'] > 0 or overall['avg_success_improvement'] > 0:
            print("⚡ PPO在某些指标上优于随机策略")
        else:
            print("⚠️ PPO表现需要进一步优化")
            
        print(f"📊 改善一致性: {summary['trends']['improvement_consistency']:.1%}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
