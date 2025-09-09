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


def _run_ppo_episode(env: SequentialNetworkSchedulerEnvironment,
                     agent: SimpleSequentialAgent,
                     temperature: float = 0.1,
                     max_steps: int = 50,
                     greedy: bool = True) -> Tuple[float, bool, Dict]:
    """运行PPO算法的Episode"""
    state = env.reset()
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
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward
    })
    
    return total_reward, success, episode_info


def _run_random_episode(env: SequentialNetworkSchedulerEnvironment,
                        max_steps: int = 50) -> Tuple[float, bool, Dict]:
    """运行随机算法的Episode"""
    state = env.reset()
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
    
    episode_info.update({
        'steps': steps,
        'success': success,
        'total_reward': total_reward
    })
    
    return total_reward, success, episode_info


def evaluate_multiple_algorithms(agent: SimpleSequentialAgent,
                                 env_config: Dict,
                                 virtual_nodes_range: List[int] = [3, 4, 5, 6, 7, 8],
                                 episodes_per_node_count: int = 30,
                                 base_seed: int = 42,
                                 temperature: float = 0.05) -> Dict:
    """评测多种算法在不同虚拟节点数量下的性能"""
    heuristic_types = ["naive", "smart"]
    algorithms = ["PPO"] + [f"{h.title()}Heuristic" for h in heuristic_types] + ["Random"]
    
    print(f"🧪 开始评测多算法性能对比")
    print(f"🎯 算法类型: {', '.join(algorithms)}")
    print(f"📊 虚拟节点范围: {virtual_nodes_range}")
    print(f"📋 每个节点数量测试 {episodes_per_node_count} 个Episode")
    
    _set_global_seed(base_seed)
    
    results = {
        'algorithms': algorithms,
        'heuristic_types': heuristic_types,
        'virtual_nodes_range': virtual_nodes_range,
        'episodes_per_node_count': episodes_per_node_count,
        'base_seed': base_seed,
        'temperature': temperature,
        'detailed_results': {},
        'summary': {}
    }
    
    # 创建启发式算法代理
    heuristic_agents = {h: create_heuristic_agent(h) for h in heuristic_types}
    
    for num_virtual_nodes in virtual_nodes_range:
        print(f"\n🔍 测试 {num_virtual_nodes} 个虚拟节点...")
        
        # 存储所有算法的结果
        algorithm_results = {}
        
        # 初始化结果存储
        for algo in algorithms:
            algorithm_results[algo] = {
                'rewards': [], 'success': [], 'steps': []
            }
        
        for ep in range(episodes_per_node_count):
            ep_seed = base_seed + ep + num_virtual_nodes * 1000
            
            # PPO
            env_ppo = _make_env_with_virtual_nodes(env_config, num_virtual_nodes, ep_seed)
            r_ppo, s_ppo, info_ppo = _run_ppo_episode(env_ppo, agent, temperature=temperature)
            algorithm_results['PPO']['rewards'].append(r_ppo)
            algorithm_results['PPO']['success'].append(1.0 if s_ppo else 0.0)
            algorithm_results['PPO']['steps'].append(info_ppo['steps'])
            
            # 启发式算法
            for h_type in heuristic_types:
                env_heu = _make_env_with_virtual_nodes(env_config, num_virtual_nodes, ep_seed)
                r_heu, s_heu, info_heu = run_heuristic_episode(env_heu, heuristic_agents[h_type])
                algo_name = f"{h_type.title()}Heuristic"
                algorithm_results[algo_name]['rewards'].append(r_heu)
                algorithm_results[algo_name]['success'].append(1.0 if s_heu else 0.0)
                algorithm_results[algo_name]['steps'].append(info_heu['steps'])
            
            # 随机算法
            env_rnd = _make_env_with_virtual_nodes(env_config, num_virtual_nodes, ep_seed)
            r_rnd, s_rnd, info_rnd = _run_random_episode(env_rnd)
            algorithm_results['Random']['rewards'].append(r_rnd)
            algorithm_results['Random']['success'].append(1.0 if s_rnd else 0.0)
            algorithm_results['Random']['steps'].append(info_rnd['steps'])
            
            if (ep + 1) % 10 == 0:
                print(f"   完成 {ep + 1}/{episodes_per_node_count} episodes")
        
        # 统计结果
        def _stats(arr):
            return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
        
        node_result = {'num_virtual_nodes': num_virtual_nodes}
        
        for algo in algorithms:
            rewards = algorithm_results[algo]['rewards']
            success = algorithm_results[algo]['success']
            steps = algorithm_results[algo]['steps']
            
            reward_mean, reward_std, reward_min, reward_max = _stats(rewards)
            success_rate = float(np.mean(success))
            avg_steps = float(np.mean(steps))
            
            node_result[algo.lower().replace('heuristic', '')] = {
                'avg_reward': reward_mean,
                'std_reward': reward_std,
                'min_reward': reward_min,
                'max_reward': reward_max,
                'success_rate': success_rate,
                'avg_steps': avg_steps
            }
        
        results['detailed_results'][num_virtual_nodes] = node_result
        
        print(f"✅ {num_virtual_nodes} 虚拟节点测试完成:")
        for algo in algorithms:
            key = algo.lower().replace('heuristic', '')
            data = node_result[key]
            print(f"   {algo:<15} -> 奖励: {data['avg_reward']:.3f}±{data['std_reward']:.3f}, 成功率: {data['success_rate']:.1%}, 平均步数: {data['avg_steps']:.1f}")
    
    # 生成总结
    results['summary'] = _generate_summary(results)
    
    return results


def _generate_summary(results: Dict) -> Dict:
    """生成测试结果的总结分析"""
    detailed = results['detailed_results']
    virtual_nodes_range = results['virtual_nodes_range']
    algorithms = results['algorithms']
    
    # 收集所有算法的平均性能
    overall_performance = {}
    for algo in algorithms:
        key = algo.lower().replace('heuristic', '')
        rewards = [detailed[n][key]['avg_reward'] for n in virtual_nodes_range]
        success_rates = [detailed[n][key]['success_rate'] for n in virtual_nodes_range]
        
        overall_performance[algo] = {
            'avg_reward': float(np.mean(rewards)),
            'avg_success_rate': float(np.mean(success_rates))
        }
    
    # 排名
    rankings = {
        'by_reward': sorted([(algo, perf['avg_reward'], perf['avg_success_rate']) 
                           for algo, perf in overall_performance.items()], 
                           key=lambda x: x[1], reverse=True),
        'by_success_rate': sorted([(algo, perf['avg_reward'], perf['avg_success_rate']) 
                                 for algo, perf in overall_performance.items()], 
                                 key=lambda x: x[2], reverse=True)
    }
    
    # 计算相对于随机策略的改善
    random_performance = overall_performance['Random']
    improvements = {}
    for algo, perf in overall_performance.items():
        if algo != 'Random':
            improvements[algo] = {
                'reward_improvement': perf['avg_reward'] - random_performance['avg_reward'],
                'success_rate_improvement': perf['avg_success_rate'] - random_performance['avg_success_rate']
            }
    
    summary = {
        'overall_performance': overall_performance,
        'rankings': rankings,
        'improvements_over_random': improvements,
        'best_algorithm': rankings['by_reward'][0][0],
        'algorithm_tiers': _classify_algorithms(rankings['by_reward'])
    }
    
    return summary


def _classify_algorithms(rankings_by_reward):
    """将算法按性能分层"""
    if len(rankings_by_reward) < 2:
        return {"top_tier": [rankings_by_reward[0][0]], "middle_tier": [], "bottom_tier": []}
    
    rewards = [r[1] for r in rankings_by_reward]
    max_reward = max(rewards)
    min_reward = min(rewards)
    range_reward = max_reward - min_reward
    
    if range_reward < 0.1:  # 如果差距很小，都归为一类
        return {"top_tier": [r[0] for r in rankings_by_reward], "middle_tier": [], "bottom_tier": []}
    
    # 分层标准
    top_threshold = max_reward - 0.2 * range_reward
    bottom_threshold = min_reward + 0.2 * range_reward
    
    tiers = {"top_tier": [], "middle_tier": [], "bottom_tier": []}
    
    for name, reward, _ in rankings_by_reward:
        if reward >= top_threshold:
            tiers["top_tier"].append(name)
        elif reward <= bottom_threshold:
            tiers["bottom_tier"].append(name)
        else:
            tiers["middle_tier"].append(name)
    
    return tiers


def _save_results(results: Dict, out_dir: str = 'test_results') -> str:
    """保存测试结果到JSON文件"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'multiple_algorithms_comparison_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


def _create_visualization(results: Dict, out_dir: str = 'test_results') -> List[str]:
    """创建可视化图表"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    virtual_nodes_range = results['virtual_nodes_range']
    detailed = results['detailed_results']
    algorithms = results['algorithms']
    
    # 准备数据
    algorithm_data = {}
    for algo in algorithms:
        key = algo.lower().replace('heuristic', '')
        algorithm_data[algo] = {
            'rewards': [detailed[n][key]['avg_reward'] for n in virtual_nodes_range],
            'success_rates': [detailed[n][key]['success_rate'] * 100 for n in virtual_nodes_range],
            'reward_stds': [detailed[n][key]['std_reward'] for n in virtual_nodes_range]
        }
    
    saved_files = []
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    
    # 图1: 奖励对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 奖励曲线
    for i, algo in enumerate(algorithms):
        data = algorithm_data[algo]
        ax1.errorbar(virtual_nodes_range, data['rewards'], yerr=data['reward_stds'], 
                    label=algo, marker='o', linewidth=2, capsize=3, color=colors[i])
    
    ax1.set_xlabel('Virtual Nodes Number')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward Comparison - All Algorithms')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 成功率对比
    for i, algo in enumerate(algorithms):
        data = algorithm_data[algo]
        ax2.plot(virtual_nodes_range, data['success_rates'], 
                label=algo, marker='o', linewidth=2, color=colors[i])
    
    ax2.set_xlabel('Virtual Nodes Number')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate Comparison - All Algorithms')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_comparison_path = os.path.join(out_dir, f'multiple_algorithms_performance_{ts}.png')
    plt.savefig(performance_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(performance_comparison_path)
    
    # 图2: 算法排名柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 平均奖励排名
    summary = results['summary']
    reward_rankings = summary['rankings']['by_reward']
    names = [r[0] for r in reward_rankings]
    rewards = [r[1] for r in reward_rankings]
    
    bars1 = ax1.bar(names, rewards, color=colors[:len(names)], alpha=0.7)
    ax1.set_title('Average Reward Ranking')
    ax1.set_ylabel('Average Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, reward in zip(bars1, rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{reward:.3f}', ha='center', va='bottom')
    
    # 平均成功率排名
    success_rankings = summary['rankings']['by_success_rate']
    names = [r[0] for r in success_rankings]
    success_rates = [r[2] * 100 for r in success_rankings]
    
    bars2 = ax2.bar(names, success_rates, color=colors[:len(names)], alpha=0.7)
    ax2.set_title('Average Success Rate Ranking')
    ax2.set_ylabel('Success Rate (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, rate in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    ranking_comparison_path = os.path.join(out_dir, f'multiple_algorithms_ranking_{ts}.png')
    plt.savefig(ranking_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(ranking_comparison_path)
    
    return saved_files


def print_detailed_results(results: Dict):
    """打印详细的测试结果"""
    print("\n" + "="*100)
    print("📊 多算法性能对比详细结果")
    print("="*100)
    
    # 测试配置
    algorithms = results['algorithms']
    print(f"🔧 测试配置:")
    print(f"   算法类型: {', '.join(algorithms)}")
    print(f"   虚拟节点范围: {results['virtual_nodes_range']}")
    print(f"   每个数量测试: {results['episodes_per_node_count']} episodes")
    print(f"   基础种子: {results['base_seed']}")
    
    # 详细结果表格
    print(f"\n📋 详细结果:")
    print("-"*100)
    header = f"{'节点数':<6} "
    for algo in algorithms:
        header += f"{algo[:10]+'奖励':<12} "
    header += "| "
    for algo in algorithms:
        header += f"{algo[:10]+'成功率':<12} "
    print(header)
    print("-"*100)
    
    for num_nodes in results['virtual_nodes_range']:
        detail = results['detailed_results'][num_nodes]
        row = f"{num_nodes:<6} "
        
        # 奖励部分
        for algo in algorithms:
            key = algo.lower().replace('heuristic', '')
            row += f"{detail[key]['avg_reward']:<12.3f} "
        row += "| "
        
        # 成功率部分
        for algo in algorithms:
            key = algo.lower().replace('heuristic', '')
            row += f"{detail[key]['success_rate']:<12.1%} "
        
        print(row)
    
    # 总结分析
    summary = results['summary']
    print(f"\n📈 总结分析:")
    print("-"*60)
    overall = summary['overall_performance']
    
    print("平均性能排名:")
    for i, (algo, reward, success_rate) in enumerate(summary['rankings']['by_reward'], 1):
        print(f"   {i}. {algo:<15} -> 奖励: {reward:.3f}, 成功率: {success_rate:.1%}")
    
    # 算法分层
    tiers = summary['algorithm_tiers']
    print(f"\n🏆 算法分层:")
    if tiers['top_tier']:
        print(f"   顶层算法: {', '.join(tiers['top_tier'])}")
    if tiers['middle_tier']:
        print(f"   中层算法: {', '.join(tiers['middle_tier'])}")
    if tiers['bottom_tier']:
        print(f"   底层算法: {', '.join(tiers['bottom_tier'])}")
    
    # 相对于随机策略的改善
    improvements = summary['improvements_over_random']
    print(f"\n📊 相对于随机策略的改善:")
    for algo, imp in improvements.items():
        print(f"   {algo:<15} -> 奖励改善: {imp['reward_improvement']:+.3f}, 成功率改善: {imp['success_rate_improvement']:+.1%}")


def main():
    print("🧪 多算法性能对比测试 (PPO vs 两种启发式 vs Random)")
    print("="*80)

    try:
        # 加载PPO模型
        ckpt = _find_latest_checkpoint('checkpoints')
        agent, env_config, _ = _load_agent_and_configs(ckpt)

        # 测试参数
        virtual_nodes_range = [3, 4, 5, 6, 7, 8]  # 测试3-8个虚拟节点
        episodes_per_node_count = 30  # 每个节点数量测试30次
        base_seed = 42
        temperature = 0.05  # 低温度，更贪心

        # 执行评测
        results = evaluate_multiple_algorithms(
            agent=agent,
            env_config=env_config,
            virtual_nodes_range=virtual_nodes_range,
            episodes_per_node_count=episodes_per_node_count,
            base_seed=base_seed,
            temperature=temperature
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
        best_algorithm = summary['best_algorithm']
        tiers = summary['algorithm_tiers']
        
        print(f"\n🎯 最终结论:")
        print("-"*50)
        print(f"🥇 最佳算法: {best_algorithm}")
        
        if 'PPO' in tiers['top_tier']:
            print("✅ PPO在顶层算法中，表现优秀!")
            if len(tiers['top_tier']) == 1:
                print("🏆 PPO是唯一的顶层算法，明显优于所有启发式算法!")
            else:
                print(f"🤝 PPO与 {', '.join([a for a in tiers['top_tier'] if a != 'PPO'])} 处于同一性能层级")
        elif 'PPO' in tiers['middle_tier']:
            print("⚡ PPO在中层算法中，表现中等")
            if tiers['top_tier']:
                print(f"📈 可考虑向 {', '.join(tiers['top_tier'])} 学习改进策略")
        else:
            print("⚠️ PPO在底层算法中，需要重新训练或调整")
        
        # 启发式算法表现分析
        heuristic_in_top = [a for a in tiers['top_tier'] if 'Heuristic' in a]
        if heuristic_in_top:
            print(f"💡 启发式算法 {', '.join(heuristic_in_top)} 表现出色，证明了好的规则设计的价值")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
