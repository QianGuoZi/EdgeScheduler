#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import json
import time
import random
from typing import Dict, Tuple

import numpy as np
import torch

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


def _make_env(env_config: Dict, seed: int) -> SequentialNetworkSchedulerEnvironment:
    """基于给定配置与种子构建环境。"""
    cfg = dict(env_config)
    cfg['seed'] = seed
    # 兼容性：SequentialNetworkSchedulerEnvironment 构造函数参数与保存的 env_config 对齐
    return SequentialNetworkSchedulerEnvironment(**cfg)


def _select_action_greedy_with_fallback(env: SequentialNetworkSchedulerEnvironment,
                                        agent: SimpleSequentialAgent,
                                        state: Dict) -> Tuple[int, list]:
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
                 max_steps: int = 30,
                 greedy: bool = True,
                 record_details: Dict = None) -> Tuple[float, bool]:
    """运行单Episode，返回(总奖励, 是否成功)。
    成功判定：所有虚拟节点已映射且总奖励 > 0
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    # only record final actions
    mapping_actions = []
    bandwidth_actions = []

    while not done and steps < max_steps:
        if agent is not None:
            if greedy:
                action = _select_action_greedy_with_fallback(env, agent, state)
            else:
                action, _, _ = agent.select_action(state, temperature)
                topk_list = [] # No topk_list for random
        else:
            # 随机策略：阶段判定映射/带宽
            if state.get('mapping_phase', True):
                action = np.random.randint(0, state.get('num_physical_nodes', 1))
            else:
                # 使用环境的带宽等级
                action = np.random.randint(0, env.bandwidth_levels)
            topk_list = [] # No topk_list for random

        # only record final actions
        if state.get('mapping_phase', True):
            mapping_actions.append(int(action))
        else:
            bandwidth_actions.append(int(action))

        state, reward, done, _ = env.step(int(action))
        total_reward += float(reward)
        steps += 1

    # 成功：所有虚拟节点映射完成且总奖励为正
    all_nodes_mapped = all(node != -1 for node in (env.partial_mapping or []))
    success = bool(all_nodes_mapped and total_reward > 0)

    if record_details is not None:
        # only keep action sequences and summary
        record_details['mapping_actions'] = mapping_actions
        record_details['bandwidth_actions'] = bandwidth_actions
        record_details['success'] = success
        record_details['total_reward'] = total_reward
    return total_reward, success


def evaluate(agent: SimpleSequentialAgent,
             env_config: Dict,
             base_seed: int = 42,
             num_episodes: int = 50,
             temperature: float = 0.05,
             greedy: bool = True,
             record_episodes: int = 3) -> Dict:
    """用相同种子分布评测 PPO 与 随机。"""
    _set_global_seed(base_seed)

    ppo_rewards, ppo_success = [], []
    rnd_rewards, rnd_success = [], []
    details = []

    for ep in range(num_episodes):
        ep_seed = base_seed + ep

        # PPO环境
        env_ppo = _make_env(env_config, ep_seed)
        # 随机环境（相同种子）
        env_rnd = _make_env(env_config, ep_seed)

        # 运行PPO
        ppo_detail = {} if len(details) < record_episodes else None
        r, s = _run_episode(env_ppo, agent=agent, temperature=temperature, greedy=greedy, record_details=ppo_detail)
        ppo_rewards.append(r)
        ppo_success.append(1.0 if s else 0.0)

        # 运行随机
        rnd_detail = {} if len(details) < record_episodes else None
        r, s = _run_episode(env_rnd, agent=None, temperature=1.0, record_details=rnd_detail)
        rnd_rewards.append(r)
        rnd_success.append(1.0 if s else 0.0)

        if ppo_detail is not None and rnd_detail is not None:
            details.append({'episode': ep, 'ppo': ppo_detail, 'random': rnd_detail})

    def _stats(arr):
        return float(np.mean(arr)), float(np.std(arr))

    ppo_reward_mean, ppo_reward_std = _stats(ppo_rewards)
    rnd_reward_mean, rnd_reward_std = _stats(rnd_rewards)
    ppo_sr = float(np.mean(ppo_success))
    rnd_sr = float(np.mean(rnd_success))

    return {
        'num_episodes': num_episodes,
        'base_seed': base_seed,
        'ppo': {
            'avg_reward': ppo_reward_mean,
            'std_reward': ppo_reward_std,
            'success_rate': ppo_sr
        },
        'random': {
            'avg_reward': rnd_reward_mean,
            'std_reward': rnd_reward_std,
            'success_rate': rnd_sr
        },
        'details': details
    }


def _save_results(results: Dict, out_dir: str = 'test_results') -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'sequential_ppo_vs_random_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


def main():
    print("🧪 Sequential PPO vs 随机 对比测试")
    print("=" * 60)

    try:
        ckpt = _find_latest_checkpoint('checkpoints')
        agent, env_config, _ = _load_agent_and_configs(ckpt)

        # 评测参数（可按需修改）
        base_seed = 42
        num_episodes = 100
        temperature = 0.1  # 低温度，更贪心

        results = evaluate(
            agent, env_config,
            base_seed=base_seed,
            num_episodes=num_episodes,
            temperature=temperature,
            greedy=True,
            record_episodes=3
        )

        # 打印摘要
        print("\n📋 结果摘要")
        print("-" * 60)
        print(f"Episodes: {results['num_episodes']} | Seed: {results['base_seed']}")
        print(f"PPO    -> Reward: {results['ppo']['avg_reward']:.3f} ± {results['ppo']['std_reward']:.3f} | Success: {results['ppo']['success_rate']:.1%}")
        print(f"Random -> Reward: {results['random']['avg_reward']:.3f} ± {results['random']['std_reward']:.3f} | Success: {results['random']['success_rate']:.1%}")

        better_reward = results['ppo']['avg_reward'] >= results['random']['avg_reward']
        better_success = results['ppo']['success_rate'] >= results['random']['success_rate']

        verdict = "✅ PPO优于随机" if (better_reward or better_success) else "⚠️ PPO未优于随机（本次设定）"
        print(f"结论: {verdict}")


        # 保存
        out_path = _save_results(results)
        print(f"💾 结果已保存: {out_path}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    main()


