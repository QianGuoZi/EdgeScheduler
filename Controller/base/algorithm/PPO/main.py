import os
import torch
import yaml
from PPO_trainer import PPOTrainer
from virtual_edge_env import VirtualEdgeEnv
from production_scheduler import PPOScheduler

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_performance(policy, env_config):
    """评估策略性能"""
    eval_env = VirtualEdgeEnv(env_config)
    total_reward = 0
    num_episodes = 10
    
    for _ in range(num_episodes):
        state = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                action, _, _ = policy.act(state, exploration=False)
            state, reward, done = eval_env.apply_schedule(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    avg_reward = total_reward / num_episodes
    print(f"Evaluation: Average Reward = {avg_reward:.2f}")
    return avg_reward

def main():
    # 加载配置
    dirName = '/home/qianguo/Edge-Scheduler/Controller/base/algorithm/PPO'
    config = load_config(os.path.join(dirName, 'config.yaml'))
    train_config = config['training']
    
    # 环境配置
    env_config = {
        'num_phys_nodes': train_config['num_phys_nodes'],
        'cpu_min': 8,
        'cpu_max': 32,
        'ram_min': 16,
        'ram_max': 64,
        'bw_min': 100,
        'bw_max': 1000,
        'link_sparsity': 0.3,
        'min_tasks': train_config['num_virtual_tasks'][0],
        'max_tasks': train_config['num_virtual_tasks'][1],
        'gamma1': 0.6,
        'gamma2': 0.4,

        'max_steps_per_episode': 50,

        'task_node_dim': 2,  # [cpu_req, ram_req]
        'phys_node_dim': 4,  # [cpu_cap, cpu_used, ram_cap, ram_used]
        'task_edge_dim': 2,  # [min_bw, max_bw]
        'phys_edge_dim': 2,  # [bw_cap, bw_used]
        'gnn_hidden_dim': 128,
        **train_config['ppo_params']
    }
    
    # 训练器配置
    ppo_config = {
        'task_node_dim': 2,  # [cpu_req, ram_req]
        'phys_node_dim': 4,  # [cpu_cap, cpu_used, ram_cap, ram_used]
        'task_edge_dim': 2,  # [min_bw, max_bw]
        'phys_edge_dim': 2,  # [bw_cap, bw_used]
        'gnn_hidden_dim': 128,
        **train_config['ppo_params']
    }
    
    # 创建训练环境
    env = VirtualEdgeEnv(env_config)
    
    # 创建训练器
    trainer = PPOTrainer(env_config, env)
    
    # 训练循环
    max_updates = 1000
    for update in range(max_updates):
        print(f"Starting update {update + 1}/{max_updates}...")
        # 收集经验并更新策略
        avg_reward = trainer.collect_experience()
        policy_loss, value_loss = trainer.update_policy()
        
        # 日志记录
        print(f"Update {update}/{max_updates} | "
              f"Avg Reward: {avg_reward:.2f} | "
              f"Policy Loss: {policy_loss:.4f} | "
              f"Value Loss: {value_loss:.4f}")
        
        # 定期保存和评估
        if update % 100 == 0:
            torch.save(trainer.policy.state_dict(), f"models/policy_{update}.pt")
        if update % 50 == 0:
            evaluate_performance(trainer.policy, env_config)
    
    # 生产调度器使用示例
    # scheduler = PPOScheduler(ppo_config, "models/policy_final.pt")
    # sample_request = {
    #     'task_nodes': [
    #         {'id': 'task1', 'cpu_req': 4, 'ram_req': 8},
    #         {'id': 'task2', 'cpu_req': 2, 'ram_req': 4}
    #     ],
    #     'task_links': [
    #         {'id': 'link1', 'source': 'task1', 'target': 'task2', 'min_bw': 50, 'max_bw': 200}
    #     ]
    # }
    # schedule_plan = scheduler.schedule(sample_request)
    # print("Production Schedule:", schedule_plan)

if __name__ == "__main__":
    main()