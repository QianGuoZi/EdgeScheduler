# train.py
import torch

from Controller.base.algorithm.PPO.PPO_trainer import PPOTrainer
from Controller.base.algorithm.PPO.virtual_edge_env import VirtualEdgeEnv

def main():
    # 环境配置示例
    env_config = {
        # 物理节点配置
        'num_phys_nodes': 10,
        'cpu_min': 8,
        'cpu_max': 32,
        'ram_min': 16,
        'ram_max': 64,
        
        # 物理链路配置
        'bw_min': 100,
        'bw_max': 1000,
        'link_sparsity': 0.3,  # 30%的节点间没有直接连接
        
        # 任务配置
        'min_tasks': 3,
        'max_tasks': 8,
        
        # 奖励权重
        'gamma1': 0.6,  # 负载均衡权重
        'gamma2': 0.4   # 带宽满足度权重
    }
    config = {
        # GNN配置
        'gnn_hidden_dim': 128,
        
        # 网络头配置
        'task_head_hidden': 256,
        'bw_head_hidden': 256,
        'path_head_hidden': 256,
        'value_head_hidden': 256,
        
        # 动作空间权重
        'task_weight': 0.5,
        'bw_weight': 0.3,
        'path_weight': 0.2,
        
        # PPO 参数
        'gamma': 0.99,  # 折扣因子
        'gae_lambda': 0.95,  # GAE参数
        'clip_epsilon': 0.2,  # PPO裁剪参数
        'ppo_epochs': 4,  # PPO更新次数
        'mini_batch_size': 64,  # 小批量大小
        'max_grad_norm': 0.5,  # 梯度裁剪阈值
        
        # 动作权重
        'task_weight': 0.5,  # 任务分配权重
        'bw_weight': 0.3,  # 带宽分配权重
        'path_weight': 0.2,  # 路径选择权重
        
        # 经验收集
        'episodes_per_update': 10,  # 每次更新收集的经验数
        'buffer_size': 2048,  # 经验缓冲区大小
    }
        


    # 创建环境实例
    env = VirtualEdgeEnv(env_config)
    state = env.reset()


    config = load_config('config.yaml')
    trainer = PPOTrainer(config['training'])
    
    for update in range(config['training']['max_updates']):
        trainer.collect_experience()
        trainer.update_policy()
        
        # 每100次更新保存一次模型
        if update % 100 == 0:
            torch.save(trainer.policy.state_dict(), 
                      f"models/policy_{update}.pt")
            
        # 每50次更新评估一次
        if update % 50 == 0:
            evaluate_performance(trainer.policy)

if __name__ == "__main__":
    main()