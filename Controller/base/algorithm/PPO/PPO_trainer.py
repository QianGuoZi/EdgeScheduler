import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_

from PPO_buffer import PPOBuffer
from policy_network import PolicyNetwork
from virtual_edge_env import VirtualEdgeEnv

class PPOTrainer:
    def __init__(self, config, env):
        self.config = config
        
        # 策略网络
        self.policy = PolicyNetwork(config)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), 
                                   lr=float(config['lr']),
                                   eps=1e-5)
        
        # 环境
        self.env = env if env is not None else VirtualEdgeEnv(config)
        
        # 经验缓冲区
        self.buffer = PPOBuffer(config['buffer_size'])
        
        # 训练参数
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
    
    def collect_experience(self):
        """ 使用当前策略收集经验 """
        episode_rewards = []
        i = 0
        for _ in range(self.config['episodes_per_update']):
            print(f"Collecting experience for episode {i + 1}...")
            i += 1
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # 获取动作掩码
                masks = self.env.get_action_masks()
                
                # 使用当前策略获取动作
                with torch.no_grad():
                    action, log_prob, value = self.policy.act(
                        state, 
                        task_mask=masks['task'],
                        # path_mask=masks['path'],
                        exploration=True
                    )
                
                # 环境执行
                next_state, reward, done = self.env.apply_schedule(action)
                
                # 存储经验
                self.buffer.add(state, action, reward, value, log_prob, done)
                
                # 更新统计
                episode_reward += reward
                episode_length += 1
                
                # 移动到下一个状态
                state = next_state if not done else None
            
            episode_rewards.append(episode_reward)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
        
        # 返回平均奖励用于监控
        return np.mean(episode_rewards)
    
    def update_policy(self):
        """ 执行PPO更新 """
        if len(self.buffer) == 0:
            return 0, 0
        
        # 计算优势函数和回报
        advantages = self.buffer.compute_advantages(self.gamma, self.gae_lambda)
        states, actions, rewards, values, old_log_probs, dones = self.buffer.sample()
        
        # 计算回报 = 优势 + 价值估计
        returns = advantages + np.array(values)
        
        # 转换为张量
        states = torch.stack(states)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # 动作转换为张量
        task_actions = torch.tensor([a['task'] for a in actions], dtype=torch.long)
        bw_actions = torch.tensor([a['bw'] for a in actions], dtype=torch.float32)
        # path_actions = torch.tensor([a['path'] for a in actions], dtype=torch.long)
        
        # 初始化损失记录
        total_policy_loss = 0
        total_value_loss = 0
        
        # 多次更新策略
        for _ in range(self.ppo_epochs):
            # 随机打乱索引
            indices = torch.randperm(len(states))
            
            # 小批量更新
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                
                # 获取小批量数据
                batch_states = states[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                batch_task_actions = task_actions[idx]
                batch_bw_actions = bw_actions[idx]
                # batch_path_actions = path_actions[idx]
                
                # 计算新策略的输出
                task_logits, bw_mean, bw_std, values = self.policy(
                    batch_states
                )
                
                # 计算新策略的对数概率
                # 任务分配概率
                task_dist = torch.distributions.Categorical(logits=task_logits)
                task_log_probs = task_dist.log_prob(batch_task_actions)
                
                # 带宽分配概率
                bw_dist = torch.distributions.Normal(bw_mean, bw_std)
                bw_log_probs = bw_dist.log_prob(batch_bw_actions)
                
                # 路径选择概率
                # path_dist = torch.distributions.Categorical(logits=path_score)
                # path_log_probs = path_dist.log_prob(batch_path_actions)
                
                # 组合对数概率（加权）
                new_log_probs = (
                    self.config.get('task_weight', 0.6) * task_log_probs +
                    self.config.get('bw_weight', 0.4) * bw_log_probs
                    # self.config.get('path_weight', 0.2) * path_log_probs
                )
                
                # 计算概率比
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算策略损失 (裁剪PPO目标)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios, 
                    1.0 - self.clip_epsilon, 
                    1.0 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值函数损失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # 记录损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 计算平均损失
        num_batches = len(states) // self.mini_batch_size * self.ppo_epochs
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        return avg_policy_loss, avg_value_loss
    
    def train(self, total_timesteps):
        """ 完整的训练循环 """
        timestep = 0
        best_reward = -float('inf')
        
        while timestep < total_timesteps:
            # 收集经验
            avg_episode_reward = self.collect_experience()
            timestep += self.config['episodes_per_update'] * np.mean(self.episode_lengths[-self.config['episodes_per_update']:])
            
            # 更新策略
            policy_loss, value_loss = self.update_policy()
            
            # 日志记录
            print(f"Timestep: {timestep}/{total_timesteps} | "
                  f"Avg Reward: {avg_episode_reward:.2f} | "
                  f"Policy Loss: {policy_loss:.4f} | "
                  f"Value Loss: {value_loss:.4f}")
            
            # 保存最佳模型
            if avg_episode_reward > best_reward:
                best_reward = avg_episode_reward
                torch.save(self.policy.state_dict(), "best_policy.pth")
        
        # 保存最终模型
        torch.save(self.policy.state_dict(), "final_policy.pth")
        return best_reward
    
    def gaussian_log_prob(self, mean, std, x):
        """ 计算高斯分布的对数概率 """
        return -0.5 * ((x - mean) / std).pow(2) - 0.5 * torch.log(2 * torch.pi * std.pow(2))