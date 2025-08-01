import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class PPOBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def add(self, state, action, reward, value, log_prob, done):
        """ 添加经验到缓冲区 """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, value, log_prob, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size=None):
        """ 采样经验 """
        if batch_size is None:
            # 返回所有经验
            states, actions, rewards, values, log_probs, dones = zip(*self.buffer)
            return states, actions, rewards, values, log_probs, dones
        else:
            # 小批量采样
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[i] for i in indices]
            states, actions, rewards, values, log_probs, dones = zip(*samples)
            return states, actions, rewards, values, log_probs, dones
    
    def compute_advantages(self, gamma=0.99, gae_lambda=0.95):
        """ 计算广义优势估计(GAE) """
        states, actions, rewards, values, log_probs, dones = self.sample()
        values = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        rewards = [r.item() if isinstance(r, torch.Tensor) else r for r in rewards]
        dones = [d.item() if isinstance(d, torch.Tensor) else d for d in dones]
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # 反向计算优势
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0
            else:
                next_value = values[t+1]
                next_non_terminal = 1.0 - dones[t+1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def prepare_batches(self, batch_size):
        """ 准备训练批次 """
        states, actions, rewards, values, old_log_probs, dones = self.sample()
        advantages = self.compute_advantages()
        
        # 转换为张量
        states = torch.stack(states)
        # old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        old_log_probs = torch.tensor(
            [lp.item() if isinstance(lp, torch.Tensor) and lp.numel() == 1 else lp for lp in old_log_probs],
            dtype=torch.float32
        )
        returns = torch.tensor(advantages + np.array(values), dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # 处理动作 - 假设动作是字典
        action_dict = {}
        if isinstance(actions[0], dict):
            for key in actions[0].keys():
                action_dict[key] = torch.tensor([a[key] for a in actions])
        else:
            action_dict['action'] = torch.tensor(actions)
        
        # 创建数据集
        dataset = TensorDataset(states, advantages, returns, old_log_probs, *action_dict.values())
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def clear(self):
        """ 清空缓冲区 """
        self.buffer = []
        self.position = 0
    
    def __len__(self):
        """ 当前缓冲区大小 """
        return len(self.buffer)