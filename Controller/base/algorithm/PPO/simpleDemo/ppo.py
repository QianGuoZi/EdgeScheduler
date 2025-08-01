import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import deque
import random
import math

class GraphEncoder(nn.Module):
    """图神经网络编码器，用于编码物理节点和虚拟工作节点"""
    
    def __init__(self, node_features: int, hidden_dim: int = 128, num_layers: int = 3):
        super(GraphEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 图卷积层
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(node_features, hidden_dim, heads=4, concat=False))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: 节点特征 [num_nodes, node_features]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_features]
        """
        h = x
        
        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        
        return self.output_proj(h)

class Actor(nn.Module):
    """Actor网络，负责生成任务映射和带宽分配动作"""
    
    def __init__(self, 
                 physical_node_dim: int,
                 virtual_node_dim: int,
                 hidden_dim: int = 128,
                 num_physical_nodes: int = 10,
                 bandwidth_levels: int = 10):
        super(Actor, self).__init__()
        
        self.num_physical_nodes = num_physical_nodes
        self.bandwidth_levels = bandwidth_levels
        
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 注意力机制，用于计算物理节点和虚拟节点的匹配度
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 任务映射策略网络
        self.mapping_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_physical_nodes)
        )
        
        # 带宽分配策略网络
        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, bandwidth_levels)
        )
        
    def forward(self, 
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr,
                virtual_node_idx: int):
        """
        Args:
            physical_features: 物理节点特征
            physical_edge_index: 物理网络边索引
            physical_edge_attr: 物理网络边特征
            virtual_features: 虚拟节点特征
            virtual_edge_index: 虚拟网络边索引
            virtual_edge_attr: 虚拟网络边特征
            virtual_node_idx: 当前要映射的虚拟节点索引
        """
        # 编码物理网络和虚拟网络
        physical_encoded = self.physical_encoder(physical_features, physical_edge_index, physical_edge_attr)
        virtual_encoded = self.virtual_encoder(virtual_features, virtual_edge_index, virtual_edge_attr)
        
        # 获取当前虚拟节点的编码
        current_virtual_node = virtual_encoded[virtual_node_idx].unsqueeze(0)  # [1, hidden_dim]
        
        # 计算注意力权重
        attn_output, attn_weights = self.attention(
            current_virtual_node.unsqueeze(0),  # [1, 1, hidden_dim]
            physical_encoded.unsqueeze(0),      # [1, num_physical_nodes, hidden_dim]
            physical_encoded.unsqueeze(0)       # [1, num_physical_nodes, hidden_dim]
        )
        
        # 融合特征
        fused_features = torch.cat([
            current_virtual_node.squeeze(0),  # [hidden_dim]
            attn_output.squeeze(0).squeeze(0)  # [hidden_dim]
        ], dim=-1)  # [hidden_dim * 2]
        
        # 生成动作概率
        mapping_logits = self.mapping_head(fused_features)
        bandwidth_logits = self.bandwidth_head(fused_features)
        
        return mapping_logits, bandwidth_logits

class Critic(nn.Module):
    """Critic网络，用于评估状态价值"""
    
    def __init__(self, 
                 physical_node_dim: int,
                 virtual_node_dim: int,
                 hidden_dim: int = 128):
        super(Critic, self).__init__()
        
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 全局池化后的价值网络
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, 
                physical_features, physical_edge_index, physical_edge_attr,
                virtual_features, virtual_edge_index, virtual_edge_attr):
        """
        计算状态价值
        """
        # 编码物理网络和虚拟网络
        physical_encoded = self.physical_encoder(physical_features, physical_edge_index, physical_edge_attr)
        virtual_encoded = self.virtual_encoder(virtual_features, virtual_edge_index, virtual_edge_attr)
        
        # 全局池化
        physical_global = torch.mean(physical_encoded, dim=0)  # [hidden_dim]
        virtual_global = torch.mean(virtual_encoded, dim=0)    # [hidden_dim]
        
        # 融合特征
        fused_features = torch.cat([physical_global, virtual_global], dim=-1)
        
        # 计算价值
        value = self.value_head(fused_features)
        
        return value

class NetworkSchedulerEnvironment:
    """网络调度环境"""
    
    def __init__(self, 
                 num_physical_nodes: int = 10,
                 max_virtual_nodes: int = 8,
                 bandwidth_levels: int = 10,
                 # 物理节点资源范围
                 physical_cpu_range: Tuple[float, float] = (50.0, 200.0),
                 physical_memory_range: Tuple[float, float] = (100.0, 400.0),
                 physical_bandwidth_range: Tuple[float, float] = (100.0, 1000.0),
                 # 虚拟节点资源范围
                 virtual_cpu_range: Tuple[float, float] = (10.0, 50.0),
                 virtual_memory_range: Tuple[float, float] = (20.0, 100.0),
                 virtual_bandwidth_range: Tuple[float, float] = (10.0, 200.0),
                 # 网络连接概率
                 physical_connectivity_prob: float = 0.3,
                 virtual_connectivity_prob: float = 0.4):
        self.num_physical_nodes = num_physical_nodes
        self.max_virtual_nodes = max_virtual_nodes
        self.bandwidth_levels = bandwidth_levels
        
        # 资源范围参数
        self.physical_cpu_range = physical_cpu_range
        self.physical_memory_range = physical_memory_range
        self.physical_bandwidth_range = physical_bandwidth_range
        self.virtual_cpu_range = virtual_cpu_range
        self.virtual_memory_range = virtual_memory_range
        self.virtual_bandwidth_range = virtual_bandwidth_range
        self.physical_connectivity_prob = physical_connectivity_prob
        self.virtual_connectivity_prob = virtual_connectivity_prob
        
        # 物理网络状态
        self.physical_cpu = None      # CPU资源
        self.physical_memory = None   # 内存资源
        self.physical_links = None    # 物理链路带宽
        
        # 虚拟工作状态
        self.virtual_cpu = None       # CPU需求
        self.virtual_memory = None    # 内存需求
        self.virtual_links = None     # 虚拟链路带宽需求范围
        
        # 调度结果
        self.node_mapping = {}        # 节点映射
        self.bandwidth_allocation = {} # 带宽分配
        
        # 当前调度进度
        self.current_virtual_node = 0
        self.scheduled_nodes = set()
        
    def reset(self, physical_state, virtual_work):
        """重置环境"""
        self.physical_cpu = physical_state['cpu'].copy()
        self.physical_memory = physical_state['memory'].copy()
        self.physical_links = physical_state['links'].copy()
        
        self.virtual_cpu = virtual_work['cpu'].copy()
        self.virtual_memory = virtual_work['memory'].copy()
        self.virtual_links = virtual_work['links'].copy()
        
        self.node_mapping = {}
        self.bandwidth_allocation = {}
        self.current_virtual_node = 0
        self.scheduled_nodes = set()
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        # 构建物理节点特征
        physical_features = []
        for i in range(self.num_physical_nodes):
            features = [
                self.physical_cpu[i],
                self.physical_memory[i],
                # 可以添加更多特征，如节点负载、网络连接数等
            ]
            physical_features.append(features)
        
        # 构建虚拟节点特征
        virtual_features = []
        for i in range(len(self.virtual_cpu)):
            features = [
                self.virtual_cpu[i],
                self.virtual_memory[i],
                # 可以添加更多特征，如任务优先级、依赖关系等
            ]
            virtual_features.append(features)
        
        # 构建边特征（这里简化处理）
        physical_edge_index = self._get_physical_edges()
        virtual_edge_index = self._get_virtual_edges()
        
        return {
            'physical_features': torch.tensor(physical_features, dtype=torch.float32),
            'virtual_features': torch.tensor(virtual_features, dtype=torch.float32),
            'physical_edge_index': physical_edge_index,
            'virtual_edge_index': virtual_edge_index,
            'current_virtual_node': self.current_virtual_node
        }
    
    def _get_physical_edges(self):
        """获取物理网络边（部分连接）"""
        edges = []
        # 使用部分连接
        for i in range(self.num_physical_nodes):
            for j in range(i + 1, self.num_physical_nodes):
                if np.random.random() < self.physical_connectivity_prob:
                    edges.append([i, j])
                    edges.append([j, i])  # 添加反向边
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.tensor([[0], [0]], dtype=torch.long)
    
    def _get_virtual_edges(self):
        """获取虚拟网络边"""
        edges = []
        for link in self.virtual_links:
            edges.append([link['from'], link['to']])
        return torch.tensor(edges, dtype=torch.long).t()
    
    def step(self, action):
        """执行动作"""
        mapping_action, bandwidth_action = action
        
        # 检查动作是否有效
        if not self._is_valid_action(mapping_action, bandwidth_action):
            return self._get_state(), -100, True, {"error": "Invalid action"}
        
        # 执行节点映射
        self.node_mapping[self.current_virtual_node] = mapping_action
        self.scheduled_nodes.add(self.current_virtual_node)
        
        # 执行带宽分配
        allocated_bandwidth = self._action_to_bandwidth(bandwidth_action)
        self.bandwidth_allocation[self.current_virtual_node] = allocated_bandwidth
        
        # 更新资源
        self._update_resources(mapping_action, allocated_bandwidth)
        
        # 移动到下一个虚拟节点
        self.current_virtual_node += 1
        
        # 检查是否完成
        done = self.current_virtual_node >= len(self.virtual_cpu)
        
        # 计算奖励
        reward = self._calculate_reward()
        
        return self._get_state(), reward, done, {}
    
    def _is_valid_action(self, mapping_action, bandwidth_action):
        """检查动作是否有效"""
        # 检查物理节点资源是否足够
        if mapping_action >= self.num_physical_nodes:
            return False
        
        cpu_needed = self.virtual_cpu[self.current_virtual_node]
        memory_needed = self.virtual_memory[self.current_virtual_node]
        
        if (self.physical_cpu[mapping_action] < cpu_needed or 
            self.physical_memory[mapping_action] < memory_needed):
            return False
        
        # 检查带宽分配是否合理
        allocated_bandwidth = self._action_to_bandwidth(bandwidth_action)
        virtual_links = [link for link in self.virtual_links 
                        if link['from'] == self.current_virtual_node or 
                           link['to'] == self.current_virtual_node]
        
        for link in virtual_links:
            min_bandwidth = link['min_bandwidth']
            max_bandwidth = link['max_bandwidth']
            if not (min_bandwidth <= allocated_bandwidth <= max_bandwidth):
                return False
        
        return True
    
    def _action_to_bandwidth(self, bandwidth_action):
        """将带宽动作转换为实际带宽值"""
        # 假设带宽范围是0到1000Mbps，分为10个等级
        return bandwidth_action * 100
    
    def _update_resources(self, physical_node, allocated_bandwidth):
        """更新物理资源"""
        # 更新CPU和内存
        self.physical_cpu[physical_node] -= self.virtual_cpu[self.current_virtual_node]
        self.physical_memory[physical_node] -= self.virtual_memory[self.current_virtual_node]
        
        # 更新网络带宽（简化处理）
        # 这里需要根据实际的网络拓扑和路由来更新带宽
    
    def _calculate_reward(self):
        """计算奖励"""
        if len(self.scheduled_nodes) == 0:
            return 0
        
        # 1. 资源负载均衡奖励
        cpu_utilization = [1 - cpu / max(self.physical_cpu) for cpu in self.physical_cpu]
        memory_utilization = [1 - mem / max(self.physical_memory) for mem in self.physical_memory]
        
        cpu_balance = 1 - np.std(cpu_utilization)
        memory_balance = 1 - np.std(memory_utilization)
        
        # 2. 带宽满足度奖励
        bandwidth_satisfaction = 0
        for link in self.virtual_links:
            if link['from'] in self.scheduled_nodes and link['to'] in self.scheduled_nodes:
                allocated = self.bandwidth_allocation.get(link['from'], 0)
                min_req = link['min_bandwidth']
                max_req = link['max_bandwidth']
                
                if min_req <= allocated <= max_req:
                    # 奖励：分配的带宽越接近最大值越好
                    satisfaction = (allocated - min_req) / (max_req - min_req)
                    bandwidth_satisfaction += satisfaction
        
        # 3. 网络效率奖励（同一物理节点上的虚拟节点间通信成本为0）
        network_efficiency = 0
        for v1 in self.scheduled_nodes:
            for v2 in self.scheduled_nodes:
                if v1 != v2:
                    p1 = self.node_mapping[v1]
                    p2 = self.node_mapping[v2]
                    if p1 == p2:  # 同一物理节点
                        network_efficiency += 1
        
        # 综合奖励
        reward = (0.3 * cpu_balance + 
                 0.3 * memory_balance + 
                 0.3 * bandwidth_satisfaction + 
                 0.1 * network_efficiency)
        
        return reward

class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, 
                 physical_node_dim: int,
                 virtual_node_dim: int,
                 num_physical_nodes: int,
                 bandwidth_levels: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.actor = Actor(physical_node_dim, virtual_node_dim, 
                          num_physical_nodes=num_physical_nodes,
                          bandwidth_levels=bandwidth_levels).to(self.device)
        self.critic = Critic(physical_node_dim, virtual_node_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state, virtual_node_idx):
        """选择动作"""
        with torch.no_grad():
            physical_features = state['physical_features'].to(self.device)
            virtual_features = state['virtual_features'].to(self.device)
            physical_edge_index = state['physical_edge_index'].to(self.device)
            virtual_edge_index = state['virtual_edge_index'].to(self.device)
            
            mapping_logits, bandwidth_logits = self.actor(
                physical_features, physical_edge_index, None,
                virtual_features, virtual_edge_index, None,
                virtual_node_idx
            )
            
            # 采样动作
            mapping_probs = F.softmax(mapping_logits, dim=-1)
            bandwidth_probs = F.softmax(bandwidth_logits, dim=-1)
            
            mapping_dist = torch.distributions.Categorical(mapping_probs)
            bandwidth_dist = torch.distributions.Categorical(bandwidth_probs)
            
            mapping_action = mapping_dist.sample()
            bandwidth_action = bandwidth_dist.sample()
            
            mapping_log_prob = mapping_dist.log_prob(mapping_action)
            bandwidth_log_prob = bandwidth_dist.log_prob(bandwidth_action)
            
            # 计算价值
            value = self.critic(physical_features, physical_edge_index, None,
                              virtual_features, virtual_edge_index, None)
            
            return (mapping_action.item(), bandwidth_action.item()), \
                   (mapping_log_prob.item(), bandwidth_log_prob.item()), \
                   value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def update(self):
        """更新策略"""
        if len(self.states) == 0:
            return
        
        # 计算优势函数
        advantages = self._compute_advantages()
        
        # 转换为张量
        states_batch = self.states
        actions_batch = self.actions
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # 多次更新
        for _ in range(10):  # 通常更新10次
            # 计算新的动作概率和价值
            new_log_probs = []
            new_values = []
            
            for i, state in enumerate(states_batch):
                physical_features = state['physical_features'].to(self.device)
                virtual_features = state['virtual_features'].to(self.device)
                physical_edge_index = state['physical_edge_index'].to(self.device)
                virtual_edge_index = state['virtual_edge_index'].to(self.device)
                
                mapping_logits, bandwidth_logits = self.actor(
                    physical_features, physical_edge_index, None,
                    virtual_features, virtual_edge_index, None,
                    state['current_virtual_node']
                )
                
                mapping_probs = F.softmax(mapping_logits, dim=-1)
                bandwidth_probs = F.softmax(bandwidth_logits, dim=-1)
                
                mapping_dist = torch.distributions.Categorical(mapping_probs)
                bandwidth_dist = torch.distributions.Categorical(bandwidth_probs)
                
                mapping_action, bandwidth_action = actions_batch[i]
                mapping_log_prob = mapping_dist.log_prob(torch.tensor(mapping_action).to(self.device))
                bandwidth_log_prob = bandwidth_dist.log_prob(torch.tensor(bandwidth_action).to(self.device))
                
                new_log_probs.append(mapping_log_prob + bandwidth_log_prob)
                
                value = self.critic(physical_features, physical_edge_index, None,
                                  virtual_features, virtual_edge_index, None)
                new_values.append(value)
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(new_values, advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device))
            
            # 熵损失
            entropy_loss = -(mapping_probs * torch.log(mapping_probs + 1e-8)).sum() - \
                          (bandwidth_probs * torch.log(bandwidth_probs + 1e-8)).sum()
            
            # 总损失
            total_loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 更新网络
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def _compute_advantages(self):
        """计算优势函数"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages

def train_ppo_scheduler(env, agent, num_episodes=1000):
    """训练PPO调度器"""
    
    for episode in range(num_episodes):
        # 生成随机环境状态
        physical_state = generate_random_physical_state(env.num_physical_nodes)
        virtual_work = generate_random_virtual_work(env.max_virtual_nodes)
        
        state = env.reset(physical_state, virtual_work)
        episode_reward = 0
        
        while True:
            # 选择动作
            action, log_prob, value = agent.select_action(state, state['current_virtual_node'])
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 更新策略
        agent.update()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {episode_reward}")

def generate_random_physical_state(num_nodes):
    """生成随机物理网络状态"""
    return {
        'cpu': np.random.uniform(50, 100, num_nodes),
        'memory': np.random.uniform(100, 200, num_nodes),
        'links': []  # 简化处理
    }

def generate_random_virtual_work(num_nodes):
    """生成随机虚拟工作"""
    return {
        'cpu': np.random.uniform(10, 30, num_nodes),
        'memory': np.random.uniform(20, 50, num_nodes),
        'links': [
            {
                'from': i,
                'to': j,
                'min_bandwidth': np.random.uniform(10, 50),
                'max_bandwidth': np.random.uniform(50, 100)
            }
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j and np.random.random() < 0.5  # 50%的连接概率
        ]
    }

# 使用示例
if __name__ == "__main__":
    # 创建环境和智能体
    env = NetworkSchedulerEnvironment(num_physical_nodes=10, max_virtual_nodes=8)
    agent = PPOAgent(physical_node_dim=2, virtual_node_dim=2, 
                    num_physical_nodes=10, bandwidth_levels=10)
    
    # 训练
    train_ppo_scheduler(env, agent, num_episodes=1000)
