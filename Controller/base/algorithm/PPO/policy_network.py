import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from gnn_encoder import GNNEncoder

class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 任务拓扑编码器
        self.task_gnn = GNNEncoder(
            node_dim=config['task_node_dim'],
            edge_dim=config['task_edge_dim'],
            hidden_dim=config['gnn_hidden_dim']
        )
        
        # 物理拓扑编码器
        self.phys_gnn = GNNEncoder(
            node_dim=config['phys_node_dim'],
            edge_dim=config['phys_edge_dim'],
            hidden_dim=config['gnn_hidden_dim']
        )
        
        # 联合决策层
        self.task_head = nn.Sequential(
            nn.Linear(2 * config['gnn_hidden_dim'], 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, config['num_phys_nodes'])
        )
        
        # 带宽分配头（输出均值和标准差）
        self.bw_head = nn.Sequential(
            nn.Linear(2 * config['gnn_hidden_dim'], 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 2)  # 输出均值和log(std)
        )
        
        # # 路径选择头（动态处理路径数量）
        # self.path_head = nn.Sequential(
        #     nn.Linear(2 * config['gnn_hidden_dim'], 256),
        #     nn.ReLU(),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, 1)  # 输出单个值，用于路径评分
        # )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(2 * config['gnn_hidden_dim'], 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )
        
        # 初始化
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, task_data, phys_data):
        # 编码任务拓扑
        task_emb = self.task_gnn(task_data)  # [batch_size, hidden_dim]
        
        # 编码物理拓扑
        phys_emb = self.phys_gnn(phys_data)  # [batch_size, hidden_dim]
        
        # 联合特征
        joint_feat = torch.cat([task_emb, phys_emb], dim=1)  # [batch_size, 2*hidden_dim]
        
        # 输出头
        task_logits = self.task_head(joint_feat)  # [batch_size, num_phys_nodes]
        bw_params = self.bw_head(joint_feat)  # [batch_size, 2]
        # path_score = self.path_head(joint_feat)  # [batch_size, 1]
        value = self.value_head(joint_feat)  # [batch_size, 1]
        
        # 分离带宽均值和标准差
        bw_mean = torch.sigmoid(bw_params[:, 0])  # [0,1]范围
        bw_log_std = bw_params[:, 1]
        bw_std = torch.exp(bw_log_std) + 1e-6  # 确保正值
        
        return task_logits, bw_mean, bw_std, value
    
    def act(self, state, task_mask=None, path_mask=None, exploration=True):
        task_data, phys_data = state
        
        # 获取网络输出
        # task_logits, bw_mean, bw_std, path_score, value = self(task_data, phys_data)
        task_logits, bw_mean, bw_std, value = self(task_data, phys_data)
        
        # 应用任务分配掩码
        if task_mask is not None:
            task_logits = task_logits.masked_fill(~task_mask.bool(), -1e9)
        
        # 任务分配（离散动作）
        task_dist = Categorical(logits=task_logits)
        if exploration:
            task_action = task_dist.sample()
        else:
            task_action = torch.argmax(task_logits, dim=-1)
        task_log_prob = task_dist.log_prob(task_action)
        
        # 带宽分配（连续动作）
        bw_dist = Normal(bw_mean, bw_std)
        bw_action = bw_dist.sample()
        bw_log_prob = bw_dist.log_prob(bw_action)
        
        # # 路径选择（带掩码）
        # if path_mask is not None:
        #     # 应用路径掩码
        #     path_score = path_score.masked_fill(~path_mask.bool(), -1e9)
        
        # path_dist = Categorical(logits=path_score)
        # if exploration:
        #     path_action = path_dist.sample()
        # else:
        #     path_action = torch.argmax(path_score, dim=-1)
        # path_log_prob = path_dist.log_prob(path_action)
        
        action = {
            'task': task_action,
            'bw': bw_action,
            # 'path': path_action
        }
        
        # 加权对数概率（考虑不同动作空间）
        log_prob = (
            0.6 * task_log_prob +
            0.4 * bw_log_prob 
            # 0.2 * path_log_prob
        )
        
        return action, log_prob, value