import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        # edge_nn1: 输入edge_dim，输出hidden_dim*node_dim
        self.edge_nn1 = nn.Linear(edge_dim, hidden_dim * node_dim)
        self.conv1 = NNConv(node_dim, hidden_dim, nn=self.edge_nn1)
        # edge_nn2: 输入edge_dim，输出hidden_dim*hidden_dim
        self.edge_nn2 = nn.Linear(edge_dim, hidden_dim * hidden_dim)
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn=self.edge_nn2)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        graph_emb = global_mean_pool(x, data.batch)
        return graph_emb