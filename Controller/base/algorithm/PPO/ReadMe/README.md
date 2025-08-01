# PPO网络资源调度算法

## 概述

本项目实现了一个基于PPO（Proximal Policy Optimization）算法的网络资源调度器，用于解决虚拟网络功能（VNF）的部署和资源分配问题。

## 问题描述

### 物理环境
- 多个物理节点，每个节点具有CPU和内存资源
- 物理节点间有网络连接，具有上行和下行带宽
- 网络拓扑可以是任意的连接结构

### 虚拟工作
- 包含多个虚拟任务节点，每个节点有CPU和内存需求
- 虚拟节点间有网络连接需求，带宽需求为区间值（最小值和最大值）
- 需要将虚拟节点映射到物理节点，并分配具体的带宽

### 调度目标
1. **资源负载均衡**：平衡物理节点的CPU和内存使用率
2. **带宽满足度**：尽可能满足虚拟链路的带宽需求
3. **网络效率**：同一物理节点上的虚拟节点间通信成本为0
4. **网络利用率平衡**：平衡上行和下行带宽的使用

## 算法设计

### 1. 状态表示
使用图神经网络（GNN）编码物理网络和虚拟网络：

- **物理节点特征**：CPU利用率、内存利用率
- **虚拟节点特征**：CPU需求、内存需求
- **边特征**：网络连接关系
- **图结构**：使用GAT（Graph Attention Network）进行特征提取

### 2. 动作空间
采用两个离散动作：
- **节点映射动作**：将虚拟节点映射到物理节点
- **带宽分配动作**：为虚拟链路分配具体带宽值

### 3. 网络架构

#### Actor网络
- 图编码器：编码物理和虚拟网络
- 注意力机制：计算虚拟节点与物理节点的匹配度
- 策略头：生成节点映射和带宽分配的概率分布

#### Critic网络
- 图编码器：编码物理和虚拟网络
- 价值头：评估当前状态的价值

### 4. 奖励函数
综合多个目标的奖励函数：

```
Reward = 0.25 × CPU负载均衡 + 
         0.25 × 内存负载均衡 +  
         0.25 × 带宽满足度 + 
         0.15 × 网络效率 + 
         0.10 × 网络利用率平衡
```

### 5. 约束处理
- **资源约束**：确保物理节点有足够的CPU和内存资源
- **带宽约束**：使用Dijkstra算法找到最短路径，确保路径上所有链路都有足够带宽
- **动作约束**：无效动作给予负奖励

## 文件结构

```
PPO/
├── ppo.py                 # 主要的PPO算法实现
├── network_scheduler.py   # 网络调度器和拓扑管理
├── replaybuffer.py        # 经验回放缓冲区
├── train_ppo.py          # 训练脚本
├── test_ppo.py           # 测试脚本
├── test_training.py      # 训练测试脚本
├── load_and_test.py      # 模型加载和测试脚本
├── demo.py               # 完整演示脚本
├── requirements.txt      # 依赖包列表
├── README.md            # 说明文档
├── BUGFIX.md           # Bug修复说明
├── models/              # 模型文件目录
│   ├── ppo_model_episode_*.pth
│   └── ppo_model_final.pth
└── stats/               # 训练统计文件目录
    ├── training_stats_episode_*.json
    └── training_stats_final.json
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本训练

```python
from train_ppo import PPOTrainer

# 创建训练器
trainer = PPOTrainer(
    num_physical_nodes=10,
    max_virtual_nodes=8,
    bandwidth_levels=10,
    hidden_dim=128,
    lr=3e-4
)

# 开始训练
trainer.train(num_episodes=1000)
```

### 2. 自定义网络拓扑

```python
from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler

# 创建自定义拓扑
topology = NetworkTopology(num_nodes=10)

# 设置节点资源
for i in range(10):
    topology.set_node_resources(i, cpu=100, memory=200)

# 添加链路
topology.add_link(0, 1, uplink=500, downlink=500)
topology.add_link(1, 2, uplink=300, downlink=300)

# 创建虚拟工作
virtual_work = VirtualWork(num_nodes=5)
virtual_work.set_node_requirement(0, cpu=20, memory=40)
virtual_work.add_link_requirement(0, 1, min_bandwidth=10, max_bandwidth=50)

# 创建调度器
scheduler = NetworkScheduler(topology)
```

### 3. 使用训练好的模型

```python
# 加载模型
trainer.load_model("ppo_model_final.pth")  # 会自动从models/目录加载

# 进行推理
state = get_current_state()
action, _, _ = trainer.agent.select_action(state, virtual_node_idx)
```

### 4. 自定义节点数量和资源范围

```python
# 创建训练器，使用自定义节点范围和资源范围
trainer = PPOTrainer(
    # 自定义节点数量范围
    num_physical_nodes_range=(6, 12),      # 物理节点范围：6-12
    max_virtual_nodes_range=(4, 8),        # 虚拟节点范围：4-8
    # 自定义物理节点资源范围
    physical_cpu_range=(100.0, 300.0),
    physical_memory_range=(200.0, 600.0),
    physical_bandwidth_range=(200.0, 1500.0),
    physical_connectivity_prob=0.5,
    # 自定义虚拟节点资源范围
    virtual_cpu_range=(20.0, 80.0),
    virtual_memory_range=(40.0, 150.0),
    virtual_bandwidth_range=(20.0, 300.0),
    virtual_connectivity_prob=0.6
)
```

### 5. 不对称带宽支持

算法支持物理网络和虚拟网络的不对称带宽：

```python
# 物理网络不对称链路
topology.add_link(0, 1, 500, 450)  # 0->1: 500, 1->0: 450

# 虚拟网络不对称链路需求
virtual_work.add_link_requirement(0, 1, 
                                50, 100,  # 0->1: 最小50，最大100
                                60, 90)   # 1->0: 最小60，最大90
```

### 6. 快速训练和测试

```bash

# 快速训练（20个episodes）
python quick_train.py
```

## 配置参数

### 网络参数
- `num_physical_nodes_range`: 物理节点数量范围 (默认: (8, 15))
- `max_virtual_nodes_range`: 虚拟节点数量范围 (默认: (5, 12))
- `bandwidth_levels`: 带宽分配等级数

### 资源范围参数
- `physical_cpu_range`: 物理节点CPU资源范围 (默认: (50.0, 200.0))
- `physical_memory_range`: 物理节点内存资源范围 (默认: (100.0, 400.0))
- `physical_bandwidth_range`: 物理节点带宽资源范围 (默认: (100.0, 1000.0))
- `virtual_cpu_range`: 虚拟节点CPU需求范围 (默认: (10.0, 50.0))
- `virtual_memory_range`: 虚拟节点内存需求范围 (默认: (20.0, 100.0))
- `virtual_bandwidth_range`: 虚拟节点带宽需求范围 (默认: (10.0, 200.0))

### 连接概率参数
- `physical_connectivity_prob`: 物理网络连接概率 (默认: 0.3)
- `virtual_connectivity_prob`: 虚拟网络连接概率 (默认: 0.4)

### 文件管理参数
- `model_dir`: 模型文件保存目录（默认："models"）
- `stats_dir`: 统计文件保存目录（默认："stats"）

### 训练参数
- `lr`: 学习率
- `gamma`: 折扣因子
- `gae_lambda`: GAE参数
- `clip_ratio`: PPO裁剪比例
- `value_loss_coef`: 价值损失系数
- `entropy_coef`: 熵损失系数
- `update_epochs`: 每次更新的轮数
- `batch_size`: 批次大小

### 网络架构参数
- `hidden_dim`: 隐藏层维度
- `num_layers`: 图卷积层数
- `num_heads`: 注意力头数

## 训练监控

训练过程中会记录以下指标：
- 训练奖励
- Episode长度
- Actor损失
- Critic损失
- 熵损失

可以通过以下方式查看训练曲线：

```python
trainer.plot_training_curves()
```

## 性能优化建议

1. **GPU加速**：使用GPU可以显著加速训练
2. **批量大小**：根据GPU内存调整batch_size
3. **学习率调度**：使用学习率衰减可以提高收敛性
4. **经验回放**：使用优先级经验回放可以提高样本效率
5. **多进程**：使用多进程生成环境数据

## 扩展功能

### 1. 支持连续动作空间
可以将带宽分配改为连续动作，使用正态分布采样。

### 2. 多目标优化
可以使用Pareto优化或加权方法处理多个冲突目标。

### 3. 动态环境
支持物理网络状态的动态变化。

### 4. 迁移学习
在不同规模的网络上进行迁移学习。

## 故障排除

### 常见问题

1. **内存不足**：减少batch_size或hidden_dim
2. **训练不收敛**：调整学习率或奖励函数权重
3. **动作无效**：检查约束条件设置
4. **图结构错误**：确保边索引格式正确

### 调试技巧

1. 打印中间状态和动作
2. 可视化网络拓扑
3. 监控奖励分解
4. 检查梯度范数

## 参考文献

1. Schulman, J., et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
2. Veličković, P., et al. "Graph attention networks." arXiv preprint arXiv:1710.10903 (2017).
3. Kipf, T. N., & Welling, M. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016). 