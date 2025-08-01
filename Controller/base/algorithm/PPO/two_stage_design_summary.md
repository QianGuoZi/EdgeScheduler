# 两阶段PPO网络调度器设计总结

## 🎯 设计目标

将原有的单阶段PPO算法修改为两阶段独立Actor的联合训练架构：

1. **MappingActor**: 负责输出所有虚拟任务节点的映射结果
2. **BandwidthActor**: 负责输出所有虚拟链路的带宽分配结果（10个离散等级）
3. **联合训练**: 两个Actor协同工作，共享一个Critic网络

## 🏗️ 架构设计

### 1. **网络架构**

#### **MappingActor (映射Actor)**
```python
class MappingActor(nn.Module):
    def __init__(self, physical_node_dim, virtual_node_dim, hidden_dim=128, 
                 num_physical_nodes=10, max_virtual_nodes=8):
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 全局映射策略网络
        self.global_mapping_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_virtual_nodes * num_physical_nodes)
        )
        
        # 约束检查层
        self.constraint_checker = nn.Sequential(...)
```

**功能特点**：
- 一次性输出所有虚拟节点的映射决策
- 使用注意力机制计算虚拟节点对物理节点的匹配度
- 包含约束检查层确保资源满足
- 输出维度：`[num_virtual_nodes, num_physical_nodes]`

#### **BandwidthActor (带宽Actor)**
```python
class BandwidthActor(nn.Module):
    def __init__(self, physical_node_dim, virtual_node_dim, hidden_dim=128,
                 bandwidth_levels=10, max_virtual_nodes=8):
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 链路编码器
        self.link_encoder = nn.Sequential(
            nn.Linear(virtual_node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 全局带宽分配策略网络
        self.global_bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_links * bandwidth_levels)
        )
```

**功能特点**：
- 接收映射结果作为输入
- 一次性输出所有虚拟链路的带宽分配
- 考虑映射结果对物理路径的影响
- 输出维度：`[num_links, bandwidth_levels]`

#### **Critic (价值网络)**
```python
class Critic(nn.Module):
    def __init__(self, physical_node_dim, virtual_node_dim, hidden_dim=128):
        # 图编码器
        self.physical_encoder = GraphEncoder(physical_node_dim, hidden_dim)
        self.virtual_encoder = GraphEncoder(virtual_node_dim, hidden_dim)
        
        # 全局价值评估网络
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
```

**功能特点**：
- 评估整体状态的价值
- 为两个Actor提供价值信号
- 使用全局特征聚合

### 2. **智能体架构**

#### **TwoStagePPOAgent**
```python
class TwoStagePPOAgent:
    def __init__(self, ...):
        # 两个独立的Actor
        self.mapping_actor = MappingActor(...)
        self.bandwidth_actor = BandwidthActor(...)
        self.critic = Critic(...)
        
        # 独立的优化器
        self.mapping_optimizer = torch.optim.Adam(self.mapping_actor.parameters(), lr=lr)
        self.bandwidth_optimizer = torch.optim.Adam(self.bandwidth_actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 经验缓冲区
        self.states = []
        self.mapping_actions = []
        self.bandwidth_actions = []
        self.rewards = []
        self.values = []
        self.mapping_log_probs = []
        self.bandwidth_log_probs = []
        self.dones = []
```

**训练流程**：
1. **动作选择**: 先执行映射Actor，再执行带宽Actor
2. **经验存储**: 分别存储映射和带宽的经验
3. **网络更新**: 分别更新两个Actor和Critic

### 3. **环境设计**

#### **TwoStageNetworkSchedulerEnvironment**
```python
class TwoStageNetworkSchedulerEnvironment:
    def step(self, mapping_action, bandwidth_action):
        """
        执行两阶段动作
        
        Args:
            mapping_action: [num_virtual_nodes] 物理节点索引
            bandwidth_action: [num_links] 带宽等级
        
        Returns:
            next_state, reward, done, info
        """
        # 验证动作有效性
        is_valid, constraint_violations = self._validate_actions(mapping_action, bandwidth_action)
        
        if not is_valid:
            reward = -10.0  # 无效动作惩罚
        else:
            reward = self._calculate_reward(mapping_action, bandwidth_action)
        
        return next_state, reward, done, info
```

**特点**：
- 一步完成所有映射和带宽分配
- 严格的约束验证
- 多目标奖励函数

## 📊 动作空间设计

### 1. **映射动作空间**
```python
# 映射动作：每个虚拟节点选择一个物理节点
mapping_action = [2, 0, 1, 3]  # 4个虚拟节点分别映射到物理节点2,0,1,3
action_space_size = num_virtual_nodes * num_physical_nodes
```

### 2. **带宽动作空间**
```python
# 带宽动作：每个虚拟链路选择一个带宽等级
bandwidth_action = [5, 3, 7, 2, 8, 1]  # 6个链路分别选择带宽等级5,3,7,2,8,1
action_space_size = num_links * bandwidth_levels
```

### 3. **带宽等级映射**
```python
# 10个离散带宽等级
bandwidth_mapping = {
    0: 10.0,   # 最小带宽
    1: 31.1,
    2: 52.2,
    3: 73.3,
    4: 94.4,
    5: 115.6,  # 中等带宽
    6: 136.7,
    7: 157.8,
    8: 178.9,
    9: 200.0   # 最大带宽
}
```

## 🎯 奖励函数设计

### 1. **多目标奖励**
```python
def _calculate_reward(self, mapping_action, bandwidth_action):
    reward = 0.0
    
    # 1. 资源利用率奖励 (30%)
    resource_utilization_reward = self._calculate_resource_utilization(mapping_action)
    reward += resource_utilization_reward * 0.3
    
    # 2. 负载均衡奖励 (30%)
    load_balancing_reward = self._calculate_load_balancing(mapping_action)
    reward += load_balancing_reward * 0.3
    
    # 3. 带宽满足度奖励 (40%)
    bandwidth_satisfaction_reward = self._calculate_bandwidth_satisfaction(bandwidth_action)
    reward += bandwidth_satisfaction_reward * 0.4
    
    return reward
```

### 2. **约束处理**
```python
# 无效动作给予负奖励
if not is_valid:
    reward = -10.0
```

## 🔧 训练策略

### 1. **联合训练**
- 两个Actor同时训练
- 共享Critic网络的价值信号
- 使用相同的奖励函数

### 2. **经验回放**
```python
def store_transition(self, state, mapping_action, bandwidth_action, 
                    reward, value, mapping_log_prob, bandwidth_log_prob, done):
    self.states.append(state)
    self.mapping_actions.append(mapping_action)
    self.bandwidth_actions.append(bandwidth_action)
    self.rewards.append(reward)
    self.values.append(value)
    self.mapping_log_probs.append(mapping_log_prob)
    self.bandwidth_log_probs.append(bandwidth_log_prob)
    self.dones.append(done)
```

### 3. **网络更新**
```python
def update(self):
    # 计算优势函数
    advantages = self._compute_advantages()
    
    # 分别更新两个Actor
    self._update_mapping_actor(states, mapping_actions, old_mapping_log_probs, advantages)
    self._update_bandwidth_actor(states, bandwidth_actions, old_bandwidth_log_probs, advantages)
    
    # 更新Critic
    self._update_critic(states, returns)
```

## 📈 优势分析

### 1. **相比单阶段设计的优势**
- **更精确的决策**: 映射和带宽分配分别优化
- **更好的可解释性**: 可以分析每个阶段的性能
- **更灵活的架构**: 可以独立调整两个Actor
- **更低的动作空间复杂度**: 分解为两个较小的动作空间

### 2. **相比两个独立系统的优势**
- **协同优化**: 两个Actor共享价值信号
- **端到端训练**: 整体目标优化
- **更好的泛化**: 学习到映射和带宽的关联关系

## 🚀 使用方法

### 1. **训练**
```python
# 创建训练器
trainer = TwoStagePPOTrainer(
    num_physical_nodes_range=(5, 8),
    max_virtual_nodes_range=(3, 6),
    bandwidth_levels=10,
    # ... 其他参数
)

# 开始训练
trainer.train(num_episodes=500, save_interval=100, eval_interval=50)
```

### 2. **测试**
```python
# 测试智能体
test_stats = trainer.test_agent(num_test_episodes=10)
```

### 3. **模型保存和加载**
```python
# 保存模型
trainer.save_model(episode=500)

# 加载模型
trainer.load_model(episode=500)
```

## 📁 文件结构

```
Controller/base/algorithm/PPO/
├── two_stage_actor_design.py      # 两阶段Actor架构
├── two_stage_environment.py       # 两阶段环境
├── train_two_stage_ppo.py         # 训练脚本
├── two_stage_design_summary.md    # 设计总结
├── models/                        # 模型保存目录
└── stats/                         # 训练统计目录
```

## 🎯 预期效果

### 1. **性能提升**
- 更高的资源利用率
- 更好的负载均衡
- 更高的带宽满足度
- 更低的约束违反率

### 2. **训练稳定性**
- 更快的收敛速度
- 更稳定的训练过程
- 更好的泛化能力

### 3. **可扩展性**
- 易于添加新的约束
- 易于调整奖励权重
- 易于集成新的网络架构

## 🔮 未来改进方向

### 1. **架构优化**
- 添加注意力可视化
- 实现自适应学习率
- 添加经验回放优先级

### 2. **约束处理**
- 实现更复杂的路径约束
- 添加动态资源约束
- 实现软约束处理

### 3. **评估指标**
- 添加更多评估指标
- 实现A/B测试框架
- 添加性能基准测试

这个两阶段设计为网络资源调度提供了一个强大而灵活的解决方案！ 