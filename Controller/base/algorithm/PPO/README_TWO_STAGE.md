# 两阶段PPO网络调度器使用指南

## 🎯 概述

这是一个基于两阶段独立Actor的PPO网络资源调度算法，专门用于解决虚拟任务到物理节点的映射和带宽分配问题。

### 核心特点
- **两阶段设计**: 映射Actor负责节点映射，带宽Actor负责带宽分配
- **联合训练**: 两个Actor协同工作，共享Critic网络
- **约束感知**: 内置资源约束和带宽约束检查
- **灵活配置**: 支持动态节点数量和资源范围

## 📁 文件结构

```
Controller/base/algorithm/PPO/
├── two_stage_actor_design.py      # 两阶段Actor架构
├── two_stage_environment.py       # 两阶段环境
├── train_two_stage_ppo.py         # 训练脚本
├── two_stage_design_summary.md    # 设计总结
├── simple_two_stage_test.py       # 简化测试
├── README_TWO_STAGE.md           # 使用指南
├── models/                        # 模型保存目录
└── stats/                         # 训练统计目录
```

## 🚀 快速开始

### 1. 环境准备

确保已安装所需依赖：
```bash
pip install torch torch-geometric networkx matplotlib scikit-learn tqdm
```

### 2. 基本测试

运行简化测试验证基本功能：
```bash
python simple_two_stage_test.py
```

运行完整架构测试：
```bash
python two_stage_actor_design.py
```

运行环境测试：
```bash
python two_stage_environment.py
```

### 3. 开始训练

```python
from train_two_stage_ppo import TwoStagePPOTrainer

# 创建训练器
trainer = TwoStagePPOTrainer(
    num_physical_nodes_range=(5, 8),
    max_virtual_nodes_range=(3, 6),
    bandwidth_levels=10,
    physical_cpu_range=(50.0, 200.0),
    physical_memory_range=(100.0, 400.0),
    physical_bandwidth_range=(100.0, 1000.0),
    virtual_cpu_range=(10.0, 50.0),
    virtual_memory_range=(20.0, 100.0),
    virtual_bandwidth_range=(10.0, 200.0),
    physical_connectivity_prob=0.3,
    virtual_connectivity_prob=0.4
)

# 开始训练
trainer.train(num_episodes=500, save_interval=100, eval_interval=50)

# 测试智能体
trainer.test_agent(num_test_episodes=10)
```

## ⚙️ 配置参数

### 环境参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_physical_nodes_range` | Tuple[int, int] | (5, 10) | 物理节点数量范围 |
| `max_virtual_nodes_range` | Tuple[int, int] | (3, 8) | 虚拟节点数量范围 |
| `bandwidth_levels` | int | 10 | 带宽离散等级数 |
| `physical_cpu_range` | Tuple[float, float] | (50.0, 200.0) | 物理节点CPU范围 |
| `physical_memory_range` | Tuple[float, float] | (100.0, 400.0) | 物理节点内存范围 |
| `physical_bandwidth_range` | Tuple[float, float] | (100.0, 1000.0) | 物理链路带宽范围 |
| `virtual_cpu_range` | Tuple[float, float] | (10.0, 50.0) | 虚拟节点CPU需求范围 |
| `virtual_memory_range` | Tuple[float, float] | (20.0, 100.0) | 虚拟节点内存需求范围 |
| `virtual_bandwidth_range` | Tuple[float, float] | (10.0, 200.0) | 虚拟链路带宽需求范围 |
| `physical_connectivity_prob` | float | 0.3 | 物理网络连接概率 |
| `virtual_connectivity_prob` | float | 0.4 | 虚拟网络连接概率 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lr` | float | 3e-4 | 学习率 |
| `gamma` | float | 0.99 | 折扣因子 |
| `gae_lambda` | float | 0.95 | GAE参数 |
| `clip_ratio` | float | 0.2 | PPO裁剪比例 |
| `value_loss_coef` | float | 0.5 | 价值损失系数 |
| `entropy_coef` | float | 0.01 | 熵损失系数 |

## 🏗️ 架构详解

### MappingActor (映射Actor)

**功能**: 为所有虚拟节点选择物理节点映射

**输入**:
- 物理网络状态: `[num_physical_nodes, physical_node_dim]`
- 虚拟网络状态: `[num_virtual_nodes, virtual_node_dim]`

**输出**:
- 映射logits: `[num_virtual_nodes, num_physical_nodes]`
- 约束分数: `[num_virtual_nodes, num_physical_nodes]`

**网络结构**:
```python
GraphEncoder(physical_node_dim, hidden_dim)
GraphEncoder(virtual_node_dim, hidden_dim)
MultiheadAttention(hidden_dim, num_heads=8)
Linear(hidden_dim * 2, num_physical_nodes)  # 映射头
Linear(hidden_dim * 2, 1)  # 约束检查器
```

### BandwidthActor (带宽Actor)

**功能**: 为所有虚拟链路分配带宽等级

**输入**:
- 网络状态 + 映射结果

**输出**:
- 带宽logits: `[num_links, bandwidth_levels]`
- 约束分数: `[num_links, bandwidth_levels]`

**网络结构**:
```python
GraphEncoder(physical_node_dim, hidden_dim)
GraphEncoder(virtual_node_dim, hidden_dim)
Linear(virtual_node_dim * 2, hidden_dim)  # 链路编码器
MultiheadAttention(hidden_dim, num_heads=8)
Linear(hidden_dim * 3, bandwidth_levels)  # 带宽头
Linear(hidden_dim * 3, 1)  # 约束检查器
```

### Critic (价值网络)

**功能**: 评估整体状态价值

**网络结构**:
```python
GraphEncoder(physical_node_dim, hidden_dim)
GraphEncoder(virtual_node_dim, hidden_dim)
Linear(hidden_dim * 2, 1)  # 价值头
```

## 🎯 动作空间

### 映射动作
```python
mapping_action = [4, 3, 1, 4]  # 4个虚拟节点分别映射到物理节点4,3,1,4
```

### 带宽动作
```python
bandwidth_action = [4, 4, 4, 2, 5, 1]  # 6个链路分别选择带宽等级4,4,4,2,5,1
```

### 带宽等级映射
```python
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

## 🏆 奖励函数

### 多目标奖励
```python
reward = (
    resource_utilization_reward * 0.3 +    # 资源利用率 (30%)
    load_balancing_reward * 0.3 +          # 负载均衡 (30%)
    bandwidth_satisfaction_reward * 0.4     # 带宽满足度 (40%)
)
```

### 约束处理
- 无效动作给予 -10.0 的惩罚
- 资源超载、带宽不足等约束违反会被检测并惩罚

## 📊 训练监控

### 训练统计
- Episode奖励
- 约束违反率
- 资源利用率
- 负载均衡度
- 带宽满足度

### 可视化
训练完成后会自动生成训练曲线图：
- Episode奖励曲线
- 约束违反率曲线
- 平均奖励移动平均
- 约束违反数量曲线

## 💾 模型管理

### 保存模型
```python
trainer.save_model(episode=500)
# 保存到: models/two_stage_ppo_model_episode_500.pth
```

### 加载模型
```python
trainer.load_model(episode=500)
```

### 保存统计
```python
trainer.save_training_stats(episode=500)
# 保存到: stats/two_stage_training_stats_episode_500.json
```

## 🧪 测试和评估

### 智能体测试
```python
test_stats = trainer.test_agent(num_test_episodes=10)
print(f"平均奖励: {test_stats['rewards'].mean():.3f}")
print(f"有效动作率: {test_stats['valid_actions']/test_stats['total_actions']:.2%}")
```

### 性能指标
- **平均奖励**: 整体性能指标
- **约束违反率**: 约束满足情况
- **资源利用率**: 物理资源使用效率
- **负载均衡度**: 节点负载分布均匀性
- **带宽满足度**: 虚拟链路带宽需求满足情况

## 🔧 自定义配置

### 修改奖励权重
```python
# 在 two_stage_environment.py 中修改
def _calculate_reward(self, mapping_action, bandwidth_action):
    reward = 0.0
    
    # 调整权重
    resource_utilization_reward = self._calculate_resource_utilization(mapping_action)
    reward += resource_utilization_reward * 0.4  # 增加资源利用率权重
    
    load_balancing_reward = self._calculate_load_balancing(mapping_action)
    reward += load_balancing_reward * 0.3
    
    bandwidth_satisfaction_reward = self._calculate_bandwidth_satisfaction(bandwidth_action)
    reward += bandwidth_satisfaction_reward * 0.3  # 减少带宽满足度权重
    
    return reward
```

### 添加新约束
```python
# 在 _validate_actions 方法中添加新约束检查
def _validate_actions(self, mapping_action, bandwidth_action):
    constraint_violations = []
    
    # 现有约束检查...
    
    # 添加新约束
    for i, physical_node_idx in enumerate(mapping_action):
        # 检查新约束
        if not self._check_new_constraint(i, physical_node_idx):
            constraint_violations.append(f"新约束违反: 节点{i}")
    
    return len(constraint_violations) == 0, constraint_violations
```

## 🚨 常见问题

### 1. 维度不匹配错误
**问题**: `RuntimeError: The size of tensor a must match the size of tensor b`
**解决**: 检查虚拟节点数量是否与配置的`max_virtual_nodes`一致

### 2. 约束违反率高
**问题**: 智能体经常产生无效动作
**解决**: 
- 增加约束检查的奖励权重
- 调整资源范围参数
- 增加训练轮数

### 3. 训练不收敛
**问题**: 奖励不上升或波动很大
**解决**:
- 调整学习率
- 增加熵损失系数
- 检查奖励函数设计

### 4. 内存不足
**问题**: CUDA out of memory
**解决**:
- 减少`hidden_dim`
- 减少`max_virtual_nodes`
- 使用CPU训练

## 📈 性能优化建议

### 1. 网络架构优化
- 调整`hidden_dim`大小
- 修改注意力头数量
- 增加或减少网络层数

### 2. 训练策略优化
- 使用学习率调度
- 实现经验回放优先级
- 添加梯度裁剪

### 3. 奖励函数优化
- 根据具体需求调整权重
- 添加稀疏奖励
- 实现奖励塑形

### 4. 约束处理优化
- 实现软约束
- 添加约束违反惩罚的衰减
- 使用约束满足的奖励信号

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 开发环境
```bash
git clone <repository>
cd Controller/base/algorithm/PPO
pip install -r requirements.txt
```

### 代码规范
- 使用中文注释
- 遵循PEP 8代码风格
- 添加适当的类型注解
- 编写单元测试

## 📄 许可证

本项目采用MIT许可证。

---

🎉 **恭喜！您现在已经掌握了完整的两阶段PPO网络调度器使用方法！** 