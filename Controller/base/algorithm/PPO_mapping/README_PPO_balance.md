# PPO_balance算法说明

## 概述

PPO_balance算法结合了PPO_mapping的PPO算法实现和A2C算法的状态/奖励设计，旨在通过负载均衡奖励机制优化网络资源调度。

## 算法特点

### 1. 算法融合设计
- **算法框架**: 基于PPO_mapping的PPO（Proximal Policy Optimization）算法
- **状态表示**: 采用A2C的简化状态表示
- **奖励设计**: 使用A2C的负载均衡奖励函数

### 2. 状态空间设计（来自A2C）

#### 状态组成
```python
state = {
    'physical_resources': tensor([num_physical_nodes, 2]),  # 每个物理节点的[CPU可用量, 内存可用量]
    'task_requirements': tensor([2]),                      # 当前任务的[CPU需求, 内存需求] 
    'valid_actions': tensor([num_physical_nodes]),         # 有效动作掩码
    'current_task_index': int,                            # 当前任务索引
    'total_tasks': int,                                   # 总任务数
    'task_mapping': dict                                  # 已完成的任务映射
}
```

#### 状态编码
- 物理节点资源: `[num_physical_nodes * 2]` - 每个节点的CPU和内存可用量
- 任务需求: `[2]` - 当前任务的CPU和内存需求
- 总状态维度: `2 * num_physical_nodes + 2`

### 3. 动作空间
- **动作类型**: 选择物理节点索引
- **动作范围**: `[0, num_physical_nodes-1]`
- **动作掩码**: 根据资源可用性限制有效动作

### 4. 奖励设计（来自A2C）

#### 即时奖励
1. **预调度检查**: 验证资源是否满足任务需求
2. **负载均衡奖励**: 基于资源分布均匀性的数学公式

#### 负载均衡奖励公式
```
τ = max(每个物理节点的可用资源量)
ϕ = mean(所有物理节点的可用资源量)  
ω = ϕ / (τ · M), 其中M是物理节点数
R = ω_cpu + ω_memory
```

#### 最终奖励
- **带宽分配成功**: +0.5
- **带宽分配失败**: -0.3
- **无成功映射**: -0.5

### 5. PPO算法特性（来自PPO_mapping）

#### 网络架构
```python
# 状态编码器
state_encoder: Linear(state_dim, hidden_dim*2) -> ReLU -> Dropout -> Linear(hidden_dim*2, hidden_dim) -> ReLU

# 策略网络 
policy: Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_physical_nodes)

# 价值网络
critic: Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)
```

#### PPO损失函数
```python
# 策略损失（PPO clip）
ratio = exp(current_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1-ε, 1+ε) * advantage
policy_loss = -min(surr1, surr2)

# 价值损失
value_loss = MSE(predicted_value, target_value)

# 熵损失（鼓励探索）
entropy_loss = -entropy_coefficient * entropy

# 总损失
total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

## 文件结构

```
PPO_mapping/
├── balance_environment.py    # PPO_balance环境实现
├── balance_agent.py         # PPO_balance智能体实现  
├── train_balance.py         # 训练脚本
├── README_PPO_balance.md    # 说明文档（本文件）
└── balance_results/         # 训练结果目录（运行时创建）
    └── balance_ppo_YYYYMMDD_HHMMSS_XXXXXXXX/
        ├── checkpoints/     # 模型检查点
        │   ├── best_model.pth           # 最佳模型（基于接受率）
        │   ├── best_model_info.json     # 最佳模型信息
        │   ├── model_ep_XXX.pth         # 定期检查点
        │   ├── model_final.pth          # 最终模型
        │   ├── stats_ep_XXX.npz         # 定期统计数据
        │   └── stats_final.npz          # 最终统计数据
        ├── plots/          # 训练曲线图
        │   └── training_curves.png      # 训练曲线图
        └── training_summary.json       # 完整训练总结
```

## 使用方法

### 1. 训练模型
```bash
cd /home/qianguo/Edge-Scheduler/Controller/base/algorithm/PPO_mapping
python train_balance.py
```

### 2. 自定义配置
```python
from train_balance import BalancePPOTrainer

# 环境配置
env_config = {
    'num_physical_nodes': 10,
    'task_nodes_range': (3, 6),
    'physical_cpu_range': (50, 100),
    'physical_memory_range': (50, 100),
    # ... 其他配置
}

# 智能体配置  
agent_config = {
    'hidden_dim': 128,
    'lr': 3e-4,
}

# 训练配置
training_config = {
    'total_episodes': 2000,
    'batch_size': 32,
    'update_frequency': 32,
    # ... 其他配置
}

# 创建训练器并训练
trainer = BalancePPOTrainer(env_config, agent_config, training_config)
trainer.train()
```

## 算法优势

### 1. 结合两者优势
- **PPO稳定性**: 使用PPO的稳定训练机制，避免策略更新过大
- **A2C简洁性**: 采用A2C的简化状态表示，降低计算复杂度
- **负载均衡**: 专注于负载均衡优化，提高资源利用效率

### 2. 实际应用价值
- **资源均衡**: 通过负载均衡奖励促进资源均匀分布
- **预调度**: 提前验证资源约束，减少无效映射
- **智能带宽**: 使用启发式智能带宽分配策略

### 3. 训练稳定性
- **动作掩码**: 限制无效动作，提高训练效率
- **数值稳定**: 多重数值稳定性检查，防止训练崩溃
- **梯度裁剪**: 防止梯度爆炸问题

## 关键参数

### 环境参数
- `num_physical_nodes`: 物理节点数量 (默认: 10)
- `task_nodes_range`: 任务节点数范围 (默认: (3, 6))
- `physical_cpu_range`: 物理CPU资源范围 (默认: (50, 100))
- `physical_memory_range`: 物理内存资源范围 (默认: (50, 100))

### 训练参数
- `total_episodes`: 训练总轮数 (默认: 2000)
- `batch_size`: 批次大小 (默认: 32)
- `update_frequency`: 更新频率 (默认: 32)
- `ppo_clip`: PPO裁剪参数 (默认: 0.2)
- `lr`: 学习率 (默认: 3e-4)

## 模型保存机制

### 保存策略（与PPO_mapping一致）
1. **最佳模型保存**: 基于任务接受率自动保存性能最好的模型
2. **定期检查点**: 每隔一定episodes保存训练检查点
3. **最终模型**: 训练结束时保存最终模型
4. **统计数据**: 同步保存训练统计数据（.npz格式）
5. **错误恢复**: 支持Ctrl+C中断和异常情况下的进度保存

### 保存的文件类型
- **模型文件**: .pth格式，包含模型参数和优化器状态
- **统计数据**: .npz格式，包含所有训练指标历史
- **模型信息**: .json格式，记录最佳模型的详细信息
- **训练总结**: .json格式，完整的训练配置和结果

## 输出指标

### 训练指标
- **Episode奖励**: 每轮的累积奖励
- **任务接受率**: 成功调度的任务比例
- **带宽分配成功率**: 带宽分配成功的episode比例
- **负载均衡奖励**: 平均负载均衡奖励值
- **训练损失**: PPO各项损失函数值

### 可视化
训练过程中自动生成以下曲线图：
1. Episode奖励变化
2. 任务接受率变化  
3. 带宽分配成功率变化
4. 负载均衡奖励变化
5. 训练损失变化
6. Episode长度变化

## 算法比较

| 算法 | 状态表示 | 奖励设计 | 算法框架 | 特点 |
|------|----------|----------|----------|------|
| **PPO_mapping** | 复杂图状态 | 映射+资源效率 | PPO | 节点映射专用 |
| **A2C** | 简化资源状态 | 负载均衡 | A2C | 负载均衡专用 |
| **PPO_balance** | 简化资源状态 | 负载均衡 | PPO | 结合两者优势 |

## 注意事项

1. **依赖性**: 需要network_scheduler模块支持
2. **资源要求**: 建议GPU加速训练
3. **随机种子**: 建议设置固定种子以确保可重现性
4. **参数调优**: 根据具体应用场景调整超参数

## 扩展方向

1. **多目标优化**: 结合更多优化目标（能耗、延迟等）
2. **动态调整**: 支持在线学习和动态环境
3. **分层架构**: 结合分层决策架构
4. **注意力机制**: 引入注意力机制提高表征能力
