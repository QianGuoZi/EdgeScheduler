# PPO算法资源参数化功能说明

## 概述

为了支持更灵活的网络资源调度场景，PPO算法现在支持自定义物理节点和虚拟节点的资源范围参数。这使得用户可以根据不同的应用场景调整资源分布，提高算法的适应性和实用性。

## 新增参数

### 1. 物理节点资源范围参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `physical_cpu_range` | Tuple[float, float] | (50.0, 200.0) | 物理节点CPU资源范围 |
| `physical_memory_range` | Tuple[float, float] | (100.0, 400.0) | 物理节点内存资源范围 |
| `physical_bandwidth_range` | Tuple[float, float] | (100.0, 1000.0) | 物理节点带宽资源范围 |

### 2. 虚拟节点资源范围参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `virtual_cpu_range` | Tuple[float, float] | (10.0, 50.0) | 虚拟节点CPU需求范围 |
| `virtual_memory_range` | Tuple[float, float] | (20.0, 100.0) | 虚拟节点内存需求范围 |
| `virtual_bandwidth_range` | Tuple[float, float] | (10.0, 200.0) | 虚拟节点带宽需求范围 |

### 3. 网络连接概率参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `physical_connectivity_prob` | float | 0.3 | 物理网络连接概率 |
| `virtual_connectivity_prob` | float | 0.4 | 虚拟网络连接概率 |

## 使用方法

### 1. 基本使用

```python
from train_ppo import PPOTrainer

# 使用默认参数
trainer = PPOTrainer(
    num_physical_nodes=10,
    max_virtual_nodes=8
)
```

### 2. 自定义资源范围

```python
# 创建高性能计算场景的训练器
trainer = PPOTrainer(
    num_physical_nodes=10,
    max_virtual_nodes=8,
    # 高CPU和内存配置
    physical_cpu_range=(200.0, 500.0),
    physical_memory_range=(400.0, 1000.0),
    physical_bandwidth_range=(500.0, 2000.0),
    physical_connectivity_prob=0.6,
    # 高计算需求
    virtual_cpu_range=(50.0, 150.0),
    virtual_memory_range=(100.0, 300.0),
    virtual_bandwidth_range=(50.0, 400.0),
    virtual_connectivity_prob=0.7
)
```

### 3. 边缘计算场景

```python
# 创建边缘计算场景的训练器
trainer = PPOTrainer(
    num_physical_nodes=15,
    max_virtual_nodes=10,
    # 中等资源配置
    physical_cpu_range=(80.0, 200.0),
    physical_memory_range=(150.0, 400.0),
    physical_bandwidth_range=(200.0, 800.0),
    physical_connectivity_prob=0.4,
    # 轻量级应用
    virtual_cpu_range=(15.0, 60.0),
    virtual_memory_range=(30.0, 120.0),
    virtual_bandwidth_range=(20.0, 150.0),
    virtual_connectivity_prob=0.5
)
```

### 4. 物联网场景

```python
# 创建物联网场景的训练器
trainer = PPOTrainer(
    num_physical_nodes=20,
    max_virtual_nodes=12,
    # 低功耗设备
    physical_cpu_range=(30.0, 100.0),
    physical_memory_range=(50.0, 200.0),
    physical_bandwidth_range=(50.0, 300.0),
    physical_connectivity_prob=0.2,
    # 微服务应用
    virtual_cpu_range=(5.0, 25.0),
    virtual_memory_range=(10.0, 50.0),
    virtual_bandwidth_range=(5.0, 80.0),
    virtual_connectivity_prob=0.3
)
```

## 修改的类和函数

### 1. PPOTrainer类

- **新增参数**: 所有资源范围参数
- **新增方法**: 
  - `_create_custom_physical_topology()`: 创建自定义物理网络拓扑
  - `_create_custom_virtual_work()`: 创建自定义虚拟工作
- **修改方法**: 
  - `_get_physical_edges()`: 使用自定义连接概率
  - `_get_virtual_edges()`: 使用自定义连接概率

### 2. NetworkSchedulerEnvironment类

- **新增参数**: 所有资源范围参数
- **修改方法**: 
  - `_get_physical_edges()`: 使用自定义连接概率

### 3. network_scheduler模块

- **修改函数**:
  - `create_sample_topology()`: 添加资源范围参数
  - `create_sample_virtual_work()`: 添加资源范围参数

## 参数影响分析

### 1. 资源范围对调度的影响

| 参数调整 | 对调度的影响 |
|----------|-------------|
| 增大物理资源范围 | 提高资源利用率，减少资源竞争 |
| 增大虚拟需求范围 | 增加调度难度，需要更智能的分配策略 |
| 减小资源范围 | 增加资源竞争，需要更精细的负载均衡 |

### 2. 连接概率对网络的影响

| 连接概率 | 网络特征 | 调度挑战 |
|----------|----------|----------|
| 高连接概率 (>0.5) | 密集连接，路径选择多 | 优化路径选择，减少网络拥塞 |
| 中等连接概率 (0.3-0.5) | 平衡的连接密度 | 平衡资源利用和网络效率 |
| 低连接概率 (<0.3) | 稀疏连接，路径受限 | 避免网络瓶颈，优化关键路径 |

## 测试和验证

### 1. 运行测试脚本

```bash
# 测试自定义资源范围功能
python test_custom_resources.py
```

### 2. 验证要点

- **资源分布**: 检查生成的资源是否在指定范围内
- **连接密度**: 验证网络连接概率是否符合预期
- **调度性能**: 测试不同参数配置下的调度效果
- **训练稳定性**: 确保参数化不影响训练收敛

## 最佳实践

### 1. 参数选择建议

- **资源范围**: 根据实际硬件配置和应用需求设置
- **连接概率**: 根据网络拓扑和通信模式调整
- **比例关系**: 确保虚拟需求不超过物理资源的最大值

### 2. 性能优化

- **资源匹配**: 虚拟需求范围应与物理资源范围匹配
- **连接优化**: 根据应用通信模式调整连接概率
- **动态调整**: 在训练过程中可以动态调整参数

### 3. 故障排除

- **资源不足**: 如果调度失败率高，考虑增加物理资源范围
- **网络拥塞**: 如果网络效率低，考虑降低连接概率
- **训练不稳定**: 如果训练不收敛，检查参数范围是否合理

## 扩展功能

### 1. 动态参数调整

未来可以支持在训练过程中动态调整资源参数，以适应不同的训练阶段。

### 2. 多场景配置

可以预定义多种场景的配置模板，如高性能计算、边缘计算、物联网等。

### 3. 自适应参数

可以实现根据训练效果自动调整参数范围的机制。

## 总结

资源参数化功能大大提高了PPO算法的灵活性和实用性，使其能够适应不同的网络环境和应用场景。通过合理设置参数，可以优化调度性能，提高资源利用率，并更好地满足实际应用需求。 