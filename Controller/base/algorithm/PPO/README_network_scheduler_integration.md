# Network Scheduler 集成指南

## 概述

本文档说明如何在 `TwoStageNetworkSchedulerEnvironment` 中集成之前设计的 `network_scheduler` 功能，以实现更强大的网络调度能力。

## 主要特性

### 1. 可选集成
- 通过 `use_network_scheduler` 参数控制是否使用 `network_scheduler`
- 保持向后兼容性，原有功能不受影响
- 可以动态切换使用不同的调度策略

### 2. 增强的验证能力
- 使用 `NetworkTopology` 进行路径验证
- 支持最短路径计算和带宽可用性检查
- 更准确的资源约束验证

### 3. 改进的奖励计算
- 利用 `NetworkScheduler` 的奖励计算机制
- 支持负载均衡、带宽满足度、网络效率等多维度评估
- 更符合实际网络调度的评估标准

## 使用方法

### 基本用法

```python
from two_stage_environment import TwoStageNetworkSchedulerEnvironment

# 创建使用network_scheduler的环境
env = TwoStageNetworkSchedulerEnvironment(
    num_physical_nodes=10,
    max_virtual_nodes=8,
    bandwidth_levels=10,
    use_network_scheduler=True  # 启用network_scheduler集成
)

# 重置环境
state = env.reset()

# 执行调度动作
mapping_action = np.array([0, 1, 2, 3])  # 虚拟节点到物理节点的映射
bandwidth_action = np.array([5, 3, 7])   # 带宽等级分配

next_state, reward, done, info = env.step(mapping_action, bandwidth_action)

# 获取调度结果
result = env.get_scheduling_result()
```

### 高级用法

```python
# 创建环境时配置更多参数
env = TwoStageNetworkSchedulerEnvironment(
    num_physical_nodes=15,
    max_virtual_nodes=10,
    bandwidth_levels=12,
    use_network_scheduler=True,
    physical_connectivity_prob=0.4,  # 物理网络连接概率
    virtual_connectivity_prob=0.5,   # 虚拟网络连接概率
    physical_cpu_range=(100.0, 500.0),
    physical_memory_range=(200.0, 1000.0),
    virtual_cpu_range=(20.0, 100.0),
    virtual_memory_range=(40.0, 200.0)
)

# 获取详细的调度结果
result = env.get_scheduling_result()

# 访问network_scheduler的结果
if 'network_scheduler_result' in result:
    network_result = result['network_scheduler_result']
    print(f"节点映射: {network_result['node_mapping']}")
    print(f"带宽分配: {network_result['bandwidth_allocation']}")
    print(f"已调度节点: {network_result['scheduled_nodes']}")

# 获取网络利用率
if env.network_topology:
    network_util = env.network_topology.get_network_utilization()
    print(f"网络带宽利用率: {network_util['bandwidth_utilization']:.2%}")
```

## 核心组件

### 1. NetworkTopology
- 管理物理网络拓扑
- 提供最短路径计算
- 处理带宽分配和释放
- 监控网络资源使用情况

### 2. VirtualWork
- 定义虚拟工作需求
- 管理虚拟节点和链路需求
- 支持不对称带宽需求

### 3. NetworkScheduler
- 执行实际的调度操作
- 验证资源约束
- 计算调度质量指标

## 验证机制

### 1. 节点映射验证
- 检查物理节点资源是否足够
- 验证CPU和内存约束
- 确保映射的唯一性

### 2. 带宽分配验证
- 检查虚拟链路带宽需求
- 验证物理路径的带宽可用性
- 支持最短路径计算

### 3. 路径验证
- 使用Dijkstra算法计算最短路径
- 检查路径上的带宽约束
- 处理网络连通性问题

## 奖励计算

当启用 `network_scheduler` 时，奖励计算包括：

1. **资源负载均衡** (25%)
   - CPU利用率均衡
   - 内存利用率均衡

2. **带宽满足度** (25%)
   - 虚拟链路带宽需求满足程度
   - 支持不对称带宽需求

3. **网络效率** (15%)
   - 同一物理节点上的虚拟节点比例
   - 减少网络跳数

4. **网络利用率** (10%)
   - 整体网络带宽利用率
   - 避免资源浪费

## 配置选项

### 环境参数
- `use_network_scheduler`: 是否启用network_scheduler集成
- `num_physical_nodes`: 物理节点数量
- `max_virtual_nodes`: 最大虚拟节点数量
- `bandwidth_levels`: 带宽等级数量

### 资源范围
- `physical_cpu_range`: 物理节点CPU范围
- `physical_memory_range`: 物理节点内存范围
- `virtual_cpu_range`: 虚拟节点CPU需求范围
- `virtual_memory_range`: 虚拟节点内存需求范围

### 网络拓扑
- `physical_connectivity_prob`: 物理网络连接概率
- `virtual_connectivity_prob`: 虚拟网络连接概率

## 示例场景

### 场景1: 简单调度
```python
# 创建简单环境
env = TwoStageNetworkSchedulerEnvironment(
    num_physical_nodes=5,
    max_virtual_nodes=3,
    use_network_scheduler=True
)

state = env.reset()
# 执行调度...
```

### 场景2: 复杂网络
```python
# 创建复杂网络环境
env = TwoStageNetworkSchedulerEnvironment(
    num_physical_nodes=20,
    max_virtual_nodes=15,
    bandwidth_levels=15,
    use_network_scheduler=True,
    physical_connectivity_prob=0.3,
    virtual_connectivity_prob=0.4
)

state = env.reset()
# 执行调度...
```

## 注意事项

1. **性能考虑**: 启用 `network_scheduler` 会增加计算开销，特别是在大规模网络中
2. **内存使用**: 集成功能会占用更多内存来存储网络拓扑信息
3. **验证严格性**: `network_scheduler` 的验证更加严格，可能会拒绝一些原来认为有效的动作
4. **奖励差异**: 使用 `network_scheduler` 的奖励计算可能与原有方法不同

## 故障排除

### 常见问题

1. **导入错误**
   ```python
   # 确保network_scheduler.py在同一目录下
   from network_scheduler import NetworkTopology, VirtualWork, NetworkScheduler
   ```

2. **验证失败**
   - 检查物理节点资源是否足够
   - 验证网络连通性
   - 确认带宽约束

3. **性能问题**
   - 考虑减少网络规模
   - 优化验证算法
   - 使用缓存机制

## 总结

通过集成 `network_scheduler`，`TwoStageNetworkSchedulerEnvironment` 获得了：

1. **更强的验证能力**: 基于实际网络拓扑的约束验证
2. **更准确的奖励**: 多维度、更符合实际的调度质量评估
3. **更好的扩展性**: 支持复杂的网络调度场景
4. **向后兼容性**: 可以随时切换回原有模式

这种集成方式为PPO算法提供了更真实和有效的网络调度环境，有助于训练出更好的调度策略。 