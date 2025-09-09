# PPO_mapping算法实现

## 算法概述

PPO_mapping是基于PPO_my改进的网络调度算法，主要特点是：
- **Actor只负责节点调度**：智能体仅决策虚拟节点到物理节点的映射
- **贪心带宽分配**：带宽分配使用贪心策略自动完成，无需强化学习
- **简化决策空间**：相比PPO_my减少了决策复杂度，提高训练效率

## 核心设计思想

### 1. 问题分解
将原始的网络调度问题分解为两个子问题：
- **节点映射问题**：由PPO智能体负责，学习最优的虚拟节点到物理节点的映射策略
- **带宽分配问题**：由贪心算法负责，为每条虚拟链路分配满足需求的最小带宽

### 2. 贪心带宽分配策略
对每条虚拟链路，按以下优先级分配带宽：
1. **同节点映射**：如果虚拟链路的两端映射到同一物理节点，自动获得无限带宽
2. **最小需求优先**：尝试分配满足最小带宽需求的资源
3. **逐步增加**：如果最小需求无法满足，尝试中等需求，最后尝试最大需求
4. **路径检查**：检查物理路径上所有链路的可用带宽

### 3. 优势分析
- **降低复杂度**：决策空间从O(N×B)减少到O(N)，其中N为物理节点数，B为带宽等级数
- **提高效率**：减少训练时间，加快收敛速度
- **保证可行性**：贪心策略确保带宽分配的可行性
- **同节点奖励**：自然地鼓励同节点映射，减少网络通信开销

## 文件结构

```
PPO_my/
├── mapping_environment.py     # PPO_mapping环境实现
├── mapping_agent.py          # PPO_mapping智能体实现
├── train_mapping.py          # PPO_mapping训练脚本
├── test_mapping.py           # 功能测试脚本
├── compare_algorithms.py     # 算法比较脚本
└── README_PPO_mapping.md     # 本文档
```

## 核心组件

### MappingNetworkSchedulerEnvironment
- **功能**：网络调度环境，专注于节点映射
- **状态空间**：物理网络状态 + 虚拟网络需求 + 当前映射进度
- **动作空间**：物理节点索引 [0, num_physical_nodes)
- **奖励设计**：
  - 即时奖励：资源利用效率 + 同节点映射潜力
  - 最终奖励：网络调度器的综合评分

### MappingAgent
- **架构**：状态编码器 + 映射策略头 + 价值网络
- **输入**：固定维度的状态向量（物理节点特征 + 虚拟节点特征 + 决策状态）
- **输出**：物理节点选择的概率分布
- **训练**：PPO算法，支持剪切、价值函数、熵正则化

### 贪心带宽分配算法
```python
def greedy_bandwidth_allocation(virtual_links):
    for link in virtual_links:
        src, dst = link.endpoints
        min_bw, max_bw = link.requirements
        
        if same_physical_node(src, dst):
            allocate(max_bw)  # 同节点映射，免费带宽
        else:
            # 尝试不同带宽等级
            for bw in [min_bw, (min_bw + max_bw) // 2, max_bw]:
                if path_bandwidth_sufficient(src, dst, bw):
                    allocate(bw)
                    break
```

## 使用说明

### 1. 快速测试
```bash
cd PPO_my
python test_mapping.py
```

### 2. 开始训练
```bash
python train_mapping.py
```

### 3. 算法比较
```bash
python compare_algorithms.py
```

### 4. 自定义配置
修改训练脚本中的配置：
```python
env_config = {
    'num_physical_nodes': 6,
    'max_virtual_nodes': 5,
    # ... 其他配置
}

training_config = {
    'total_episodes': 3000,
    'batch_size': 64,
    # ... 其他配置
}
```

## 训练配置

### 环境参数
- `num_physical_nodes`: 物理节点数量（默认5）
- `max_virtual_nodes`: 最大虚拟节点数（默认6）
- `virtual_nodes_range`: 虚拟节点数范围（默认(4,6)）
- `curriculum_enabled`: 是否启用课程学习（默认True）

### Agent参数
- `hidden_dim`: 隐藏层维度（默认128）
- `lr`: 学习率（默认3e-4）
- `ppo_clip`: PPO剪切参数（默认0.2）

### 训练参数
- `total_episodes`: 总训练轮数（默认2000）
- `batch_size`: 批次大小（默认32）
- `update_frequency`: 更新频率（默认32）

## 输出结果

### 训练输出
- `mapping_results/`: 训练结果目录
  - `checkpoints/`: 模型检查点
  - `plots/`: 训练曲线图
  - `training_summary.json`: 训练总结

### 比较结果
- `comparison_results/`: 算法比较结果
  - `algorithm_comparison.png`: 比较图表
  - `comparison_results.json`: 详细比较数据

## 性能指标

### 主要指标
- **成功率**：成功完成映射和带宽分配的比例
- **平均奖励**：episode平均奖励值
- **训练效率**：收敛速度和训练时间
- **带宽成功率**：贪心带宽分配的成功比例

### 与PPO_my的比较
- **决策步数**：PPO_mapping约为PPO_my的50%
- **训练速度**：预期提升2-3倍
- **成功率**：在简单场景下相当或更优
- **收敛稳定性**：预期更稳定

## 扩展方向

### 1. 改进贪心策略
- 考虑路径长度优化
- 动态调整带宽分配优先级
- 支持带宽复用和QoS需求

### 2. 混合策略
- 对复杂场景使用PPO进行带宽分配
- 根据问题规模自适应选择策略

### 3. 多目标优化
- 同时优化延迟、吞吐量、成本等多个目标
- 支持不同的网络拓扑类型

## 注意事项

1. **环境一致性**：确保与PPO_my使用相同的网络生成逻辑
2. **奖励设计**：贪心策略可能需要调整奖励函数权重
3. **超参数调优**：不同网络规模可能需要不同的超参数
4. **收敛判断**：关注成功率和带宽分配成功率两个指标

## 依赖要求

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- NetworkX（通过network_scheduler模块）

## 文献引用

本算法基于以下工作：
1. PPO (Proximal Policy Optimization)
2. Network Function Virtualization
3. Resource Allocation in Edge Computing

---

## 更新日志

- **v1.0**: 初始实现，支持基本的节点映射和贪心带宽分配
- **v1.1**: 添加课程学习和算法比较功能
- **v1.2**: 优化状态表示和奖励函数设计
