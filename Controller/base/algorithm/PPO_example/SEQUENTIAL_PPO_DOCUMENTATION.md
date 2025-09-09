# Sequential PPO for Network Resource Allocation

这是一个基于Sequential PPO的网络资源分配调度系统，用于解决任务节点到物理节点的映射和带宽分配问题。

## 📖 目录

- [系统概述](#系统概述)
- [核心组件](#核心组件)
- [环境说明](#环境说明)
- [智能体架构](#智能体架构)
- [训练流程](#训练流程)
- [使用指南](#使用指南)
- [测试说明](#测试说明)
- [配置参数](#配置参数)

## 🎯 系统概述

### 问题定义
解决网络资源分配中的联合调度问题：
- **输入**: 物理网络拓扑 + 虚拟任务需求
- **输出**: 任务节点到物理节点的映射 + 虚拟链路的带宽分配
- **目标**: 在满足资源约束的前提下，最大化负载均衡和带宽满足度

### Sequential决策过程
将原本的一步决策分解为多步序列决策：
```
Episode流程: 物理环境生成 → 虚拟任务生成 → 顺序映射节点 → 顺序分配带宽 → 计算奖励
```

**优势**:
- 更细粒度的学习信号
- 符合实际部署的顺序特性
- 便于处理变长的任务规模

## 🏗️ 核心组件

### 1. SequentialNetworkSchedulerEnvironment
**文件**: `sequential_environment.py`

序列化的网络调度环境，支持：
- 动态生成物理网络拓扑
- 随机生成虚拟任务需求
- 两阶段Sequential决策过程
- 实时约束检查和奖励计算
- Curriculum Learning难度调整

**关键特性**:
```python
# 两阶段决策过程
Phase 1: 映射阶段 (Mapping Phase)
- Step 0: 映射虚拟节点0 → 物理节点?
- Step 1: 映射虚拟节点1 → 物理节点?
- ...
- Step N-1: 映射虚拟节点N-1 → 物理节点?

Phase 2: 带宽分配阶段 (Bandwidth Allocation Phase) 
- Step N: 为虚拟链路0分配带宽等级
- Step N+1: 为虚拟链路1分配带宽等级
- ...
- Step N+M-1: 为虚拟链路M-1分配带宽等级
```

### 2. SimpleSequentialAgent
**文件**: `sequential_agent.py`

基于PPO的序列决策智能体：
- 使用MLP编码网络状态
- 分离的映射策略和带宽策略
- 统一的价值函数网络
- 支持温度调节的探索策略

**网络架构**:
```
输入状态 → 状态编码器 → { 映射策略头, 带宽策略头, 价值头 }
                    ↓         ↓         ↓
                  物理节点   带宽等级    状态价值
```

### 3. NetworkScheduler系统
**文件**: `network_scheduler.py`

底层的网络调度引擎：
- 物理拓扑管理 (NetworkTopology)
- 虚拟工作表示 (VirtualWork)  
- 资源约束验证
- 多种奖励函数实现

## 🌍 环境说明

### 状态空间
```python
state = {
    'physical_features': [N_phy × 4],      # 物理节点特征 [cpu, memory, cpu_usage, mem_usage]
    'virtual_features': [N_vir × 2],       # 虚拟节点需求 [cpu_req, mem_req]
    'virtual_edges': [2 × E_vir],          # 虚拟边连接 [src_nodes, dst_nodes]
    'virtual_edge_features': [E_vir × 2],  # 虚拟边需求 [min_bw, max_bw]
    'partial_mapping': [N_vir],            # 部分映射结果 [-1=未映射, >=0=物理节点索引]
    'partial_bandwidth': [E_vir],          # 部分带宽分配 [0=未分配, >0=带宽等级]
    'current_step': int,                   # 当前步数
    'mapping_phase': bool,                 # 当前阶段标识
    'virtual_num_nodes': int,              # 虚拟节点数量
    'virtual_num_edges': int               # 虚拟边数量
}
```

### 动作空间
- **映射阶段**: 选择物理节点索引 `action ∈ [0, N_physical)`
- **带宽阶段**: 选择带宽等级 `action ∈ [0, bandwidth_levels)`

### 奖励设计
```python
# 映射阶段奖励
mapping_reward = resource_efficiency_bonus  # 基于资源利用率

# 带宽分配阶段奖励  
bandwidth_reward = satisfaction_ratio       # 基于需求满足程度

# 最终奖励（Episode结束时）
final_reward = load_balance + bandwidth_satisfaction + resource_efficiency
```

## 🤖 智能体架构

### 网络结构
```python
# 状态编码器
state_encoder: 66 → 256 → 128

# 策略网络
mapping_policy: 128 → 128 → N_physical_nodes    # 映射概率分布
bandwidth_policy: 128 → 128 → bandwidth_levels  # 带宽概率分布

# 价值网络  
value_network: 128 → 128 → 1                    # 状态价值
```

### PPO训练特性
- **经验收集**: 批量收集episode经验
- **优势估计**: GAE (Generalized Advantage Estimation)
- **策略更新**: Clipped PPO objective
- **价值学习**: MSE loss with baseline
- **探索策略**: 温度调节的softmax采样

## 🚀 训练流程

### Episode流程
```python
1. env.reset() → 生成物理环境和虚拟任务
2. 映射阶段: 逐个为虚拟节点选择物理节点
3. 带宽阶段: 逐个为虚拟链路分配带宽
4. 计算最终奖励和成功率
5. 存储episode经验
```

### 批量更新
```python
1. 收集batch_size个episodes的经验
2. 计算优势函数和回报
3. PPO策略更新
4. 价值函数更新
5. 清空经验缓冲区
```

### Curriculum Learning
- 根据成功率动态调整任务难度
- 成功率低 → 降低难度（减少虚拟节点/增加物理资源）
- 成功率高 → 提高难度（增加虚拟节点/减少物理资源）

## 📚 使用指南

### 基础训练
```bash
# 使用默认配置训练
cd 自己写的ppo
python train_sequential_original.py

# 使用原始问题配置训练（推荐）
python train_sequential_original.py
```

### 快速测试
```bash
# 简单功能测试
python test_sequential_simple.py

# 原始配置测试
python test_sequential_original_simple.py

# 策略对比测试
python test_original_config.py
```

### 自定义配置
```python
from train_sequential_original import OriginalSequentialPPOTrainer

# 环境配置
env_config = {
    'num_physical_nodes': 10,
    'virtual_nodes_range': (3, 8),
    'physical_cpu_range': (50, 200),
    'physical_memory_range': (100, 400),
    # ... 更多配置
}

# 训练配置
training_config = {
    'total_episodes': 3000,
    'batch_size': 64,
    'update_frequency': 64,
    'print_frequency': 50,
}

# 创建训练器
trainer = OriginalSequentialPPOTrainer(
    env_config=env_config,
    training_config=training_config
)

# 开始训练
stats = trainer.train()
```

## 🧪 测试说明

### 1. test_sequential_simple.py
**用途**: 基础功能验证
- 测试环境reset和step功能
- 验证状态和动作空间
- 检查约束验证逻辑
- 快速调试用（小规模参数）

### 2. test_sequential_original_simple.py  
**用途**: 原始配置验证
- 使用define_problem.md的标准参数
- 训练500个episodes验证学习能力
- 输出详细的训练统计

### 3. test_original_config.py
**用途**: 策略对比分析
- 测试随机、贪心、平衡三种策略
- 使用原始奖励函数评估
- 分析不同策略的性能差异

### 运行示例
```bash
# 基础测试 (约1-2分钟)
python test_sequential_simple.py

# 学习能力验证 (约5-10分钟)  
python test_sequential_original_simple.py

# 策略对比 (约2-3分钟)
python test_original_config.py
```

## ⚙️ 配置参数

### 环境参数
```python
# 网络规模
num_physical_nodes: int = 10        # 物理节点数量
max_virtual_nodes: int = 8          # 最大虚拟节点数
virtual_nodes_range: Tuple = (3,8)  # 虚拟节点数量范围

# 资源范围
physical_cpu_range: Tuple = (50,200)     # 物理CPU资源范围
physical_memory_range: Tuple = (100,400) # 物理内存资源范围
virtual_cpu_range: Tuple = (10,50)       # 虚拟CPU需求范围
virtual_memory_range: Tuple = (20,100)   # 虚拟内存需求范围

# 网络参数
physical_connectivity_prob: float = 0.3  # 物理网络连接概率
virtual_connectivity_prob: float = 0.4   # 虚拟网络连接概率
bandwidth_levels: int = 10               # 带宽分配等级数
```

### 训练参数
```python
# 训练规模
total_episodes: int = 3000    # 总训练轮数
batch_size: int = 64          # 批次大小
update_frequency: int = 64    # 更新频率

# 学习参数
lr: float = 5e-4              # 学习率
hidden_dim: int = 128         # 隐藏层维度
temperature: float = 1.0      # 探索温度

# Curriculum Learning
warmup_episodes: int = 100    # 热身阶段
curriculum_start: int = 500   # 难度调整开始时机
```

### 奖励权重
```python
# 基于原始数学模型
w1_cpu: float = 0.4                    # CPU负载均衡权重
w2_memory: float = 0.4                 # 内存负载均衡权重  
w3_bandwidth: float = 0.2              # 带宽负载均衡权重
gamma1_load_balance: float = 0.7       # 负载均衡总权重
gamma2_bandwidth_satisfaction: float = 0.3  # 带宽满足度总权重
```

## 📊 输出说明

### 训练过程输出
```
Episode   50 | Reward:  0.234 | Success: 23.5% | Length:  8.2 | Loss: 0.1234
Episode  100 | Reward:  0.445 | Success: 45.0% | Length:  9.1 | Loss: 0.0876
...
```

### 最终统计
```
📈 训练结果分析:
前期(0-500):    平均奖励: -0.123 ± 0.234, 成功率: 12.3%
中期(500-1500): 平均奖励:  0.234 ± 0.123, 成功率: 34.5%  
后期(1500-3000):平均奖励:  0.456 ± 0.098, 成功率: 67.8%

最终100轮统计:
  平均奖励: 0.523
  成功率: 72.3%
  成功时平均奖励: 0.612
```

### 可视化输出
- 训练曲线图保存在 `plots/` 目录
- 模型检查点保存在 `checkpoints/` 目录

## 🔧 故障排除

### 常见问题


1. **内存不足**
   ```python
   # 减少批次大小和更新频率
   training_config = {
       'batch_size': 32,        # 从64减少到32
       'update_frequency': 32,  # 从64减少到32
   }
   ```

2. **收敛慢**
   ```python
   # 调整学习率和温度
   agent_config = {
       'lr': 1e-3,              # 增加学习率
   }
   # 或增加热身期
   training_config = {
       'warmup_episodes': 200,  # 延长热身期
   }
   ```

3. **成功率低**
   ```python
   # 启用Curriculum Learning
   env_config = {
       'curriculum_enabled': True,
   }
   ```

## 📋 依赖要求

```
torch >= 1.9.0
numpy >= 1.21.0  
matplotlib >= 3.5.0 (可选，用于可视化)
```



如有问题或建议，请参考：
- `MODIFICATION_PROCESS.md` - 详细的开发历程
- `define_problem/define_problem.md` - 原始问题定义
- 代码注释 - 详细的实现说明

---
**版本**: Phase 7 - Original Configuration Alignment