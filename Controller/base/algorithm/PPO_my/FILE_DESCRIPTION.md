# PPO_my 文件夹结构与功能说明

## 📋 概述

本文件夹实现了基于**PPO (Proximal Policy Optimization)** 算法的边缘计算网络资源调度系统。该系统用于解决虚拟网络嵌入（VNE）问题，将虚拟任务节点映射到物理网络节点，并分配网络带宽资源。

---

## 📁 文件结构

```
PPO_my/
├── 核心模块
│   ├── network_scheduler.py        # 网络调度器核心实现
│   ├── sequential_environment.py   # Sequential决策环境
│   ├── sequential_agent.py         # PPO智能体实现
│   ├── original_problem_config.py  # 问题配置定义
│   └── original_reward.py          # 奖励函数实现
│
├── 环境变体
│   ├── lightweight_heuristic_integration_environment.py  # 轻量级启发式环境
│   └── new_heuristic_environment.py                      # 新启发式环境
│
├── 启发式算法
│   └── heuristic_algorithm.py      # 多种启发式算法实现
│
├── 训练脚本
│   ├── train_sequential_original.py    # 原始Sequential PPO训练
│   ├── train_lightweight_ppo.py        # 轻量级启发式PPO训练
│   └── train_new_heuristic_ppo.py      # 新启发式PPO训练
│
├── 测试与评估
│   ├── two_ppo_comparison_test.py           # 两种PPO模型对比测试
│   ├── three_algorithms_comparison_test.py  # 三种算法对比测试
│   ├── four_algorithms_comparison_test.py   # 四种算法对比测试
│   └── four_algorithms_with_balance_test.py # 带负载均衡的四算法对比
│
├── 输出目录
│   ├── checkpoints/     # 模型检查点保存
│   ├── plots/           # 训练曲线图
│   └── test_results/    # 测试结果JSON和图表
│
└── 文档
    └── README_PPO_Analysis.md  # PPO算法分析报告
```

---

## 🔧 核心模块详解

### 1. `network_scheduler.py` - 网络调度器核心

**核心类：**

| 类名 | 功能描述 |
|------|----------|
| `NetworkTopology` | 物理网络拓扑管理，包括节点资源、链路带宽的增删改查 |
| `VirtualWork` | 虚拟任务定义，包含节点资源需求和链路带宽需求 |
| `NetworkScheduler` | 调度器主类，执行节点映射和带宽分配 |

**主要功能：**
- 物理网络拓扑的创建与管理
- 资源（CPU/内存/带宽）的分配与释放
- 最短路径计算（Dijkstra算法）
- 奖励计算（负载均衡度、带宽满足度）

**辅助函数：**
- `create_sample_topology()` - 创建示例物理拓扑
- `create_sample_virtual_work()` - 创建示例虚拟任务

---

### 2. `sequential_environment.py` - Sequential决策环境

**类：** `SequentialNetworkSchedulerEnvironment`

**决策流程：**
```
Episode开始
    ↓
映射阶段（Mapping Phase）
    ├── 逐个虚拟节点选择物理节点
    └── 完成所有节点映射后进入下一阶段
    ↓
带宽分配阶段（Bandwidth Phase）
    ├── 逐条虚拟链路选择带宽等级
    └── 完成所有链路分配后Episode结束
    ↓
Episode结束，计算最终奖励
```

**核心特性：**
- **Curriculum Learning**：根据历史成功率动态调整难度
- **状态表示**：物理网络特征 + 虚拟网络特征 + 决策状态
- **即时奖励**：每步给予即时反馈，最后给予全局奖励

**环境参数：**
```python
num_physical_nodes: int = 5       # 物理节点数
max_virtual_nodes: int = 6        # 最大虚拟节点数
bandwidth_levels: int = 5         # 带宽等级数
physical_cpu_range: (40, 80)      # 物理CPU范围
virtual_cpu_range: (12, 25)       # 虚拟CPU需求范围
curriculum_enabled: bool = True   # 是否启用课程学习
```

---

### 3. `sequential_agent.py` - PPO智能体

**类：** `SimpleSequentialAgent`

**网络架构：**
```
状态输入
    ↓
状态编码器（State Encoder）
    ├── Linear(state_dim → hidden*2)
    ├── ReLU + Dropout(0.1)
    └── Linear(hidden*2 → hidden)
    ↓
    ├── 映射策略头（Mapping Policy）
    │   └── 输出：物理节点概率分布
    │
    ├── 带宽策略头（Bandwidth Policy）
    │   └── 输出：带宽等级概率分布
    │
    └── 价值网络（Critic）
        └── 输出：状态价值V(s)
```

**状态编码：**
- 物理节点特征：`[CPU可用量, 内存可用量, 链路可用带宽均值]` × max_physical_nodes
- 虚拟节点特征：`[CPU需求, 内存需求, 链路带宽需求均值]` × max_virtual_nodes
- 决策状态特征：10维（当前步骤、阶段、进度等）

**关键方法：**
- `select_action()` - 根据当前状态选择动作
- `calculate_loss()` - 计算PPO损失（策略损失 + 价值损失）
- `update()` - 更新网络参数（含梯度裁剪）

---

### 4. `original_problem_config.py` - 问题配置

定义了标准的问题参数：

```python
ORIGINAL_CONFIG = {
    'physical_topology': {
        'num_nodes': 10,              # 物理节点数
        'cpu_range': (50, 100),       # CPU资源范围
        'memory_range': (50, 100),    # 内存资源范围
        'bandwidth_range': (100, 1000), # 链路带宽范围
        'connectivity_prob': 0.3,     # 连接概率
    },
    'task_topology': {
        'num_nodes_range': (3, 8),    # 任务节点数范围
        'cpu_demand_range': (10, 50), # CPU需求范围
        'memory_demand_range': (10, 50),
        'bandwidth_min_range': (10, 50),
        'bandwidth_max_range': (50, 100),
    },
    'optimization_weights': {
        'w1_cpu': 0.4,                # CPU负载权重
        'w2_memory': 0.4,             # 内存负载权重
        'w3_bandwidth': 0.2,          # 带宽负载权重
        'gamma1_load_balance': 0.6,   # 负载均衡权重
        'gamma2_bandwidth_satisfaction': 0.4,  # 带宽满足度权重
    }
}
```

---

### 5. `original_reward.py` - 奖励函数

**类：** `OriginalRewardCalculator`

**优化目标：**
```
最小化: γ1 × L - γ2 × D_BW

其中:
- L = w1×L_CPU + w2×L_RAM + w3×L_BW (负载均衡度，标准差)
- D_BW = (1/|V|) × Σ δ(v) (带宽满足度)
```

**奖励转换：**
```
原始目标: Minimize γ1×L - γ2×D_BW
转换为奖励: Maximize -γ1×L + γ2×D_BW
归一化到 [-1, 1] 范围
```

---

## 🎯 启发式算法

### `heuristic_algorithm.py`

实现了四种启发式调度算法：

| 算法名称 | 类名 | 策略描述 |
|---------|------|----------|
| **Naive** | `NaiveResourceHeuristicAgent` | 随机选择前几个可用节点，固定中等带宽 |
| **Moderate** | `ModerateResourceHeuristicAgent` | 考虑资源利用率和简单负载均衡 |
| **Smart** | `SmartLoadBalanceHeuristicAgent` | 综合负载均衡、资源效率和连接性 |
| **FlexiTask** | `FlexiTaskHeuristicAgent` | 基于K8s调度思想的两阶段调度 |

**FlexiTask算法详解：**
```
第一阶段 - Predicates（预选）:
    ├── 资源过滤
    └── 负载阈值检查（2分钟平均、30分钟峰值）

第二阶段 - Priorities（优选）:
    └── NFF = w1×NLL + w2×NBB - w3×H
        ├── NLL: 资源空闲度
        ├── NBB: 资源均衡度
        └── H: 热度惩罚
```

---

## 🏋️ 训练脚本

### 1. `train_sequential_original.py`

使用原始Sequential PPO架构训练，配置：
- 训练轮数：3000 episodes
- 批次大小：64
- 学习率：5e-4
- 支持Curriculum Learning

### 2. `train_lightweight_ppo.py`

轻量级启发式PPO训练，特点：
- 保持原始PPO架构
- 增强奖励函数（添加负载均衡、资源效率奖励）
- 支持启发式权重调度（从0.6逐渐衰减到0.2）

### 3. `train_new_heuristic_ppo.py`

新启发式PPO训练，使用`NewHeuristicEnvironment`。

---

## 📊 测试与评估

### 测试脚本功能对比

| 脚本名称 | 对比对象 | 主要指标 |
|---------|---------|---------|
| `two_ppo_comparison_test.py` | 两种PPO模型 | 奖励、成功率、L、D_BW |
| `three_algorithms_comparison_test.py` | PPO + 启发式 | 三种算法综合对比 |
| `four_algorithms_comparison_test.py` | PPO + Mapping PPO + 启发式 | 四种算法对比 |
| `four_algorithms_with_balance_test.py` | PPO + Balance PPO + 启发式 | 带负载均衡的对比 |

### 评估指标

- **成功率 (Success Rate)**：任务完成的比例
- **平均奖励 (Average Reward)**：每个episode的平均奖励
- **负载均衡度 L**：资源利用率的标准差（越小越好）
- **带宽满足度 D_BW**：链路带宽需求的满足程度（越大越好）

---

## 📂 输出目录说明

### `checkpoints/`
保存训练过程中的模型检查点：
```
checkpoints/
├── exp_20250826_162200/
│   ├── episode_200.pt
│   ├── episode_400.pt
│   └── ...
├── exp_20250826_162200_config.json
└── ...
```

### `plots/`
保存训练曲线图：
- `*_training_curves.png` - 奖励、成功率随训练轮次变化

### `test_results/`
保存测试结果：
- `*.json` - 详细测试数据
- `*_performance_*.png` - 性能对比图
- `*_lb_bw_*.png` - 负载均衡与带宽满足度对比图

---

## 🚀 快速开始

### 1. 训练模型
```bash
# 原始PPO训练
python train_sequential_original.py

# 轻量级启发式PPO训练
python train_lightweight_ppo.py
```

### 2. 测试模型
```bash
# 两种PPO对比测试
python two_ppo_comparison_test.py

# 四种算法对比测试
python four_algorithms_comparison_test.py
```

### 3. 测试启发式算法
```bash
python heuristic_algorithm.py
```

---

## 📈 核心算法流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练循环                                  │
├─────────────────────────────────────────────────────────────────┤
│  for episode in range(total_episodes):                          │
│      state = env.reset()                                        │
│      while not done:                                            │
│          ┌─────────────────────────────────────────────────┐    │
│          │ if mapping_phase:                                │    │
│          │     action = agent.select_action(state)          │    │
│          │     # 选择物理节点                               │    │
│          │ else:                                            │    │
│          │     action = agent.select_action(state)          │    │
│          │     # 选择带宽等级                               │    │
│          └─────────────────────────────────────────────────┘    │
│          state, reward, done, info = env.step(action)           │
│          buffer.add(state, action, reward, done)                │
│                                                                  │
│      if len(buffer) >= batch_size:                              │
│          loss = agent.calculate_loss(buffer)                    │
│          agent.update(loss)                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 关键技术点

1. **Sequential决策**：将复杂的联合决策分解为多个简单步骤
2. **Curriculum Learning**：根据成功率动态调整任务难度
3. **双头策略网络**：映射和带宽分配共享编码器，独立策略头
4. **GAE优势估计**：使用Generalized Advantage Estimation
5. **动作掩码**：防止选择无效动作
6. **梯度裁剪**：防止梯度爆炸（max_norm=1.0）

---

## 📚 相关依赖

```python
torch >= 1.9.0
numpy >= 1.20.0
networkx >= 2.6
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

---

*文档生成时间：2025年12月*

