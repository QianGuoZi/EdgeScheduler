# PPO 网络调度算法设计文档

本文档详细描述了基于PPO（Proximal Policy Optimization）的网络资源调度算法设计，包括原始Sequential PPO架构和增强的NewHeuristic PPO方案。

---

## 目录

1. [问题定义](#一问题定义)
2. [原始Sequential PPO架构](#二原始sequential-ppo架构)
3. [NewHeuristic PPO增强方案](#三newheuristic-ppo增强方案)
4. [Agent设计](#四agent设计-simplesequentialagent)
5. [训练器设计](#五训练器设计)
6. [方案对比](#六方案对比)
7. [使用指南](#七使用指南)

---

## 一、问题定义

### 1.1 问题概述

网络资源调度问题是将**虚拟网络请求（Virtual Network Request）** 映射到**物理网络基础设施（Physical Network Infrastructure）** 上的优化问题。

### 1.2 数学模型

#### 物理网络

- **物理节点集合**: $P = \{p_1, p_2, ..., p_{|P|}\}$
- **物理链路集合**: $R = \{r_1, r_2, ..., r_{|R|}\}$
- **节点资源**: $CPU_P(p)$, $RAM_P(p)$
- **链路带宽**: $BW_R(r)$

#### 虚拟网络请求

- **虚拟节点集合**: $N = \{n_1, n_2, ..., n_{|N|}\}$
- **虚拟链路集合**: $V = \{v_1, v_2, ..., v_{|V|}\}$
- **节点需求**: $CPU_N(n)$, $RAM_N(n)$
- **链路带宽需求**: $[BW_{min}(v), BW_{max}(v)]$

#### 优化目标

$$\max \quad \gamma_1 \cdot L + \gamma_2 \cdot D_{BW}$$

其中：
- $L$ 为负载均衡度（Load Balance）
- $D_{BW}$ 为带宽满足度（Bandwidth Satisfaction）
- $\gamma_1, \gamma_2$ 为权重系数

### 1.3 默认配置参数

```python
ORIGINAL_CONFIG = {
    # 物理拓扑配置
    'physical_topology': {
        'num_nodes': 10,              # 10个物理节点
        'cpu_range': (50, 100),       # CPU资源范围
        'memory_range': (50, 100),    # 内存资源范围
        'bandwidth_range': (100, 1000), # 链路带宽范围
        'connectivity_prob': 0.3,     # 连接概率
        'initial_usage': (0.1, 0.5),  # 初始使用率
    },
    
    # 任务拓扑配置
    'task_topology': {
        'num_nodes_range': (3, 8),    # 虚拟节点数范围
        'cpu_demand_range': (10, 50), # CPU需求范围
        'memory_demand_range': (10, 50), # 内存需求范围
        'bandwidth_min_range': (10, 50), # 最小带宽需求
        'bandwidth_max_range': (50, 100), # 最大带宽需求
        'connectivity_prob': 0.4,     # 连接概率
    },
    
    # 优化权重
    'optimization_weights': {
        'gamma1_load_balance': 0.6,   # 负载均衡权重
        'gamma2_bandwidth_satisfaction': 0.4, # 带宽满足度权重
    },
    
    # 动作空间
    'action_space': {
        'bandwidth_levels': 10,       # 带宽等级数
    },
}
```

---

## 二、原始Sequential PPO架构

### 2.1 架构概述

原始Sequential PPO将网络调度问题分解为**两阶段顺序决策**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sequential PPO 决策流程                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              阶段1: 节点映射 (Mapping Phase)              │  │
│   │                                                         │  │
│   │   虚拟节点 n_0 ──► 选择物理节点 p_i ──► 映射完成        │  │
│   │   虚拟节点 n_1 ──► 选择物理节点 p_j ──► 映射完成        │  │
│   │   ...                                                   │  │
│   │   虚拟节点 n_k ──► 选择物理节点 p_m ──► 映射完成        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            阶段2: 带宽分配 (Bandwidth Phase)              │  │
│   │                                                         │  │
│   │   虚拟链路 v_0 ──► 选择带宽等级 l_i ──► 分配完成        │  │
│   │   虚拟链路 v_1 ──► 选择带宽等级 l_j ──► 分配完成        │  │
│   │   ...                                                   │  │
│   │   虚拟链路 v_m ──► 选择带宽等级 l_n ──► 分配完成        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                     Episode 结束，计算最终奖励                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 环境设计 (SequentialNetworkSchedulerEnvironment)

#### 2.2.1 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_physical_nodes` | 5 | 物理节点数量 |
| `max_virtual_nodes` | 6 | 最大虚拟节点数 |
| `bandwidth_levels` | 5 | 带宽分配等级数 |
| `physical_cpu_range` | (40, 80) | 物理CPU资源范围 |
| `physical_memory_range` | (40, 80) | 物理内存资源范围 |
| `physical_bandwidth_range` | (40, 80) | 物理带宽资源范围 |
| `virtual_cpu_range` | (12, 25) | 虚拟CPU需求范围 |
| `virtual_memory_range` | (12, 25) | 虚拟内存需求范围 |
| `virtual_bandwidth_range` | (8, 20) | 虚拟带宽需求范围 |
| `physical_connectivity_prob` | 0.8 | 物理网络连接概率 |
| `virtual_connectivity_prob` | 0.7 | 虚拟网络连接概率 |
| `virtual_nodes_range` | (4, 6) | 虚拟节点数范围 |
| `curriculum_enabled` | True | 是否启用课程学习 |

#### 2.2.2 状态表示

```python
state = {
    # 物理网络特征
    'physical_features': tensor,      # [num_physical_nodes, 5]
                                      # [CPU总量, 内存总量, CPU使用率, 内存使用率, 链路带宽均值]
    'physical_edges': tensor,         # [2, num_physical_edges]
    'physical_edge_features': tensor, # [num_physical_edges, 2]
    
    # 虚拟网络特征
    'virtual_features': tensor,       # [num_virtual_nodes, 3]
                                      # [CPU需求, 内存需求, 链路带宽需求均值]
    'virtual_edges': tensor,          # [2, num_virtual_edges]
    'virtual_edge_features': tensor,  # [num_virtual_edges, 2]
    
    # 决策状态
    'current_step': int,              # 当前步数
    'mapping_phase': bool,            # 是否为映射阶段
    'current_virtual_node': int,      # 当前待映射的虚拟节点
    'current_link_index': int,        # 当前待分配的链路索引
    'partial_mapping': list,          # 部分映射结果 [-1, -1, 2, 0, ...]
    'partial_bandwidth': list,        # 部分带宽分配结果
    
    # 辅助信息
    'num_physical_nodes': int,
    'num_virtual_nodes': int,
    'num_virtual_links': int,
    'bandwidth_mapping': dict,        # 链路带宽等级映射
}
```

#### 2.2.3 动作空间

- **映射阶段**: $a \in \{0, 1, ..., |P|-1\}$，选择物理节点索引
- **带宽阶段**: $a \in \{0, 1, ..., L-1\}$，选择带宽等级

#### 2.2.4 奖励函数

**映射阶段奖励**：

```python
def _calculate_mapping_reward(self, physical_node_action):
    # 基础成功奖励
    base_reward = 1.0
    
    # 资源利用效率奖励
    cpu_utilization = (used_cpu + cpu_demand) / total_cpu
    memory_utilization = (used_memory + memory_demand) / total_memory
    
    # 奖励适中的资源利用率 (0.3-0.8 范围内给高奖励)
    cpu_efficiency = 1.0 - abs(cpu_utilization - 0.55) / 0.45
    memory_efficiency = 1.0 - abs(memory_utilization - 0.55) / 0.45
    
    efficiency_reward = 0.5 * (cpu_efficiency + memory_efficiency)
    
    return base_reward + efficiency_reward
```

**带宽阶段奖励**：

```python
def _calculate_bandwidth_reward(self, bandwidth_level_action):
    # 获取分配的带宽和需求范围
    allocated_bandwidth = bandwidth_mapping[link_key][bandwidth_level]
    
    # 计算满足度
    if allocated_bandwidth >= min_bandwidth:
        if allocated_bandwidth <= max_bandwidth:
            # 在需求范围内，线性奖励
            satisfaction = (allocated - min) / (max - min)
        else:
            satisfaction = 1.0  # 超过最大需求也给奖励
    else:
        satisfaction = 0.1  # 未满足最小需求
    
    return satisfaction
```

**最终奖励**：

```python
def _calculate_final_reward(self):
    # 使用简化奖励函数
    components = network_scheduler.get_simple_reward_components(virtual_work)
    
    # 组件包括：
    # - mapping_success_rate: 映射成功率
    # - load_balance_reward: 负载均衡奖励
    # - resource_efficiency: 资源效率
    # - bandwidth_satisfaction: 带宽满足度
    # - path_length_penalty: 路径长度惩罚
    
    return weighted_sum(components)
```

#### 2.2.5 课程学习 (Curriculum Learning)

```python
def _adjust_difficulty(self):
    """根据历史成功率调整难度"""
    if len(success_history) < history_window:
        return
    
    recent_success_rate = mean(success_history[-history_window:])
    
    if recent_success_rate > 0.85:  # 太简单，增加难度
        difficulty_level = min(max_difficulty, 
                              difficulty_level + adjustment_rate)
    elif recent_success_rate < 0.4:  # 太难，降低难度
        difficulty_level = max(min_difficulty, 
                              difficulty_level - adjustment_rate)
```

难度调整影响：
- 虚拟节点数量
- 资源需求大小
- 问题复杂度

### 2.3 训练器设计 (OriginalSequentialPPOTrainer)

#### 2.3.1 训练配置

```python
training_config = {
    'total_episodes': 3000,      # 总训练回合数
    'batch_size': 64,            # 批次大小
    'update_frequency': 64,      # 更新频率
    'print_frequency': 50,       # 打印频率
    'save_frequency': 200,       # 保存频率
    'warmup_episodes': 100,      # 热身阶段
    'curriculum_enabled': False, # 课程学习
    'curriculum_start': 500,     # 课程学习开始时机
}
```

#### 2.3.2 温度调度

```python
def get_temperature(episode):
    if episode < warmup_episodes:
        return 1.5  # 热身阶段：更多探索
    else:
        # 线性衰减
        progress = (episode - warmup) / (total - warmup)
        return max(0.1, 1.0 - progress)
```

#### 2.3.3 训练循环

```python
def train(self):
    for episode in range(total_episodes):
        # 1. 获取温度
        temperature = get_temperature(episode)
        
        # 2. 收集经验
        episode_data = collect_episode(episode, temperature)
        
        # 3. 添加到缓冲区
        experience_buffer.extend(episode_data)
        
        # 4. 定期更新Agent
        if episode % update_frequency == 0:
            loss_dict = update_agent()
        
        # 5. 记录统计信息
        stats.update(episode_data)
        
        # 6. 定期保存
        if episode % save_frequency == 0:
            save_checkpoint(episode)
```

---

## 三、NewHeuristic PPO增强方案

### 3.1 设计理念

NewHeuristic PPO 是一个**基于奖励塑形（Reward Shaping）的强化学习方案**，其核心思想是：

> **保持原始PPO架构不变，仅通过增强奖励函数来引导智能体学习更好的调度策略**

### 3.2 架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    NewHeuristic PPO 架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐     ┌──────────────────────────────────┐ │
│  │ SimpleSequential │     │    NewHeuristicEnvironment       │ │
│  │      Agent       │◄────│  (继承 SequentialEnvironment)    │ │
│  │  (原始PPO架构)   │     │                                  │ │
│  └────────┬─────────┘     │  ✅ 状态表示：不变               │ │
│           │               │  ✅ 奖励函数：增强               │ │
│           │               │     - 基础奖励                   │ │
│           ▼               │     + 负载均衡奖励               │ │
│    ┌──────────────┐       │     + 资源效率奖励               │ │
│    │  动作选择    │       │     + 带宽效率奖励               │ │
│    │  (映射/带宽) │       │     + 进度奖励                   │ │
│    └──────────────┘       └──────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 环境设计 (NewHeuristicEnvironment)

#### 3.3.1 继承关系

```python
class NewHeuristicEnvironment(SequentialNetworkSchedulerEnvironment):
    """继承基础顺序环境，只修改奖励函数"""
```

#### 3.3.2 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `heuristic_reward_weight` | 0.3 | 启发式奖励的权重系数 |
| `enable_load_balance_reward` | True | 是否启用负载均衡奖励 |
| `enable_resource_efficiency_reward` | True | 是否启用资源效率奖励 |
| `enable_progress_reward` | True | 是否启用进度奖励 |

#### 3.3.3 状态表示（与基类相同，不做修改）

```python
def _get_state(self):
    """状态获取 - 直接返回父类原始状态，不添加额外特征"""
    original_state = super()._get_state()
    return original_state  # 关键：不修改状态空间
```

### 3.4 奖励函数设计（核心创新）

#### 3.4.1 奖励融合公式

$$R_{total} = R_{base} + w_{heuristic} \times R_{heuristic}$$

其中：
- $R_{base}$ 为基础奖励（来自父类）
- $R_{heuristic}$ 为启发式奖励
- $w_{heuristic}$ 为启发式权重（可调度）

#### 3.4.2 映射阶段奖励

```python
def _calculate_mapping_reward(self, physical_node_action: int) -> float:
    # 1. 计算基础奖励（来自父类）
    base_reward = super()._calculate_mapping_reward(physical_node_action)
    
    # 2. 计算启发式奖励
    heuristic_reward = 0.0
    
    if self.enable_load_balance_reward:
        load_balance_reward = self._calculate_load_balance_reward(physical_node_action)
        heuristic_reward += 0.4 * load_balance_reward  # 权重40%
    
    if self.enable_resource_efficiency_reward:
        efficiency_reward = self._calculate_resource_efficiency_reward(physical_node_action)
        heuristic_reward += 0.6 * efficiency_reward    # 权重60%
    
    # 3. 融合奖励
    total_reward = base_reward + self.heuristic_reward_weight * heuristic_reward
    return total_reward
```

#### 3.4.3 负载均衡奖励

**目标**：鼓励将虚拟节点均匀分布到不同物理节点

```python
def _calculate_load_balance_reward(self, physical_node: int) -> float:
    # 1. 计算选择前的负载均衡度
    before_balance = self._calculate_current_load_balance()
    
    # 2. 模拟选择该节点后的负载分布
    node_loads = {node: count for node in partial_mapping if node != -1}
    node_loads[physical_node] = node_loads.get(physical_node, 0) + 1
    
    # 3. 计算新的负载均衡度
    loads = list(node_loads.values())
    mean_load = np.mean(loads)
    load_variance = np.var(loads)
    max_variance = mean_load ** 2
    after_balance = 1.0 - (load_variance / max_variance)
    
    # 4. 奖励 = 均衡度改善程度 × 2（映射到[-1, +1]）
    return 2.0 * (after_balance - before_balance)
```

**计算公式**：

$$L_{balance} = 1 - \frac{\sigma^2_{load}}{\mu^2_{load}}$$

其中 $\sigma^2$ 是负载方差，$\mu$ 是平均负载。

#### 3.4.4 资源效率奖励

**目标**：鼓励选择使资源利用率接近理想值（62.5%）的物理节点

```python
def _calculate_resource_efficiency_reward(self, physical_node: int) -> float:
    # 1. 获取资源需求和当前利用率
    cpu_demand = virtual_features[current_virtual_node, 0]
    memory_demand = virtual_features[current_virtual_node, 1]
    
    # 2. 检查资源可行性
    if cpu_available < cpu_demand or memory_available < memory_demand:
        return -1.0  # 资源不足，严重惩罚
    
    # 3. 计算映射后的利用率
    new_cpu_util = (cpu_total * cpu_utilization + cpu_demand) / cpu_total
    new_memory_util = (memory_total * memory_utilization + memory_demand) / memory_total
    
    # 4. 计算与理想利用率(62.5%)的偏差
    ideal_util = 0.625
    cpu_efficiency = 1.0 - abs(new_cpu_util - ideal_util) / 0.625
    memory_efficiency = 1.0 - abs(new_memory_util - ideal_util) / 0.625
    
    # 5. 综合效率分数，映射到[-1, +1]
    efficiency_score = (cpu_efficiency + memory_efficiency) / 2.0
    return 2.0 * efficiency_score - 1.0
```

**理想利用率曲线**：

```
效率
  1 ┤      ╭────╮
    │    ╱      ╲
0.5 ┤  ╱          ╲
    │╱              ╲
  0 ┼────┬────┬────┬────
    0   0.3  0.625  1.0  利用率
         理想区间
```

#### 3.4.5 带宽分配阶段奖励

```python
def _calculate_bandwidth_reward(self, bandwidth_level_action: int) -> float:
    # 1. 计算基础奖励
    base_reward = super()._calculate_bandwidth_reward(bandwidth_level_action)
    
    # 2. 计算带宽效率奖励
    heuristic_reward = self._calculate_bandwidth_efficiency_reward(bandwidth_level_action)
    
    # 3. 融合奖励
    total_reward = base_reward + self.heuristic_reward_weight * heuristic_reward
    return total_reward
```

#### 3.4.6 带宽效率奖励

**目标**：根据链路重要性分配合适的带宽等级

```python
def _calculate_bandwidth_efficiency_reward(self, bandwidth_level: int) -> float:
    # 1. 计算链路重要性
    link_importance = self._calculate_simple_link_importance(src, dst)
    
    # 2. 根据重要性确定理想带宽比例
    if link_importance > 0.7:
        ideal_ratio = 0.8  # 重要链路需要高带宽
    elif link_importance > 0.4:
        ideal_ratio = 0.5  # 中等重要链路
    else:
        ideal_ratio = 0.2  # 低重要性链路
    
    # 3. 计算当前带宽等级与理想比例的匹配度
    current_ratio = bandwidth_level / max_level
    match_score = 1.0 - abs(current_ratio - ideal_ratio) / max(ideal_ratio, 1 - ideal_ratio)
    
    return 2.0 * match_score - 1.0
```

#### 3.4.7 链路重要性计算

```python
def _calculate_simple_link_importance(self, src: int, dst: int) -> float:
    """基于节点度数计算链路重要性"""
    # 统计源节点和目标节点的度数
    src_degree = count_edges_connected_to(src)
    dst_degree = count_edges_connected_to(dst)
    
    # 重要性 = 平均度数 / 最大可能度数
    avg_degree = (src_degree + dst_degree) / 2.0
    max_degree = num_virtual_nodes - 1
    
    return avg_degree / max_degree
```

#### 3.4.8 进度奖励

**目标**：鼓励完成更多的映射/带宽分配任务

```python
def _calculate_progress_reward(self) -> float:
    if self.mapping_phase:
        # 映射阶段：映射完成度
        mapped_count = sum(1 for x in partial_mapping if x != -1)
        progress = mapped_count / total_nodes
        return progress * 0.5
    else:
        # 带宽阶段：带宽分配完成度
        allocated_count = sum(1 for x in partial_bandwidth if x > 0)
        progress = allocated_count / total_links
        return progress * 0.5
```

### 3.5 训练器设计 (LightweightHeuristicPPOTrainer)

#### 3.5.1 启发式权重调度

**核心创新**：训练初期使用较高的启发式权重引导学习，后期逐渐降低让智能体自主探索

```python
def calculate_current_heuristic_weight(self, episode: int) -> float:
    """线性衰减的启发式权重"""
    initial_weight = 0.6   # 初始权重
    final_weight = 0.2     # 最终权重
    decay_episodes = 2000  # 衰减周期
    
    if episode >= decay_episodes:
        return final_weight
    
    # 线性衰减
    progress = episode / decay_episodes
    return initial_weight - progress * (initial_weight - final_weight)
```

**权重衰减曲线**：

```
权重
0.6 ┤────╲
    │     ╲
0.4 ┤      ╲
    │       ╲
0.2 ┤        ╲────────────
    │
  0 ┼────┬────┬────┬────┬────
    0   500  1000 1500 2000  Episode
```

#### 3.5.2 训练配置

```python
training_config = {
    'total_episodes': 4000,
    'batch_size': 64,
    'print_frequency': 50,
    'evaluation_frequency': 200,
    'save_frequency': 300,
    # 启发式权重调度
    'enable_reward_weight_schedule': True,
    'initial_heuristic_weight': 0.6,
    'final_heuristic_weight': 0.2,
    'weight_decay_episodes': 2000,
}
```

#### 3.5.3 训练循环

```python
def train(self):
    for episode in range(total_episodes):
        # 1. 调整启发式权重
        current_weight = self.calculate_current_heuristic_weight(episode)
        self.env.heuristic_reward_weight = current_weight
        
        # 2. 收集episode数据
        episode_data = self.collect_episode(episode)
        batch_data.append(episode_data)
        
        # 3. 批量更新
        if len(batch_data) >= batch_size:
            loss_dict = self.update_agent(batch_data)
            batch_data.clear()
        
        # 4. 定期评估和保存
        if episode % eval_frequency == 0:
            self.evaluate_agent()
        if episode % save_frequency == 0:
            self.save_checkpoint(episode)
```

---

## 四、Agent设计 (SimpleSequentialAgent)

### 4.1 网络架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleSequentialAgent                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入状态 (Dict)                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐                                           │
│  │   状态编码器     │  Linear(state_dim → 256) → ReLU           │
│  │  state_encoder   │  → Dropout(0.1)                           │
│  │                  │  → Linear(256 → 128) → ReLU               │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    共享特征 (128维)                      │    │
│  └────────────────────────────────────────────────────────┘    │
│           │                    │                    │           │
│           ▼                    ▼                    ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 映射策略头   │    │ 带宽策略头   │    │   价值网络   │      │
│  │mapping_policy│    │bandwidth_pol │    │   critic     │      │
│  │Linear→ReLU  │    │Linear→ReLU  │    │Linear→ReLU  │      │
│  │→Linear(N_p) │    │→Linear(N_bw)│    │→Linear(1)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│           │                    │                    │           │
│           ▼                    ▼                    ▼           │
│     物理节点概率          带宽等级概率           状态价值        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 状态编码

```python
def _encode_state(self, state: Dict) -> torch.Tensor:
    """将字典状态编码为固定长度向量"""
    features = []
    
    # 1. 物理网络特征 [max_physical_nodes × 3]
    # 转换：[CPU总量, 内存总量, 使用率...] → [CPU可用量, 内存可用量, 带宽均值]
    cpu_available = cpu_total * (1 - cpu_utilization)
    memory_available = memory_total * (1 - memory_utilization)
    avg_available_bandwidth = physical_features[:, 4]
    
    # 2. 虚拟网络特征 [max_virtual_nodes × 3]
    # [CPU需求, 内存需求, 带宽需求均值]
    
    # 3. 决策状态特征 [10维]
    # [步数, 阶段, 当前虚拟节点, 当前链路, 节点总数, 链路总数, 映射完成度, ...]
    
    return torch.cat(features)  # 总维度 = N_p×3 + N_v×3 + 10
```

### 4.3 状态维度计算

```python
def _calculate_state_dim(self):
    # 物理节点：每个节点3个特征（CPU可用量，内存可用量，链路可用带宽均值）
    physical_summary_dim = max_physical_nodes * 3
    
    # 虚拟节点：每个节点3个特征（CPU需求，内存需求，链路带宽需求均值）
    virtual_summary_dim = max_virtual_nodes * 3
    
    # 决策状态：10个特征
    decision_state_dim = 10
    
    return physical_summary_dim + virtual_summary_dim + decision_state_dim
```

### 4.4 动作选择

```python
def select_action(self, state, temperature=1.0):
    # 1. 前向传播获取logits
    action_logits, value, _ = self.forward(state)
    
    # 2. 温度缩放
    scaled_logits = action_logits / temperature
    
    # 3. 应用动作掩码（无效节点设为-inf）
    if action_mask is not None:
        scaled_logits = scaled_logits.masked_fill(~action_mask, float('-inf'))
    
    # 4. Softmax + 采样
    action_probs = F.softmax(scaled_logits, dim=-1)
    action = Categorical(action_probs).sample()
    
    return action, log_prob, value
```

### 4.5 PPO损失计算

```python
def calculate_loss(self, states, actions, rewards, dones, gamma=0.99):
    # 1. 计算GAE优势
    advantages = []
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
        else:
            next_value = values[t + 1] if t + 1 < len(rewards) else 0
        
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * 0.95 * last_advantage * (not dones[t])
        last_advantage = advantages[t]
    
    # 2. 计算目标值
    targets = advantages + values.detach()
    
    # 3. 策略损失 (REINFORCE风格)
    policy_loss = -mean(log_prob(action) * advantage)
    
    # 4. 价值损失
    value_loss = MSE(values, targets)
    
    # 5. 总损失
    total_loss = policy_loss + 0.5 * value_loss
    
    return {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss
    }
```

### 4.6 网络更新

```python
def update(self, loss_dict):
    self.optimizer.zero_grad()
    loss_dict['total_loss'].backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    
    self.optimizer.step()
```

---

## 五、训练器设计

### 5.1 原始训练器 vs NewHeuristic训练器

| 特性 | OriginalSequentialPPOTrainer | LightweightHeuristicPPOTrainer |
|------|------------------------------|--------------------------------|
| 环境类 | SequentialNetworkSchedulerEnvironment | NewHeuristicEnvironment |
| 奖励函数 | 基础奖励 | 基础 + 启发式奖励 |
| 权重调度 | ❌ | ✅ 线性衰减 |
| 温度调度 | ✅ | ✅ |
| 课程学习 | ✅ | ✅ |
| 评估频率 | 无定期评估 | 每200轮评估 |

### 5.2 经验收集

```python
def collect_episode(self, episode_idx, temperature=1.0):
    state = env.reset()
    done = False
    episode_data = {
        'states': [], 'actions': [], 'rewards': [],
        'dones': [], 'values': [], 'log_probs': []
    }
    
    while not done and step_count < max_steps:
        # 选择动作
        action, log_prob, value = agent.select_action(state, temperature)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 存储经验
        episode_data['states'].append(state)
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        episode_data['dones'].append(done)
        episode_data['values'].append(value)
        episode_data['log_probs'].append(log_prob)
        
        state = next_state
        step_count += 1
    
    return episode_data
```

### 5.3 检查点保存

```python
def save_checkpoint(self, episode):
    checkpoint = {
        'episode': episode,
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'stats': stats,
        'config': {
            'env_config': env_config,
            'agent_config': agent_config,
            'training_config': training_config
        },
        'experiment_info': {
            'experiment_timestamp': timestamp,
            'experiment_name': name
        }
    }
    
    torch.save(checkpoint, f'{checkpoint_dir}/episode_{episode}.pt')
```

---

## 六、方案对比

### 6.1 三种环境对比

| 特性 | Sequential | Lightweight | NewHeuristic |
|------|------------|-------------|--------------|
| 状态空间修改 | ❌ | ✅ (+5特征) | ❌ |
| 奖励函数增强 | ❌ | ✅ | ✅ |
| 负载均衡奖励 | ❌ | ✅ | ✅ |
| 资源效率奖励 | ❌ | ✅ | ✅ |
| 进度奖励 | ❌ | ✅ | ✅ |
| 权重调度 | ❌ | ✅ | ✅ |
| 状态复杂度 | 低 | 高 | 低 |
| 训练稳定性 | 中 | 高 | 高 |

### 6.2 设计优势总结

| 特性 | 说明 |
|------|------|
| **架构简洁** | NewHeuristic不修改状态空间，仅增强奖励函数 |
| **训练稳定** | 启发式奖励提供更密集的学习信号 |
| **权重调度** | 初期引导 → 后期自主，平衡探索与利用 |
| **多目标优化** | 同时考虑负载均衡、资源效率、带宽效率 |
| **可解释性** | 奖励分解清晰，便于调试和分析 |

---

## 七、使用指南

### 7.1 文件结构

```
PPO_my/
├── sequential_environment.py         # 基础顺序环境
├── sequential_agent.py               # PPO Agent
├── train_sequential_original.py      # 原始PPO训练脚本
├── new_heuristic_environment.py      # NewHeuristic环境
├── train_new_heuristic_ppo.py        # NewHeuristic训练脚本
├── lightweight_heuristic_integration_environment.py  # Lightweight环境
├── train_lightweight_ppo.py          # Lightweight训练脚本
├── original_problem_config.py        # 问题配置
├── network_scheduler.py              # 网络调度器核心
├── original_reward.py                # 原始奖励函数
├── heuristic_algorithm.py            # 启发式算法
├── checkpoints/                      # 模型检查点
├── plots/                            # 训练曲线图
└── test_results/                     # 测试结果
```

### 7.2 训练原始PPO

```bash
cd Controller/base/algorithm/PPO_my
python train_sequential_original.py
```

### 7.3 训练NewHeuristic PPO

```bash
cd Controller/base/algorithm/PPO_my
python train_new_heuristic_ppo.py
```

### 7.4 自定义配置训练

```python
from train_new_heuristic_ppo import LightweightHeuristicPPOTrainer

# 自定义环境配置
custom_env_config = {
    'heuristic_reward_weight': 0.5,
    'enable_load_balance_reward': True,
    'enable_resource_efficiency_reward': True,
    'enable_progress_reward': True
}

# 自定义训练配置
custom_training_config = {
    'total_episodes': 5000,
    'batch_size': 128,
    'initial_heuristic_weight': 0.7,
    'final_heuristic_weight': 0.1,
    'weight_decay_episodes': 3000,
}

# 创建训练器
trainer = LightweightHeuristicPPOTrainer(
    env_config=custom_env_config,
    training_config=custom_training_config
)

# 开始训练
trainer.train()
```

### 7.5 加载模型进行评估

```python
import torch
from new_heuristic_environment import NewHeuristicEnvironment
from sequential_agent import SimpleSequentialAgent

# 加载检查点
checkpoint = torch.load('checkpoints/new_heuristic_xxx/final.pt')

# 恢复配置
env_config = checkpoint['config']['env_config']
agent_config = checkpoint['config']['agent_config']

# 创建环境和Agent
env = NewHeuristicEnvironment(**env_config)
agent = SimpleSequentialAgent(**agent_config)
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval()

# 运行评估
state = env.reset()
done = False
total_reward = 0

while not done:
    action, _, _ = agent.select_action(state, temperature=0.1)
    state, reward, done, info = env.step(action)
    total_reward += reward

print(f"评估奖励: {total_reward}")
```

### 7.6 训练输出示例

```
🚀 轻量级启发式PPO训练器初始化完成
   实验时间: 20250101_120000
   实验名称: new_heuristic_20250101_120000
   检查点目录: checkpoints/new_heuristic_20250101_120000
   图表目录: plots/new_heuristic_20250101_120000
   使用原始PPO架构: ✅
   启发式奖励权重: 0.4

🚀 开始轻量级启发式PPO训练
   总episodes: 4000
   批次大小: 64

Episode   50: 奖励= 2.345, 成功率= 45.0%, 启发式权重=0.590, 负载均衡=0.823, 资源效率=0.567
Episode  100: 奖励= 3.567, 成功率= 62.0%, 启发式权重=0.580, 负载均衡=0.856, 资源效率=0.612
...
Episode 2000: 奖励= 5.234, 成功率= 85.0%, 启发式权重=0.200, 负载均衡=0.912, 资源效率=0.734
...

✅ 训练完成! 用时: 1234.5秒

🏆 最终评估:
   评估结果: 奖励=5.456±0.234, 成功率=87.5%
```

---

## 附录

### A. 辅助指标计算

#### A.1 负载均衡度计算

```python
def _calculate_current_load_balance(self) -> float:
    """计算当前负载均衡程度（0-1，越高越均衡）"""
    # 统计每个物理节点的负载
    node_loads = {}
    for physical_node in self.partial_mapping:
        if physical_node != -1:
            node_loads[physical_node] = node_loads.get(physical_node, 0) + 1
    
    # 计算负载方差
    loads = list(node_loads.values())
    mean_load = np.mean(loads)
    load_variance = np.var(loads)
    
    # 转换为0-1分数
    max_possible_variance = mean_load ** 2
    balance_score = 1.0 - (load_variance / max_possible_variance)
    
    return max(0.0, min(1.0, balance_score))
```

#### A.2 资源效率计算

```python
def _calculate_current_resource_efficiency(self) -> float:
    """计算当前资源效率（0-1，越高越好）"""
    efficiency_scores = []
    
    for node in physical_nodes:
        cpu_utilization = node.cpu_utilization
        memory_utilization = node.memory_utilization
        
        # 理想利用率在50%-75%之间，最优点62.5%
        ideal_util = 0.625
        cpu_efficiency = 1.0 - abs(cpu_utilization - ideal_util) / 0.625
        memory_efficiency = 1.0 - abs(memory_utilization - ideal_util) / 0.625
        
        node_efficiency = (cpu_efficiency + memory_efficiency) / 2.0
        efficiency_scores.append(max(0.0, node_efficiency))
    
    return np.mean(efficiency_scores)
```

#### A.3 连通性计算

```python
def _calculate_simple_connectivity(self) -> float:
    """计算简化的连通性指标"""
    mapped_nodes = [node for node in partial_mapping if node != -1]
    
    # 计算已映射节点间的平均距离
    total_distance = 0
    count = 0
    for i, node1 in enumerate(mapped_nodes):
        for node2 in mapped_nodes[i+1:]:
            distance = abs(node1 - node2)
            total_distance += distance
            count += 1
    
    # 平均距离越小，连通性越好
    avg_distance = total_distance / count
    max_distance = num_physical_nodes - 1
    
    connectivity = 1.0 - (avg_distance / max_distance)
    return max(0.0, min(1.0, connectivity))
```

### B. 超参数建议

| 超参数 | 推荐范围 | 说明 |
|--------|----------|------|
| `learning_rate` | 1e-4 ~ 5e-4 | 学习率 |
| `hidden_dim` | 64 ~ 256 | 隐藏层维度 |
| `batch_size` | 32 ~ 128 | 批次大小 |
| `temperature` | 0.1 ~ 1.5 | 动作采样温度 |
| `gamma` | 0.95 ~ 0.99 | 折扣因子 |
| `heuristic_weight` | 0.2 ~ 0.6 | 启发式奖励权重 |
| `weight_decay_episodes` | 1000 ~ 3000 | 权重衰减周期 |

---

*文档版本: 1.0*
*最后更新: 2025年*
