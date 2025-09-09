# Sequential PPO for Network Resource Allocation

基于Sequential PPO的网络资源分配调度系统，用于解决虚拟网络到物理网络的映射和带宽分配问题。

## 🎯 核心特性

- **Sequential决策**: 将复杂的联合优化问题分解为序列决策
- **PPO训练**: 使用Proximal Policy Optimization进行策略学习  
- **约束处理**: 实时验证资源约束和连通性约束
- **Curriculum Learning**: 自适应难度调整机制
- **原始问题对齐**: 严格遵循mathematical model定义

## 🚀 快速开始

### 1. 基础测试 (30秒)
```bash
python test_sequential_simple.py
```

### 2. 学习验证 (5-10分钟)
```bash
python test_sequential_original_simple.py
```

### 3. 完整训练 (30-60分钟)
```bash
python train_sequential_original.py
```

## 📁 核心文件

### 🧠 核心组件
- **`sequential_environment.py`** - Sequential环境实现
- **`sequential_agent.py`** - PPO智能体实现  
- **`network_scheduler.py`** - 底层调度引擎

### 🎮 训练脚本
- **`train_sequential_original.py`** - 完整训练流程(推荐)
- **`train_sequential_ppo.py`** - 旧版训练脚本

### 🧪 测试脚本
- **`test_sequential_simple.py`** - 基础功能测试
- **`test_sequential_original_simple.py`** - 学习能力验证
- **`test_original_config.py`** - 策略对比测试

### ⚙️ 配置模块
- **`original_problem_config.py`** - 原始问题标准配置
- **`original_reward.py`** - 原始奖励函数实现

## 📊 预期结果

### 训练指标
```
良好的训练结果:
├── 平均奖励: 0.4+ (从负值提升)
├── 成功率: 50%+ (能找到可行解)
├── 约束违反: <1.0 (很少违反约束)
└── 收敛时间: 1000-2000 episodes
```

### 输出文件
```
训练完成后:
├── plots/original_sequential_ppo_training_curves.png  # 训练曲线
├── checkpoints/*.pt                                   # 模型检查点
└── 详细训练统计                                      # 控制台输出
```

## 🏗️ 系统架构

### 环境 (SequentialNetworkSchedulerEnvironment)
```
Episode: 物理网络 + 虚拟任务
├── Phase 1: 节点映射 (虚拟节点 → 物理节点)
└── Phase 2: 带宽分配 (虚拟链路 → 带宽等级)
```

### 智能体 (SimpleSequentialAgent)
```
输入状态 → 状态编码器 → { 映射策略, 带宽策略, 价值函数 }
```

### 决策流程
```
Step 0-N: 逐个映射虚拟节点到物理节点
Step N-M: 逐个为虚拟链路分配带宽等级
Final: 计算奖励和成功率
```

## 🔧 自定义配置

### 调整问题规模
```python
env_config = {
    'num_physical_nodes': 10,        # 物理节点数
    'virtual_nodes_range': (3, 8),   # 虚拟节点数范围
    'physical_cpu_range': (50, 200), # 物理CPU资源
    'virtual_cpu_range': (10, 50),   # 虚拟CPU需求
}
```

### 调整训练参数
```python
training_config = {
    'total_episodes': 3000,          # 总训练轮数
    'batch_size': 64,               # 批次大小
    'update_frequency': 64,          # PPO更新频率
}
```

## 🐛 常见问题

### Q1: 训练不收敛？
```python
# 降低难度
env_config = {
    'virtual_nodes_range': (2, 3),   # 减少虚拟节点
    'physical_cpu_range': (100, 300) # 增加物理资源
}
```

### Q2: NumPy兼容性问题？
```bash
# 使用简化测试版本(不依赖matplotlib)
python test_sequential_original_simple.py
```

### Q3: 内存不足？
```python
# 减少批次和网络大小
training_config = {'batch_size': 32}
agent_config = {'hidden_dim': 64}
```

## 📚 详细文档

- **[SEQUENTIAL_PPO_DOCUMENTATION.md](../SEQUENTIAL_PPO_DOCUMENTATION.md)** - 完整技术文档
- **[USAGE_GUIDE.md](../USAGE_GUIDE.md)** - 详细使用指南
- **[MODIFICATION_PROCESS.md](../MODIFICATION_PROCESS.md)** - 开发历程记录
- **[define_problem.md](../define_problem/define_problem.md)** - 原始问题定义

## 🎓 技术细节

### 状态空间维度
```
总维度: 66 (可变长度适配)
├── 物理节点特征: [N_phy × 4]
├── 虚拟节点需求: [N_vir × 2]  
├── 虚拟边信息: [E_vir × 2]
├── 部分映射结果: [N_vir]
└── 元信息: [10+]
```

### 奖励机制
```
映射阶段: 资源效率奖励
带宽阶段: 需求满足奖励
最终奖励: 负载均衡 + 带宽满足度 + 资源效率
```

---

**开发状态**: 已完成Phase 7 - 原始问题配置对齐  
**最后更新**: 2025-01-21  
**推荐入口**: `train_sequential_original.py`