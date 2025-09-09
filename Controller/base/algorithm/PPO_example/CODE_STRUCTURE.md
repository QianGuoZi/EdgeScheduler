# Sequential PPO 代码结构总览

本文档提供整个项目的代码结构和组织方式说明。

## 📂 项目目录结构

```
C:\Users\shash\Desktop\ppo\
├── 📁 自己写的ppo/                    # 🎯 核心实现目录
│   ├── 🧠 sequential_environment.py   # Sequential环境实现
│   ├── 🧠 sequential_agent.py         # PPO智能体实现
│   ├── 🧠 network_scheduler.py        # 底层调度引擎
│   ├── 🎮 train_sequential_original.py # 主训练脚本(推荐)
│   ├── 🧪 test_sequential_original_simple.py # 学习验证测试
│   ├── 🧪 test_sequential_simple.py   # 基础功能测试
│   ├── 🧪 test_original_config.py     # 策略对比测试
│   ├── ⚙️ original_problem_config.py  # 原始问题配置
│   ├── ⚙️ original_reward.py          # 原始奖励函数
│   ├── 📄 README.md                   # 核心模块说明
│   └── 🗂️ [其他支持文件]
├── 📁 define_problem/                 # 问题定义文档
│   └── 📋 define_problem.md           # 数学模型定义
├── 📁 plots/                          # 训练结果可视化
├── 📁 checkpoints/                    # 模型检查点
├── 📋 SEQUENTIAL_PPO_DOCUMENTATION.md # 完整技术文档
├── 📋 USAGE_GUIDE.md                  # 详细使用指南
├── 📋 MODIFICATION_PROCESS.md         # 开发历程记录
└── 📋 CODE_STRUCTURE.md               # 本文档
```

## 🧩 核心模块详解

### 1. 🧠 sequential_environment.py
**作用**: Sequential网络调度环境
```python
class SequentialNetworkSchedulerEnvironment:
    """将决策过程分解为多个步骤的环境"""
    
    # 核心方法:
    def reset() -> state                    # 生成新的物理环境和虚拟任务
    def step(action) -> state,reward,done   # 执行单步决策
    def _step_mapping(action)               # 处理节点映射动作
    def _step_bandwidth(action)             # 处理带宽分配动作
    def _check_constraints()                # 验证资源约束
    def _calculate_reward()                 # 计算即时奖励
```

**特点**:
- 两阶段Sequential决策过程
- 实时约束验证
- 支持Curriculum Learning
- 详细的统计信息记录

### 2. 🧠 sequential_agent.py  
**作用**: 基于PPO的Sequential智能体
```python
class SimpleSequentialAgent:
    """简化的Sequential PPO Agent"""
    
    # 核心组件:
    state_encoder: nn.Module              # 状态编码器
    mapping_policy: nn.Module             # 节点映射策略
    bandwidth_policy: nn.Module           # 带宽分配策略
    value_network: nn.Module              # 价值函数网络
    
    # 核心方法:
    def select_action(state, temp)        # 根据状态选择动作
    def calculate_loss(experiences)       # 计算PPO损失
    def update(loss_dict)                # 更新网络参数
```

**特点**:
- 分离的策略头设计
- 温度调节的探索策略
- 标准PPO训练框架

### 3. 🧠 network_scheduler.py
**作用**: 底层网络调度和资源管理
```python
class NetworkTopology:
    """物理网络拓扑管理"""
    # 节点和链路资源管理
    # 最短路径计算
    # 资源可用性查询

class VirtualWork:
    """虚拟任务表示"""
    # 节点需求规格
    # 链路带宽需求
    # 连通性要求

class NetworkScheduler:
    """调度决策执行"""
    # 映射决策应用
    # 约束验证
    # 奖励计算
```

## 🎮 训练脚本层次

### train_sequential_original.py (🌟 推荐)
```python
class OriginalSequentialPPOTrainer:
    """使用原始配置的训练器"""
    
    # 特点:
    ✅ 使用原始问题参数
    ✅ 完整的训练流程
    ✅ Curriculum Learning
    ✅ 详细的训练统计
    ✅ 自动保存和可视化
    
    # 训练流程:
    collect_episode()    # 收集episode经验
    update_agent()       # PPO参数更新  
    print_progress()     # 训练进度显示
    save_checkpoint()    # 模型保存
    plot_training_curves() # 结果可视化
```

### 其他训练脚本
- `train_sequential_ppo.py`: 旧版训练脚本，保留作参考

## 🧪 测试脚本分工

### test_sequential_simple.py
```python
# 🎯 目标: 基础功能验证
# ⏱️ 时间: 30秒
# 🔍 测试内容:
def test_environment_reset()      # 环境重置功能
def test_action_space()          # 动作空间验证
def test_constraint_checking()   # 约束检查逻辑
def test_reward_calculation()    # 奖励计算正确性
```

### test_sequential_original_simple.py
```python
# 🎯 目标: 学习能力验证  
# ⏱️ 时间: 5-10分钟
# 🔍 测试内容:
def test_original_config()       # 原始配置测试
def short_training_loop()        # 500 episodes训练
def analyze_learning_progress()  # 学习趋势分析
def performance_statistics()     # 性能统计报告
```

### test_original_config.py
```python
# 🎯 目标: 策略对比分析
# ⏱️ 时间: 2-3分钟  
# 🔍 测试内容:
def random_policy()              # 随机基准策略
def greedy_policy()              # 贪心策略
def balanced_policy()            # 均衡策略
def strategy_comparison()        # 策略性能对比
```

## ⚙️ 配置系统

### original_problem_config.py
```python
ORIGINAL_CONFIG = {
    'physical_topology': {
        'num_nodes': 10,                    # 物理节点数
        'cpu_range': (50, 200),            # CPU资源范围
        'memory_range': (100, 400),        # 内存资源范围
        'bandwidth_range': (100, 1000),    # 带宽资源范围
        'connectivity_prob': 0.3,          # 连接概率
    },
    'task_topology': {
        'num_nodes_range': (3, 8),         # 虚拟节点数量
        'cpu_demand_range': (10, 50),      # CPU需求范围
        'memory_demand_range': (20, 100),  # 内存需求范围
        'bandwidth_min_range': (10, 100),  # 最小带宽
        'bandwidth_max_range': (100, 200), # 最大带宽
        'connectivity_prob': 0.4,          # P2P连接概率
    },
    'optimization_weights': {
        'w1_cpu': 0.4,                     # CPU均衡权重
        'w2_memory': 0.4,                  # 内存均衡权重  
        'w3_bandwidth': 0.2,               # 带宽均衡权重
        'gamma1_load_balance': 0.7,        # 负载均衡权重
        'gamma2_bandwidth_satisfaction': 0.3, # 带宽满足权重
    }
}

# 辅助函数:
get_two_stage_env_config()        # 转换为TwoStage格式
get_sequential_env_config()       # 转换为Sequential格式
get_reward_weights()              # 获取奖励权重
```

### original_reward.py
```python
class OriginalRewardCalculator:
    """基于原始数学模型的奖励计算"""
    
    def calculate_utilization()         # 计算资源利用率
    def calculate_load_balance()        # 计算负载均衡度L
    def calculate_bandwidth_satisfaction() # 计算带宽满足度D_BW
    def calculate_reward()              # 综合奖励计算
    
    # 目标函数: Minimize γ1*L - γ2*D_BW
    # 转换奖励: Maximize -γ1*L + γ2*D_BW
```

## 🔄 数据流向

### 训练时数据流
```
1. Environment.reset()
   ├── 生成物理拓扑 (NetworkTopology)
   ├── 生成虚拟任务 (VirtualWork)  
   └── 返回初始状态

2. Agent.select_action(state)
   ├── 状态编码 (state_encoder)
   ├── 策略推理 (mapping_policy/bandwidth_policy)
   └── 返回动作和价值

3. Environment.step(action)
   ├── 应用动作 (NetworkScheduler)
   ├── 检查约束 (constraint validation)
   ├── 计算奖励 (reward calculation)
   └── 返回新状态和奖励

4. Agent.calculate_loss(experiences)
   ├── 优势估计 (GAE)
   ├── PPO损失计算
   └── 梯度更新

5. 重复2-4直到episode结束
```

### 测试时数据流
```
Test Script → Environment → Agent → Results Analysis
     ↓             ↓          ↓           ↓
   配置参数    状态生成    动作选择    性能统计
```

## 🔗 模块依赖关系

```
训练脚本层
├── train_sequential_original.py
├── test_*.py
└── [其他脚本]
    ↓ 依赖
核心模块层  
├── sequential_environment.py
├── sequential_agent.py  
└── network_scheduler.py
    ↓ 依赖
配置系统层
├── original_problem_config.py
├── original_reward.py
└── [配置文件]
    ↓ 依赖
基础依赖层
├── torch (深度学习框架)
├── numpy (数值计算)
└── matplotlib (可视化)
```

## 📊 关键接口定义

### Environment接口
```python
def reset() -> Dict[str, Any]:
    """返回初始状态字典"""
    
def step(action: int) -> Tuple[Dict, float, bool, Dict]:
    """返回(next_state, reward, done, info)"""
    
def get_action_space() -> int:
    """返回当前步骤的动作空间大小"""
```

### Agent接口  
```python
def select_action(state: Dict, temperature: float) -> Tuple[int, float, float]:
    """返回(action, log_prob, value)"""
    
def calculate_loss(experiences: Dict) -> Dict[str, torch.Tensor]:
    """返回各项损失值"""
    
def update(loss_dict: Dict) -> None:
    """执行参数更新"""
```

## 🛠️ 扩展指南

### 添加新的奖励函数
```python
# 1. 在network_scheduler.py中添加新方法
def calculate_custom_reward(self, virtual_work):
    # 实现自定义奖励逻辑
    pass

# 2. 在environment中调用
if hasattr(self.network_scheduler, 'calculate_custom_reward'):
    reward = self.network_scheduler.calculate_custom_reward(self.virtual_work_obj)
```

### 添加新的约束
```python
# 在sequential_environment.py中扩展
def _check_custom_constraints(self, action):
    # 实现自定义约束检查
    violations = []
    # ... 约束逻辑
    return violations
```

### 修改网络架构
```python
# 在sequential_agent.py中修改
def __init__(self):
    # 添加新的网络层
    self.custom_layer = nn.Linear(hidden_dim, custom_output_dim)
```

---

**文档版本**: Phase 7  
**最后更新**: 2025-01-21  
**维护说明**: 本文档与代码实现保持同步更新