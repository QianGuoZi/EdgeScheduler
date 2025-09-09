# Sequential PPO 使用指南

本文档提供了Sequential PPO系统的详细使用说明，包括快速开始、完整训练和高级配置。

## 🚀 快速开始 (5分钟)

### 1. 环境检查
```bash
cd C:\Users\shash\Desktop\ppo\自己写的ppo

# 检查Python环境
python --version  # 建议Python 3.8+

# 检查核心依赖
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import numpy; print('NumPy版本:', numpy.__version__)"
```

### 2. 快速功能测试
```bash
# 基础功能验证 (30秒)
python test_sequential_simple.py

# 预期输出:
# ✅ Sequential环境功能正常
# ✅ Agent能正确选择动作
# ✅ 奖励计算正确
```

### 3. 短期训练测试
```bash
# 学习能力验证 (5-10分钟, 500 episodes)
python test_sequential_original_simple.py

# 预期结果:
# - 初期成功率: 0-20%
# - 最终成功率: 20-50% 
# - 明显的学习趋势
```

## 🎯 完整训练 (30-60分钟)

### 标准训练流程
```bash
# 使用原始配置进行完整训练
python train_sequential_original.py

# 训练参数:
# - Episodes: 3000
# - 物理节点: 10个
# - 虚拟任务: 3-8个节点
# - 预计时间: 30-60分钟
```

### 训练过程监控
```bash
# 训练时的输出示例:
Episode   50 | Reward:  0.123 | Success: 24.0% | Violations:  2.1 | Loss: 0.0543
Episode  100 | Reward:  0.234 | Success: 36.0% | Violations:  1.8 | Loss: 0.0432
Episode  500 | Reward:  0.345 | Success: 45.0% | Violations:  1.2 | Loss: 0.0321
...

# 关键指标:
# Reward: 平均奖励 (目标: 从负值提升到0.4+)
# Success: 成功率 (目标: 从0%提升到60%+) 
# Violations: 约束违反数量 (目标: 逐渐减少)
# Loss: 训练损失 (目标: 逐渐收敛)
```

### 训练结果文件
```
训练完成后生成:
├── plots/
│   └── original_sequential_ppo_training_curves.png  # 训练曲线图
├── checkpoints/
│   ├── original_sequential_ppo_episode_200.pt      # 模型检查点
│   ├── original_sequential_ppo_episode_400.pt
│   └── ...
└── 训练日志输出
```

## 📊 结果分析

### 成功的训练指标
```bash
✅ 良好训练结果:
📈 最终100轮统计:
  平均奖励: 0.400+     # 正奖励表示找到可行解
  成功率: 50.0%+       # 一半以上episodes成功
  平均长度: 8-15步     # 合理的决策步数
  约束违反: <1.0       # 很少违反约束
```

### 训练曲线分析
优秀的训练曲线应该显示：
- **奖励曲线**: 从负值逐渐上升到正值
- **成功率**: 从0%逐渐提升到40%+
- **约束违反**: 从高频违反逐渐减少
- **损失函数**: 逐渐收敛到较低值

## 🔧 自定义配置

### 1. 调整问题难度

#### 简单配置 (初学者推荐)
```python
# 创建custom_train.py
from train_sequential_original import OriginalSequentialPPOTrainer

easy_env_config = {
    'num_physical_nodes': 8,         # 减少物理节点
    'virtual_nodes_range': (2, 4),   # 减少虚拟节点
    'physical_cpu_range': (80, 200), # 增加物理资源
    'virtual_cpu_range': (10, 30),   # 减少虚拟需求
    'seed': 42
}

easy_training_config = {
    'total_episodes': 1000,          # 减少训练轮数
    'batch_size': 32,               # 较小批次
}

trainer = OriginalSequentialPPOTrainer(
    env_config=easy_env_config,
    training_config=easy_training_config
)
stats = trainer.train()
```

#### 困难配置 (高级用户)
```python
hard_env_config = {
    'num_physical_nodes': 12,        # 增加物理节点
    'virtual_nodes_range': (5, 10),  # 增加虚拟节点
    'physical_cpu_range': (30, 100), # 减少物理资源
    'virtual_cpu_range': (15, 60),   # 增加虚拟需求
    'virtual_nodes_range': (6, 10),  # 更大的虚拟网络
}

hard_training_config = {
    'total_episodes': 5000,          # 增加训练轮数
    'batch_size': 128,              # 更大批次
    'lr': 1e-4,                     # 较小学习率
}
```

### 2. 超参数调优

#### 学习率调整
```python
# 快速学习 (可能不稳定)
agent_config = {'lr': 1e-3}

# 稳定学习 (推荐)
agent_config = {'lr': 5e-4}

# 精细调优
agent_config = {'lr': 1e-4}
```

#### 网络容量调整
```python
# 小网络 (快速训练)
agent_config = {'hidden_dim': 64}

# 标准网络 (推荐)
agent_config = {'hidden_dim': 128}

# 大网络 (复杂问题)
agent_config = {'hidden_dim': 256}
```

### 3. Curriculum Learning配置

#### 启用难度自适应
```python
env_config = {
    'curriculum_enabled': True,      # 启用Curriculum Learning
}

training_config = {
    'warmup_episodes': 200,          # 热身期长度
    'curriculum_start': 300,         # 难度调整开始时机
}
```

#### 手动难度控制
```python
# 在训练循环中手动调整
if episode > 500 and success_rate > 0.6:
    env.increase_difficulty()        # 提高难度
elif episode > 200 and success_rate < 0.2:
    env.decrease_difficulty()        # 降低难度
```

## 📋 不同使用场景

### 场景1: 研究验证
```bash
# 目标: 验证算法有效性
# 推荐: 使用原始配置

python train_sequential_original.py

# 关注指标:
# - 是否能找到可行解 (成功率>0)
# - 学习曲线是否收敛
# - 与随机策略的对比
```

### 场景2: 性能基准测试
```bash
# 目标: 建立性能基准
# 推荐: 多次运行取平均值

# 运行多次实验
for i in {1..5}; do
    python train_sequential_original.py --seed $((42+i))
done

# 分析平均性能和方差
```

### 场景3: 算法改进
```bash
# 目标: 改进算法性能
# 推荐: 渐进式改进

# 1. 建立基准
python train_sequential_original.py

# 2. 修改算法(如改进奖励函数)
# 3. 对比测试
python your_improved_version.py

# 4. 性能对比分析
```

### 场景4: 实际部署准备
```bash
# 目标: 准备实际应用
# 推荐: 使用真实数据分布

real_world_config = {
    # 基于实际数据中心的参数
    'num_physical_nodes': 50,
    'virtual_nodes_range': (10, 30),
    # ... 其他真实参数
}
```

## 🐛 故障排除指南

### 问题1: 训练不收敛
**现象**: 奖励始终为负，成功率接近0%

**解决方案**:
```python
# 1. 降低问题难度
env_config = {
    'virtual_nodes_range': (2, 3),   # 减少虚拟节点
    'physical_cpu_range': (100, 300) # 增加物理资源
}

# 2. 延长热身期
training_config = {
    'warmup_episodes': 500,          # 更长的高温度探索
}

# 3. 调整学习率
agent_config = {'lr': 1e-3}         # 提高学习率
```

### 问题2: 训练过慢
**现象**: 训练速度很慢，CPU利用率低

**解决方案**:
```python
# 1. 减少批次大小
training_config = {
    'batch_size': 16,                # 从64减少到16
    'update_frequency': 16,
}

# 2. 简化网络
agent_config = {'hidden_dim': 64}   # 从128减少到64

# 3. 减少训练轮数进行调试
training_config = {'total_episodes': 500}
```

### 问题3: 内存不足
**现象**: 出现OutOfMemoryError

**解决方案**:
```python
# 1. 减少经验缓存
training_config = {
    'batch_size': 16,
    'update_frequency': 16,
}

# 2. 限制环境规模
env_config = {
    'max_virtual_nodes': 5,          # 限制最大虚拟节点数
    'num_physical_nodes': 8,         # 限制物理节点数
}
```

### 问题4: NumPy兼容性问题
**现象**: matplotlib相关的ImportError

**解决方案**:
```bash
# 使用不依赖matplotlib的简化版本
python test_sequential_original_simple.py

# 或者降级numpy
# pip install numpy<2.0
```

## 📈 性能期望

### 不同配置下的典型性能

#### 简单配置 (2-3个虚拟节点)
- **最终成功率**: 70-90%
- **收敛时间**: 500-1000 episodes
- **最终奖励**: 0.5-0.7

#### 标准配置 (3-8个虚拟节点)
- **最终成功率**: 40-70%
- **收敛时间**: 1000-2000 episodes  
- **最终奖励**: 0.3-0.5

#### 困难配置 (6-10个虚拟节点)
- **最终成功率**: 20-50%
- **收敛时间**: 2000-4000 episodes
- **最终奖励**: 0.2-0.4

## 🎓 高级技巧

### 1. 预训练策略
```python
# 先在简单配置上预训练
easy_trainer = OriginalSequentialPPOTrainer(env_config=easy_config)
easy_trainer.train()

# 保存模型
torch.save(easy_trainer.agent.state_dict(), 'pretrained_model.pt')

# 在困难配置上继续训练
hard_trainer = OriginalSequentialPPOTrainer(env_config=hard_config)
hard_trainer.agent.load_state_dict(torch.load('pretrained_model.pt'))
hard_trainer.train()
```

### 2. 多种子实验
```python
# 运行多个随机种子的实验
seeds = [42, 123, 456, 789, 999]
results = []

for seed in seeds:
    env_config = {'seed': seed}
    trainer = OriginalSequentialPPOTrainer(env_config=env_config)
    stats = trainer.train()
    results.append(stats['success_rates'][-100:])  # 最后100轮成功率

# 计算平均性能和置信区间
mean_performance = np.mean([np.mean(r) for r in results])
std_performance = np.std([np.mean(r) for r in results])
```

### 3. 超参数搜索
```python
# 网格搜索示例
hyperparams = {
    'lr': [1e-4, 5e-4, 1e-3],
    'hidden_dim': [64, 128, 256],
    'batch_size': [32, 64, 128]
}

best_performance = 0
best_params = None

for lr in hyperparams['lr']:
    for hidden_dim in hyperparams['hidden_dim']:
        for batch_size in hyperparams['batch_size']:
            # 训练并评估
            config = {'lr': lr, 'hidden_dim': hidden_dim}
            trainer = OriginalSequentialPPOTrainer(agent_config=config)
            stats = trainer.train()
            
            # 记录最佳配置
            performance = np.mean(stats['success_rates'][-100:])
            if performance > best_performance:
                best_performance = performance
                best_params = config
```

---

**最后更新**: 2025-01-21  
**支持**: 请参考SEQUENTIAL_PPO_DOCUMENTATION.md获取详细技术文档