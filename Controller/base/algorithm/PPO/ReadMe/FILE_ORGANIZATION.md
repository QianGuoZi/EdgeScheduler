# PPO算法文件组织说明

## 概述

为了更好的文件管理，PPO算法现在将训练过程中生成的文件按类型分别存储在不同的目录中。

## 目录结构

```
PPO/
├── 核心算法文件
│   ├── ppo.py                 # 主要的PPO算法实现
│   ├── network_scheduler.py   # 网络调度器和拓扑管理
│   └── replaybuffer.py        # 经验回放缓冲区
│
├── 训练和测试脚本
│   ├── train_ppo.py          # 完整训练脚本
│   ├── test_ppo.py           # 单元测试脚本
│   ├── test_training.py      # 训练过程测试脚本
│
├── 演示脚本
│   └── demo.py               # 完整演示脚本
│
├── 文档文件
│   ├── README.md            # 主要说明文档
│   ├── BUGFIX.md           # Bug修复说明
│   └── FILE_ORGANIZATION.md # 本文档
│
├── 配置文件
│   └── requirements.txt      # 依赖包列表
│
├── 模型文件目录
│   └── models/
│       ├── ppo_model_episode_100.pth
│       ├── ppo_model_episode_200.pth
│       ├── ppo_model_episode_300.pth
│       └── ppo_model_final.pth
│
└── 训练统计目录
    └── stats/
        ├── training_stats_episode_100.json
        ├── training_stats_episode_200.json
        ├── training_stats_episode_300.json
        └── training_stats_final.json
```

## 文件类型说明

### 1. 模型文件 (.pth)
- **位置**: `models/` 目录
- **内容**: PyTorch模型权重和优化器状态
- **命名规则**: 
  - `ppo_model_episode_{N}.pth` - 第N个episode的检查点
  - `ppo_model_final.pth` - 最终训练完成的模型
- **大小**: 约3-4MB（取决于网络规模）

### 2. 训练统计文件 (.json)
- **位置**: `stats/` 目录
- **内容**: 训练过程中的各种统计信息
  - `training_rewards`: 每个episode的奖励
  - `episode_lengths`: 每个episode的长度
  - `actor_losses`: Actor网络的损失
  - `critic_losses`: Critic网络的损失
  - `entropy_losses`: 熵损失
- **命名规则**: 
  - `training_stats_episode_{N}.json` - 第N个episode的统计
  - `training_stats_final.json` - 最终训练统计

## 使用方法

### 1. 训练时自动创建目录

```python
from train_ppo import PPOTrainer

# 创建训练器时会自动创建目录
trainer = PPOTrainer(
    model_dir="models",    # 模型保存目录
    stats_dir="stats"      # 统计保存目录
)
```

### 2. 保存文件

```python
# 保存模型
trainer.save_model("ppo_model_final.pth")
# 实际保存到: models/ppo_model_final.pth

# 保存统计
trainer.save_training_stats("training_stats_final.json")
# 实际保存到: stats/training_stats_final.json
```

### 3. 加载文件

```python
# 加载模型
trainer.load_model("ppo_model_final.pth")
# 自动从: models/ppo_model_final.pth 加载

# 加载统计（手动）
import json
with open("stats/training_stats_final.json", 'r') as f:
    stats = json.load(f)
```

## 优势

### 1. 文件组织清晰
- 模型文件和统计文件分开存储
- 避免根目录文件过多
- 便于管理和备份

### 2. 自动化管理
- 训练时自动创建目录
- 保存时自动使用正确路径
- 加载时自动查找正确位置

### 3. 易于扩展
- 可以轻松添加新的文件类型
- 支持多个实验的模型管理
- 便于版本控制

## 迁移说明

如果您有旧版本的训练文件，可以手动移动到对应目录：

```bash
# 移动模型文件
mkdir -p models
mv *.pth models/

# 移动统计文件
mkdir -p stats
mv *.json stats/
```

## 注意事项

1. **目录权限**: 确保程序有创建目录和写入文件的权限
2. **路径一致性**: 加载模型时需要使用相同的网络参数
3. **文件清理**: 定期清理不需要的检查点文件以节省空间
4. **备份策略**: 建议定期备份重要的模型文件

## 故障排除

### 1. 目录不存在
```python
# 手动创建目录
import os
os.makedirs("models", exist_ok=True)
os.makedirs("stats", exist_ok=True)
```

### 2. 权限错误
```bash
# 检查目录权限
ls -la models/ stats/

# 修改权限（如果需要）
chmod 755 models/ stats/
```

### 3. 模型加载失败
- 检查模型文件是否存在
- 确认网络参数是否匹配
- 查看错误信息中的具体参数差异 