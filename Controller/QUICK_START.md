# 实验自动化快速开始指南

## 概述

本工具集提供了自动化实验流程和数据记录功能，帮助您便捷地进行多次实验并收集数据。

## 文件说明

- **experiment_automation.py** - 主自动化脚本（自动执行完整实验流程）
- **manual_record_experiment.py** - 手动记录工具（当自动化失败时使用）
- **extract_scheduling_metrics.py** - 指标提取工具
- **visualize_experiment_results.py** - 数据可视化工具
- **EXPERIMENT_GUIDE.md** - 详细使用指南

## 快速开始

### 方法1：完全自动化（推荐）

```bash
cd /home/qianguo/Edge-Scheduler/Controller

# 运行所有数据集，每个数据集1次
python3 experiment_automation.py --datasets 1 2 3 4 5 --runs 1
```

### 方法2：手动执行 + 手动记录

如果自动化脚本无法正常工作，可以：

1. **手动执行实验流程**（参考 EXPERIMENT_GUIDE.md）

2. **手动记录数据**：
```bash
python3 manual_record_experiment.py \
  --dataset 1 \
  --L 0.1234 \
  --D_BW 0.5678
```

### 方法3：从日志提取指标

```bash
# 从日志文件提取指标
python3 extract_scheduling_metrics.py --log /path/to/log --output metrics.json

# 然后手动记录
python3 manual_record_experiment.py --dataset 1 --L <value> --D_BW <value>
```

## 数据可视化

实验完成后，生成图表：

```bash
# 生成所有图表
python3 visualize_experiment_results.py --all

# 或生成特定图表
python3 visualize_experiment_results.py --load-balance --bandwidth --composite
```

图表将保存在 `experiment_results/plots/` 目录。

## 结果文件

- **experiment_results/experiment_results.csv** - CSV格式数据
- **experiment_results/experiment_results.json** - JSON格式数据
- **experiment_results/plots/** - 可视化图表

## 清理功能

实验自动化脚本会在每次实验结束后自动清理容器和停止agents。如果需要手动清理：

```bash
bash cleanup_experiment.sh
```

## 常见问题

### Q: 自动化脚本无法提取指标怎么办？

A: 使用手动记录工具：
```bash
python3 manual_record_experiment.py --dataset 1 --L <值> --D_BW <值>
```

### Q: 如何查看实验结果？

A: 
1. 查看CSV文件：`cat experiment_results/experiment_results.csv`
2. 生成图表：`python3 visualize_experiment_results.py --all`

### Q: 如何修改联合参数的计算权重？

A: 编辑 `experiment_automation.py` 中的 `calculate_composite_score` 函数，修改 `l_weight` 和 `dbw_weight` 参数。

### Q: 实验结束后如何清理环境？

A: 脚本会自动清理，也可以手动执行：
```bash
bash cleanup_experiment.sh
```

## 下一步

详细说明请参考 **EXPERIMENT_GUIDE.md**
