# 实验自动化使用指南

本指南介绍如何使用自动化脚本进行批量实验并记录数据。

## 文件说明

1. **experiment_automation.py** - 主实验自动化脚本
2. **extract_scheduling_metrics.py** - 指标提取工具
3. **EXPERIMENT_GUIDE.md** - 本使用指南

## 实验流程

每次实验包含以下步骤：

1. 选择数据集（从 `workload_datasets/` 中选择）
2. 更新配置文件：
   - `workload_config.json` - 负载数据
   - `task_links/1/links_range.json` - 任务请求数据
3. 启动10个边缘节点的 `agent.py`（通过 `run_agents.sh`）
4. 在 222.201.187.50 运行 `test.py`
5. 发送 curl 命令启动负载
6. 等待负载启动完成
7. 发送 curl 命令获取调度结果
8. 记录数据（负载均衡度L、带宽满足度D_BW、联合参数）

## 使用方法

### 方法1：使用自动化脚本（推荐）

```bash
cd /home/qianguo/Edge-Scheduler/Controller

# 运行所有数据集，每个数据集运行1次
python3 experiment_automation.py --datasets 1 2 3 4 5 --runs 1

# 运行指定数据集，每个数据集运行3次
python3 experiment_automation.py --datasets 1 2 3 --runs 3

# 只运行数据集1，运行5次
python3 experiment_automation.py --datasets 1 --runs 5
```

### 方法2：手动运行单次实验

如果自动化脚本无法正常工作，可以手动执行以下步骤：

#### 步骤1：更新配置文件

```bash
cd /home/qianguo/Edge-Scheduler/Controller

# 选择数据集ID（例如：1）
DATASET_ID=1

# 更新workload_config.json
cp workload_datasets/workload_config_${DATASET_ID}.json workload_config.json

# 更新task_links配置
cp workload_datasets/task_links_config_${DATASET_ID}.json task_links/1/links_range.json
```

#### 步骤2：启动agents

```bash
cd /home/qianguo/Edge-Scheduler
bash run_agents.sh start
```

#### 步骤3：启动test.py

在 222.201.187.50 的终端中：

```bash
cd /home/qianguo/Edge-Scheduler/Controller
python3 test.py
```

#### 步骤4：启动负载

在另一个终端（也在 222.201.187.50）：

```bash
curl -X POST http://222.201.187.50:3333/bgload/add \
  -H "Content-Type: application/json" \
  -d '{"use_config": true, "auto_launch": true}'
```

等待负载启动完成（约10-15秒）。

#### 步骤5：获取调度结果

```bash
curl 'http://localhost:3333/startupTask?taskId=1'
```

等待调度完成（约15-20秒）。

#### 步骤6：提取和记录指标

指标通常会在 test.py 的输出中打印，格式类似：
```
从环境获取指标: L=0.1234, D_BW=0.5678
```

或者使用指标提取工具：

```bash
# 从日志文件提取
python3 extract_scheduling_metrics.py --log /path/to/log/file --output metrics.json

# 从JSON响应提取
python3 extract_scheduling_metrics.py --json /path/to/response.json --output metrics.json
```

## 数据记录

### 数据格式

每次实验记录以下数据：

- **experiment_id**: 实验ID
- **dataset_id**: 数据集ID
- **timestamp**: 时间戳
- **load_balance_degree (L)**: 负载均衡度（越小越好）
- **bandwidth_satisfaction (D_BW)**: 带宽满足度（越大越好）
- **composite_score**: 联合参数（综合指标）

### 联合参数计算

联合参数计算公式：

```
composite_score = l_weight * (1 - L) + dbw_weight * D_BW
```

其中：
- `l_weight = 0.5`（默认）
- `dbw_weight = 0.5`（默认）
- `(1 - L)` 将负载均衡度转换为"越大越好"的方向

### 结果文件

实验结果保存在 `experiment_results/` 目录：

- **experiment_results.csv** - CSV格式，方便用Excel或Python分析
- **experiment_results.json** - JSON格式，包含完整信息

## 指标计算参考

指标的计算方法参考 `base/algorithm/PPO_my/four_algorithms_with_balance_test.py`：

1. **负载均衡度 (L)**:
   - 通过 `network_scheduler.get_original_reward_components(virtual_work)` 获取
   - 返回字典中的 `'L'` 键

2. **带宽满足度 (D_BW)**:
   - 通过 `network_scheduler.get_original_reward_components(virtual_work)` 获取
   - 返回字典中的 `'D_BW'` 键

3. **联合参数**:
   - 综合指标，平衡负载均衡和带宽满足度
   - 计算公式见上方

## 清理功能

实验自动化脚本会在每次实验结束后自动清理：

1. **停止并删除负载容器** (stress:latest)
2. **停止并删除任务容器** (task1:v1.0)
3. **停止边缘设备agents**

所有实验完成后，会再次执行清理操作，确保环境干净。

### 手动清理

如果需要手动清理，可以使用：

```bash
# 使用清理脚本
bash /home/qianguo/Edge-Scheduler/Controller/cleanup_experiment.sh

# 或使用原始脚本
bash manage_containers.sh stop stress:latest
bash manage_containers.sh rm stress:latest
bash manage_containers.sh stop task1:v1.0
bash manage_containers.sh rm task1:v1.0
bash run_agents.sh stop
```

## 注意事项

1. **确保服务就绪**: 在发送调度请求前，确保 test.py 已完全启动
2. **等待时间**: 负载启动和调度完成需要一定时间，脚本中已设置等待时间，可根据实际情况调整
3. **数据提取**: 如果自动化脚本无法正确提取指标，需要手动从 test.py 的输出或日志中提取
4. **错误处理**: 如果某次实验失败，脚本会记录错误信息并继续下一个实验
5. **自动清理**: 每次实验结束后会自动清理容器和停止agents，确保下次实验环境干净

## 改进建议

如果需要从 test.py 的输出中自动提取指标，可以考虑：

1. **修改 test.py**: 在调度完成后，将指标输出到标准输出或日志文件
2. **添加API端点**: 在 test.py 中添加一个API端点，返回调度结果和指标
3. **日志解析**: 改进 `extract_scheduling_metrics.py`，更好地解析日志文件

## 数据可视化

实验完成后，可以使用以下工具进行数据可视化：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('experiment_results/experiment_results.csv')

# 绘制负载均衡度
plt.figure(figsize=(10, 6))
plt.plot(df['experiment_id'], df['load_balance_degree'], 'o-')
plt.xlabel('实验ID')
plt.ylabel('负载均衡度 (L)')
plt.title('负载均衡度变化')
plt.grid(True)
plt.savefig('load_balance.png')

# 绘制带宽满足度
plt.figure(figsize=(10, 6))
plt.plot(df['experiment_id'], df['bandwidth_satisfaction'], 'o-')
plt.xlabel('实验ID')
plt.ylabel('带宽满足度 (D_BW)')
plt.title('带宽满足度变化')
plt.grid(True)
plt.savefig('bandwidth_satisfaction.png')

# 绘制综合指标
plt.figure(figsize=(10, 6))
plt.plot(df['experiment_id'], df['composite_score'], 'o-')
plt.xlabel('实验ID')
plt.ylabel('综合指标')
plt.title('综合指标变化')
plt.grid(True)
plt.savefig('composite_score.png')
```

## 故障排除

### 问题1：无法连接到服务

- 检查 test.py 是否正在运行
- 检查防火墙设置
- 确认 IP 地址和端口正确

### 问题2：无法提取指标

- 检查 test.py 的输出是否包含指标信息
- 尝试手动运行一次实验，观察输出格式
- 根据实际输出格式修改 `extract_scheduling_metrics.py`

### 问题3：agents启动失败

- 检查 `run_agents.sh` 脚本权限：`chmod +x run_agents.sh`
- 检查SSH连接是否正常
- 检查远程设备的 agent.py 路径是否正确

## 联系与支持

如有问题，请检查：
1. 脚本日志输出
2. test.py 的标准输出
3. 实验结果的 error 字段
