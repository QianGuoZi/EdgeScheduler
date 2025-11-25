# CPU粒度判断逻辑更新说明

## 更新概述

已将CPU粒度（cpu_granularity）集成到资源分配和检查的整个流程中，确保CPU资源的计算和判断都基于统一的粒度标准。

## 核心概念

### 三个关键指标

1. **物理核心数（Physical Cores）**
   - `emulator.cpu`: int类型，表示物理CPU核心数
   - 例如：4核CPU，则 `cpu = 4`

2. **CPU份数（CPU Shares）**
   - `node.cpu`: int类型，表示分配的CPU份数
   - `emulator.cpuPreMap`: int类型，表示已分配的CPU份数总和
   - 例如：25份，则 `cpu = 25`

3. **CPU粒度（CPU Granularity）**
   - `emulator.cpu_granularity`: float类型，表示每份CPU对应的物理核心数
   - 默认值：0.04（即25份=1核心）
   - 例如：`cpu_granularity = 0.04`

### 转换公式

```
实际CPU核心数 = CPU份数 × CPU粒度
actual_cores = cpu_shares × cpu_granularity
```

## 修改详情

### 1. `node.py` - Emulator类

#### 1.1 构造函数更新

```python
def __init__(self, ID: int, name: str, ip: str, cpu: int, ram: int, 
             ip_controller: str, cpu_granularity: float = 0.04):
    self.cpu: int = cpu  # 物理核心数
    self.cpuPreMap: int = 0  # 已分配的CPU份数
    self.cpu_granularity: float = cpu_granularity  # CPU粒度
```

**变化：**
- 新增 `cpu_granularity` 参数，默认值0.04
- 明确注释：`cpu`是物理核心数，`cpuPreMap`是份数

#### 1.2 check_resource方法更新

```python
def check_resource(self, name: str, cpu: int, ram: int, cpu_granularity: float = None):
    """检查资源是否足够
    
    Args:
        cpu: CPU份数
        ram: 内存(MB)
        cpu_granularity: CPU粒度，为None时使用emulator的默认值
    """
    if cpu_granularity is None:
        cpu_granularity = self.cpu_granularity
    
    # 计算需要的实际CPU核心数
    required_cpu_cores = cpu * cpu_granularity
    # 计算已分配的实际CPU核心数
    allocated_cpu_cores = self.cpuPreMap * cpu_granularity
    # 计算剩余可用的CPU核心数
    available_cpu_cores = self.cpu - allocated_cpu_cores
    
    assert available_cpu_cores >= required_cpu_cores and ...
```

**变化：**
- 使用粒度将份数转换为实际核心数进行比较
- 提供更详细的错误信息，显示份数和实际核心数

#### 1.3 新增辅助方法

```python
def get_available_cpu_shares(self) -> int:
    """获取可用的CPU份数"""
    total_shares = int(self.cpu / self.cpu_granularity)
    return total_shares - self.cpuPreMap

def get_available_cpu_cores(self) -> float:
    """获取可用的实际CPU核心数"""
    allocated_cores = self.cpuPreMap * self.cpu_granularity
    return self.cpu - allocated_cores
```

**用途：**
- 方便调度器查询可用资源
- 支持以份数或核心数两种方式查询

### 2. `controller.py` - Controller类

#### 2.1 add_emulator方法更新

```python
def add_emulator(self, name: str, ip: str, cpu: int, ram: int, unit: str, 
                 cpu_granularity: float = 0.04) -> Emulator:
    """添加模拟器
    
    Args:
        cpu: CPU核心数（物理核心）
        cpu_granularity: CPU分配的最小粒度，默认0.04
    """
    e = Emulator(wid, name, ip, cpu, ram, self.ip, cpu_granularity)
```

**变化：**
- 新增 `cpu_granularity` 参数
- 传递给Emulator构造函数

#### 2.2 add_emulated_node方法更新

```python
def add_emulated_node(self, name: str, taskID: int, working_dir: str, 
                      cmd: List[str], image: str, cpu: int, ram: int, 
                      unit: str, nic: str = 'eth0', 
                      emulator: Emulator = None) -> EmulatedNode:
    """添加模拟节点
    
    Args:
        cpu: CPU份数（注意：这是份数，不是核心数）
    """
    if emulator:
        # check_resource会自动使用emulator的cpu_granularity
        emulator.check_resource(name, cpu, ram)
```

**变化：**
- 明确注释：`cpu`参数是份数
- `check_resource`自动使用emulator的粒度

### 3. `scheduler.py` - Scheduler类

#### 3.1 resource_schedule方法更新

```python
for emulator in self.controller.emulator.values():
    # 计算可用的CPU份数
    available_cpu_shares = emulator.get_available_cpu_shares()
    available_ram = emulator.ram - emulator.ramPreMap
    physical_nodes.append({
        'name': emulator.nameW,
        'cpu': available_cpu_shares,  # CPU份数
        'ram': available_ram
    })
    # 显示实际核心数以便调试
    available_cores = emulator.get_available_cpu_cores()
    print(f"Emulator: {emulator.nameW}, "
          f"CPU: {available_cpu_shares} shares ({available_cores:.2f} cores), "
          f"RAM: {available_ram} MB")
```

**变化：**
- 使用 `get_available_cpu_shares()` 获取份数
- 同时显示份数和实际核心数方便调试

## 使用示例

### 示例1：4核CPU，粒度0.04

```python
# 创建4核emulator，粒度0.04（即100份=4核）
emulator = controller.add_emulator(
    name='emulator-1',
    ip='100.68.2.1',
    cpu=4,                    # 4个物理核心
    ram=16,
    unit='G',
    cpu_granularity=0.04      # 每份=0.04核心
)

# 总可用份数 = 4 / 0.04 = 100份

# 添加第1个节点：37份 = 1.48核心
node1 = controller.add_emulated_node(
    name='1_p1',
    taskID=1,
    working_dir='/path/to/app',
    cmd=['python3', 'app.py'],
    image='task1:v1.0',
    cpu=37,                   # 37份
    ram=5120,
    unit='M',
    emulator=emulator
)
# check_resource: 需要1.48核心，可用4.0核心 ✓

# 添加第2个节点：32份 = 1.28核心
node2 = controller.add_emulated_node(
    name='1_n1',
    taskID=1,
    working_dir='/path/to/app',
    cmd=['python3', 'app.py'],
    image='task1:v1.0',
    cpu=32,                   # 32份
    ram=5120,
    unit='M',
    emulator=emulator
)
# check_resource: 需要1.28核心，可用2.52核心(4-1.48) ✓

# 添加第3个节点：31份 = 1.24核心
node3 = controller.add_emulated_node(
    name='1_n3',
    taskID=1,
    working_dir='/path/to/app',
    cmd=['python3', 'app.py'],
    image='task1:v1.0',
    cpu=31,                   # 31份
    ram=5120,
    unit='M',
    emulator=emulator
)
# check_resource: 需要1.24核心，可用1.24核心(4-1.48-1.28) ✓

# 总计：37+32+31=100份 = 4.0核心（完美利用）
```

### 示例2：资源不足的情况

```python
# 已经分配了100份（4.0核心）
# 尝试再添加一个节点：5份 = 0.2核心

node4 = controller.add_emulated_node(
    name='1_n4',
    taskID=1,
    working_dir='/path/to/app',
    cmd=['python3', 'app.py'],
    image='task1:v1.0',
    cpu=5,                    # 5份
    ram=5120,
    unit='M',
    emulator=emulator
)
# ❌ AssertionError: 
# emulator-1's cpu or ram is not enough for 1_n4.
# Required: 0.20 CPU cores (5 shares), 5120MB RAM.
# Available: 0.00 CPU cores, 11264MB RAM.
```

### 示例3：不同粒度的对比

```python
# 粒度0.04：适合精细分配
emulator1 = controller.add_emulator(
    'emu1', '100.68.2.1', cpu=4, ram=16, unit='G',
    cpu_granularity=0.04  # 100份 = 4核
)

# 粒度0.1：适合中等分配
emulator2 = controller.add_emulator(
    'emu2', '100.68.2.2', cpu=4, ram=16, unit='G',
    cpu_granularity=0.1   # 40份 = 4核
)

# 粒度0.5：适合粗粒度分配
emulator3 = controller.add_emulator(
    'emu3', '100.68.2.3', cpu=4, ram=16, unit='G',
    cpu_granularity=0.5   # 8份 = 4核
)
```

## 资源计算流程图

```
添加节点请求
    ↓
controller.add_emulated_node(cpu=37)  # 37份
    ↓
emulator.check_resource(cpu=37)
    ↓
required_cores = 37 × 0.04 = 1.48     # 转换为核心数
allocated_cores = cpuPreMap × 0.04    # 已分配核心数
available_cores = 4 - allocated_cores  # 可用核心数
    ↓
available_cores >= required_cores ?
    ├─ Yes → 通过检查
    └─ No  → 抛出异常（显示份数和核心数）
    ↓
emulator.add_node()
    ↓
cpuPreMap += 37  # 累加份数
```

## 调度器视角

调度器看到的是"份数"：

```python
# 调度器获取资源信息
physical_nodes = [
    {
        'name': 'emulator-1',
        'cpu': 100,      # 100份可用（对应4.0核心）
        'ram': 16384
    }
]

virtual_nodes = [
    {'name': '1_p1', 'cpu': 37, 'ram': 5120},  # 需要37份
    {'name': '1_n1', 'cpu': 32, 'ram': 5120},  # 需要32份
    {'name': '1_n3', 'cpu': 31, 'ram': 5120}   # 需要31份
]

# 调度算法只需要处理份数
# 底层的Emulator会自动处理份数到核心数的转换
```

## 优势总结

1. **统一标准**：所有CPU计算都基于相同的粒度
2. **精确控制**：可以精确到0.04核心（或更小）
3. **清晰分离**：调度器处理份数，Emulator处理核心数转换
4. **灵活配置**：可以为不同emulator设置不同粒度
5. **易于调试**：错误信息同时显示份数和核心数
6. **向后兼容**：默认粒度0.04，现有代码无需修改

## 注意事项

1. **参数含义**：
   - `emulator.cpu` = 物理核心数（整数）
   - `node.cpu` = CPU份数（整数）
   - `emulator.cpu_granularity` = 每份对应的核心数（小数）

2. **粒度选择**：
   - 0.04：适合4核机器分100份
   - 0.01：超细粒度，适合严格控制
   - 0.1：中等粒度，适合简单场景

3. **总份数计算**：
   ```python
   total_shares = int(cpu / cpu_granularity)
   # 例如：4核 / 0.04 = 100份
   ```

4. **浮点精度**：由于浮点运算，可能有微小误差，但在实际使用中可忽略

