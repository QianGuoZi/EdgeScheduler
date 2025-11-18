# 容器连接问题排查指南

## 问题描述

Controller 尝试向 Worker 节点（如 `100.68.2.2:8001`）发送配置文件时失败，错误信息：
```
[Errno 113] No route to host
```

## 诊断结果

### ✅ 已确认正常的部分
1. **网络连通性**: 可以 ping 通所有 emulator 主机 (100.68.2.1, 100.68.2.2, 100.68.2.3)
2. **Agent 服务**: 所有主机的 agent 服务 (端口 3333) 正常运行
3. **配置文件**: yml 文件配置正确，已生成在 Controller 目录

### ❌ 问题所在
1. **容器端口**: 所有容器端口 (8001, 8002, 8003, 8004, 8005) 都无法连接
2. **容器状态**: 容器可能没有成功启动

## 根本原因分析

根据代码分析，问题可能在于：

1. **docker-compose 启动失败**: 
   - agent.py 的 `route_emulated_launch()` 使用 `Popen` 异步启动容器
   - 没有等待容器启动完成就返回了
   - 可能因为权限、镜像、NFS 挂载等问题导致启动失败

2. **时序问题**:
   - Controller 在调用 `launch_all_emulated()` 后立即调用 `__send_conf()`
   - 容器可能还在启动过程中，端口还未就绪

3. **可能的具体原因**:
   - Docker 镜像 `task1:v1.0` 不存在
   - NFS 挂载失败 (100.68.1.50)
   - 端口已被占用
   - 容器启动时健康检查失败

## 解决方案

### 方案 1: 手动检查和修复（推荐首先执行）

#### 步骤 1: 检查容器状态
```bash
# 在各个 emulator 上检查容器
ssh 100.68.2.2 'sudo docker ps -a | grep 1_'
ssh 100.68.2.1 'sudo docker ps -a | grep 1_'
ssh 100.68.2.3 'sudo docker ps -a | grep 1_'
```

#### 步骤 2: 检查 yml 文件是否传输成功
```bash
ssh 100.68.2.2 'ls -la /home/qianguo/Edge-Scheduler/Worker/1/'
ssh 100.68.2.1 'ls -la /home/qianguo/Edge-Scheduler/Worker/1/'
ssh 100.68.2.3 'ls -la /home/qianguo/Edge-Scheduler/Worker/1/'
```

#### 步骤 3: 检查 Docker 镜像
```bash
ssh 100.68.2.2 'sudo docker images | grep task1'
ssh 100.68.2.1 'sudo docker images | grep task1'
ssh 100.68.2.3 'sudo docker images | grep task1'
```

#### 步骤 4: 查看容器日志（如果容器已创建）
```bash
ssh 100.68.2.2 'sudo docker logs 1_p1'
ssh 100.68.2.1 'sudo docker logs 1_n1'
ssh 100.68.2.3 'sudo docker logs 1_n2'
```

#### 步骤 5: 手动启动容器（如果容器未运行）
```bash
# 在 emulator-3 (100.68.2.2) 上
ssh 100.68.2.2
cd /home/qianguo/Edge-Scheduler/Worker/1/
sudo docker-compose -f emulator-3_1.yml up -d

# 在 emulator-2 (100.68.2.1) 上
ssh 100.68.2.1
cd /home/qianguo/Edge-Scheduler/Worker/1/
sudo docker-compose -f emulator-2_1.yml up -d

# 在 emulator-1 (100.68.2.3) 上
ssh 100.68.2.3
cd /home/qianguo/Edge-Scheduler/Worker/1/
sudo docker-compose -f emulator-1_1.yml up -d
```

#### 步骤 6: 验证容器启动
```bash
# 检查容器是否运行
ssh 100.68.2.2 'sudo docker ps | grep 1_p1'

# 测试容器端口
curl http://100.68.2.2:8001/hi

# 如果成功，应该返回: "this is node 1_p1"
```

### 方案 2: 代码修复（长期解决方案）

需要修改以下文件来添加容器启动等待机制：

#### 修改 1: Worker/agent.py - 添加容器状态检查 API

在 agent.py 中添加：
```python
@app.route('/container/status', methods=['GET'])
def route_container_status():
    """检查容器状态"""
    taskID = request.args.get('taskID')
    container_name = request.args.get('name')
    
    try:
        cmd = f'sudo docker ps --filter name={container_name} --format "{{{{.Status}}}}"'
        result = sp.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            return result.stdout.strip()
        else:
            return 'not found', 404
    except Exception as e:
        return str(e), 500
```

#### 修改 2: Controller/base/controller.py - 添加启动等待

在 `launch_all_emulated()` 后添加等待逻辑：
```python
def wait_for_containers(self, taskID: int, timeout=300):
    """等待所有容器启动完成"""
    import time
    start_time = time.time()
    
    for en in self.task[taskID].eNode.values():
        container_name = en.name
        emulator_ip = en.ip
        
        while time.time() - start_time < timeout:
            try:
                # 检查容器健康状态
                response = requests.get(
                    f'http://{emulator_ip}:{en.hostPort}/hi',
                    timeout=2
                )
                if response.status_code == 200:
                    print(f'容器 {container_name} 启动成功')
                    break
            except:
                time.sleep(5)
                continue
        else:
            raise TimeoutError(f'容器 {container_name} 启动超时')
```

#### 修改 3: 在 deploy_task() 中调用等待

在 `controller.py` 的 `deploy_task()` 方法中：
```python
self.launch_all_emulated(taskID)
self.wait_for_containers(taskID)  # 添加这一行
```

### 方案 3: 添加防火墙规则（如果需要）

如果是防火墙问题：
```bash
# 在各个 emulator 上
sudo firewall-cmd --zone=public --add-port=8001-8005/tcp --permanent
sudo firewall-cmd --reload

# 或者使用 iptables
sudo iptables -A INPUT -p tcp --dport 8001:8005 -j ACCEPT
```

## 快速修复命令

如果您只是想快速让系统工作，运行以下命令：

```bash
# 1. 检查并构建 Docker 镜像（如果不存在）
for host in 100.68.2.1 100.68.2.2 100.68.2.3; do
    ssh $host 'sudo docker images | grep task1' || echo "主机 $host 缺少镜像"
done

# 2. 手动启动所有容器
ssh 100.68.2.2 'cd /home/qianguo/Edge-Scheduler/Worker/1/ && sudo docker-compose -f emulator-3_1.yml up -d'
ssh 100.68.2.1 'cd /home/qianguo/Edge-Scheduler/Worker/1/ && sudo docker-compose -f emulator-2_1.yml up -d'
ssh 100.68.2.3 'cd /home/qianguo/Edge-Scheduler/Worker/1/ && sudo docker-compose -f emulator-1_1.yml up -d'

# 3. 等待 30 秒让容器启动
echo "等待容器启动..."
sleep 30

# 4. 验证所有容器
for host in 100.68.2.1 100.68.2.2 100.68.2.3; do
    echo "主机 $host 的容器:"
    ssh $host 'sudo docker ps | grep 1_'
done

# 5. 重新发送配置
curl "http://localhost:3333/conf/dataset?taskId=1"
curl "http://localhost:3333/conf/structure?taskId=1"
```

## 预防措施

为避免将来出现类似问题：

1. **在 Controller 添加启动检查**: 确保容器启动后再发送配置
2. **添加重试机制**: 发送配置时添加重试逻辑
3. **改进错误处理**: 捕获并记录详细的错误信息
4. **添加健康检查**: 定期检查容器和服务状态

## 相关文件

- `/home/qianguo/Edge-Scheduler/Worker/agent.py` - Agent 服务代码
- `/home/qianguo/Edge-Scheduler/Controller/base/controller.py` - Controller 主代码
- `/home/qianguo/Edge-Scheduler/Controller/base/manager.py` - 配置发送代码
- `/home/qianguo/Edge-Scheduler/Controller/emulator-*_1.yml` - Docker Compose 配置

## 联系支持

如果问题仍未解决，请提供：
1. 容器状态输出 (`docker ps -a`)
2. 容器日志 (`docker logs <container>`)
3. Agent 日志
4. Controller 完整错误信息

