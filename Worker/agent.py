from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor
import os
import socket
import subprocess as sp
import threading
import time
from typing import Dict

from flask import Flask, json, request
import requests


class classInfo:
    def __init__(self,node_name: str, node: str):
        self.formNode = node_name
        self.toNode = node

# DO NOT change this port number.
agent_port = 3333
executor = ThreadPoolExecutor ()
lock = threading.RLock ()
app = Flask (__name__)
dirname = os.path.abspath (os.path.dirname (__file__))
hostname = socket.gethostname ()
heartbeat = {}
tc_data = {}
physical_nic = ''
ctl_addr = ''
dml_p: sp.Popen
classid : Dict[classInfo,int]={}
classNum : Dict[str,int]={}
linkDict : Dict[str,set]={}



@app.route ('/hi', methods=['GET'])
def route_hi ():
	# 返回主机名称
	return 'this is agent ' + hostname + '\n'


@app.route ('/heartbeat', methods=['GET'])
def route_heartbeat ():
	# 收集心跳，假如是第一次从仿真节点接收，则需要部署容器的tc设置，用于模拟网络
	"""
	listen message from worker/worker_utils.py, heartbeat ().
	it will store the time of nodes heartbeat.
	when it receives the heartbeat of an emulated node for the first time,
	it will deploy the container's tc settings.
	"""
	name = request.args.get ('name')
	
	# 添加调试和错误处理
	if not name:
		print("错误: heartbeat 请求中缺少 name 参数")
		return 'Missing name parameter', 400
	
	try:
		# 从节点名称中提取 taskID (格式: taskID_nodename)
		taskID = name.split("_")[0]
		print(f"收到节点 {name} 的心跳, taskID: {taskID}")
	except Exception as e:
		print(f"错误: 从节点名称 {name} 提取 taskID 失败: {str(e)}")
		return 'Invalid name format', 400
	
	t_time = time.time ()
	with lock:
		# deploy the emulated node's tc settings.
		if name not in heartbeat and name in tc_data:
			print(f"首次收到节点 {name} 的心跳，开始部署 tc 设置")
			ret = {}
			deploy_emulated_tc (name, ret)
			# this request can be received by controller/base/node.py, route_emulated_tc ().
			try:
				print(f"发送 tc 响应到 controller: taskID={taskID}, data={ret}")
				response = requests.post ('http://' + ctl_addr + '/emulated/tc', 
				                         data={'taskID': taskID,'data': json.dumps (ret)})
				print(f"Controller 响应状态码: {response.status_code}")
				if response.status_code != 200:
					print(f"Controller 响应内容: {response.text}")
			except Exception as e:
				print(f"发送 tc 响应到 controller 失败: {str(e)}")
		heartbeat [name] = t_time
	return ''

def deploy_emulated_tc(name: str, ret: Dict):
    # 部署仿真节点的tc设置
    node_name = name
    time_start = time.time()
    data = tc_data[name]
    prefix = 'sudo docker exec ' + name + ' '
    # 清理旧的tc设置
    clear_old_tc(prefix, data['NET_NODE_NIC'])
    # 配置新的tc设置
    msg = create_new_tc(prefix, data['NET_NODE_NIC'], data['NET_NODE_TC'],
                        data['NET_NODE_TC_IP'], data['NET_NODE_TC_PORT'], node_name)
    if msg == '':
        print(name + ' tc succeed')
        with lock:
            ret[name] = {'number': len(data['NET_NODE_TC'])}
    else:
        print(name + ' tc failed, err:')
        print(msg)
        with lock:
            ret[name] = {'msg': msg}
    time_end = time.time()
    print('all time cost', time_end - time_start, 's')

def clear_old_tc(prefix: str, nic: str):
    # 清除旧的tc设置
    cmd = prefix + ' tc qdisc show dev %s' % nic
    p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, shell=True)
    msg = p.communicate()[0].decode()
    if "priomap" not in msg and "noqueue" not in msg:
        cmd = prefix + ' tc qdisc del dev %s root' % nic
        sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, shell=True).wait()


def create_new_tc(prefix: str, nic: str, tc: Dict[str, str], tc_ip: Dict[str, str],
                  tc_port: Dict[str, int], node_name: str):
    global classNum
    global classid
    # 配置新的tc设置
    if not tc:
        return ''

    # 使用 default 2 让未匹配流量走管理 class，避免被限速
    cmd = ['%s tc qdisc add dev %s root handle 1: htb default 2' % (prefix, nic),
           '%s tc class add dev %s parent 1: classid 1:1 htb rate 10gbps ceil 10gbps burst 1gb' % (prefix, nic),
           # 为管理流量创建豁免class (高优先级，不限速) - 用于心跳、Controller通信、ARP等
           '%s tc class add dev %s parent 1:1 classid 1:2 htb rate 10gbps ceil 10gbps burst 1gb' % (prefix, nic),
           # 为 class 1:2 添加叶子 qdisc (pfifo)，用于实际排队发送数据包
           '%s tc qdisc add dev %s parent 1:2 handle 20: pfifo limit 1000' % (prefix, nic),
           # 添加匹配所有 IP 流量的 filter (prio 10, 低优先级) - 确保所有流量都有出路
           '%s tc filter add dev %s protocol ip parent 1: prio 10 u32 match ip dst 0.0.0.0/0 flowid 1:2' % (prefix, nic),
           # 为ICMP流量(ping)创建filter，使用prio 1(高优先级)
           '%s tc filter add dev %s protocol ip parent 1: prio 1 u32 match ip protocol 1 0xff flowid 1:2' % (prefix, nic),
           # 为到Agent的流量(端口3333)创建filter，使用prio 1(高优先级)
           '%s tc filter add dev %s protocol ip parent 1: prio 1 u32 match ip dport 3333 0xffff flowid 1:2' % (prefix, nic),
           # 为从外部访问容器的流量创建filter(端口范围8000-9000)，用于Controller访问
           '%s tc filter add dev %s protocol ip parent 1: prio 1 u32 match ip sport 8000 0xf000 flowid 1:2' % (prefix, nic)]
    num = classNum.get(node_name,10)
    nodeset = set()
    if linkDict.get(node_name) != None:
        nodeset = linkDict[node_name]
    for name in tc.keys():
        bw = tc[name]
        ip = tc_ip[name]
        port = tc_port[name]
        # 创建限速 class，使用合理的 burst 值（至少 32KB）
        cmd.append('%s tc class add dev %s parent 1:1 classid ' % (prefix, nic)
                   + '1:%d htb rate %s ceil %s burst 32kb' % (num, bw, bw))
        # 为每个限速 class 添加叶子 qdisc (pfifo)
        cmd.append('%s tc qdisc add dev %s parent 1:%d handle %d0: pfifo limit 1000' % (prefix, nic, num, num))
        # 创建 filter 匹配目标 IP 和端口
        cmd.append('%s tc filter add dev %s protocol ip parent 1: prio 2 u32 match ip dst ' % (prefix, nic)
                   + '%s/32 match ip dport %d 0xffff flowid 1:%d' % (ip, port, num))
        nodepair = classInfo(node_name,name)
        classid[nodepair] = num
        print(node_name + ' to ' + name + ' link id is: ' + str(classid[nodepair]))
        nodeset.add(name)
        num += 1
    classNum[node_name] = num
    linkDict[node_name] = nodeset
    print(node_name + 'classNum: ' + str(classNum[node_name]))
    for name in linkDict[node_name]:
        print(node_name + 'has: ' + name)
    p = sp.Popen(' && '.join(cmd), stdout=sp.PIPE, stderr=sp.STDOUT, shell=True, close_fds=True)
    msg = p.communicate()[0].decode()
    return msg

@app.route ('/heartbeat/all', methods=['GET'])
def route_heartbeat_all ():
	# 通过一个GET请求来查看发送一个heartbeat的时间开销
	"""
	you can send a GET request to this /heartbeat/all to get
	how much time has passed since nodes last sent a heartbeat.
	"""
	s = 'the last heartbeat of nodes are:\n'
	now = time.time ()
	for name in heartbeat:
		_time = now - heartbeat [name]
		s = s + name + ' was ' + str (_time) + ' seconds ago. ' \
		    + 'it should be less than 30s.\n'
	return s

@app.route ('/heartbeat/abnormal', methods=['GET'])
def route_abnormal_heartbeat ():
	# 通过发送GET请求获取可能异常的节点  
	"""
	you can send a GET request to this /heartbeat/abnormal to get
	the likely abnormal nodes.
	"""
	s = 'the last heartbeat of likely abnormal nodes are:\n'
	now = time.time ()
	for name in heartbeat:
		_time = now - heartbeat [name]
		if _time > 30:
			s = s + name + ' was ' + str (_time) + ' seconds ago. ' \
			    + 'it should be less than 30s.\n'
	return s

@app.route ('/emulator/info', methods=['GET'])
def route_emulator_info ():
	# 从controller层获取controller的ip、port和模拟器的名称
	"""
	listen message from controller/base/node.py, send_emulator_info ().
	save the ${ip:port} of ctl and emulator's name.
	"""
	global ctl_addr, hostname
	ctl_addr = request.args.get ('address')
	hostname = request.args.get ('name')
	return ''

@app.route ('/emulated/tc', methods=['POST'])
def route_emulated_tc ():
	# 从controller层获取tc设置的内容
	"""
	listen message from controller/base/node.py, send_emulated_tc ().
	after emulated nodes are ready, it will deploy emulated nodes' tc settings.
	"""
	data = json.loads (request.form ['data'])
	print (data)
	tc_data.update (data)
	return ''

@app.route ('/emulated/tc/update', methods=['POST'])
def route_emulated_tc_update ():
	# 从controller层获取tc设置的更新
	"""
	listen message from controller/base/manager.py, update_emulated_tc ().
	after emulated nodes are ready, it will deploy emulated nodes' tc settings.
	"""
	data = json.loads (request.form ['data'])
	print (data)
	tc_data.update (data)

	ret = {}
	tasks = []
	for name in data:
		tasks.append (executor.submit (deploy_emulated_tc, name, ret))
	os.wait (tasks, return_when=ALL_COMPLETED)
	return json.dumps (ret)

@app.route('/emulated/build', methods=['POST'])
def route_emulated_build():
    """从controller层获取模拟器docker相关的信息，创建image"""
    taskID = request.form.get('taskID')
    if not taskID:
        print('Error: No taskID provided')
        return '-1'

    try:
        task_dir = os.path.join(dirname, str(taskID))
        # 确保目标目录存在
        os.makedirs(task_dir, exist_ok=True)
        
        # 保存文件
        path = os.path.join(task_dir, 'Dockerfile')
        dockerfile = request.files.get('Dockerfile')
        dml_req = request.files.get('dml_req')
        
        if not dockerfile or not dml_req:
            print('Error: Missing required files')
            return '-1'
            
        dockerfile.save(path)
        dml_req_path = os.path.join(task_dir, 'dml_req.txt')
        dml_req.save(dml_req_path)
        
        # 构建镜像
        tag = request.form.get('tag')
        if not tag:
            print('Error: No tag provided')
            return '-1'
            
        # 修改构建命令,指定正确的构建上下文路径和平台
        cmd = f'cd {task_dir} && sudo docker build --platform linux/arm64 -t {tag} .'
        print(f'执行命令: {cmd}')
        
        p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
        msg = p.communicate()[0].decode()
        print(msg)
        
        if 'Successfully tagged' in msg:
            print('构建镜像成功')
            return '1'
        else:
            print('构建镜像失败') 
            return '-1'
            
    except Exception as e:
        print(f'构建过程出错: {str(e)}')
        return '-1'

# @app.route ('/emulated/build', methods=['POST'])
# def route_emulated_build ():
# 	# 从controller层获取模拟器docker相关的信息ym文件，创建image
# 	"""
# 	listen file from controller/base/node.py, build_emulated_env ().
# 	it will use these files to build a docker image.
# 	"""
# 	taskID = request.form.get('taskID')
# 	task_dir = os.path.join(dirname, taskID)
#     # 确保目标目录存在
# 	os.makedirs(task_dir, exist_ok=True)
#     # 保存文件
# 	path = os.path.join(task_dir, 'Dockerfile')
# 	request.files.get('Dockerfile').save(path)
# 	request.files.get('dml_req').save(os.path.join(task_dir, 'dml_req.txt'))
	
# 	tag = request.form ['tag']
# 	cmd = 'sudo docker build -t ' + tag + ' -f ' + path + ' .'
# 	print (cmd)
# 	p = sp.Popen (cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
# 	msg = p.communicate () [0].decode ()
# 	print (msg)
# 	if 'Successfully tagged' in msg:
# 		print ('build image succeed')
# 		return '1'
# 	else:
# 		print ('build image failed')
# 		return '-1'
	
@app.route ('/emulated/launch', methods=['POST'])
def route_emulated_launch ():
	# 从controller层获取yml文件，heartbeat会清空，用docker-compose开启容器
	"""
	listen file from controller/base/node.py, launch_emulated ().
	it will launch the yml file.
	"""
	print('===== route_emulated_launch 开始 =====')
	print(f'request.form: {request.form}')
	print(f'request.files: {request.files}')
	
	heartbeat.clear ()
	taskID = request.form.get('taskID')
	print(f'taskID: {taskID}, type: {type(taskID)}')
	
	if not taskID:
		print('错误: taskID 为空')
		return 'Error: taskID is required', 400
		
	yml_file = request.files.get('yml')
	if not yml_file:
		print('错误: yml 文件为空')
		return 'Error: yml file is required', 400
	
	task_dir = os.path.join(dirname, taskID)
	os.makedirs(task_dir, exist_ok=True)
	filename = os.path.join (task_dir, hostname + '_' + str(taskID) + '.yml')
	print(f'保存文件到: {filename}')
	yml_file.save (filename)
	cmd = 'sudo COMPOSE_HTTP_TIMEOUT=120 docker-compose -f ' + filename + ' up'
	print (cmd)
	sp.Popen (cmd, shell=True, stderr=sp.STDOUT)
	print('===== route_emulated_launch 完成 =====')
	return ''

@app.route ('/emulated/stop', methods=['GET'])
def route_emulated_stop ():
	# 从controller层获取暂停指令，heartbeat会清空，用docker-compose暂停容器
	"""
	listen message from controller/base/manager.py, stop_emulated ().
	it will stop the above yml file.
	"""
	taskID = request.args.get ('taskID')
	task_dir = os.path.join(dirname, taskID)
	filename = os.path.join (task_dir, hostname + '.yml')
	cmd = 'sudo docker-compose -f ' + filename + ' stop'
	print (cmd)
	sp.Popen (cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait ()
	heartbeat.clear ()
	return ''

@app.route ('/emulated/clear', methods=['GET'])
def route_emulated_clear ():
	# 从controller层获取终止指令，heartbeat会清空，用docker-compose终止容器，删除yml文件
	"""
	listen message from controller/base/manager.py, clear_emulated ().
	it will clear the above yml file.
	"""
	taskID = request.args.get ('taskID')
	task_dir = os.path.join(dirname, taskID)
	filename = os.path.join (task_dir, hostname + '.yml')
	cmd = 'sudo docker-compose -f ' + filename + '.yml down -v'
	print (cmd)
	sp.Popen (cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait ()
	heartbeat.clear ()
	return ''

#目前用不上
@app.route('/emulated/node/remove', methods=['GET'])
def route_emulated_node_remove():
    # 从controller层获取移除指令，heartbeat会清空，用docker-compose移除某个容器
	taskID = request.args.get ('taskID')
	node_name = request.args.get('node_name')
	task_dir = os.path.join(dirname, taskID)
	filename = os.path.join (task_dir, hostname + '.yml')
	cmd = 'sudo docker-compose -f ' + filename + ' stop ' + node_name + ' && sudo docker-compose -f ' \
          + filename + ' rm -f ' + node_name
	print(cmd)
	sp.Popen(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait()
	return ''

@app.route ('/emulated/reset', methods=['GET'])
def route_emulated_reset ():
	# 删除所有docker容器、网络和数据卷
	"""
	listen message from controller/base/manager.py, reset_emulated ().
	it will remove all docker containers, networks and volumes.
	"""
	cmd = ['sudo docker rm -f $(docker ps -aq)',
	       'sudo docker network rm $(docker network ls -q)',
	       'sudo docker volume rm $(docker volume ls -q)']
	for c in cmd:
		print (c)
		sp.Popen (c, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait ()
	heartbeat.clear ()
	return ''


# ===================== 背景负载相关路由 =====================

@app.route('/bgload/launch', methods=['POST'])
def route_bgload_launch():
	"""
	从controller接收背景负载yml文件，启动背景负载容器
	这些容器用于模拟emulator上已有的负载
	"""
	print('===== route_bgload_launch 开始 =====')
	print(f'request.files: {request.files}')
	
	yml_file = request.files.get('yml')
	if not yml_file:
		print('错误: yml 文件为空')
		return 'Error: yml file is required', 400
	
	# 保存yml文件到bgload目录
	bgload_dir = os.path.join(dirname, 'bgload')
	os.makedirs(bgload_dir, exist_ok=True)
	filename = os.path.join(bgload_dir, hostname + '_bgload.yml')
	print(f'保存文件到: {filename}')
	yml_file.save(filename)
	
	# 使用docker-compose启动背景负载容器（后台运行，使用-d参数）
	cmd = f'sudo COMPOSE_HTTP_TIMEOUT=120 docker-compose -f {filename} up -d'
	print(cmd)
	p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	msg = p.communicate()[0].decode()
	print(f'启动结果: {msg}')
	
	print('===== route_bgload_launch 完成 =====')
	return ''


@app.route('/bgload/stop', methods=['GET'])
def route_bgload_stop():
	"""
	停止背景负载容器
	"""
	print('===== route_bgload_stop 开始 =====')
	
	bgload_dir = os.path.join(dirname, 'bgload')
	filename = os.path.join(bgload_dir, hostname + '_bgload.yml')
	
	if not os.path.exists(filename):
		print(f'背景负载yml文件不存在: {filename}')
		return 'No background load yml file found', 404
	
	cmd = f'sudo docker-compose -f {filename} stop'
	print(cmd)
	p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	msg = p.communicate()[0].decode()
	print(f'停止结果: {msg}')
	
	print('===== route_bgload_stop 完成 =====')
	return ''


@app.route('/bgload/clear', methods=['GET'])
def route_bgload_clear():
	"""
	清理背景负载容器（停止并删除容器和yml文件）
	"""
	print('===== route_bgload_clear 开始 =====')
	
	bgload_dir = os.path.join(dirname, 'bgload')
	filename = os.path.join(bgload_dir, hostname + '_bgload.yml')
	
	if not os.path.exists(filename):
		print(f'背景负载yml文件不存在: {filename}')
		return 'No background load yml file found', 404
	
	# 停止并删除容器
	cmd = f'sudo docker-compose -f {filename} down -v'
	print(cmd)
	p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	msg = p.communicate()[0].decode()
	print(f'清理结果: {msg}')
	
	# 删除yml文件
	try:
		os.remove(filename)
		print(f'已删除yml文件: {filename}')
	except Exception as e:
		print(f'删除yml文件失败: {e}')
	
	print('===== route_bgload_clear 完成 =====')
	return ''


@app.route('/bgload/status', methods=['GET'])
def route_bgload_status():
	"""
	获取背景负载容器的状态
	"""
	print('===== route_bgload_status 开始 =====')
	
	bgload_dir = os.path.join(dirname, 'bgload')
	filename = os.path.join(bgload_dir, hostname + '_bgload.yml')
	
	if not os.path.exists(filename):
		return json.dumps({'status': 'no_config', 'containers': []})
	
	# 获取容器状态
	cmd = f'sudo docker-compose -f {filename} ps --format json'
	p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	msg = p.communicate()[0].decode()
	
	try:
		containers = json.loads(msg) if msg.strip() else []
	except:
		containers = []
	
	print('===== route_bgload_status 完成 =====')
	return json.dumps({'status': 'ok', 'containers': containers})


app.run (host='0.0.0.0', port=agent_port, threaded=True)