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
	t_time = time.time ()
	with lock:
		# deploy the emulated node's tc settings.
		if name not in heartbeat and name in tc_data:
			ret = {}
			deploy_emulated_tc (name, ret)
			# this request can be received by controller/base/node.py, route_emulated_tc ().
			requests.post ('http://' + ctl_addr + '/emulated/tc', data={'data': json.dumps (ret)})
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

    cmd = ['%s tc qdisc add dev %s root handle 1: htb default 1' % (prefix, nic),
           '%s tc class add dev %s parent 1: classid 1:1 htb rate 10gbps ceil 10gbps burst 15k' % (prefix, nic)]
    num = classNum.get(node_name,10)
    nodeset = set()
    if linkDict.get(node_name) != None:
        nodeset = linkDict[node_name]
    for name in tc.keys():
        bw = tc[name]
        ip = tc_ip[name]
        port = tc_port[name]
        cmd.append('%s tc class add dev %s parent 1:1 classid ' % (prefix, nic)
                   + '1:%d htb rate %s ceil %s burst 15k' % (num, bw, bw))
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

@app.route('/emulated/node/stop', methods=['GET'])
def route_emulated_node_stop():
    # 从controller层获取暂停指令，heartbeat会清空，用docker-compose暂停某个容器
    time_start = time.time()
    node_name = request.args.get('node_name')
    print(node_name)
    cmd = 'sudo docker-compose -f ' + hostname + '.yml stop ' + node_name
    print(cmd)
    sp.Popen(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait()
    time_end = time.time()
    print('stop time cost', time_end - time_start, 's')
    return ''

@app.route ('/emulated/build', methods=['POST'])
def route_emulated_build ():
	# 从controller层获取模拟器docker相关的信息ym文件，创建image
	"""
	listen file from controller/base/node.py, build_emulated_env ().
	it will use these files to build a docker image.
	"""
	path = os.path.join (dirname, 'Dockerfile')
	request.files.get ('Dockerfile').save (path)
	request.files.get ('dml_req').save (os.path.join (dirname, 'dml_req.txt'))
	tag = request.form ['tag']
	cmd = 'sudo docker build -t ' + tag + ' -f ' + path + ' .'
	print (cmd)
	p = sp.Popen (cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	msg = p.communicate () [0].decode ()
	print (msg)
	if 'Successfully tagged' in msg:
		print ('build image succeed')
		return '1'
	else:
		print ('build image failed')
		print ('build image failed')
		return '-1'
	
@app.route ('/emulated/launch', methods=['POST'])
def route_emulated_launch ():
	# 从controller层获取yml文件，heartbeat会清空，用docker-compose开启容器
	"""
	listen file from controller/base/node.py, launch_emulated ().
	it will launch the yml file.
	"""
	heartbeat.clear ()
	taskId = request.form['taskId']
	filename = os.path.join (dirname, hostname + '_' + str(taskId) + '.yml')
	request.files.get ('yml').save (filename)
	cmd = 'sudo COMPOSE_HTTP_TIMEOUT=120 docker-compose -f ' + filename + ' up'
	print (cmd)
	sp.Popen (cmd, shell=True, stderr=sp.STDOUT)
	return ''

@app.route ('/emulated/stop', methods=['GET'])
def route_emulated_stop ():
	# 从controller层获取暂停指令，heartbeat会清空，用docker-compose暂停容器
	"""
	listen message from controller/base/manager.py, stop_emulated ().
	it will stop the above yml file.
	"""
	cmd = 'sudo docker-compose -f ' + hostname + '.yml stop'
	print (cmd)
	sp.Popen (cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait ()
	heartbeat.clear ()
	return ''

@app.route('/emulated/node/remove', methods=['GET'])
def route_emulated_node_remove():
    time_start = time.time()
    # 从controller层获取移除指令，heartbeat会清空，用docker-compose移除某个容器
    node_name = request.args.get('node_name')
    cmd = 'sudo docker-compose -f ' + hostname + '.yml stop ' + node_name + ' && sudo docker-compose -f ' \
          + hostname + '.yml rm -f ' + node_name
    print(cmd)
    sp.Popen(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait()
    time_end = time.time()
    print('docker remove time cost', time_end - time_start, 's')
    return ''

@app.route ('/emulated/clear', methods=['GET'])
def route_emulated_clear ():
	# 从controller层获取终止指令，heartbeat会清空，用docker-compose终止容器，删除yml文件
	"""
	listen message from controller/base/manager.py, clear_emulated ().
	it will clear the above yml file.
	"""
	cmd = 'sudo docker-compose -f ' + hostname + '.yml down -v'
	print (cmd)
	sp.Popen (cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait ()
	heartbeat.clear ()
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

app.run (host='0.0.0.0', port=agent_port, threaded=True)