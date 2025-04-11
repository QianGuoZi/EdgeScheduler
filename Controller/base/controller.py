from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor
from logging import Manager
import os
from queue import Queue
import threading
import time
from typing import Dict, List, Type
import inspect
import importlib.util
from .taskManger import TaskManager

from flask import Flask, json, request
from .link import VirtualLink
from .nfs import Nfs
from .node import EmulatedNode, Emulator, Node, PhysicalNode
from .scheduler import Scheduler
from .task import Task
from .utils import read_json, send_data


dirName = '/home/qianguo/controller/'
class Controller(object):
    """
    任务接收器，负责和用户进行交互
    """
    def __init__(self, ip: str, base_host_port: int, dir_name: str, manager: Manager, scheduler: Scheduler):
        self.currWID: int = 0  # build-in worker ID.
        self.currRID: int = 0  # build-in real link ID.
        self.currNID: int = 0  # build-in node ID.
        self.currVID: int = 0  # build-in virtual link ID.
        self.currTID: int = 0  # build-in task ID.


        self.flask = Flask(__name__)
        self.ip: str = ip
        self.port: int = 3333  # DO NOT change this port number.
        self.agentPort: int = 3333  # DO NOT change this port number.
        self.nodePort: int = 4444  # DO NOT change this port number.
        # emulated node maps dml port to emulator's host port starting from $(base_host_port).
        self.hostPort: int = base_host_port
        self.address: str = self.ip + ':' + str(self.port)
        self.dirName: str = dir_name

        self.nfs: Dict[str, Nfs] = {}  # nfs tag to nfs object.
        self.pNode: Dict[str, PhysicalNode] = {}  # physical node's name to physical node object.
        self.emulator: Dict[str, Emulator] = {}  # emulator's name to emulator object.
        self.eNode: Dict[str, EmulatedNode] = {}  # emulated node's name to emulated node object.
        self.vLink: Dict[int, VirtualLink] = {}  # virtual link ID to virtual link object.
        self.virtualLinkNumber: int = 0

        self.task: Dict[int, Task] = {} # task ID to task object.

        # for auto deployment.
        self.W: Dict[int, Dict] = {}  # worker ID to {name, cpu, MB of ram}.
        self.N: Dict[int, Dict] = {}  # node ID to {name, cpu, MB of ram}.
        self.RConnect: List[List[List[int]]]  # workers adjacency matrix.
        self.VConnect: List[List[List[int]]]  # nodes adjacency matrix.
        self.preMap: Dict[int, int] = {}  # node ID to worker ID.

        # for default manager.
        self.manager = manager(self)
        self.deployedCount: int = 0
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor()
        
        # scheduler
        self.scheduler = scheduler

        # 添加三个队列
        self.pending_tasks = Queue()  # 待调度任务队列
        self.scheduled_tasks = Queue()  # 已调度未部署任务队列
        self.deployed_tasks = Queue()  # 已部署任务队列
        
        # 启动调度和部署循环
        self.stop_processing = False
        self.schedule_thread = threading.Thread(target=self._schedule_loop)
        self.deploy_thread = threading.Thread(target=self._deploy_loop)
        self.schedule_thread.daemon = True
        self.deploy_thread.daemon = True
        self.schedule_thread.start()
        self.deploy_thread.start()

    def __next_w_id(self):
        self.currWID += 1
        return self.currWID

    def __next_r_id(self):
        self.currRID += 1
        return self.currRID

    def __next_n_id(self):
        self.currNID += 1
        return self.currNID

    def __next_v_id(self):
        self.currVID += 1
        return self.currVID

    def next_task_id(self):
        """
        获取下一个任务id
        """
        self.currTID += 1
        return self.currTID
    
    def add_nfs(self, tag: str, path: str, ip: str = '', mask: int = 16) -> Nfs:
        assert tag != '', Exception('tag cannot be empty')
        assert tag not in self.nfs, Exception(tag + ' has been used')
        assert 0 < mask <= 32, Exception(str(mask) + ' is not in range (0, 32]')
        assert path[0] == '/', Exception(path + ' is not an absolute path')
        if ip == '':
            ip = self.ip
        nfs = Nfs(tag, path, ip, mask)
        self.nfs[tag] = nfs
        return nfs
       
    def _schedule_loop(self):
        """持续检查待调度队列并进行调度"""
        while not self.stop_processing:
            if not self.pending_tasks.empty():
                with self.lock:
                    task_id = self.pending_tasks.get()
                    try:
                        # 调用scheduler进行调度
                        allocation = self.scheduler.resource_schedule(task_id)
                        # 将任务和其分配结果放入已调度队列
                        self.scheduled_tasks.put((task_id, allocation))
                    except Exception as e:
                        print(f"调度任务 {task_id} 失败: {str(e)}")
            time.sleep(1)

    def _deploy_loop(self):
        """持续检查已调度队列并进行部署"""
        while not self.stop_processing:
            if not self.scheduled_tasks.empty():
                with self.lock:
                    task_info = self.scheduled_tasks.get()
                    if task_info:
                        task_id, allocation = task_info
                        try:
                            # 进行部署
                            success = self.deploy_task(task_id, allocation)
                            if success:
                                # 部署成功，加入已部署队列
                                self.deployed_tasks.put(task_id)
                            else:
                                # 部署失败，重新加入已调度队列
                                self.scheduled_tasks.put((task_id, allocation))
                        except Exception as e:
                            print(f"部署任务 {task_id} 失败: {str(e)}")
                            # 部署失败，重新加入已调度队列
                            self.scheduled_tasks.put((task_id, allocation))
            time.sleep(1)

    def add_pending_task(self, task_id: int):
        """添加待调度任务"""
        self.pending_tasks.put(task_id)

    def get_scheduled_task(self) -> tuple:
        """获取已调度的任务"""
        if not self.scheduled_tasks.empty():
            return self.scheduled_tasks.get()
        return None

    def get_deployed_tasks(self) -> list:
        """获取所有已部署的任务ID列表"""
        return list(self.deployed_tasks.queue)

    def shutdown(self):
        """关闭所有处理循环"""
        self.stop_processing = True
        if self.schedule_thread.is_alive():
            self.schedule_thread.join()
        if self.deploy_thread.is_alive():
            self.deploy_thread.join()

    def add_emulator(self, name: str, ip: str, cpu: int, ram: int, unit: str) -> Emulator:
        assert name != '', Exception('name cannot be empty')
        assert name not in self.emulator, Exception(name + ' has been used')
        assert cpu > 0 and ram > 0, Exception('cpu or ram is not bigger than 0')
        assert unit in ['M', 'G'], Exception(unit + ' is not in ["M", "G"]')
        if unit == 'G':
            ram *= 1024
        wid = self.__next_w_id()
        e = Emulator(wid, name, ip, cpu, ram, self.ip)
        self.emulator[name] = e
        for tag in self.nfs.values():  # mount all nfs tags by default.
            e.mount_nfs(tag)

        self.W[wid] = {'name': name, 'cpu': cpu, 'ram': ram}
        return e
    
    def send_emulator_info(self):
        """
        send the ${ip:port} and emulator's name to emulators.
        this request can be received by worker/agent.py, route_emulator_info ().
        """
        for e in self.emulator.values():
            print('send_emulator_info: send to ' + e.nameW)
            send_data('GET', '/emulator/info?address=' + self.address + '&name=' + e.nameW,
                      e.ipW, self.agentPort)

    def add_emulated_node(self, name: str, taskID: int,working_dir: str, cmd: List[str], image: str,
                          cpu: int, ram: int, unit: str, nic: str = 'eth0', emulator: Emulator = None) -> EmulatedNode:
        assert name != '', Exception('name cannot be empty')
        assert name not in self.eNode, Exception(name + ' has been used')
        assert name not in self.pNode, Exception(name + ' has been used')
        assert cpu > 0 and ram > 0, Exception('cpu or ram is not bigger than 0')
        assert unit in ['M', 'G'], Exception(unit + ' is not in ["M", "G"]')
        if unit == 'G':
            ram *= 1024

        if emulator:
            emulator.check_resource(name, cpu, ram)
        nid = self.__next_n_id()
        en = EmulatedNode(nid, name, taskID, nic, working_dir, cmd, self.nodePort, self.hostPort, image, cpu, ram)

        if emulator:
            self.assign_emulated_node(en, emulator)

        self.eNode[name] = en
        self.N[nid] = {'name': name, 'cpu': cpu, 'ram': ram}
        return en
    
    def delete_emulated_node(self, en: EmulatedNode, emulator: Emulator = None):
        assert en.name in self.eNode, Exception(en.name + ' is not existed')
        if emulator:
            emulator.delete_node(en)
        del self.eNode[en.name]
        del self.N[en.id]
        del self.preMap[en.id]

    def assign_emulated_node(self, en: EmulatedNode, emulator: Emulator):
        assert en.id not in self.preMap, Exception(en.name + ' has been assigned')
        emulator.add_node(en)
        en.add_var({
            'NET_CTL_ADDRESS': self.address,
            'NET_AGENT_ADDRESS': emulator.ipW + ':' + str(self.agentPort),
            'NET_TASK_ID': str(en.tid)
        })
        self.preMap[en.id] = emulator.idW

    def load_link(self,taskId: int, links_json: Dict):
        for name in links_json:
            nodeName = str(taskId) + '_' + name
            src = self.name_to_node(nodeName)
            print(f"{nodeName}, {src}")
            for dest_json in links_json[name]:
                destName = str(taskId) + '_' + dest_json['dest']
                dest = self.name_to_node(destName)
                unit = dest_json['bw'][-4:]
                _bw = int(dest_json['bw'][:-4])
                print(f"{destName}, {dest}")
                self.__add_virtual_link(src, dest, _bw, unit)

    def name_to_node(self, name: str) -> Node:
        """
        get node by name.
        """
        if name in self.pNode:
            return self.pNode[name]
        elif name in self.eNode:
            return self.eNode[name]
        else:
            Exception('no such node called ' + name)

    def __add_virtual_link(self, n1: Node, n2: Node, bw: int, unit: str):
        """
        parameters will be passed to Linux Traffic Control.
        n1-----bw----->>n2
        """
        assert bw > 0, Exception('bw is not bigger than 0')
        assert unit in ['kbps', 'mbps'], Exception(
            unit + ' is not in ["kbps", "mbps"]')
        self.virtualLinkNumber += 1
        n1.link_to(n2.name, str(bw) + unit, n2.ip, n2.hostPort)

    def save_yml(self, taskID: int):
        """
        save the deployment of emulated nodes as yml files.
        """
        for cs in self.task[taskID].emulator.values():
            cs.save_yml(self.dirName, taskID)
    
    def save_node_info(self, taskID: int):
        """
        save the node's information as json file.
        """
        # 修改为通过taskID进行保存
        emulator = {}
        e_node = {}
        p_node = {}
        for e in self.task[taskID].emulator.values():
            emulator[e.nameW] = {'ip': e.ipW}
            for en in e.eNode.values():
                if en.tid == taskID:
                    e_node[en.name] = {'ip': en.ip, 'port': str(en.hostPort), 'emulator': e.nameW}
        for pn in self.task[taskID].pNode.values():
            p_node[pn.name] = {'ip': pn.ip, 'port': str(pn.hostPort)}
        file_name = (os.path.join(self.dirName, 'node_info_'+ str(taskID) +'.json'))
        data = {'emulator': emulator, 'emulated_node': e_node, 'physical_node': p_node}
        with open(file_name, 'w') as f:
            f.writelines(json.dumps(data, indent=2))

    def send_tc(self):
        self.__set_emulated_tc_listener()
        if self.virtualLinkNumber > 0:
            # send the tc settings to emulators.
            self.__send_emulated_tc()
            # send the tc settings to physical nodes.
            self.__send_physical_tc()
        else:
            print('tc finish')

    def __set_emulated_tc_listener(self):
        """
        listen message from worker/agent.py, deploy_emulated_tc ().
        it will save the result of deploying emulated tc settings.
        """

        @self.flask.route('/emulated/tc', methods=['POST'])
        def route_emulated_tc():
            data: Dict = json.loads(request.form['data'])
            for name, ret in data.items():
                if 'msg' in ret:
                    print('emulated node ' + name + ' tc failed, err:')
                    print(ret['msg'])
                elif 'number' in ret:
                    print('emulated node ' + name + ' tc succeed')
                    with self.lock:
                        self.deployedCount += int(ret['number'])
                        if self.deployedCount == self.virtualLinkNumber:
                            print('tc finish')
            return ''

    def __send_emulated_tc(self, taskID: int):
        """
        send the tc settings to emulators.
        this request can be received by worker/agent.py, route_emulated_tc ().
        """
        for emulator in self.task[taskID].emulator.values():
            data = {}
            # collect tc settings of each emulated node in this emulator.
            for en in emulator.eNode.values():
                data[en.name] = {
                    'NET_NODE_NIC': en.nic,
                    'NET_NODE_TC': en.tc,
                    'NET_NODE_TC_IP': en.tcIP,
                    'NET_NODE_TC_PORT': en.tcPort
                }
            # the emulator will deploy all tc settings of its emulated nodes.
            print('send_emulated_tc: send to ' + emulator.nameW)
            send_data('POST', '/emulated/tc', emulator.ipW, self.agentPort,
                      data={'data': json.dumps(data)})

    def __send_physical_tc(self):
        """
        send the tc settings to physical nodes.
        this request can be received by worker/agent.py, route_physical_tc ().
        """
        for pn in self.pNode.values():
            if not pn.tc:
                print('physical node ' + pn.name + ' tc succeed')
                continue
            data = {
                'NET_NODE_NIC': pn.nic,
                'NET_NODE_TC': pn.tc,
                'NET_NODE_TC_IP': pn.tcIP,
                'NET_NODE_TC_PORT': pn.tcPort
            }
            print('physical_tc_update: send to ' + pn.name)
            res = send_data('POST', '/physical/tc', pn.ip, self.agentPort,
                            data={'data': json.dumps(data)})
            if res == '':
                print('physical node ' + pn.name + ' tc succeed')
                with self.lock:
                    self.deployedCount += len(pn.tc)
                    if self.deployedCount == self.virtualLinkNumber:
                        print('tc finish')
            else:
                print('physical node ' + pn.name + ' tc failed, err:')
                print(res)

    def launch_all_emulated(self, taskID: int):
        """
        send the yml files to emulators to launch all emulated node and the dml application.
        this request can be received by worker/agent.py, route_emulated_launch ().
        """
        tasks = []
        for s in self.task[taskID].emulator.values():
            if s.eNode:
                tasks.append(self.executor.submit(self.__launch_emulated, s, self.dirName))
        os.wait(tasks, return_when=ALL_COMPLETED)

    def __launch_emulated(self, emulator: Emulator, taskID: int,path: str):
        with open(os.path.join(path, emulator.nameW + '_' + str(taskID) + '.yml'), 'r') as f:
            send_data('POST', '/emulated/launch', emulator.ipW, self.agentPort, files={'yml': f})

    def __build_emulated_env(self, tag: str, path1: str, path2: str):
        """
        send the Dockerfile and pip requirements.txt to emulators to build the execution environment.
        this request can be received by worker/agent.py, route_emulated_build ().
        @param tag: docker image name:version.
        @param path1: path of Dockerfile.
        @param path2: path of pip requirements.txt.
        @return:
        """
        tasks = [self.executor.submit(self.__build_emulated_env_helper, e, tag, path1, path2)
                 for e in self.emulator.values()]
        os.wait(tasks, return_when=ALL_COMPLETED)

    def __build_emulated_env_helper(self, emulator: Emulator, tag: str, path1: str, path2: str):
        with open(path1, 'r') as f1, open(path2, 'r') as f2:
            print('build_emulated_env: send to ' + emulator.nameW)
            res = send_data('POST', '/emulated/build', emulator.ipW, self.agentPort,
                            data={'tag': tag}, files={'Dockerfile': f1, 'dml_req': f2})
            if res == '1':
                print(emulator.nameW + ' build succeed')

    # 好像大概初步改完了
    def deploy_task(self, taskID: int, allocation: Dict, build_emulated_env: bool = False):
        """
        启动相应的容器
        """
        try:
            # 挂载
            nfsApp = self.nfs['dml_app']
            nfsDataset = self.nfs['dataset']

            # 动态加载用户的TaskManager子类
            manager_file_path = os.path.join(self.dirName, 'task_manager', str(taskID), 'task_manager.py')
            manager_class = load_task_manager_class(manager_file_path)
            
            # 添加节点
            # TODO 要动态获取类，创造实例
            task = Task(taskID, self.dirName, taskManager=manager_class)
            self.task[taskID] = task
            added_emulators = set()

            for node_name, node_info in allocation.items():
                emu = self.emulator[node_info['emulator']]
                # 如果这个emulator还没有添加到task中，就添加它
                if emu.nameW not in added_emulators:
                    task.add_emulator(emu)
                    added_emulators.add(emu.nameW)

                en = self.add_emulated_node (node_name, taskID, '/home/qianguo/worker/dml_app/'+str(taskID),
                    ['python3', 'gl_peer.py'], 'task'+str(taskID)+':v1.0', cpu=node_info['cpu'], ram=node_info['ram'], unit='G', emulator=emu)
                task.add_emulator_node(en)
                en.mount_local_path ('./dml_file', '/home/qianguo/worker/dml_file')
                en.mount_nfs (nfsApp, '/home/qianguo/worker/dml_app')
                en.mount_nfs (nfsDataset, '/home/qianguo/worker/dataset')
                    
            if build_emulated_env:
                path_dockerfile = os.path.join(self.dirName, 'dml_app/'+ str(taskID) +'/Dockerfile')
                path_req = os.path.join(self.dirName, 'dml_app/'+ str(taskID) +'/dml_req.txt')
                self.__build_emulated_env('task'+str(taskID)+':v1.0', path_dockerfile, path_req)
            # 解析links
            links_json = read_json (os.path.join (dirName, "task_links", str(taskID),'links.json'))
            self.load_link (taskID, links_json)

            # 保存信息
            self.save_yml(taskID) # 保存yml文件到controller
            self.save_node_info(taskID) # 保存节点信息到testbed
            # 修改
            self.send_tc(taskID) # 将tc信息发送给worker，没有的添加，有的更新
            self.launch_all_emulated(taskID, dirName)
            return True
        
        except Exception as e:
            print(f"部署任务 {taskID} 失败: {str(e)}")
            return False

def load_task_manager_class(file_path: str) -> Type[TaskManager]:
    """动态加载用户的TaskManager子类"""
    try:
        # 获取模块名称
        module_name = file_path.split('/')[-1].replace('.py', '')
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 查找继承自taskManager的类
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, TaskManager) and 
                obj != TaskManager):
                return obj
                
        raise ValueError("未找到taskManager的子类")
    except Exception as e:
        print(f"加载用户管理器类失败: {str(e)}")
        raise