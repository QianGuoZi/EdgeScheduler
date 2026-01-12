import abc
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import os
from queue import Queue
import threading
import time
from typing import Dict, List, Type
import inspect
import importlib.util
import subprocess as sp
from .taskManger import TaskManager

from flask import Flask, json, request
from .link import VirtualLink
from .nfs import Nfs
from .node import BackgroundLoadNode, EmulatedNode, Emulator, Node, PhysicalNode
from .manager import Manager
from .scheduler import Scheduler
from .task import Task
from .utils import read_json, send_data


dirName = '/home/qianguo/Edge-Scheduler/Controller'
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

        self._task_id_counter = 1  # 添加任务ID计数器
        self._task_id_lock = threading.Lock()  # 添加锁以保证线程安全


        self.flask = Flask(__name__)
        self.ip: str = ip
        self.port: int = 3333  # DO NOT change this port number.
        self.agentPort: int = 3333  # DO NOT change this port number.
        self.nodePort: int = 4444  # DO NOT change this port number.
        # emulated node maps dml port to emulator's host port starting from $(base_host_port).
        self.hostPort: int = base_host_port
        self.taskPort: int = 6000
        self.address: str = self.ip + ':' + str(self.port)
        self.dirName: str = dir_name

        self.nfs: Dict[str, Nfs] = {}  # nfs tag to nfs object.
        self.pNode: Dict[str, PhysicalNode] = {}  # physical node's name to physical node object.
        self.emulator: Dict[str, Emulator] = {}  # emulator's name to emulator object.
        self.eNode: Dict[str, EmulatedNode] = {}  # emulated node's name to emulated node object.
        self.vLink: Dict[int, VirtualLink] = {}  # virtual link ID to virtual link object.
        self.virtualLinkNumber: int = 0

        self.emulatorLink = {}  # 存储emulator之间的带宽信息
        self.emulatorLinkPreMap = {} 

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
        self.scheduler = scheduler(self)

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
        # self.currTID += 1
        # return self.currTID
        with self._task_id_lock:
            task_id = self._task_id_counter
            self._task_id_counter += 1
            return task_id

    def add_emulator_bw(self, emulator1: str, emulator2: str, bw: int):
        """添加两个节点间的带宽信息"""
        if emulator1 not in self.emulatorLink:
            self.emulatorLink[emulator1] = {}
        # if emulator2 not in self.emulatorLink:
        #     self.emulatorLink[emulator2] = {}
            
        self.emulatorLink[emulator1][emulator2] = bw
        # self.emulatorLink[emulator2][emulator1] = bw  # 对称存储
        
    def get_bw(self, emulator1: str, emulator2: str) -> int:
        """获取两个节点间剩余的带宽"""
        try:
            # 检查 emulator1 是否存在
            if emulator1 not in self.emulatorLink:
                print(f"emulator1 {emulator1} 不在 emulatorLink 中")
                return 0
                
            # 检查 emulator2 是否存在
            if emulator2 not in self.emulatorLink[emulator1]:
                print(f"emulator2 {emulator2} 不在 emulatorLink[{emulator1}] 中")
                return 0
                
            # 检查预分配映射
            if emulator1 not in self.emulatorLinkPreMap:
                print(f"emulator1 {emulator1} 不在 emulatorLinkPreMap 中")
                return self.emulatorLink[emulator1][emulator2]
                
            if emulator2 not in self.emulatorLinkPreMap[emulator1]:
                print(f"emulator2 {emulator2} 不在 emulatorLinkPreMap[{emulator1}] 中")
                return self.emulatorLink[emulator1][emulator2]
                
            # 计算剩余带宽
            total = self.emulatorLink[emulator1][emulator2]
            used = self.emulatorLinkPreMap[emulator1][emulator2]
            remaining = total - used
            
            print(f"节点 {emulator1} -> {emulator2} 的带宽情况:")
            print(f"总带宽: {total}")
            print(f"已用带宽: {used}")
            print(f"剩余带宽: {remaining}")
            
            return remaining
            
        except KeyError as e:
            print(f"KeyError: {str(e)}")
            print(f"emulatorLink: {self.emulatorLink}")
            print(f"emulatorLinkPreMap: {self.emulatorLinkPreMap}")
            return 0
        except Exception as e:
            print(f"计算带宽时出错: {str(e)}")
            return 0
    
    def add_emulator_bw_pre_map(self, emulator1: str, emulator2: str, bw: int):
        """添加两个节点间的带宽预分配信息"""
        if emulator1 not in self.emulatorLinkPreMap:
            self.emulatorLinkPreMap[emulator1] = {}
        # if emulator2 not in self.emulatorLinkPreMap:
        #     self.emulatorLinkPreMap[emulator2] = {}
            
        self.emulatorLinkPreMap[emulator1][emulator2] = bw
        # self.emulatorLinkPreMap[emulator2][emulator1] = bw
    
    def iter_bandwidth(self):
        """遍历所有节点间的带宽
        
        Yields:
            tuple: (emulator1, emulator2, bandwidth, used_bandwidth)
        """
        for emu1 in self.emulatorLink:
            for emu2, bw in self.emulatorLink[emu1].items():
                used_bw = (self.emulatorLinkPreMap.get(emu1, {}).get(emu2, 0) 
                        if emu1 in self.emulatorLinkPreMap else 0)
                yield emu1, emu2, bw, used_bw

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
                    task_info = self.pending_tasks.get()
                    # 兼容旧格式：如果只传入了task_id，则使用默认方法
                    if isinstance(task_info, tuple):
                        task_id, method = task_info
                    else:
                        task_id = task_info
                        method = "ppo"
                    try:
                        # 调用scheduler进行调度
                        print(f"开始调度任务: {task_id}, 使用调度方法: {method}")
                        # 使用事件来等待调度完成
                        scheduling_event = threading.Event()
                        
                        def schedule_task():
                            nonlocal allocation
                            try:
                                allocation = self.scheduler.resource_schedule(task_id, method=method)
                            finally:
                                scheduling_event.set()
                        
                        allocation = None
                        scheduling_thread = threading.Thread(target=schedule_task)
                        scheduling_thread.start()
                        
                        # 等待调度完成或超时
                        if scheduling_event.wait(timeout=30):  # 30秒超时
                            if allocation:
                                print(f"任务 {task_id} 调度成功")
                                self.scheduled_tasks.put((task_id, allocation))
                            else:
                                print(f"任务 {task_id} 调度失败: 未获得分配方案")
                        else:
                            print(f"任务 {task_id} 调度超时")
                            
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
                            # success = True
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

    def add_pending_task(self, task_id: int, method: str = "ppo"):
        """添加待调度任务
        
        Args:
            task_id: 任务ID
            method: 调度方法，默认为 "ppo"
        """
        print(f"添加待调度任务: {task_id}, 调度方法: {method}")
        self.pending_tasks.put((task_id, method))

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

    def add_emulator(self, name: str, ip: str, cpu: int, ram: int, unit: str = 'G', cpu_granularity: float = 0.04) -> Emulator:
        """添加模拟器
        
        Args:
            name: 模拟器名称
            ip: IP地址
            cpu: CPU核心数
            ram: 内存大小
            unit: 内存单位 ('M' 或 'G')，内部统一存储为GB
            cpu_granularity: CPU分配的最小粒度，默认0.04
        
        Returns:
            Emulator对象
        """
        assert name != '', Exception('name cannot be empty')
        assert name not in self.emulator, Exception(name + ' has been used')
        assert cpu > 0 and ram > 0, Exception('cpu or ram is not bigger than 0')
        assert unit in ['M', 'G'], Exception(unit + ' is not in ["M", "G"]')
        if unit == 'M':
            ram = ram // 1024  # 将MB转换为GB（整数）
        wid = self.__next_w_id()
        e = Emulator(wid, name, ip, cpu, ram, self.ip, cpu_granularity)
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
                          cpu: int, ram: int, unit: str = 'G', nic: str = 'eth0', emulator: Emulator = None) -> EmulatedNode:
        """添加模拟节点
        
        Args:
            name: 节点名称
            taskID: 任务ID
            working_dir: 工作目录
            cmd: 启动命令
            image: Docker镜像
            cpu: CPU份数（注意：这是份数，不是核心数）
            ram: 内存大小
            unit: 内存单位 ('M' 或 'G')，内部统一存储为GB
            nic: 网卡名称
            emulator: 模拟器对象
        
        Returns:
            EmulatedNode对象
        """
        assert name != '', Exception('name cannot be empty')
        assert name not in self.eNode, Exception(name + ' has been used')
        assert name not in self.pNode, Exception(name + ' has been used')
        assert cpu > 0 and ram > 0, Exception('cpu or ram is not bigger than 0')
        assert unit in ['M', 'G'], Exception(unit + ' is not in ["M", "G"]')
        if unit == 'M':
            ram = ram // 1024  # 将MB转换为GB（整数）

        if emulator:
            # check_resource会自动使用emulator的cpu_granularity进行检查
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

    def load_link(self, taskID: int, links_json: Dict):
        for name in links_json:
            nodeName = str(taskID) + '_' + name
            src = self.name_to_node(nodeName)
            for dest_json in links_json[name]:
                destName = str(taskID) + '_' + dest_json['dest']
                dest = self.name_to_node(destName)
                unit = dest_json['bw'][-4:]
                _bw = int(dest_json['bw'][:-4])
                # print(f"{destName}, {dest}")
                self.task[taskID].add_virtual_link(src, dest, _bw, unit)

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

    # def __add_virtual_link(self, n1: Node, n2: Node, bw: int, unit: str):
    #     """
    #     parameters will be passed to Linux Traffic Control.
    #     n1-----bw----->>n2
    #     """
    #     assert bw > 0, Exception('bw is not bigger than 0')
    #     assert unit in ['kbps', 'mbps'], Exception(
    #         unit + ' is not in ["kbps", "mbps"]')
    #     self.virtualLinkNumber += 1
    #     n1.link_to(n2.name, str(bw) + unit, n2.ip, n2.hostPort)

    def save_yml(self, taskID: int):
        """
        save the deployment of emulated nodes as yml files.
        """
        for cs in self.task[taskID].emulator.values():
            cs.save_yml(self.dirName, taskID)
    
    def save_yml_with_cpus(self, taskID: int, cpu_granularity: float = 0.04):
        """使用cpus参数保存yml文件，支持更细粒度的CPU分配
        
        Args:
            taskID: 任务ID
            cpu_granularity: CPU分配的最小粒度，默认0.04（即每份CPU占用0.04个核心）
        """
        for cs in self.task[taskID].emulator.values():
            cs.save_yml_with_cpus(self.dirName, taskID, cpu_granularity)
    
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
        # file_name = (os.path.join(self.dirName, '/node_info/', 'node_info_'+ str(taskID) +'.json'))
        node_info_dir = os.path.join(self.dirName, 'node_info')
        os.makedirs(node_info_dir, exist_ok=True)
        file_name = os.path.join(node_info_dir, f'node_info_{taskID}.json')
        data = {'emulator': emulator, 'emulated_node': e_node, 'physical_node': p_node}
        with open(file_name, 'w') as f:
            f.writelines(json.dumps(data, indent=2))

    def send_tc(self, taskID: int):
        if self.task[taskID].virtualLinkNumber > 0:
            # send the tc settings to emulators.
            self.__send_emulated_tc(taskID)
            # send the tc settings to physical nodes.
            self.__send_physical_tc(taskID)
        else:
            print('tc finish')

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
                      data={'taskID': taskID,'data': json.dumps(data)})

    def __send_physical_tc(self, taskID: int):
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
                    self.task[taskID].deployedCount += len(pn.tc)
                    if self.task[taskID].deployedCount == self.task[taskID].virtualLinkNumber:
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
                print('launch_all_emulated: send to ' + s.nameW)
                tasks.append(self.executor.submit(self.__launch_emulated, s, taskID, self.dirName))
        wait(tasks, return_when=ALL_COMPLETED)

    # def __launch_emulated(self, emulator: Emulator, taskID: int, path: str):
    #     path = os.path.join(path, emulator.nameW + '_' + str(taskID) + '.yml')
    #     print(f'launch_all_emulated: send to {emulator.nameW}, path: {path}, ip:{emulator.ipW}, port:{self.agentPort}')
    #     with open(os.path.join(path, emulator.nameW + '_' + str(taskID) + '.yml'), 'r') as f:
    #         send_data('POST', '/emulated/launch', emulator.ipW, self.agentPort, files={'yml': f})
    def __launch_emulated(self, emulator: Emulator, taskID: int, path: str):
        """启动单个模拟器的节点"""
        try:
            # 构造yml文件路径
            yml_filename = f"{emulator.nameW}_{taskID}.yml"
            yml_path = os.path.join(path, yml_filename)
            
            print(f'launch_all_emulated: 准备发送到 {emulator.nameW}')
            print(f'文件路径: {yml_path}')
            
            # 检查文件是否存在
            if not os.path.exists(yml_path):
                raise FileNotFoundError(f"找不到YML文件: {yml_path}")
                
            # 打开并发送文件
            with open(yml_path, 'rb') as f:
                print(f'正在发送请求到 {emulator.ipW}:{self.agentPort}')
                print(f'taskID: {taskID}, type: {type(taskID)}')
                print(f'发送数据: taskID={str(taskID)}')
                filename = os.path.basename(yml_path)
                response = send_data('POST',
                    '/emulated/launch',
                    emulator.ipW,
                    self.agentPort,
                    data={'taskID': str(taskID)},
                    files={'yml': (filename, f, 'text/yaml')}
                )
                print(f'请求响应: {response}')
                
        except FileNotFoundError as e:
            print(f"文件错误: {str(e)}")
            raise
        except Exception as e:
            print(f"发送请求失败: {str(e)}")
            raise

    def __build_emulated_env(self, taskID: int, tag: str, path1: str, path2: str):
        """
        send the Dockerfile and pip requirements.txt to emulators to build the execution environment.
        this request can be received by worker/agent.py, route_emulated_build ().
        @param tag: docker image name:version.
        @param path1: path of Dockerfile.
        @param path2: path of pip requirements.txt.
        @return:
        """
        tasks = [self.executor.submit(self.__build_emulated_env_helper, e, tag, path1, path2, taskID)
                 for e in self.emulator.values()]
        wait(tasks, return_when=ALL_COMPLETED)

    def __build_emulated_env_helper(self, emulator: Emulator, tag: str, path1: str, path2: str, taskID: int):
        with open(path1, 'r') as f1, open(path2, 'r') as f2:
            print('build_emulated_env: send to ' + emulator.nameW)
            res = send_data('POST', '/emulated/build', emulator.ipW, self.agentPort,
                            data={'tag': tag, 'taskID': taskID}, files={'Dockerfile': f1, 'dml_req': f2})
            if res == '1':
                print(emulator.nameW + ' build succeed')

    def export_nfs(self):
        """
        clear all exported path and then export the defined path through nfs.
        """
        cmd = 'sudo exportfs -au'
        sp.Popen(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait()
        for nfs in self.nfs.values():
            subnet = nfs.subnet
            path = nfs.path
            # export the path.
            cmd = 'sudo exportfs ' + subnet + ':' + path
            sp.Popen(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait()
            # check result.
            cmd = 'sudo exportfs -v'
            p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
            msg = p.communicate()[0].decode()
            assert path in msg and subnet in msg, Exception(
                'share ' + path + ' to ' + subnet + ' failed')

    def __send_physical_tc(self, taskID: int):
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
                    self.task[taskID].deployedCount += len(pn.tc)
                    if self.task[taskID].deployedCount == self.task[taskID].virtualLinkNumber:
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
                print('launch_all_emulated: send to ' + s.nameW)
                tasks.append(self.executor.submit(self.__launch_emulated, s, taskID, self.dirName))
        wait(tasks, return_when=ALL_COMPLETED)

    # def __launch_emulated(self, emulator: Emulator, taskID: int, path: str):
    #     path = os.path.join(path, emulator.nameW + '_' + str(taskID) + '.yml')
    #     print(f'launch_all_emulated: send to {emulator.nameW}, path: {path}, ip:{emulator.ipW}, port:{self.agentPort}')
    #     with open(os.path.join(path, emulator.nameW + '_' + str(taskID) + '.yml'), 'r') as f:
    #         send_data('POST', '/emulated/launch', emulator.ipW, self.agentPort, files={'yml': f})
    def __launch_emulated(self, emulator: Emulator, taskID: int, path: str):
        """启动单个模拟器的节点"""
        try:
            # 构造yml文件路径
            yml_filename = f"{emulator.nameW}_{taskID}.yml"
            yml_path = os.path.join(path, yml_filename)
            
            print(f'launch_all_emulated: 准备发送到 {emulator.nameW}')
            print(f'文件路径: {yml_path}')
            
            # 检查文件是否存在
            if not os.path.exists(yml_path):
                raise FileNotFoundError(f"找不到YML文件: {yml_path}")
                
            # 打开并发送文件
            with open(yml_path, 'rb') as f:
                print(f'正在发送请求到 {emulator.ipW}:{self.agentPort}')
                print(f'taskID: {taskID}, type: {type(taskID)}')
                print(f'发送数据: taskID={str(taskID)}')
                filename = os.path.basename(yml_path)
                response = send_data('POST',
                    '/emulated/launch',
                    emulator.ipW,
                    self.agentPort,
                    data={'taskID': str(taskID)},
                    files={'yml': (filename, f, 'text/yaml')}
                )
                print(f'请求响应: {response}')
                
        except FileNotFoundError as e:
            print(f"文件错误: {str(e)}")
            raise
        except Exception as e:
            print(f"发送请求失败: {str(e)}")
            raise

    def __build_emulated_env(self, taskID: int, tag: str, path1: str, path2: str):
        """
        send the Dockerfile and pip requirements.txt to emulators to build the execution environment.
        this request can be received by worker/agent.py, route_emulated_build ().
        @param tag: docker image name:version.
        @param path1: path of Dockerfile.
        @param path2: path of pip requirements.txt.
        @return:
        """
        tasks = [self.executor.submit(self.__build_emulated_env_helper, e, tag, path1, path2, taskID)
                 for e in self.emulator.values()]
        wait(tasks, return_when=ALL_COMPLETED)

    def __build_emulated_env_helper(self, emulator: Emulator, tag: str, path1: str, path2: str, taskID: int):
        with open(path1, 'r') as f1, open(path2, 'r') as f2:
            print('build_emulated_env: send to ' + emulator.nameW)
            res = send_data('POST', '/emulated/build', emulator.ipW, self.agentPort,
                            data={'tag': tag, 'taskID': taskID}, files={'Dockerfile': f1, 'dml_req': f2})
            if res == '1':
                print(emulator.nameW + ' build succeed')

    def export_nfs(self):
        """
        clear all exported path and then export the defined path through nfs.
        """
        cmd = 'sudo exportfs -au'
        sp.Popen(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait()
        for nfs in self.nfs.values():
            subnet = nfs.subnet
            path = nfs.path
            # export the path.
            cmd = 'sudo exportfs ' + subnet + ':' + path
            sp.Popen(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT).wait()
            # check result.
            cmd = 'sudo exportfs -v'
            p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
            msg = p.communicate()[0].decode()
            assert path in msg and subnet in msg, Exception(
                'share ' + path + ' to ' + subnet + ' failed')

    def load_background_workloads(self, config_path: str = None) -> Dict[str, BackgroundLoadNode]:
        """
        从配置文件中读取各emulator的背景负载配置，并添加到对应的emulator中
        
        Args:
            config_path: 配置文件路径，默认为workload_config.json
            
        Returns:
            Dict[str, BackgroundLoadNode]: emulator名称到背景负载节点的映射
        """
        if config_path is None:
            config_path = os.path.join(self.dirName, 'workload_config.json')
        
        print(f"正在加载背景负载配置: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"配置文件不存在: {config_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"配置文件格式错误: {e}")
            return {}
        
        workloads = config.get('workloads', {})
        bg_nodes = {}
        
        for emulator_name, workload_info in workloads.items():
            # 检查是否启用
            if not workload_info.get('enabled', True):
                print(f"跳过未启用的负载: {emulator_name}")
                continue
            
            # 检查emulator是否存在
            if emulator_name not in self.emulator:
                print(f"警告: emulator {emulator_name} 不存在，跳过")
                continue
            
            emulator = self.emulator[emulator_name]
            
            # 获取负载配置
            cpu = workload_info.get('cpu', 1)
            ram = workload_info.get('ram', 1)
            unit = workload_info.get('unit', 'G')
            image = workload_info.get('image', 'stress:latest')
            
            # 内存单位转换
            if unit == 'M':
                ram = ram // 1024
            
            # 创建背景负载节点
            bg_node_name = f"{emulator_name}_bgload"
            bg_node = BackgroundLoadNode(
                name=bg_node_name,
                emulator_name=emulator_name,
                cpu=cpu,
                ram=ram,
                image=image
            )
            
            try:
                # 添加到emulator
                emulator.add_background_load(bg_node)
                bg_nodes[emulator_name] = bg_node
                print(f"成功添加背景负载到 {emulator_name}: CPU={cpu}, RAM={ram}GB")
            except Exception as e:
                print(f"添加背景负载到 {emulator_name} 失败: {e}")
        
        # 加载网络拓扑带宽信息
        links = config.get('links', {})
        if links:
            print(f"正在加载网络拓扑带宽配置...")
            for source_emulator, connections in links.items():
                # 检查源emulator是否存在
                if source_emulator not in self.emulator:
                    print(f"警告: emulator {source_emulator} 不存在，跳过其网络拓扑配置")
                    continue
                
                for connection in connections:
                    dest_emulator = connection.get('dest')
                    bw_str = connection.get('bw', '0mbps')
                    
                    # 检查目标emulator是否存在
                    if dest_emulator not in self.emulator:
                        print(f"警告: 目标emulator {dest_emulator} 不存在，跳过连接 {source_emulator} -> {dest_emulator}")
                        continue
                    
                    # 解析带宽值（支持mbps格式）
                    try:
                        bw = int(bw_str.replace('mbps', '').replace('Mbps', '').replace('MBPS', ''))
                    except (ValueError, AttributeError):
                        print(f"警告: 无法解析带宽值 '{bw_str}'，跳过连接 {source_emulator} -> {dest_emulator}")
                        continue
                    
                    # 更新带宽使用量（不创建tc链接，只更新已用带宽记录）
                    try:
                        self.add_emulator_bw_pre_map(source_emulator, dest_emulator, bw)
                        print(f"成功更新带宽使用量: {source_emulator} -> {dest_emulator} = {bw}mbps")
                    except Exception as e:
                        print(f"更新带宽使用量失败 {source_emulator} -> {dest_emulator}: {e}")
        
        return bg_nodes

    def save_background_load_ymls(self) -> Dict[str, str]:
        """
        为所有有背景负载的emulator保存yml文件
        
        Returns:
            Dict[str, str]: emulator名称到yml文件路径的映射
        """
        yml_files = {}
        
        for emulator_name, emulator in self.emulator.items():
            if emulator.bgLoadNode:
                yml_path = emulator.save_background_load_yml(self.dirName)
                if yml_path:
                    yml_files[emulator_name] = yml_path
                    print(f"保存背景负载yml文件: {yml_path}")
        
        return yml_files

    def launch_background_loads(self) -> bool:
        """
        批量启动所有emulator上的背景负载容器
        
        Returns:
            bool: 是否全部启动成功
        """
        # 先保存yml文件
        yml_files = self.save_background_load_ymls()
        
        if not yml_files:
            print("没有需要启动的背景负载")
            return True
        
        # 使用线程池并行启动
        tasks = []
        for emulator_name, yml_path in yml_files.items():
            emulator = self.emulator[emulator_name]
            print(f"准备启动 {emulator_name} 上的背景负载")
            tasks.append(self.executor.submit(self.__launch_background_load, emulator, yml_path))
        
        # 等待所有任务完成
        wait(tasks, return_when=ALL_COMPLETED)
        
        # 更新背景负载状态
        all_success = True
        for emulator_name, emulator in self.emulator.items():
            for bg_node in emulator.bgLoadNode.values():
                bg_node.is_running = True
        
        print("所有背景负载启动完成")
        return all_success

    def __launch_background_load(self, emulator: Emulator, yml_path: str):
        """
        启动单个emulator上的背景负载容器
        
        Args:
            emulator: Emulator对象
            yml_path: yml文件路径
        """
        try:
            print(f"正在发送背景负载yml到 {emulator.nameW}")
            
            if not os.path.exists(yml_path):
                raise FileNotFoundError(f"找不到YML文件: {yml_path}")
            
            with open(yml_path, 'rb') as f:
                filename = os.path.basename(yml_path)
                response = send_data('POST',
                    '/bgload/launch',
                    emulator.ipW,
                    self.agentPort,
                    files={'yml': (filename, f, 'text/yaml')}
                )
                print(f"背景负载启动响应 ({emulator.nameW}): {response}")
                
        except Exception as e:
            print(f"启动背景负载失败 ({emulator.nameW}): {e}")
            raise

    def stop_background_loads(self) -> bool:
        """
        批量停止所有emulator上的背景负载容器
        
        Returns:
            bool: 是否全部停止成功
        """
        tasks = []
        for emulator_name, emulator in self.emulator.items():
            if emulator.bgLoadNode:
                print(f"准备停止 {emulator_name} 上的背景负载")
                tasks.append(self.executor.submit(self.__stop_background_load, emulator))
        
        if not tasks:
            print("没有需要停止的背景负载")
            return True
        
        # 等待所有任务完成
        wait(tasks, return_when=ALL_COMPLETED)
        
        # 更新背景负载状态
        for emulator_name, emulator in self.emulator.items():
            for bg_node in emulator.bgLoadNode.values():
                bg_node.is_running = False
        
        print("所有背景负载已停止")
        return True

    def __stop_background_load(self, emulator: Emulator):
        """
        停止单个emulator上的背景负载容器
        
        Args:
            emulator: Emulator对象
        """
        try:
            print(f"正在停止 {emulator.nameW} 上的背景负载")
            response = send_data('GET',
                '/bgload/stop',
                emulator.ipW,
                self.agentPort
            )
            print(f"背景负载停止响应 ({emulator.nameW}): {response}")
        except Exception as e:
            print(f"停止背景负载失败 ({emulator.nameW}): {e}")
            raise

    def clear_background_loads(self) -> bool:
        """
        批量清理所有emulator上的背景负载容器（停止并删除）
        
        Returns:
            bool: 是否全部清理成功
        """
        tasks = []
        for emulator_name, emulator in self.emulator.items():
            if emulator.bgLoadNode:
                print(f"准备清理 {emulator_name} 上的背景负载")
                tasks.append(self.executor.submit(self.__clear_background_load, emulator))
        
        if not tasks:
            print("没有需要清理的背景负载")
            return True
        
        # 等待所有任务完成
        wait(tasks, return_when=ALL_COMPLETED)
        
        # 清理背景负载信息
        for emulator_name, emulator in self.emulator.items():
            # 重置背景负载资源占用
            emulator.bgCpuUsed = 0.0
            emulator.bgRamUsed = 0
            emulator.bgLoadNode.clear()
        
        print("所有背景负载已清理")
        return True

    def __clear_background_load(self, emulator: Emulator):
        """
        清理单个emulator上的背景负载容器
        
        Args:
            emulator: Emulator对象
        """
        try:
            print(f"正在清理 {emulator.nameW} 上的背景负载")
            response = send_data('GET',
                '/bgload/clear',
                emulator.ipW,
                self.agentPort
            )
            print(f"背景负载清理响应 ({emulator.nameW}): {response}")
        except Exception as e:
            print(f"清理背景负载失败 ({emulator.nameW}): {e}")
    
    def __creat_log(self, taskID: int) -> bool:
        """创建任务日志文件夹
        
        Args:
            taskID: 任务ID
            
        Returns:
            bool: 创建是否成功
        """
        try:
            print(f'开始创建日志文件夹: {taskID}')
            task = self.task.get(taskID)
            if not task:
                print(f'找不到任务 {taskID}')
                return False
            url = f'http://{self.ip}:{task.taskPort}/task/{taskID}/startTask'
            print(f'发送请求到: {url}')
            res = send_data(
                'POST', 
                f'/task/{taskID}/startTask',
                self.ip,
                self.taskPort + taskID,
                data={'taskID': taskID}
            )
            
            if res == '':
                print('日志创建成功')
                return True
            else:
                print(f'日志创建失败: {res}')
                return False
                
        except Exception as e:
            print(f'创建日志时出错: {str(e)}')
            return False

    def deploy_task(self, taskID: int, allocation: Dict, build_emulated_env: bool = False, use_cpus: bool = True, cpu_granularity: float = 0.04):
        """
        启动相应的容器
        
        Args:
            taskID: 任务ID
            allocation: 资源分配方案
            build_emulated_env: 是否构建模拟环境
            use_cpus: 是否使用cpus参数进行CPU分配（True使用cpus，False使用cpuset）
            cpu_granularity: 当use_cpus=True时，CPU分配的最小粒度，默认0.04
        """
        try:
            # 挂载
            nfsApp = self.nfs['dml_app']
            nfsDataset = self.nfs['dataset']

            # 动态加载用户的TaskManager子类
            manager_file_path = os.path.join(self.dirName, 'task_manager', str(taskID), 'task_manager.py')
            manager_class = load_task_manager_class(manager_file_path)
            
            # 添加节点
            task = Task(taskID, self.dirName, manager_class=manager_class)
            self.task[taskID] = task
            if not task.wait_for_server():
                raise Exception("任务服务器启动失败")
            print(f"任务 {taskID} 服务器启动成功")
            
            # 提前创建日志目录（带时间戳）
            import time
            log_timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            log_folder = os.path.join(self.dirName, 'dml_file/log', str(taskID), log_timestamp)
            os.makedirs(log_folder, exist_ok=True)
            task.taskManager.logFileFolder = log_folder
            print(f"日志目录已创建: {log_folder}")

            added_emulators = set()

            # 添加节点信息
            for node_name, node_info in allocation.items():
                emu = self.emulator[node_info['emulator']]
                if emu.nameW not in added_emulators:
                    task.add_emulator(emu)
                    added_emulators.add(emu.nameW)
                print(f"添加任务 {taskID} 的节点 {node_name} 到模拟器 {emu.nameW}")
                en = self.add_emulated_node (node_name, taskID, '/home/qianguo/EdgeScheduler/Worker/dml_app/'+str(taskID),
                    ['python3', 'gl_peer.py'], 'task'+ '1' +':v1.0', cpu=node_info['cpu'], ram=node_info['ram'], unit='G', emulator=emu)
                task.add_emulator_node(en)
                # 添加日志目录时间戳环境变量
                en.add_var({
                    'NET_LOG_TIMESTAMP': log_timestamp
                })
                en.mount_local_path ('../dml_file', '/home/qianguo/EdgeScheduler/Worker/dml_file')
                en.mount_nfs (nfsApp, '/home/qianguo/EdgeScheduler/Worker/dml_app')
                en.mount_nfs (nfsDataset, '/home/qianguo/EdgeScheduler/Worker/dataset')

            # 添加链路信息
            with open(os.path.join(dirName, 'task_links', str(taskID),'links.json'), 'r') as file:
                            links_data = json.load(file)            
            for node, connections in links_data.items():
                node_name = str(taskID) + '_' + node
                for dest in connections:
                    dest_node = str(taskID) + '_' + dest['dest']
                    bw = int(dest['bw'].replace('mbps', ''))
                    node_emualtor = allocation[node_name]['emulator']
                    dest_emulator = allocation[dest_node]['emulator']
                    if node_emualtor != dest_emulator:
                        if self.get_bw(node_emualtor, dest_emulator) < bw:
                            raise Exception(f"节点 {node_emualtor} 和 {dest_emulator} 之间的带宽不足: {self.get_bw(node_emualtor, dest_emulator)} < {bw}")
                        else:
                            self.add_emulator_bw_pre_map(node_emualtor,dest_emulator, bw)

            if build_emulated_env:
                path_dockerfile = os.path.join(self.dirName, 'dml_app/'+ str(taskID) +'/Dockerfile')
                path_req = os.path.join(self.dirName, 'dml_app/'+ str(taskID) +'/dml_req.txt')
                self.__build_emulated_env(taskID,'task'+str(taskID)+':v1.0', path_dockerfile, path_req)
            # 解析links
            links_json = read_json (os.path.join (dirName, "task_links", str(taskID),'links.json'))
            self.load_link (taskID, links_json)

            # 保存信息
            task.taskManager.load_node_info() # 保存节点信息到task
            # 根据use_cpus参数选择保存yml的方式
            if use_cpus:
                self.save_yml_with_cpus(taskID, cpu_granularity) # 使用cpus参数保存yml文件
            else:
                self.save_yml(taskID) # 使用cpuset参数保存yml文件（原有方式）
            self.save_node_info(taskID) # 保存节点信息到testbed
            # 修改
            self.send_tc(taskID) # 将tc信息发送给worker，没有的添加，有的更新
            self.launch_all_emulated(taskID)
            # success = self.__creat_log(taskID)
            # if not success:
            #     raise Exception("创建日志失败")
            # print("日志创建完成") 
            return True
        
        except Exception as e:
            print(f"部署任务 {taskID} 失败: {str(e)}")
            return False
    
    def start(self):
        self.flask.run(host='0.0.0.0', port=self.port, threaded=True)

def load_task_manager_class(file_path: str) -> Type[TaskManager]:
    """动态加载用户的TaskManager子类"""
    try:
        print(f"开始加载文件: {file_path}")
        
        # 确保项目根目录在 Python 路径中
        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 动态加载模块
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        print(f"模块加载成功: {module_name}")
        
        # 直接尝试获取 GlManager 类
        if hasattr(module, 'GlManager'):
            manager_class = getattr(module, 'GlManager')
            try:
                # 修改判断逻辑
                if any(base.__name__ == 'TaskManager' for base in manager_class.__bases__):
                    print(f"找到TaskManager子类: GlManager")
                    return manager_class
            except Exception as e:
                print(f"检查GlManager继承关系时出错: {str(e)}")
        
        # 如果找不到 GlManager，遍历所有成员
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                try:
                    # 修改判断逻辑
                    if any(base.__name__ == 'TaskManager' for base in obj.__bases__):
                        print(f"找到TaskManager子类: {name}")
                        return obj
                except Exception as e:
                    print(f"检查类 {name} 继承关系时出错: {str(e)}")
                    
        raise ValueError(f"在{module_name}中未找到TaskManager的子类")
        
    except Exception as e:
        print(f"加载用户管理器类失败: {str(e)}")
        print(f"文件路径: {file_path}")
        print(f"sys.path: {sys.path}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        raise