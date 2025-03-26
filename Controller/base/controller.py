from concurrent.futures import ThreadPoolExecutor
from logging import Manager
from queue import Queue
import threading
import time
from typing import Dict, List, Type

from flask import Flask
from Controller.base.link import VirtualLink
from Controller.base.nfs import Nfs
from Controller.base.node import EmulatedNode, Emulator, PhysicalNode
from Controller.base.scheduler import Scheduler
from Controller.base.task import Task

class Controller(object):
    """
    任务接收器，负责和用户进行交互
    """
    def __init__(self, ip: str, base_host_port: int, dir_name: str, manager: Manager):
        self.currWID: int = 0  # build-in worker ID.
        self.currRID: int = 0  # build-in real link ID.
        self.currNID: int = 0  # build-in node ID.
        self.currVID: int = 0  # build-in virtual link ID.
        self.currTID: int = 0  # build-in task ID.


        self.flask = Flask(__name__)
        self.ip: str = ip
        self.port: int = 3333  # DO NOT change this port number.
        self.agentPort: int = 3333  # DO NOT change this port number.
        self.dmlPort: int = 4444  # DO NOT change this port number.
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
        self.manager = manager
        self.deployedCount: int = 0
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor()
        
        # scheduler
        self.scheduler = Scheduler(self)

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
                            # 通知manager进行部署
                            success = self.manager.deploy_task(task_id, allocation)
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


    # TODO：修改以下方法
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
    
    def add_emulated_node(self, name: str, working_dir: str, cmd: List[str], image: str,
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
        en = EmulatedNode(nid, name, nic, working_dir, cmd, self.dmlPort, self.hostPort, image, cpu, ram)

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
            'NET_AGENT_ADDRESS': emulator.ipW + ':' + str(self.agentPort)
        })
        self.preMap[en.id] = emulator.idW