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
    def __init__(self, ip: str, base_host_port: int, dir_name: str, manager_class: Type[Manager]):
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
        self.manager = manager_class(self)
        self.deployedCount: int = 0
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor()
        
        # scheduler
        self.scheduler = Scheduler(self)

        # 添加两个队列
        self.pending_tasks = Queue()  # 待调度任务队列
        self.scheduled_tasks = Queue()  # 已调度任务队列
        
        # 启动调度循环
        self.stop_scheduling = False
        self.schedule_thread = threading.Thread(target=self._schedule_loop)
        self.schedule_thread.daemon = True
        self.schedule_thread.start()
    
    def next_task_id(self):
        """
        获取下一个任务id
        """
        self.currTID += 1
        return self.currTID
       
    def _schedule_loop(self):
        """持续检查待调度队列并进行调度"""
        while not self.stop_scheduling:
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
            time.sleep(1)  # 避免空转消耗CPU
    
    def add_pending_task(self, task_id: int):
        """添加待调度任务"""
        self.pending_tasks.put(task_id)
    
    def get_scheduled_task(self) -> tuple:
        """获取已调度的任务"""
        if not self.scheduled_tasks.empty():
            return self.scheduled_tasks.get()
        return None

    def shutdown(self):
        """关闭调度循环"""
        self.stop_scheduling = True
        if self.schedule_thread.is_alive():
            self.schedule_thread.join()