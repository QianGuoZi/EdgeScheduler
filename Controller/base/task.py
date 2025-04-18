from concurrent.futures import ThreadPoolExecutor
from logging import Manager
import threading
from typing import Dict, List, Type
from .link import VirtualLink
from .nfs import Nfs
from .node import EmulatedNode, Emulator, PhysicalNode

from flask import Flask, request

from .taskManger import TaskManager

dirName = '/home/qianguo/Edge-Scheduler/Controller'
class Task(object):
    """
    管理某个任务的生命过程
    """
    def __init__(self, ID: int, dir_name: str, manager_class: Type[TaskManager]):
        self.flask = Flask(__name__)
        self.ID: int = ID
        self.dirName: str = dir_name
        self.agentPort: int = 3333  # DO NOT change this port number.
        self.taskPort = 5000 + ID

        self.url_prefix = f'/task/{ID}'

        self.nfs: Dict[str, Nfs] = {}  # nfs tag to nfs object.
        self.pNode: Dict[str, PhysicalNode] = {}  # physical node's name to physical node object.
        self.emulator: Dict[str, Emulator] = {}  # emulator's name to emulator object.
        self.eNode: Dict[str, EmulatedNode] = {}  # emulated node's name to emulated node object.
        self.vLink: Dict[int, VirtualLink] = {}  # virtual link ID to virtual link object.
        self.virtualLinkNumber: int = 0

        # for auto deployment.
        self.W: Dict[int, Dict] = {}  # worker ID to {name, cpu, MB of ram}.
        self.N: Dict[int, Dict] = {}  # node ID to {name, cpu, MB of ram}.
        self.RConnect: List[List[List[int]]]  # workers adjacency matrix.
        self.VConnect: List[List[List[int]]]  # nodes adjacency matrix.
        self.preMap: Dict[int, int] = {}  # node ID to worker ID.

        self.executor = ThreadPoolExecutor()

        # for default manager.
        self.taskManager = manager_class(self)

        self.start_server()
        self._wait_for_server()
    
    def add_emulator_node(self, en : EmulatedNode):
        """
        添加一个emulated node
        """
        self.eNode[en.name] = en
        self.N[en.id] = {'name': en.name, 'cpu': en.cpu, 'ram': en.ram}

    def add_emulator(self, e : Emulator):
        """
        添加一个emulator
        """
        self.emulator[e.nameW] = e
        self.W[e.idW] = {'name': e.nameW, 'cpu': e.cpu, 'ram': e.ram}

    def start_server(self):
        """在新线程中启动Flask服务器"""
        def run_flask():
            self.flask.run(host='0.0.0.0', port=self.taskPort)
            
        self.server_thread = threading.Thread(target=run_flask)
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def _wait_for_server(self, timeout=30):
            """等待服务器启动完成"""
            import time
            import requests
            from requests.exceptions import RequestException
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"http://localhost:{self.taskPort}/health")
                    if response.status_code == 200:
                        print(f"任务 {self.ID} 服务器启动成功")
                        return True
                except RequestException:
                    time.sleep(0.1)
                    continue
            
            raise TimeoutError(f"任务 {self.ID} 服务器启动超时")