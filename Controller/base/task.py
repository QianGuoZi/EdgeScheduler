from concurrent.futures import ThreadPoolExecutor
from logging import Manager
import threading
from typing import Dict, List, Type
from Controller.base.link import VirtualLink
from Controller.base.nfs import Nfs
from Controller.base.node import EmulatedNode, Emulator, PhysicalNode

from flask import Flask, request

class Task(object):
    """
    管理某个任务的生命过程
    """
    def __init__(self, ID: int, ip: str, base_host_port: int, dir_name: str, manager_class: Type[Manager]):
        self.flask = Flask(__name__)
        self.ID: int = ID
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
        #self.scheduler = scheduler(self)