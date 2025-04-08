from concurrent.futures import ThreadPoolExecutor
from logging import Manager
import threading
from typing import Dict, List, Type
from Controller.base.link import VirtualLink
from Controller.base.nfs import Nfs
from Controller.base.node import EmulatedNode, Emulator, PhysicalNode

from flask import Flask, request

from Controller.base.taskManger import taskManager

dirName = '/home/qianguo/controller/'
class Task(object):
    """
    管理某个任务的生命过程
    """
    def __init__(self, ID: int, dir_name: str, manager_class: Type[taskManager]):
        self.flask = Flask(__name__)
        self.ID: int = ID
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