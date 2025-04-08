import abc
import os
import threading
import time
from typing import Dict, List

from flask import request
from Controller.base.task import Task
from Controller.base.utils import send_data
from manager import Manager

class NodeInfo(object):
    def __init__(self, name: str, ip: str, port: int):
        self.name: str = name
        self.ip: str = ip
        self.port: int = port

class taskManager(metaclass=abc.ABCMeta):
    """
    负责任务的部署、资源调整、任务状态监控等
    """
    def __init__(self, task: Task):
        self.task = task
        self.eNode: Dict[str, NodeInfo] = {}
        self.pNode: Dict[str, NodeInfo] = {}
        self.nodeNumber: int = 0
        self.logFile: List[str] = []
        self.logFileFolder: str = ''
        self.lock = threading.RLock()
    
    def load_node_info(self):
        for name, en in self.task.eNode.items():
            self.eNode[name] = NodeInfo(name, en.ip, en.hostPort)
        for name, pn in self.task.pNode.items():
            self.pNode[name] = NodeInfo(name, pn.ip, pn.hostPort)
        self.nodeNumber = len(self.eNode) + len(self.pNode)

    def __load_default_route(self):
        @self.testbed.flask.route('/startTask', methods=['GET'])
        def route_start_task():
            """
            开始任务
            """
            taskId = request.args.get('taskId')
            if self.logFileFolder == '':
                self.logFileFolder = os.path.join(self.task.dirName, 'dml_file/log', taskId,
                                                  time.strftime('-%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
            msg = self.on_route_start(request)
            # return str explicitly is necessary.
            return str(msg)

        @self.testbed.flask.route('/finishTask', methods=['GET'])
        def route_finish_task():
            """
            when finished, ask node for log file.
            user need to implement self.on_route_finish () by extend this class.
            """
            all_finished = self.on_route_finish(request)
            if all_finished:
                print('training completed')
                os.makedirs(self.logFileFolder)
                for en in self.eNode.values():
                    send_data('GET', '/log', en.ip, en.port)
            return ''
        
        #TODO: 修改log方法
        @self.testbed.flask.route('/log', methods=['POST'])
        def route_log():
            """
            this function can listen log files from worker/worker_utils.py, send_log ().
            log files will be saved on ${self.logFileFolder}.
            when total_number files are received, it will parse these files into pictures
            and save them on ${self.logFileFolder}/png.
            user need to implement self.parse_log_file () by extend this class.
            """
            name = request.args.get('name')
            print('get ' + name + '\'s log')
            request.files.get('log').save(os.path.join(self.logFileFolder, name + '.log'))
            with self.lock:
                self.logFile.append(name + '.log')
                if len(self.logFile) == self.nodeNumber:
                    print('log files collection completed, saved on ' + self.logFileFolder)
                    full_path = os.path.join(self.logFileFolder, 'png/')
                    if not os.path.exists(full_path):
                        os.mkdir(full_path)
                    for filename in self.logFile:
                        self.parse_log_file(request, filename)
                    print('log files parsing completed, saved on ' + self.logFileFolder + '/png')
                    self.logFile.clear()
                    self.testbed.executor.submit(self.__after_log)
            return ''
    @abc.abstractmethod
    def on_route_start(self, req: request) -> str:
        pass

    @abc.abstractmethod
    def on_route_finish(self, req: request) -> bool:
        pass

    @abc.abstractmethod
    def parse_log_file(self, req: request, filename: str):
        pass