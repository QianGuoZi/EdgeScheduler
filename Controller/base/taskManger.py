import abc
from concurrent.futures import wait, ALL_COMPLETED
import os
import threading
import time
from typing import Dict, List

from flask import request
from .utils import send_data

class NodeInfo(object):
    def __init__(self, name: str, ip: str, port: int):
        self.name: str = name
        self.ip: str = ip
        self.port: int = port

class TaskManager(metaclass=abc.ABCMeta):
    """
    负责任务的部署、资源调整、任务状态监控等
    """
    def __init__(self, task):
        self.task = task
        self.eNode: Dict[str, NodeInfo] = {}
        self.pNode: Dict[str, NodeInfo] = {}
        self.nodeNumber: int = 0
        self.logFile: List[str] = []
        self.logFileFolder: str = ''
        self.lock = threading.RLock()
        self.__load_default_route()
    
    def load_node_info(self):
        for name, en in self.task.eNode.items():
            self.eNode[name] = NodeInfo(name, en.ip, en.hostPort)
        for name, pn in self.task.pNode.items():
            self.pNode[name] = NodeInfo(name, pn.ip, pn.hostPort)
        self.nodeNumber = len(self.eNode) + len(self.pNode)

    #这几个端口不知道挂哪去了，草拟的
    def __load_default_route(self):
        @self.task.flask.route('/health')
        def health_check():
            return 'OK'
        
        prefix = self.task.url_prefix
        @self.task.flask.route(f'{prefix}/startTask', methods=['POST'])
        def route_start_task():
            """
            开始任务
            """
            taskID = request.form.get('taskID')
            print(f'start task {taskID}')
            if self.logFileFolder == '':
                self.logFileFolder = os.path.join(self.task.dirName, 'dml_file/log', str(taskID),
                                                  time.strftime('-%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
                os.makedirs(self.logFileFolder, exist_ok=True)
            msg = self.on_route_start(request)
            # return str explicitly is necessary.
            return str(msg)

        @self.task.flask.route(f'{prefix}/finishTask', methods=['GET'])
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
        
        @self.task.flask.route('/log', methods=['POST'])
        def route_log():
            """
            this function can listen log files from worker/worker_utils.py, send_log ().
            log files will be saved on ${self.logFileFolder}.
            when total_number files are received, it will parse these files into pictures
            and save them on ${self.logFileFolder}/png.
            user need to implement self.parse_log_file () by extend this class.
            """
            name = request.args.get('name')
            taskID = request.args.get('taskId')
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
                    self.task.executor.submit(self.__after_log)
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
    
    def __stop_all_emulated(self):
        def stop_emulated(_emulator_ip: str, _agent_port: int):
            send_data('GET', '/emulated/stop', _emulator_ip, _agent_port)

        tasks = []
        for s in self.task.emulator.values():
            if s.eNode:
                tasks.append(self.task.executor.submit(stop_emulated, s.ipW, self.task.agentPort))
        wait(tasks, return_when=ALL_COMPLETED)

    def __after_log(self):
        time.sleep(5)
        print('try to stop all emulated nodes')
        self.__stop_all_emulated()