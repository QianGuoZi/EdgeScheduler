import os
from typing import Dict
from flask import json


dirName = '/home/qianguo/Edge-Scheduler/Controller'
class Scheduler(object):
    def __init__(self, controller):
        self.controller = controller

    def resource_schedule(self, taskId : int) -> Dict:
        """
        需要访问Controller获取目前的资源，还有现有的需求，然后根据调度算法提供
        """
        # self.testbed.emulater
        with open(os.path.join(dirName, 'task_links', str(taskId),'links.json'), 'r') as file:
            links_data = json.load(file)
        
        allocation = {}

        # 调度过程
        for node, connections in links_data.items():
            node_name = str(taskId) + '_' + node
            # print(f"Node: {node}")
            for e_name, e_obj in self.controller.emulator.items():
                # print(f"Emulator name: {e_name}, Emulator object: {e_obj}")
                if e_obj.cpu - e_obj.cpuPreMap > 5 and e_obj.ram - e_obj.ramPreMap > 2:
                    allocation[node_name] = {'emulator': e_name, 'cpu': 5, 'ram': 2}
                else :
                    continue
        

        return allocation