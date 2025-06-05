import os
from typing import Dict
from flask import json

from .algorithm.GA import NodeMappingGA


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
        physical_nodes = []
        virtual_nodes = []
        physical_links = []
        virtual_links = []

        for emulator in self.controller.emulator.values():
            physical_nodes.append({
                'name': emulator.nameW,
                'cpu': emulator.cpu - emulator.cpuPreMap,
                'ram': emulator.ram - emulator.ramPreMap
            })
            print(f"Emulator: {emulator.nameW}, CPU: {emulator.cpu - emulator.cpuPreMap}, RAM: {emulator.ram - emulator.ramPreMap}")
        
        for emu1, emu2, bw, used_bw in self.controller.iter_bandwidth():
            physical_links.append({
                'src': emu1,
                'dst': emu2,
                'bw': bw-used_bw
            })
            print(f"Link: {emu1} -> {emu2}, Available BW: {bw-used_bw} mbps")

        # 调度过程
        for node, connections in links_data.items():
            node_name = str(taskId) + '_' + node
            virtual_nodes.append({
                'name': node_name,
                'cpu': 5,
                'ram': 2
            })
            print(f"Virtual Node: {node_name}, CPU: 5, RAM: 2")
            for dest in connections:
                dest_node = str(taskId) + '_' + dest['dest']
                bw = int(dest['bw'].replace('mbps', ''))  # 转换带宽值为整数
                virtual_links.append({
                    'src': node_name,
                    'dst': dest_node,
                    'bw': bw
                })
            print(f"Virtual Links for {node_name}: {[f'{node_name} -> {dest['dest']}' for dest in connections]}")
            # print(f"Node: {node} ")
            # for e_name, e_obj in self.controller.emulator.items():
            #     # print(f"Emulator name: {e_name}, Emulator object: {e_obj}")
            #     if e_obj.cpu - e_obj.cpuPreMap > 5 and e_obj.ram - e_obj.ramPreMap > 2:
            #         allocation[node_name] = {'emulator': e_name, 'cpu': 5, 'ram': 2}
            #     else :
            #         continue
        ga = NodeMappingGA(physical_nodes, virtual_nodes, physical_links, virtual_links)
        best_solution = ga.run()
        print("Best Solution:", best_solution)
        for i, node in enumerate(virtual_nodes):
            if best_solution[i] is not None:
                allocation[node['name']] = {
                    'emulator': physical_nodes[best_solution[i]]['name'],
                    'cpu': node['cpu'],
                    'ram': node['ram']
                }
        print("Allocation:", allocation)
        return allocation