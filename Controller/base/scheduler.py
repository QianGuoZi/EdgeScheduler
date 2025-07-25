import os
import random  # 添加这行
from typing import Dict
from flask import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from .algorithm.GA import NodeMappingGA
from .algorithm.Rand import NodeMappingRandom


dirName = '/home/qianguo/Edge-Scheduler/Controller'
class Scheduler(object):
    def __init__(self, controller):
        self.controller = controller
        # 创建记录文件夹
        self.log_dir = os.path.join(dirName, 'scheduling_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file = None
        self.node_count = 0
        
    def record_load(self, node_count: int, allocation: Dict):
        """记录当前负载情况"""
        if self.current_log_file is None:
            # 使用时间戳创建新的日志文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_log_file = os.path.join(self.log_dir, f'load_log_{timestamp}.csv')
            # 创建表头
            with open(self.current_log_file, 'w') as f:
                f.write('nodes,cpu_load,ram_load,bw_load\n')
        
        # 计算总负载
        total_cpu_load = 0
        total_ram_load = 0
        total_bw_load = 0
        total_cpu_capacity = 0
        total_ram_capacity = 0
        
        # 计算 CPU 和 RAM 负载
        for emulator in self.controller.emulator.values():
            total_cpu_load += emulator.cpuPreMap
            total_ram_load += emulator.ramPreMap
            total_cpu_capacity += emulator.cpu
            total_ram_capacity += emulator.ram
            
        # 计算带宽负载
        total_bw_capacity = 0
        total_bw_used = 0
        for emu1, emu2, bw, used_bw in self.controller.iter_bandwidth():
            total_bw_capacity += bw
            total_bw_used += used_bw
            
        # 计算负载率
        cpu_load_ratio = total_cpu_load / total_cpu_capacity if total_cpu_capacity > 0 else 0
        ram_load_ratio = total_ram_load / total_ram_capacity if total_ram_capacity > 0 else 0
        bw_load_ratio = total_bw_used / total_bw_capacity if total_bw_capacity > 0 else 0
        
        # 记录到文件
        with open(self.current_log_file, 'a') as f:
            f.write(f'{node_count},{cpu_load_ratio},{ram_load_ratio},{bw_load_ratio}\n')
            
    def plot_load_history(self):
        """生成负载历史图表"""
        if not self.current_log_file or not os.path.exists(self.current_log_file):
            print("没有找到负载记录文件")
            return
            
        # 读取数据
        df = pd.read_csv(self.current_log_file)
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(df['nodes'], df['cpu_load'], 'r-', label='CPU Load')
        plt.plot(df['nodes'], df['ram_load'], 'b-', label='RAM Load')
        plt.plot(df['nodes'], df['bw_load'], 'g-', label='Bandwidth Load')
        
        plt.xlabel('Number of Nodes')
        plt.ylabel('Load Ratio')
        plt.title('Resource Load History')
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        plot_file = self.current_log_file.replace('.csv', '.png')
        plt.savefig(plot_file)
        plt.close()
        
        print(f"负载历史图表已保存到: {plot_file}")

    def resource_schedule(self, taskId: int) -> Dict:
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
        # node_count = 0

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

        for node, connections in links_data.items():
            node_name = str(taskId) + '_' + node
            # 使用随机生成的 CPU 和 RAM 值
            cpu_demand = random.randint(1, 5)
            ram_demand = random.randint(1, 5)
            virtual_nodes.append({
                'name': node_name,
                # 'cpu': cpu_demand,
                'cpu': 2,
                # 'ram': ram_demand
                'ram': 5
            })
            print(f"Virtual Node: {node_name}, CPU: {cpu_demand}, RAM: {ram_demand}")
            for dest in connections:
                dest_node = str(taskId) + '_' + dest['dest']
                bw = int(dest['bw'].replace('mbps', ''))  # 转换带宽值为整数
                virtual_links.append({
                    'src': node_name,
                    'dst': dest_node,
                    'bw': bw
                })
            print(f"Virtual Links for {node_name}: {[f'{node_name} -> {dest['dest']}' for dest in connections]}")

        # 调度过程
        ga = NodeMappingGA(physical_nodes, virtual_nodes, physical_links, virtual_links)
        best_solution = ga.run()
        print("Best Solution:", best_solution)

        # rand = NodeMappingRandom(physical_nodes, virtual_nodes, physical_links, virtual_links)
        # best_solution = rand.run()
        # print("Best Solution (Random):", best_solution)

        for i, node in enumerate(virtual_nodes):
            if best_solution[i] is not None:
                allocation[node['name']] = {
                    'emulator': physical_nodes[best_solution[i]]['name'],
                    'cpu': node['cpu'],
                    'ram': node['ram']
                }
                self.node_count += 1
                # 记录当前负载情况
                self.record_load(self.node_count, allocation)

        
        # 生成负载历史图表
        self.plot_load_history()
        print("Allocation:", allocation)
        return allocation