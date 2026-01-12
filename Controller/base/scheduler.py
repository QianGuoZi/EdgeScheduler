import os
import random
from typing import Dict
from flask import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from .algorithm.GA import NodeMappingGA
from .algorithm.Rand import NodeMappingRandom
from .ppo_scheduler import PPOScheduler


dirName = '/home/qianguo/Edge-Scheduler/Controller'


class Scheduler(object):
    def __init__(self, controller):
        self.controller = controller
        # 创建记录文件夹
        self.log_dir = os.path.join(dirName, 'scheduling_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file = None
        self.node_count = 0
        
        # 初始化PPO调度器（支持多种基于 PPO 环境的调度策略）
        self.ppo_scheduler = PPOScheduler(controller)
        

    def resource_schedule(self, taskId: int, method: str = "ppo") -> Dict:
        """
        使用PPO模型进行资源调度
        读取links_range.json，调度后生成links.json
        
        links_range.json 格式:
        {
            "nodes": {
                "p1": {"cpu": 10, "ram": 30},
                "n1": {"cpu": 8, "ram": 25},
                ...
            },
            "links": {
                "p1": [{"dest": "n1", "bw_min": "5mbps", "bw_max": "10mbps"}, ...],
                ...
            }
        }
        """
        # 读取带宽范围文件
        links_range_path = os.path.join(dirName, 'task_links', str(taskId), 'links_range.json')
        with open(links_range_path, 'r') as file:
            full_data = json.load(file)
        
        # 解析新格式：包含 nodes 和 links 两部分
        if 'nodes' in full_data and 'links' in full_data:
            nodes_data = full_data['nodes']  # 节点的 cpu 和 ram 需求
            links_data = full_data['links']  # 链路带宽需求
        else:
            # 兼容旧格式（只有链路信息）
            nodes_data = {}
            links_data = full_data
        
        # 根据调度方法检查相应的模型是否可用
        method_lower = method.lower()
        is_method_available = False
        
        if method_lower in ("ppo", "heuristic", "random"):
            # PPO、启发式和随机算法使用同一个环境，只需要 ppo_agent 可用
            is_method_available = self.ppo_scheduler.is_available()
        elif method_lower in ("ppo_mapping", "ppo_balance"):
            # PPO_mapping 需要 balance_agent 可用
            is_method_available = self.ppo_scheduler.is_balance_available()
        else:
            # 未知方法，尝试使用 PPO
            is_method_available = self.ppo_scheduler.is_available()
        
        # 尝试使用基于 PPO 环境的调度算法
        if is_method_available:
            try:
                allocation, bandwidth_allocation = self.ppo_scheduler.schedule(
                    taskId, nodes_data, links_data, method=method
                )
                
                if allocation:
                    # 更新节点计数
                    self.node_count += len(allocation)
                    # 生成links.json文件
                    self._generate_links_json(taskId, links_data, bandwidth_allocation)
                    print("Allocation:", allocation)
                    return allocation
                    
            except Exception as e:
                print(f"❌ {method} 调度失败，回退到GA算法: {e}")
                import traceback
                traceback.print_exc()
        else:
            # 如果方法不可用，打印警告信息
            if method_lower in ("ppo_mapping", "ppo_balance"):
                print(f"⚠️ {method} 方法不可用：PPO_balance 模型未加载，回退到GA算法")
            else:
                print(f"⚠️ {method} 方法不可用：PPO 模型未加载，回退到GA算法")
        
        # PPO失败时回退到GA算法
        return self._schedule_with_ga(taskId, nodes_data, links_data)
    
    def _schedule_with_ga(self, taskId: int, nodes_data: Dict, links_data: Dict) -> Dict:
        """
        使用GA算法进行调度（回退方案）
        
        Args:
            taskId: 任务ID
            nodes_data: 节点资源需求 {"p1": {"cpu": 10, "ram": 30}, ...}
            links_data: 链路带宽需求 {"p1": [{"dest": "n1", ...}, ...], ...}
        """
        print(f"🧬 使用GA算法调度任务 {taskId}")
        
        allocation = {}
        physical_nodes = []
        virtual_nodes = []
        physical_links = []
        virtual_links = []

        for emulator in self.controller.emulator.values():
            # 计算可用的CPU份数
            available_cpu_shares = emulator.get_available_cpu_shares()
            available_ram = emulator.ram - emulator.ramPreMap
            physical_nodes.append({
                'name': emulator.nameW,
                'cpu': available_cpu_shares,
                'ram': available_ram
            })
            # 同时显示实际核心数以便调试
            available_cores = emulator.get_available_cpu_cores()
            print(f"Emulator: {emulator.nameW}, "
                  f"CPU: {available_cpu_shares} shares ({available_cores:.2f} cores), "
                  f"RAM: {available_ram} GB")
        
        for emu1, emu2, bw, used_bw in self.controller.iter_bandwidth():
            physical_links.append({
                'src': emu1,
                'dst': emu2,
                'bw': bw-used_bw
            })
            print(f"Link: {emu1} -> {emu2}, Available BW: {bw-used_bw} mbps")

        # 用于生成links.json的带宽分配记录
        bandwidth_allocation = {}

        for node, connections in links_data.items():
            node_name = str(taskId) + '_' + node
            
            # 从 nodes_data 中读取 CPU 和 RAM 需求
            if nodes_data and node in nodes_data:
                cpu_demand = nodes_data[node].get('cpu', 10)
                ram_demand = nodes_data[node].get('ram', 30)
            else:
                # 兼容旧格式：如果没有节点资源信息，使用默认值
                cpu_demand = 10
                ram_demand = 30
            
            virtual_nodes.append({
                'name': node_name,
                'cpu': cpu_demand,
                'ram': ram_demand
            })
            print(f"Virtual Node: {node_name}, CPU: {cpu_demand} shares, RAM: {ram_demand} GB")
            
            for dest in connections:
                dest_node = str(taskId) + '_' + dest['dest']
                # 解析带宽范围
                if 'bw_min' in dest and 'bw_max' in dest:
                    min_bw = int(dest['bw_min'].replace('mbps', ''))
                    max_bw = int(dest['bw_max'].replace('mbps', ''))
                elif 'bw' in dest and 'bw_max' in dest:
                    min_bw = int(dest['bw'].replace('mbps', ''))
                    max_bw = int(dest['bw_max'].replace('mbps', ''))
                else:
                    bw = int(dest.get('bw', '10mbps').replace('mbps', ''))
                    min_bw = bw
                    max_bw = bw
                
                # GA算法使用最小带宽
                virtual_links.append({
                    'src': node_name,
                    'dst': dest_node,
                    'bw': min_bw
                })
                
                # 记录带宽分配（使用最小带宽）
                bandwidth_allocation[(node, dest['dest'])] = min_bw
                
            print(f"Virtual Links for {node_name}: {[node_name + ' -> ' + dest['dest'] for dest in connections]}")

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
                # self.record_load(self.node_count, allocation)

        # 生成links.json文件
        self._generate_links_json(taskId, links_data, bandwidth_allocation)
        
        # self.plot_load_history()
        print("Allocation:", allocation)
        return allocation
    
    def _generate_links_json(self, taskId: int, links_data: Dict, bandwidth_allocation: Dict):
        """根据调度结果生成links.json文件"""
        links_json = {}
        
        for node, connections in links_data.items():
            links_json[node] = []
            
            for dest in connections:
                dest_name = dest['dest']
                
                # 从带宽分配中获取分配的带宽值
                allocated_bw = bandwidth_allocation.get((node, dest_name))
                
                if allocated_bw is None:
                    # 如果没有找到分配的带宽，使用最小带宽作为默认值
                    if 'bw_min' in dest:
                        allocated_bw = int(dest['bw_min'].replace('mbps', ''))
                    elif 'bw' in dest:
                        allocated_bw = int(dest['bw'].replace('mbps', ''))
                    else:
                        allocated_bw = 10  # 默认值
                
                links_json[node].append({
                    'dest': dest_name,
                    'bw': f"{allocated_bw}mbps"
                })
        
        # 写入links.json文件
        links_json_path = os.path.join(dirName, 'task_links', str(taskId), 'links.json')
        with open(links_json_path, 'w', encoding='utf-8') as f:
            json.dump(links_json, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已生成links.json文件: {links_json_path}")
