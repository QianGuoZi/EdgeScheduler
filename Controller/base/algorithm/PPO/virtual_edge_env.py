import random
import numpy as np
import torch
from torch_geometric.data import Data

class VirtualEdgeEnv:
    def __init__(self, config):
        self.config = config
        self.resource_pool = self.create_resource_pool()
        self.current_state = None
        self.resource_usage = self.initialize_resource_usage()
        
    def initialize_resource_usage(self):
        """ 初始化资源使用状态 """
        return {
            "nodes": {node["id"]: {"cpu_used": 0, "ram_used": 0} for node in self.resource_pool["nodes"]},
            "links": {link["id"]: {"bandwidth_used": 0} for link in self.resource_pool["links"]}
        }
    
    def create_resource_pool(self):
        """ 创建虚拟边缘集群 """
        # 创建节点
        nodes = [
            {
                "id": i, 
                "cpu": random.randint(self.config['cpu_min'], self.config['cpu_max']),
                "ram": random.randint(self.config['ram_min'], self.config['ram_max'])
            }
            for i in range(self.config['num_phys_nodes'])
        ]
        
        # 创建链路
        links = []
        link_id = 0
        for i in range(self.config['num_phys_nodes']):
            for j in range(i+1, self.config['num_phys_nodes']):
                # 随机决定是否创建链路
                if random.random() > self.config['link_sparsity']:
                    links.append({
                        "id": link_id,
                        "source": i,
                        "target": j,
                        "bandwidth": random.randint(
                            self.config['bw_min'], 
                            self.config['bw_max'])
                    })
                    link_id += 1
        
        return {"nodes": nodes, "links": links}
    
    def reset(self):
        """ 重置环境状态 """
        self.resource_usage = self.initialize_resource_usage()
        task_request = self.generate_task_request()
        self.current_state = self.request_to_state(task_request)
        return self.current_state
    
    def generate_task_request(self):
        """ 生成虚拟任务请求 """
        # 随机确定任务数量
        num_tasks = random.randint(
            self.config['min_tasks'], 
            self.config['max_tasks'])
        
        # 创建任务节点
        task_nodes = [
            {
                "id": f"t{i}", 
                "cpu_req": random.randint(1, 4), 
                "ram_req": random.randint(1, 8)
            }
            for i in range(num_tasks)
        ]
        
        # 创建任务链路
        task_links = []
        for i in range(num_tasks):
            for j in range(i+1, num_tasks):
                # 随机决定是否创建任务链路
                if random.random() > 0.7:  # 30%的节点间有连接
                    min_bw = random.randint(10, 50)
                    max_bw = random.randint(min_bw + 10, min_bw + 100)
                    task_links.append({
                        "source": f"t{i}",
                        "target": f"t{j}",
                        "min_bw": min_bw,
                        "max_bw": max_bw
                    })
        
        return {"task_nodes": task_nodes, "task_links": task_links}
    
    def request_to_state(self, request):
        """ 将请求转换为PPO输入状态 """
        # 构建任务图
        task_data = self.build_task_graph(request)
        
        # 构建物理图
        phys_data = self.build_phys_graph()
        
        return (task_data, phys_data)
    
    def build_task_graph(self, request):
        """ 构建任务图Data对象 - 确保包含边特征 """
        # 节点特征矩阵 [cpu_req, ram_req]
        node_features = torch.tensor([
            [node['cpu_req'], node['ram_req']] 
            for node in request['task_nodes']
        ], dtype=torch.float)
        
        # 边连接和特征 [min_bw, max_bw] - 现在边特征维度为2
        edge_index = []
        edge_attr = []
        for link in request['task_links']:
            src_idx = next(i for i, n in enumerate(request['task_nodes']) 
                      if n['id'] == link['source'])
            dst_idx = next(i for i, n in enumerate(request['task_nodes']) 
                      if n['id'] == link['target'])
            edge_index.append([src_idx, dst_idx])
            edge_attr.append([link['min_bw'], link['max_bw']])  # 保持边特征为2维
        
        return Data(
            x=node_features,
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            batch=torch.zeros(len(request['task_nodes']), dtype=torch.long)
        )
    
    def build_phys_graph(self):
        """ 构建物理图Data对象 - 确保包含边特征 """
        # 节点特征矩阵 [cpu_capacity, cpu_used, ram_capacity, ram_used]
        node_features = []
        for node in self.resource_pool['nodes']:
            usage = self.resource_usage['nodes'][node['id']]
            node_features.append([
                node['cpu'], 
                usage['cpu_used'],
                node['ram'],
                usage['ram_used']
            ])
        
        # 边特征矩阵 [bandwidth_capacity, bandwidth_used] - 边特征维度为2
        edge_attr = []
        edge_index = []
        for link in self.resource_pool['links']:
            usage = self.resource_usage['links'][link['id']]
            edge_index.append([link['source'], link['target']])
            edge_attr.append([
                link['bandwidth'],
                usage['bandwidth_used']
            ])
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            batch=torch.zeros(len(self.resource_pool['nodes']), dtype=torch.long)
        )
    
    def get_action_masks(self, task_request):
        """ 生成动作掩码 """
        masks = {
            'task': [],  # 任务分配掩码
            'path': []   # 路径选择掩码
        }
        
        # 任务分配掩码：检查物理节点资源是否足够
        for task in task_request['task_nodes']:
            task_mask = []
            for node in self.resource_pool['nodes']:
                available_cpu = node['cpu'] - self.resource_usage['nodes'][node['id']]['cpu_used']
                available_ram = node['ram'] - self.resource_usage['nodes'][node['id']]['ram_used']
                sufficient = (available_cpu >= task['cpu_req'] and 
                             available_ram >= task['ram_req'])
                task_mask.append(sufficient)
            masks['task'].append(task_mask)
        
        # 路径选择掩码：检查链路带宽是否足够
        for link in task_request['task_links']:
            path_mask = []
            for phys_link in self.resource_pool['links']:
                available_bw = phys_link['bandwidth'] - self.resource_usage['links'][phys_link['id']]['bandwidth_used']
                sufficient = available_bw >= link['min_bw']
                path_mask.append(sufficient)
            masks['path'].append(path_mask)
        
        # 转换为张量
        masks['task'] = torch.tensor(masks['task'], dtype=torch.bool)
        masks['path'] = torch.tensor(masks['path'], dtype=torch.bool)
        
        return masks
    
    def scale_bandwidth_action(self, bw_action, min_bw, max_bw):
        """ 将[0,1]范围的带宽动作缩放到[min_bw, max_bw] """
        return min_bw + bw_action * (max_bw - min_bw)

    def apply_schedule(self, schedule_plan):
        """ 应用调度并返回新状态和奖励 """
        # 1. 应用任务分配
        for task_id, node_id in schedule_plan['task_assignments'].items():
            task = next(t for t in schedule_plan['task_nodes'] if t['id'] == task_id)
            self.resource_usage['nodes'][node_id]['cpu_used'] += task['cpu_req']
            self.resource_usage['nodes'][node_id]['ram_used'] += task['ram_req']
        
        # 2. 应用带宽分配（缩放动作）
        for link_id, bw_action in schedule_plan['bandwidth_actions'].items():
            link = next(l for l in schedule_plan['task_links'] if l['id'] == link_id)
            actual_bw = self.scale_bandwidth_action(
                bw_action, 
                link['min_bw'], 
                link['max_bw']
            )
            
            phys_link_id = schedule_plan['path_mappings'][link_id]
            self.resource_usage['links'][phys_link_id]['bandwidth_used'] += actual_bw
            
        # 3. 计算奖励
        reward = self.calculate_reward(schedule_plan)
        
        # 4. 检查是否完成
        done = len(schedule_plan['task_assignments']) == len(schedule_plan['task_nodes'])
        
        # 5. 获取新状态
        new_state = self.request_to_state(schedule_plan)
        self.current_state = new_state
        
        return new_state, reward, done
    
    def calculate_reward(self, schedule_plan):
        """ 计算奖励值 """
        # 1. 计算负载均衡指标
        cpu_utils = []
        ram_utils = []
        for node in self.resource_pool['nodes']:
            usage = self.resource_usage['nodes'][node['id']]
            cpu_utils.append(usage['cpu_used'] / node['cpu'])
            ram_utils.append(usage['ram_used'] / node['ram'])
        
        L_cpu = np.std(cpu_utils)
        L_ram = np.std(ram_utils)
        
        # 2. 计算带宽利用率指标
        bw_utils = []
        for link in self.resource_pool['links']:
            usage = self.resource_usage['links'][link['id']]
            bw_utils.append(usage['bandwidth_used'] / link['bandwidth'])
        L_bw = np.std(bw_utils) if bw_utils else 0
        
        # 3. 综合负载均衡度
        L_total = 0.4 * L_cpu + 0.3 * L_ram + 0.3 * L_bw
        
        # 4. 计算带宽需求满足度
        bw_satisfaction = 0
        for link in schedule_plan['task_links']:
            allocated_bw = schedule_plan['bandwidth_allocations'][link['id']]
            min_bw = link['min_bw']
            max_bw = link['max_bw']
            
            if max_bw == min_bw:
                bw_satisfaction += 1 if allocated_bw == min_bw else 0
            else:
                bw_satisfaction += (allocated_bw - min_bw) / (max_bw - min_bw)
        
        bw_satisfaction /= len(schedule_plan['task_links']) if schedule_plan['task_links'] else 1
        
        # 5. 约束违反惩罚
        penalty = 0
        # 检查节点资源超限
        for node in self.resource_pool['nodes']:
            usage = self.resource_usage['nodes'][node['id']]
            if usage['cpu_used'] > node['cpu']:
                penalty += 10 * (usage['cpu_used'] - node['cpu'])
            if usage['ram_used'] > node['ram']:
                penalty += 10 * (usage['ram_used'] - node['ram'])
        
        # 检查链路带宽超限
        for link in self.resource_pool['links']:
            usage = self.resource_usage['links'][link['id']]
            if usage['bandwidth_used'] > link['bandwidth']:
                penalty += 5 * (usage['bandwidth_used'] - link['bandwidth'])
        
        # 6. 最终奖励
        reward = -self.config['gamma1'] * L_total + self.config['gamma2'] * bw_satisfaction - penalty
        return reward