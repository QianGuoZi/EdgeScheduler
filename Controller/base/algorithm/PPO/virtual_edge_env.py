import random
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

class VirtualEdgeEnv:
    def __init__(self, config):
        self.config = config
        self.resource_pool = self.create_resource_pool()
        self.current_state = None
        self.assigned_tasks = set()
        self.resource_usage = self.initialize_resource_usage()
        self.current_task_request = None  # 添加当前任务请求属性
        self.step_count = 0 
        self.max_steps = config.get('max_steps_per_episode', 100)
        
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
        """重置环境状态"""
        self.resource_usage = self.initialize_resource_usage()
        task_request = self.generate_task_request()
        self.current_task_request = task_request  # 保存当前任务请求
        self.current_state = self.request_to_state(task_request)
        self.assigned_tasks = set()
        self.step_count = 0
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
                if random.random() > 0.5:  # 50%的节点间有连接
                    min_bw = random.randint(10, 50)
                    max_bw = random.randint(min_bw + 10, min_bw + 100)
                    task_links.append({
                        "id": len(task_links),  # 添加链路ID
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
        # print("Task graph built:")
        # print("x shape:", task_data.x.shape, task_data.x.dtype)
        # print("edge_index shape:", task_data.edge_index.shape, task_data.edge_index.dtype)
        # print("edge_attr shape:", task_data.edge_attr.shape, task_data.edge_attr.dtype)
        
        # 构建物理图
        phys_data = self.build_phys_graph()
        # print("Physical graph built:")
        # print("x shape:", phys_data.x.shape, phys_data.x.dtype)
        # print("edge_index shape:", phys_data.edge_index.shape, phys_data.edge_index.dtype)
        # print("edge_attr shape:", phys_data.edge_attr.shape, phys_data.edge_attr.dtype)
        
        return (task_data, phys_data)
    
    def build_task_graph(self, request):
        """ 构建任务图Data对象 - 确保包含边特征 """
        # 节点特征矩阵 [cpu_req, ram_req]
        node_features = torch.tensor([
            [node['cpu_req'], node['ram_req']]
            for node in request['task_nodes']
        ], dtype=torch.float)
        
        # 边连接和特征 [min_bw, max_bw] - 现在边特征维度为2
        edge_index_list = []
        edge_attr_list = []
        for link in request['task_links']:
            src_idx = next(i for i, n in enumerate(request['task_nodes'])
                           if n['id'] == link['source'])
            dst_idx = next(i for i, n in enumerate(request['task_nodes'])
                           if n['id'] == link['target'])
            edge_index_list.append([src_idx, dst_idx])
            edge_attr_list.append([link['min_bw'], link['max_bw']])
        
        if len(edge_index_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
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
        edge_index_list = []
        edge_attr_list = []
        for link in self.resource_pool['links']:
            usage = self.resource_usage['links'][link['id']]
            edge_index_list.append([link['source'], link['target']])
            edge_attr_list.append([
                link['bandwidth'],
                usage['bandwidth_used']
            ])
            
        if len(edge_index_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=torch.zeros(len(self.resource_pool['nodes']), dtype=torch.long)
        )
    
    def get_action_masks(self):
            """生成动作掩码（使用当前任务请求）"""
            if self.current_task_request is None:
                raise RuntimeError("No current task request available")
                
            masks = {
                'task': [],  # 任务分配掩码
            }
            
            # 任务分配掩码：检查物理节点资源是否足够
            for task in self.current_task_request['task_nodes']:
                task_mask = []
                for node in self.resource_pool['nodes']:
                    available_cpu = node['cpu'] - self.resource_usage['nodes'][node['id']]['cpu_used']
                    available_ram = node['ram'] - self.resource_usage['nodes'][node['id']]['ram_used']
                    sufficient = (available_cpu >= task['cpu_req'] and 
                                available_ram >= task['ram_req'])
                    task_mask.append(sufficient)
                masks['task'].append(task_mask)
            
            # 转换为张量
            masks['task'] = torch.tensor(masks['task'], dtype=torch.bool)
            
            return masks
        
    def scale_bandwidth_action(self, bw_action, min_bw, max_bw):
        """ 将[0,1]范围的带宽动作缩放到[min_bw, max_bw] """
        return min_bw + bw_action * (max_bw - min_bw)

    def apply_schedule(self, action):
        """应用调度并返回新状态和奖励"""
        if self.current_task_request is None:
            raise RuntimeError("No current task request available")
            
        task_assignments = action['task']
        bw_actions = action['bw']
        
        # 1. 应用任务分配
        for task_idx, node_id in enumerate(task_assignments):
            node_id_int = int(node_id)
            task = self.current_task_request['task_nodes'][task_idx]
            self.resource_usage['nodes'][node_id_int]['cpu_used'] += task['cpu_req']
            self.resource_usage['nodes'][node_id_int]['ram_used'] += task['ram_req']
        
        # 2. 应用带宽分配和路由
        bw_allocations = {}
        path_mappings = {}
        
        for link_idx, link in enumerate(self.current_task_request['task_links']):
            # 获取源任务和目标任务在任务列表中的索引
            src_task_idx = next(i for i, n in enumerate(self.current_task_request['task_nodes']) 
                          if n['id'] == link['source'])
            dst_task_idx = next(i for i, n in enumerate(self.current_task_request['task_nodes']) 
                          if n['id'] == link['target'])
            
            # 获取映射的物理节点
            src_phys = task_assignments[src_task_idx]
            dst_phys = task_assignments[dst_task_idx]
            
            # 如果映射到同一节点，不消耗带宽
            if src_phys == dst_phys:
                allocated_bw = link['min_bw']
                bw_allocations[link['id']] = allocated_bw
                path_mappings[link['id']] = None
                continue
                
            # 计算路径
            path = self.shortest_path(src_phys, dst_phys)
            if path is None:
                # 无可用路径，使用最小带宽
                allocated_bw = link['min_bw']
            else:
                # 检查路径上的最小剩余带宽
                min_remaining = min(link_info['remaining'] for link_info in path)
                
                # 计算实际分配的带宽（缩放动作值）
                min_bw = link['min_bw']
                max_bw = min(link['max_bw'], min_remaining)  # 不超过路径容量
                allocated_bw = self.scale_bandwidth_action(bw_actions[link_idx], min_bw, max_bw)
                
                # 更新链路使用
                for link_info in path:
                    link_id = link_info['link_id']
                    self.resource_usage['links'][link_id]['bandwidth_used'] += allocated_bw
                
                # 存储分配结果
                path_mappings[link['id']] = [link_info['link_id'] for link_info in path]
            
            bw_allocations[link['id']] = allocated_bw
        
        # 3. 计算奖励
        reward = self.calculate_reward(task_assignments, bw_allocations, path_mappings)
        
        # 4. 更新步数计数器
        self.step_count += 1
        
        # 5. 检查终止条件
        done = False
        reason = ""
        
        # 条件1: 所有任务都已分配
        all_tasks_assigned = len(self.assigned_tasks) == len(self.current_task_request['task_nodes'])
        if all_tasks_assigned:
            done = True
            reason = "All tasks assigned"
        
        # 条件2: 达到最大步数
        elif self.step_count >= self.max_steps:
            done = True
            reason = f"Max steps reached ({self.max_steps})"
        
        # 条件3: 资源耗尽 - 检查是否还能调度剩余任务
        elif not self.can_schedule_remaining_tasks():
            done = True
            reason = "Resource exhausted"
        
        # 条件4: 调度失败 - 当前动作导致资源超限
        if self.has_constraint_violation():
            done = True
            reason = "Constraint violation"
            # 给予严重惩罚
            reward = -1000

        # 6. 生成新任务请求并更新状态
        self.current_task_request = self.generate_task_request()
        new_state = self.request_to_state(self.current_task_request)
        self.current_state = new_state

         # 7. 记录终止原因（用于调试）
        if done:
            print(f"Episode done: {reason}")
    
        return new_state, reward, done
    
    def calculate_reward(self, task_assignments, bw_allocations, path_mappings):
        """计算奖励值，考虑链路约束"""
        if self.current_task_request is None:
            return 0
            
        # 1. 节点资源利用率
        cpu_utils = []
        ram_utils = []
        for node in self.resource_pool['nodes']:
            usage = self.resource_usage['nodes'][node['id']]
            cpu_utils.append(usage['cpu_used'] / node['cpu'])
            ram_utils.append(usage['ram_used'] / node['ram'])
        
        # 2. 链路带宽利用率
        bw_utils = []
        for link in self.resource_pool['links']:
            usage = self.resource_usage['links'][link['id']]
            bw_utils.append(usage['bandwidth_used'] / link['bandwidth'])
        
        # 3. 计算负载均衡指标
        L_cpu = np.std(cpu_utils) if cpu_utils else 0
        L_ram = np.std(ram_utils) if ram_utils else 0
        L_bw = np.std(bw_utils) if bw_utils else 0
        L_total = 0.4 * L_cpu + 0.3 * L_ram + 0.3 * L_bw
        
        # 4. 带宽需求满足度
        bw_satisfaction = 0
        for link in self.current_task_request['task_links']:
            allocated_bw = bw_allocations.get(link['id'], link['min_bw'])
            min_bw = link['min_bw']
            max_bw = link['max_bw']
            
            if max_bw == min_bw:
                bw_satisfaction += 1 if allocated_bw == min_bw else 0
            else:
                # 确保不会超过最大带宽
                normalized = min(1.0, max(0.0, (allocated_bw - min_bw) / (max_bw - min_bw)))
                bw_satisfaction += normalized
        
        if self.current_task_request['task_links']:
            bw_satisfaction /= len(self.current_task_request['task_links'])
        else:
            bw_satisfaction = 1.0
        
        # 5. 约束违反惩罚
        penalty = 0
        # 节点资源超限
        for node in self.resource_pool['nodes']:
            usage = self.resource_usage['nodes'][node['id']]
            if usage['cpu_used'] > node['cpu']:
                penalty += 10 * (usage['cpu_used'] - node['cpu'])
            if usage['ram_used'] > node['ram']:
                penalty += 10 * (usage['ram_used'] - node['ram'])
        
        # 链路带宽超限
        for link in self.resource_pool['links']:
            usage = self.resource_usage['links'][link['id']]
            if usage['bandwidth_used'] > link['bandwidth']:
                penalty += 5 * (usage['bandwidth_used'] - link['bandwidth'])
        
        # 6. 最终奖励
        reward = -self.config.get('gamma1', 0.5) * L_total + \
                  self.config.get('gamma2', 0.5) * bw_satisfaction - \
                  penalty
        
        return reward
    
    def shortest_path(self, src_phys, dst_phys, weight_type='bw_available'):
        """使用Dijkstra算法计算最短路径"""
        G = nx.Graph()
        
        # 添加物理节点
        for node in self.resource_pool['nodes']:
            G.add_node(node['id'])
        
        # 添加物理链路
        for link in self.resource_pool['links']:
            remaining_bw = link['bandwidth'] - self.resource_usage['links'][link['id']]['bandwidth_used']
            
            # 设置权重
            if weight_type == 'hop_count':
                weight = 1
            elif weight_type == 'bw_available':
                weight = 1 / (remaining_bw + 1e-6)  # 避免除以0
            else:
                weight = 1
                
            G.add_edge(link['source'], link['target'], 
                    weight=weight, 
                    id=link['id'],
                    capacity=link['bandwidth'],
                    remaining=remaining_bw)
        
        try:
            path_nodes = nx.shortest_path(G, source=src_phys, target=dst_phys, weight='weight')
            path_links = []
            for i in range(len(path_nodes)-1):
                edge_data = G[path_nodes[i]][path_nodes[i+1]]
                # 对于无向图，可能有多个边属性，取第一个
                if isinstance(edge_data, dict):
                    link_data = edge_data
                else:
                    # 对于多边图，取第一个边
                    link_data = list(edge_data.values())[0]
                
                path_links.append({
                    'link_id': link_data['id'],
                    'capacity': link_data['capacity'],
                    'remaining': link_data['remaining']
                })
            return path_links
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    def has_constraint_violation(self):
        """ 检查是否有资源约束违反 """
        # 检查节点资源
        for node in self.resource_pool['nodes']:
            usage = self.resource_usage['nodes'][node['id']]
            if usage['cpu_used'] > node['cpu'] or usage['ram_used'] > node['ram']:
                return True
        
        # 检查链路带宽
        for link in self.resource_pool['links']:
            usage = self.resource_usage['links'][link['id']]
            if usage['bandwidth_used'] > link['bandwidth']:
                return True
        
        return False
    def can_schedule_remaining_tasks(self):
        """ 检查是否还能调度剩余的任务 """
        unscheduled_tasks = [
            t for t in self.current_task_request['task_nodes'] 
            if t['id'] not in self.assigned_tasks
        ]
        
        # 如果没有剩余任务，返回True
        if not unscheduled_tasks:
            return True
        
        # 检查每个未调度任务是否至少有一个可用节点
        for task in unscheduled_tasks:
            has_valid_node = False
            for node in self.resource_pool['nodes']:
                available_cpu = node['cpu'] - self.resource_usage['nodes'][node['id']]['cpu_used']
                available_ram = node['ram'] - self.resource_usage['nodes'][node['id']]['ram_used']
                
                if available_cpu >= task['cpu_req'] and available_ram >= task['ram_req']:
                    has_valid_node = True
                    break
            
            # 如果有一个任务无法调度，返回False
            if not has_valid_node:
                return False
        
        return True
    def has_constraint_violation(self):
        """ 检查是否有资源约束违反 """
        # 检查节点资源
        for node in self.resource_pool['nodes']:
            usage = self.resource_usage['nodes'][node['id']]
            if usage['cpu_used'] > node['cpu'] or usage['ram_used'] > node['ram']:
                return True
        
        # 检查链路带宽
        for link in self.resource_pool['links']:
            usage = self.resource_usage['links'][link['id']]
            if usage['bandwidth_used'] > link['bandwidth']:
                return True
        
        return False