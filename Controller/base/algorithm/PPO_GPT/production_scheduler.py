import torch
# from production_env import ProductionEdgeEnv
from policy_network import PolicyNetwork

class PPOScheduler:
    def __init__(self, config, model_path):
        self.config = config
        # self.env = ProductionEdgeEnv(config)
        self.model = self.load_model(model_path)
        
    def load_model(self, path):
        """加载训练好的模型"""
        model = PolicyNetwork(self.config)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    
    def schedule(self, task_request):
        """核心调度方法"""
        # 1. 将请求转换为状态
        state = self.env.receive_request(task_request)
        
        # 2. 生成决策（无探索）
        with torch.no_grad():
            action, _, _ = self.model.act(state, exploration=False)
        
        # 3. 转换为调度方案
        schedule_plan = self.action_to_schedule(action, task_request)
        return schedule_plan
    
    def action_to_schedule(self, action, request):
        """将动作转换为调度方案"""
        plan = {
            "task_assignments": {},
            "bandwidth_allocations": {},
            "path_mappings": {}
        }
        
        # 任务节点映射
        for i, task in enumerate(request['task_nodes']):
            phys_node_idx = action['task'][i].item()
            phys_node_id = self.env.resource_pool['nodes'][phys_node_idx]['id']
            plan["task_assignments"][task['id']] = phys_node_id
        
        # 带宽分配
        for j, link in enumerate(request['task_links']):
            # 缩放带宽到实际范围
            min_bw = link['min_bw']
            max_bw = link['max_bw']
            scaled_bw = min_bw + action['bw'][j].item() * (max_bw - min_bw)
            plan["bandwidth_allocations"][link['id']] = scaled_bw
            
            # 路径映射（简化为直接物理链路）
            src_task = next(t for t in request['task_nodes'] if t['id'] == link['source'])
            dst_task = next(t for t in request['task_nodes'] if t['id'] == link['target'])
            src_node = plan["task_assignments"][src_task['id']]
            dst_node = plan["task_assignments"][dst_task['id']]
            
            # 查找直接物理链路
            direct_link = next(
                (l for l in self.env.resource_pool['links'] 
                 if (l['source'] == src_node and l['target'] == dst_node) or
                    (l['source'] == dst_node and l['target'] == src_node)),
                None
            )
            plan["path_mappings"][link['id']] = direct_link['id'] if direct_link else None
        
        return plan