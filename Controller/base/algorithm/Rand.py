import random

class NodeMappingRandom:
    """随机分配算法类"""
    def __init__(self, physical_nodes, virtual_nodes, physical_links, virtual_links):
        self.physical_nodes = physical_nodes  # 物理节点信息，包含CPU和内存资源
        self.virtual_nodes = virtual_nodes  # 虚拟节点需求，包含CPU和内存需求
        self.physical_links = physical_links  # 物理链路信息，包含带宽资源
        self.virtual_links = virtual_links  # 虚拟链路需求，包含带宽需求
        
        # 添加节点名称到索引的映射
        self.phys_name_to_idx = {node['name']: i for i, node in enumerate(physical_nodes)}
        self.virt_name_to_idx = {node['name']: i for i, node in enumerate(virtual_nodes)}
        
    def check_resources(self, allocation):
        """检查资源分配是否可行"""
        # 初始化物理节点的资源使用情况
        physical_usage = {i: {'cpu': 0, 'ram': 0} for i in range(len(self.physical_nodes))}
        
        # 计算CPU和RAM使用情况
        for v_idx, p_idx in enumerate(allocation):
            if p_idx is None:
                continue
            physical_usage[p_idx]['cpu'] += self.virtual_nodes[v_idx]['cpu']
            physical_usage[p_idx]['ram'] += self.virtual_nodes[v_idx]['ram']
            
            # 检查是否超过容量
            if (physical_usage[p_idx]['cpu'] > self.physical_nodes[p_idx]['cpu'] or
                physical_usage[p_idx]['ram'] > self.physical_nodes[p_idx]['ram']):
                return False
                
        # 检查带宽限制
        for v_link in self.virtual_links:
            src_idx = self.virt_name_to_idx[v_link['src']]
            dst_idx = self.virt_name_to_idx[v_link['dst']]
            src_phys = allocation[src_idx]
            dst_phys = allocation[dst_idx]
            
            if src_phys is None or dst_phys is None or src_phys == dst_phys:
                continue
                
            # 查找对应的物理链路
            link_found = False
            for p_link in self.physical_links:
                p_src = self.phys_name_to_idx[p_link['src']]
                p_dst = self.phys_name_to_idx[p_link['dst']]
                if (p_src == src_phys and p_dst == dst_phys) or (p_src == dst_phys and p_dst == src_phys):
                    if v_link['bw'] <= p_link['bw']:
                        link_found = True
                        break
            
            if not link_found:
                return False
                
        return True
        
    def run(self):
        """随机分配算法主函数"""
        max_attempts = 100  # 最大尝试次数
        best_allocation = None
        
        for attempt in range(max_attempts):
            # 随机生成一个分配方案
            allocation = []
            for _ in range(len(self.virtual_nodes)):
                p_idx = random.randint(0, len(self.physical_nodes) - 1)
                allocation.append(p_idx)
                
            # 检查资源限制
            if self.check_resources(allocation):
                best_allocation = allocation
                print(f"找到可行解: {allocation}")
                break
                
            if attempt % 100 == 0:
                print(f"尝试次数: {attempt}")
                
        if best_allocation is None:
            print("未找到可行解")
            return None

        return best_allocation