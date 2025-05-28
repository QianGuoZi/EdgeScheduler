import numpy as np
import random

class NodeMappingGA:
    def __init__(self, physical_nodes, virtual_nodes, physical_links, virtual_links, pop_size=100):
        self.physical_nodes = physical_nodes  # 物理节点信息，包含CPU和内存资源
        self.virtual_nodes = virtual_nodes  # 虚拟节点需求，包含CPU和内存需求
        self.physical_links = physical_links  # 物理链路信息，包含带宽资源
        self.virtual_links = virtual_links  # 虚拟链路需求，包含带宽需求
        self.pop_size = pop_size  # 种群大小
        
        self.population = self.init_population()
    
    def init_population(self):
        # 为每个个体随机分配一个物理节点ID，创建一个初始种群。
        population = []
        for _ in range(self.pop_size):
            individual = [random.randint(0, len(self.physical_nodes) - 1) for _ in self.virtual_nodes]
            population.append(individual)
            print(f"Initialized individual: {individual}")
        return population
    
    def fitness(self, individual):
        # 计算适应度值和可行性值
        # 方法计算个体的适应度值。如果资源负载超过1，则使用指数惩罚函数调整适应度值。
        cpu_load, ram_load, bw_load = self.calculate_loads(individual)
        penalty_coefficients = {'cpu': 2.0, 'ram': 2.0, 'bw': 3.0}
        fitness_cpu = 2 - cpu_load if cpu_load <= 1 else np.exp(penalty_coefficients['cpu'] * (1 - cpu_load))
        fitness_ram = 2 - ram_load if ram_load <= 1 else np.exp(penalty_coefficients['ram'] * (1 - ram_load))
        fitness_bw = 2 - bw_load if bw_load <= 1 else np.exp(penalty_coefficients['bw'] * (1 - bw_load))
        feasibility = 1 if all(load <= 1 for load in [cpu_load, ram_load, bw_load]) else 0
        print(f"Individual: {individual}, CPU Load: {cpu_load}, RAM Load: {ram_load}, BW Load: {bw_load}, Fitness CPU: {fitness_cpu}, Fitness RAM: {fitness_ram}, Fitness BW: {fitness_bw}, Feasibility: {feasibility}")
        return fitness_cpu + fitness_ram + fitness_bw, feasibility
    
    def calculate_loads(self, individual):
        # 初始化每个物理节点的负载
        physical_node_loads = {p: {'cpu': 0, 'ram': 0} for p in range(len(self.physical_nodes))}
        
        # 计算每个物理节点承担的虚拟节点负载总和
        for v_idx, p_idx in enumerate(individual):
            physical_node_loads[p_idx]['cpu'] += self.virtual_nodes[v_idx]['cpu']
            physical_node_loads[p_idx]['ram'] += self.virtual_nodes[v_idx]['ram']
        
        # 计算每个物理节点的负载率
        cpu_loads = [loads['cpu'] / self.physical_nodes[p_idx]['cpu'] 
                    for p_idx, loads in physical_node_loads.items()]
        ram_loads = [loads['ram'] / self.physical_nodes[p_idx]['ram'] 
                    for p_idx, loads in physical_node_loads.items()]
        
        # 取最大负载率
        cpu_load = max(cpu_loads)
        ram_load = max(ram_loads)
        
        # 修改带宽负载计算
        bw_load = 0
        physical_link_loads = {}  # 记录每条物理链路的带宽负载
        
        # 初始化物理链路负载
        for p_link in self.physical_links:
            physical_link_loads[(p_link['src'], p_link['dst'])] = 0
            physical_link_loads[(p_link['dst'], p_link['src'])] = 0  # 双向记录
        
        # 计算带宽负载
        for v_link in self.virtual_links:
            src_phys = individual[v_link['src']]
            dst_phys = individual[v_link['dst']]
            
            # 如果源和目的节点映射到不同的物理节点，才需要消耗带宽
            if src_phys != dst_phys:
                found = False
                for p_link in self.physical_links:
                    # 检查物理链路是否连接这两个节点
                    if ((p_link['src'] == src_phys and p_link['dst'] == dst_phys) or 
                        (p_link['src'] == dst_phys and p_link['dst'] == src_phys)):
                        # 累加带宽负载
                        physical_link_loads[(p_link['src'], p_link['dst'])] += v_link['bw'] / p_link['bw']
                        physical_link_loads[(p_link['dst'], p_link['src'])] += v_link['bw'] / p_link['bw']
                        found = True
                        break
                if not found:
                    raise ValueError(f"找不到连接物理节点 {src_phys} 和 {dst_phys} 的物理链路")
        
        # 获取所有物理链路中的最大负载作为总体带宽负载
        bw_load = max(physical_link_loads.values()) if physical_link_loads else 0
        
        return cpu_load, ram_load, bw_load
    
    def select(self):
        # 根据适应度值和可行性对种群进行排序，并选择前半部分作为父代。
        # 首先按可行性值降序排序（-1比-0小，所以1排在前面）
        # 当可行性值相同时，再按适应度值降序排序
        self.population = sorted(self.population, key=lambda ind: (-self.fitness(ind)[1], -self.fitness(ind)[0]))
        return self.population[:self.pop_size//2]
    
    def crossover(self, parents):
        # 从两个父代中随机选择基因组合生成子代。
        offspring = []
        while len(offspring) < self.pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]
            offspring.append(child)
        return offspring
    
    def mutate(self, individuals, mutation_rate=0.01):
        # 以一定概率对个体进行变异，即随机改变某些基因。
        for individual in individuals:
            if random.random() < mutation_rate:
                idx = random.randint(0, len(individual)-1)
                individual[idx] = random.randint(0, len(self.physical_nodes)-1)
        return individuals
    
    def run(self, generations=1):
        for gen in range(generations):
            selected = self.select()
            offspring = self.crossover(selected)
            mutated = self.mutate(offspring)
            self.population = mutated
            
            best_ind = max(self.population, key=lambda ind: self.fitness(ind)[0])
            print(f"Generation {gen}: Best Fitness = {self.fitness(best_ind)[0]}")
            
        best_individual = max(self.population, key=lambda ind: self.fitness(ind)[0])
        return best_individual


# 示例使用
physical_nodes = [{'cpu': 32, 'ram': 64}, {'cpu': 32, 'ram': 64}, {'cpu': 32, 'ram': 64}]
virtual_nodes = [{'cpu': 8, 'ram': 16}, {'cpu': 16, 'ram': 32}, {'cpu': 16, 'ram': 32}]
# physical_links = {(0, 1): {'bw': 1000}, (1, 2): {'bw': 1000}, (0, 2): {'bw': 1000}}
physical_links = [{'src': 0, 'dst': 1, 'bw': 1000}, {'src': 1, 'dst': 2, 'bw': 1000}, {'src': 0,'dst': 2, 'bw': 1000}]
# virtual_links = {(0, 1): {'bw': 500}}
virtual_links = [{'src': 0, 'dst': 1, 'bw': 500}, {'src': 0, 'dst': 2, 'bw': 500}, {'src': 1, 'dst': 2, 'bw': 500}]

ga = NodeMappingGA(physical_nodes, virtual_nodes, physical_links, virtual_links)
best_solution = ga.run()
print("Best Solution:", best_solution)