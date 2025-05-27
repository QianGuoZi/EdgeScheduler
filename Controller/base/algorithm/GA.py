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
        return population
    
    def fitness(self, individual):
        # 计算适应度值和可行性值
        # 方法计算个体的适应度值。如果资源负载超过1，则使用指数惩罚函数调整适应度值。
        cpu_load, ram_load, bw_load = self.calculate_loads(individual)
        penalty_coefficients = {'cpu': 1.5, 'ram': 2.0, 'bw': 2.5}
        fitness_cpu = 2 - cpu_load if cpu_load <= 1 else np.exp(penalty_coefficients['cpu'] * (1 - cpu_load))
        fitness_ram = 2 - ram_load if ram_load <= 1 else np.exp(penalty_coefficients['ram'] * (1 - ram_load))
        fitness_bw = 2 - bw_load if bw_load <= 1 else np.exp(penalty_coefficients['bw'] * (1 - bw_load))
        feasibility = 1 if all(load <= 1 for load in [cpu_load, ram_load, bw_load]) else 0
        return fitness_cpu + fitness_ram + fitness_bw, feasibility
    
    def calculate_loads(self, individual):
        # 计算各种资源负载
        # 方法根据当前个体计算各资源的最大负载。
        cpu_load = max(np.sum([self.virtual_nodes[i]['cpu'] for i, p in enumerate(individual)], axis=0) / self.physical_nodes[p]['cpu'] for p in set(individual))
        ram_load = max(np.sum([self.virtual_nodes[i]['ram'] for i, p in enumerate(individual)], axis=0) / self.physical_nodes[p]['ram'] for p in set(individual))
        bw_load = 0
        for v_link in self.virtual_links:
            src_phys = individual[v_link['src']]
            dst_phys = individual[v_link['dst']]
            if src_phys != dst_phys:  # 如果虚拟链路的两端不在同一个物理节点上
                # 找到对应的物理链路
                found = False
                for p_link in self.physical_links:
                    if (p_link['src'] == src_phys and p_link['dst'] == dst_phys) or (p_link['src'] == dst_phys and p_link['dst'] == src_phys):
                        bw_load += v_link['bw'] / p_link['bw']
                        found = True
                        break
                if not found:
                    raise ValueError("No matching physical link found for virtual link")
        
        bw_load = bw_load if bw_load > 0 else 0
        
        return cpu_load, ram_load, bw_load
    
    def select(self):
        # 根据适应度值和可行性对种群进行排序，并选择前半部分作为父代。
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
    
    def run(self, generations=100):
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
virtual_nodes = [{'cpu': 8, 'ram': 16}, {'cpu': 16, 'ram': 32}]
# physical_links = {(0, 1): {'bw': 1000}, (1, 2): {'bw': 1000}, (0, 2): {'bw': 1000}}
physical_links = [{'src': 0, 'dst': 1, 'bw': 1000}, {'src': 1, 'dst': 2, 'bw': 1000}, {'src': 0,'dst': 2, 'bw': 1000}]
# virtual_links = {(0, 1): {'bw': 500}}
virtual_links = [{'src': 0, 'dst': 1, 'bw': 500}]

ga = NodeMappingGA(physical_nodes, virtual_nodes, physical_links, virtual_links)
best_solution = ga.run()
print("Best Solution:", best_solution)