#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试scheduler_single_test在10个物理节点环境下的调度过程
"""

import sys
import os
import json
import tempfile
import shutil
from datetime import datetime

sys.path.append('/home/qianguo/Edge-Scheduler')

# 模拟Controller和Emulator类
class MockEmulator:
    def __init__(self, name, cpu, ram, cpu_used=0, ram_used=0):
        self.nameW = name
        self.cpu = cpu
        self.ram = ram
        self.cpuPreMap = cpu_used
        self.ramPreMap = ram_used

class MockController:
    def __init__(self, num_nodes=10):
        """创建指定数量的物理节点"""
        self.emulator = {}
        
        # 创建10个物理节点，资源配置各不相同
        node_configs = [
            ('node_01', 100, 200, 10, 20),   # 高性能节点
            ('node_02', 80, 150, 15, 30),    # 中等性能节点
            ('node_03', 120, 250, 20, 50),   # 高性能节点
            ('node_04', 60, 120, 5, 15),     # 低性能节点
            ('node_05', 90, 180, 25, 40),    # 中等性能节点
            ('node_06', 110, 220, 30, 60),   # 高性能节点
            ('node_07', 70, 140, 10, 25),    # 中低性能节点
            ('node_08', 95, 190, 20, 35),    # 中等性能节点
            ('node_09', 85, 170, 15, 30),    # 中等性能节点
            ('node_10', 130, 280, 40, 80)    # 最高性能节点
        ]
        
        for name, cpu, ram, cpu_used, ram_used in node_configs[:num_nodes]:
            self.emulator[name] = MockEmulator(name, cpu, ram, cpu_used, ram_used)
        
        # 创建全连接的带宽网络（每对节点之间都有连接）
        self.bandwidth_data = []
        nodes = list(self.emulator.keys())
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j:  # 避免重复连接
                    # 随机生成带宽配置，模拟真实网络环境
                    base_bw = 1000 + (i + j) * 100  # 基础带宽1000-2800 Mbps
                    used_bw = 50 + (i * j) % 200     # 已用带宽50-250 Mbps
                    self.bandwidth_data.append((node1, node2, base_bw, used_bw))
    
    def iter_bandwidth(self):
        """模拟带宽迭代器"""
        for emu1, emu2, bw, used_bw in self.bandwidth_data:
            yield emu1, emu2, bw, used_bw

def create_test_task_links(task_id, num_virtual_nodes=6):
    """创建测试用的虚拟网络拓扑文件"""
    # 创建临时任务目录
    task_dir = f'/home/qianguo/Edge-Scheduler/Controller/task_links/{task_id}'
    os.makedirs(task_dir, exist_ok=True)
    
    # 生成虚拟网络拓扑
    # 创建一个包含num_virtual_nodes个虚拟节点的网络
    virtual_nodes = [f"vn_{i+1}" for i in range(num_virtual_nodes)]
    
    links_data = {}
    
    # 为每个虚拟节点创建连接
    for i, node in enumerate(virtual_nodes):
        links_data[node] = []
        
        # 每个节点连接到2-3个其他节点
        for j in range(num_virtual_nodes):
            if i != j and (j == (i+1) % num_virtual_nodes or j == (i+2) % num_virtual_nodes):
                # 随机生成带宽需求
                bw_min = 5 + (i + j) % 10  # 5-15 Mbps
                bw_max = bw_min + 5 + (i * j) % 10  # 比最小带宽高5-15 Mbps
                
                links_data[node].append({
                    "dest": virtual_nodes[j],
                    "bw_min": f"{bw_min}mbps",
                    "bw_max": f"{bw_max}mbps"
                })
    
    # 保存到文件
    links_file = os.path.join(task_dir, 'links_range.json')
    with open(links_file, 'w', encoding='utf-8') as f:
        json.dump(links_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 创建测试任务文件: {links_file}")
    print(f"🔗 虚拟节点数量: {num_virtual_nodes}")
    print(f"🌐 虚拟链路数量: {sum(len(links) for links in links_data.values())}")
    
    return links_file

def test_scheduler_10_nodes():
    """测试scheduler在10个物理节点环境下的调度过程"""
    print("🧪 测试scheduler在10个物理节点环境下的调度过程")
    print("="*80)
    
    try:
        # 导入调度器
        from Controller.base.scheduler_single_test import SingleTestScheduler
        
        print("步骤1: 创建10个物理节点的模拟环境")
        print("-"*50)
        
        # 创建包含10个物理节点的模拟控制器
        controller = MockController(num_nodes=10)
        
        print(f"✅ 创建了 {len(controller.emulator)} 个物理节点:")
        for name, emu in controller.emulator.items():
            available_cpu = emu.cpu - emu.cpuPreMap
            available_ram = emu.ram - emu.ramPreMap
            print(f"   {name}: CPU {available_cpu}/{emu.cpu}, RAM {available_ram}/{emu.ram}")
        
        print(f"🌐 网络连接数量: {len(controller.bandwidth_data)} 条链路")
        
        print("\n步骤2: 初始化PPO调度器")
        print("-"*50)
        
        # 创建调度器
        scheduler = SingleTestScheduler(controller)
        
        print("✅ 调度器初始化成功")
        print(f"📡 PPO模型状态: {'已加载' if scheduler.ppo_agent else '未加载'}")
        if hasattr(scheduler, 'ppo_env_config'):
            print(f"🔧 PPO环境配置: {len(scheduler.ppo_env_config)} 个参数")
        if hasattr(scheduler, 'ppo_agent_config'):
            print(f"🤖 PPO代理配置: {len(scheduler.ppo_agent_config)} 个参数")
        
        print("\n步骤3: 创建测试任务")
        print("-"*50)
        
        # 测试不同规模的虚拟网络
        test_cases = [
            {"task_id": 1001, "virtual_nodes": 3, "description": "小规模任务(3个虚拟节点)"},
            {"task_id": 1002, "virtual_nodes": 5, "description": "中等规模任务(5个虚拟节点)"},
            {"task_id": 1003, "virtual_nodes": 8, "description": "大规模任务(8个虚拟节点)"}
        ]
        
        results = []
        
        for test_case in test_cases:
            task_id = test_case["task_id"]
            num_virtual_nodes = test_case["virtual_nodes"]
            description = test_case["description"]
            
            print(f"\n🚀 测试案例: {description}")
            print(f"📋 任务ID: {task_id}, 虚拟节点数: {num_virtual_nodes}")
            
            # 创建测试任务文件
            links_file = create_test_task_links(task_id, num_virtual_nodes)
            
            # 记录调度前的资源状态
            total_available_cpu = sum(emu.cpu - emu.cpuPreMap for emu in controller.emulator.values())
            total_available_ram = sum(emu.ram - emu.ramPreMap for emu in controller.emulator.values())
            
            print(f"📊 调度前可用资源: CPU {total_available_cpu}, RAM {total_available_ram}")
            
            # 执行调度
            start_time = datetime.now()
            allocation = scheduler.resource_schedule(task_id)
            end_time = datetime.now()
            
            scheduling_time = (end_time - start_time).total_seconds()
            
            print(f"⏱️  调度耗时: {scheduling_time:.3f} 秒")
            print(f"🎯 调度结果: {len(allocation)} 个虚拟节点已映射")
            
            if allocation:
                print("📋 详细分配结果:")
                total_allocated_cpu = 0
                total_allocated_ram = 0
                
                for virtual_node, mapping in allocation.items():
                    print(f"   {virtual_node} -> {mapping['emulator']} "
                          f"(CPU: {mapping['cpu']}, RAM: {mapping['ram']})")
                    total_allocated_cpu += mapping['cpu']
                    total_allocated_ram += mapping['ram']
                
                print(f"📈 总分配资源: CPU {total_allocated_cpu}, RAM {total_allocated_ram}")
                
                # 计算资源利用率
                cpu_utilization = (total_allocated_cpu / total_available_cpu) * 100 if total_available_cpu > 0 else 0
                ram_utilization = (total_allocated_ram / total_available_ram) * 100 if total_available_ram > 0 else 0
                
                print(f"📊 资源利用率: CPU {cpu_utilization:.1f}%, RAM {ram_utilization:.1f}%")
                
                success = True
            else:
                print("⚠️  没有成功分配任何虚拟节点")
                success = False
            
            # 记录结果
            results.append({
                'task_id': task_id,
                'description': description,
                'virtual_nodes': num_virtual_nodes,
                'allocated_nodes': len(allocation),
                'success': success,
                'scheduling_time': scheduling_time
            })
            
            print("-" * 60)
        
        print("\n步骤4: 生成性能报告")
        print("-"*50)
        
        # 生成总结报告
        scheduler.generate_summary_report()
        
        print("\n📊 测试结果汇总:")
        print("="*80)
        
        for result in results:
            status = "✅ 成功" if result['success'] else "❌ 失败"
            print(f"{result['description']}: {status}")
            print(f"   分配节点: {result['allocated_nodes']}/{result['virtual_nodes']}")
            print(f"   调度耗时: {result['scheduling_time']:.3f} 秒")
            print()
        
        # 计算总体统计
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        avg_scheduling_time = sum(r['scheduling_time'] for r in results) / total_tests
        
        print(f"🎯 总体统计:")
        print(f"   测试案例总数: {total_tests}")
        print(f"   成功案例数: {successful_tests}")
        print(f"   成功率: {successful_tests/total_tests*100:.1f}%")
        print(f"   平均调度耗时: {avg_scheduling_time:.3f} 秒")
        
        print("\n✅ 10个物理节点调度测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """清理测试产生的临时文件"""
    test_task_ids = [1001, 1002, 1003]
    for task_id in test_task_ids:
        task_dir = f'/home/qianguo/Edge-Scheduler/Controller/task_links/{task_id}'
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
            print(f"🧹 清理测试文件: {task_dir}")

if __name__ == "__main__":
    print("🎯 开始10个物理节点的scheduler调度测试")
    print("="*80)
    
    try:
        success = test_scheduler_10_nodes()
        
        if success:
            print("\n🎊 测试通过! scheduler在10个物理节点环境下运行正常")
        else:
            print("\n💥 测试失败，请检查错误信息")
    
    finally:
        # 清理测试文件
        print("\n🧹 清理测试文件...")
        cleanup_test_files()
        print("✅ 清理完成")
