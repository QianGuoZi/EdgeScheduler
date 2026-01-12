#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验自动化脚本
用于自动化执行多次实验并记录数据
"""

import os
import json
import time
import subprocess
import requests
import csv
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# 配置路径
CONTROLLER_DIR = '/home/qianguo/Edge-Scheduler/Controller'
WORKLOAD_CONFIG_PATH = os.path.join(CONTROLLER_DIR, 'workload_config.json')
TASK_LINKS_CONFIG_PATH = os.path.join(CONTROLLER_DIR, 'task_links', '1', 'links_range.json')
WORKLOAD_DATASETS_DIR = os.path.join(CONTROLLER_DIR, 'workload_datasets')
DATASETS_INDEX_PATH = os.path.join(WORKLOAD_DATASETS_DIR, 'datasets_index.json')
RUN_AGENTS_SCRIPT = os.path.join(CONTROLLER_DIR, '..', 'run_agents.sh')
MANAGE_CONTAINERS_SCRIPT = os.path.join(CONTROLLER_DIR, '..', 'manage_containers.sh')
TEST_PY_PATH = os.path.join(CONTROLLER_DIR, 'test.py')

# 服务器配置
CONTROLLER_IP = '222.201.187.50'
CONTROLLER_PORT = 3333
CONTROLLER_URL = f'http://{CONTROLLER_IP}:{CONTROLLER_PORT}'

# 实验配置（根结果目录，实际保存目录在类初始化时再细分）
EXPERIMENT_RESULTS_DIR = os.path.join(CONTROLLER_DIR, 'experiment_results')


class ExperimentAutomation:
    """实验自动化类"""
    
    def __init__(self, scheduler_method: str = "ppo"):
        self.results = []
        self.current_experiment_id = 0
        # 记录调度算法类型，用于在调用 Controller 接口时传递
        # 可选值示例: "ppo", "heuristic", "random"
        self.scheduler_method = scheduler_method

        # 创建根结果目录
        os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)

        # 为当前一次实验批次创建独立子目录：
        # 目录名格式：方法_年月日_时分，例如：ppo_20260112_1530
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_name = f"{self.scheduler_method}_{run_timestamp}"
        self.run_results_dir = os.path.join(EXPERIMENT_RESULTS_DIR, self.run_name)
        os.makedirs(self.run_results_dir, exist_ok=True)

        # 当前批次结果文件路径
        self.experiment_csv_file = os.path.join(self.run_results_dir, 'experiment_results.csv')
        self.experiment_json_file = os.path.join(self.run_results_dir, 'experiment_results.json')

        print(f"本次实验结果将保存在目录: {self.run_results_dir}")
    
    def load_datasets_index(self) -> Dict:
        """加载数据集索引"""
        with open(DATASETS_INDEX_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_workload_config(self, dataset_id: int) -> bool:
        """更新workload_config.json"""
        workload_file = os.path.join(WORKLOAD_DATASETS_DIR, f'workload_config_{dataset_id}.json')
        if not os.path.exists(workload_file):
            print(f"❌ 负载数据文件不存在: {workload_file}")
            return False
        
        with open(workload_file, 'r', encoding='utf-8') as f:
            workload_data = json.load(f)
        
        with open(WORKLOAD_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(workload_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 已更新 workload_config.json (数据集 {dataset_id})")
        return True
    
    def update_task_links_config(self, dataset_id: int) -> bool:
        """更新task_links/1/links_range.json"""
        task_links_file = os.path.join(WORKLOAD_DATASETS_DIR, f'task_links_config_{dataset_id}.json')
        if not os.path.exists(task_links_file):
            print(f"❌ 任务请求数据文件不存在: {task_links_file}")
            return False
        
        with open(task_links_file, 'r', encoding='utf-8') as f:
            task_links_data = json.load(f)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(TASK_LINKS_CONFIG_PATH), exist_ok=True)
        
        with open(TASK_LINKS_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(task_links_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 已更新 task_links/1/links_range.json (数据集 {dataset_id})")
        return True
    
    def start_agents(self) -> bool:
        """启动边缘节点的agent.py"""
        print("正在启动边缘节点agents...")
        try:
            result = subprocess.run(
                ['bash', RUN_AGENTS_SCRIPT, 'start'],
                cwd=os.path.dirname(RUN_AGENTS_SCRIPT),
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✓ Agents启动成功")
                return True
            else:
                print(f"⚠️  Agents启动可能有问题: {result.stderr}")
                return True  # 即使有警告也继续
        except subprocess.TimeoutExpired:
            print("❌ Agents启动超时")
            return False
        except Exception as e:
            print(f"❌ Agents启动失败: {e}")
            return False
    
    def start_test_py(self) -> Optional[subprocess.Popen]:
        """
        在后台启动test.py
        
        test.py 会启动一个 Flask 服务器，提供以下功能：
        - 接收任务请求 (/taskRequestFile)
        - 启动任务调度 (/startupTask)
        - 管理后台负载 (/bgload/add)
        - 其他 Controller API 端点
        """
        print("正在启动test.py (Flask 服务器)...")
        try:
            # 使用 subprocess.Popen 在后台启动，但保留输出以便调试
            process = subprocess.Popen(
                ['python3', TEST_PY_PATH],
                cwd=CONTROLLER_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1  # 行缓冲
            )
            # 给 test.py 一些时间启动 Flask 服务器
            print("  等待 Flask 服务器初始化...")
            time.sleep(3)
            
            # 检查进程是否还在运行
            if process.poll() is not None:
                # 进程已经退出
                stdout, stderr = process.communicate()
                print(f"❌ test.py 启动失败，进程已退出")
                if stderr:
                    print(f"   错误信息: {stderr.decode('utf-8', errors='ignore')[:500]}")
                return None
            
            print("✓ test.py 进程已启动 (Flask 服务器正在初始化...)")
            return process
        except Exception as e:
            print(f"❌ test.py启动失败: {e}")
            return None
    
    def wait_for_service(self, max_wait: int = 60) -> bool:
        """
        等待 test.py 启动的 Flask 服务就绪
        
        等待的服务：
        - test.py 启动的 Flask 服务器
        - 监听地址: http://222.201.187.50:3333
        - 这是 Controller 的主服务，提供调度、负载管理等 API 端点
        """
        print("等待 Flask 服务就绪 (test.py)...")
        print(f"  服务地址: {CONTROLLER_URL}")
        print(f"  最大等待时间: {max_wait}秒")
        
        for i in range(max_wait):
            try:
                # 尝试访问根路径或任何端点来检查服务是否启动
                response = requests.get(f"{CONTROLLER_URL}/", timeout=3)
                # 200, 404, 405 等状态码都说明服务在运行（只是路径可能不存在）
                if response.status_code in [200, 404, 405]:
                    print(f"✓ Flask 服务已就绪 (响应码: {response.status_code})")
                    return True
            except requests.exceptions.ConnectionError:
                # 连接错误说明服务还没启动
                pass
            except requests.exceptions.Timeout:
                # 超时也说明服务可能还没启动
                pass
            except Exception as e:
                # 其他异常，可能是服务在启动中
                pass
            
            time.sleep(1)
            if i % 5 == 0 and i > 0:
                print(f"  等待中... ({i}/{max_wait}秒)")
        
        print(f"❌ Flask 服务启动超时 ({max_wait}秒)")
        print(f"   请检查:")
        print(f"   1. test.py 是否正常启动")
        print(f"   2. 端口 3333 是否被占用")
        print(f"   3. 防火墙是否阻止了连接")
        print(f"   4. test.py 的输出是否有错误信息")
        return False
    
    def start_load(self) -> bool:
        """启动负载"""
        print("正在启动负载...")
        try:
            response = requests.post(
                f"{CONTROLLER_URL}/bgload/add",
                json={"use_config": True, "auto_launch": True},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if response.status_code == 200:
                print("✓ 负载启动命令已发送")
                # 等待负载启动完成
                print("等待负载启动完成...")
                time.sleep(10)  # 可以根据实际情况调整等待时间
                return True
            else:
                print(f"❌ 负载启动失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ 负载启动失败: {e}")
            return False
    
    def get_scheduling_result(self, task_id: int = 1) -> Optional[Dict]:
        """获取调度结果"""
        print(f"正在获取任务 {task_id} 的调度结果...")
        try:
            response = requests.get(
                f"{CONTROLLER_URL}/startupTask",
                params={
                    "taskId": task_id,
                    # 将实验中选择的调度算法传递给 Controller
                    # Controller 侧的 API 需要支持可选参数 scheduler_method
                    "scheduler_method": self.scheduler_method
                },
                timeout=60
            )
            if response.status_code == 200:
                print("✓ 调度请求已发送")
                # 等待调度完成
                print("等待调度完成...")
                time.sleep(15)  # 可以根据实际情况调整等待时间
                
                # 尝试获取调度结果（这里需要根据实际API调整）
                # 可能需要轮询任务状态或从日志中提取
                return {"status": "completed", "task_id": task_id}
            else:
                print(f"⚠️  调度请求返回: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"❌ 获取调度结果失败: {e}")
            return None
    
    def extract_metrics_from_logs(self, task_id: int = 1, test_process=None) -> Dict:
        """从日志或调度结果中提取指标"""
        metrics = {
            "load_balance_degree": None,
            "bandwidth_satisfaction": None,
            "composite_score": None
        }
        
        # 方法1: 尝试从test.py的标准输出中提取
        if test_process and test_process.stdout:
            try:
                # 读取最近的输出（尝试非阻塞读取）
                output = ""
                try:
                    import sys
                    if sys.platform != 'win32':
                        # 非阻塞读取（Linux/Mac）
                        import fcntl
                        flags = fcntl.fcntl(test_process.stdout, fcntl.F_GETFL)
                        fcntl.fcntl(test_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                        output = test_process.stdout.read().decode('utf-8', errors='ignore')
                    else:
                        # Windows平台
                        output = test_process.stdout.read().decode('utf-8', errors='ignore')
                except (ImportError, OSError):
                    # 如果fcntl不可用，尝试普通读取
                    try:
                        output = test_process.stdout.read().decode('utf-8', errors='ignore')
                    except:
                        pass
                
                # 尝试从输出中提取指标
                import re
                l_match = re.search(r'L[=:]\s*([0-9.]+)', output)
                dbw_match = re.search(r'D_BW[=:]\s*([0-9.]+)', output)
                
                if l_match:
                    metrics["load_balance_degree"] = float(l_match.group(1))
                if dbw_match:
                    metrics["bandwidth_satisfaction"] = float(dbw_match.group(1))
            except Exception as e:
                print(f"⚠️  从标准输出提取指标失败: {e}")
        
        # 方法2: 尝试从日志文件提取
        log_files = [
            os.path.join(CONTROLLER_DIR, 'logs', f'scheduling_{task_id}.log'),
            os.path.join(CONTROLLER_DIR, 'scheduling.log'),
            os.path.join(CONTROLLER_DIR, 'test.log'),
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    from extract_scheduling_metrics import MetricsExtractor
                    extractor = MetricsExtractor()
                    file_metrics = extractor.extract_from_log_file(log_file)
                    if file_metrics and file_metrics.get("load_balance_degree") is not None:
                        metrics.update(file_metrics)
                        break
                except Exception as e:
                    print(f"⚠️  从日志文件 {log_file} 提取指标失败: {e}")
        
        # 如果仍然没有提取到，返回默认值
        if metrics["load_balance_degree"] is None:
            metrics["load_balance_degree"] = 0.0
        if metrics["bandwidth_satisfaction"] is None:
            metrics["bandwidth_satisfaction"] = 0.0
        
        # 计算综合指标
        metrics["composite_score"] = self.calculate_composite_score(
            metrics["load_balance_degree"],
            metrics["bandwidth_satisfaction"]
        )
        
        return metrics
    
    def calculate_composite_score(self, l: float, d_bw: float, 
                                  l_weight: float = 0.5, dbw_weight: float = 0.5) -> float:
        """计算联合参数（综合指标）"""
        import math
        if math.isnan(l) or math.isnan(d_bw):
            return float('nan')
        # L越小越好，D_BW越大越好
        # 转换为统一的方向：越大越好
        normalized_l = max(0, 1 - l)  # 假设L在[0,1]范围内
        return l_weight * normalized_l + dbw_weight * d_bw
    
    def cleanup_containers_and_agents(self):
        """清理容器和停止agents"""
        print("\n" + "="*60)
        print("开始清理：停止容器和agents")
        print("="*60)
        
        # 1. 停止并删除负载容器（stress:latest）
        print("\n1. 清理负载容器 (stress:latest)...")
        try:
            result = subprocess.run(
                ['bash', MANAGE_CONTAINERS_SCRIPT, 'stop', 'stress:latest'],
                cwd=os.path.dirname(MANAGE_CONTAINERS_SCRIPT),
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✓ 负载容器已停止")
            else:
                print(f"⚠️  停止负载容器时出现问题: {result.stderr}")
        except Exception as e:
            print(f"⚠️  停止负载容器失败: {e}")
        
        try:
            # 自动确认删除操作（输入 y）
            result = subprocess.run(
                ['bash', MANAGE_CONTAINERS_SCRIPT, 'rm', 'stress:latest'],
                cwd=os.path.dirname(MANAGE_CONTAINERS_SCRIPT),
                input='y\n',  # 自动输入 y 确认删除
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✓ 负载容器已删除")
            else:
                print(f"⚠️  删除负载容器时出现问题: {result.stderr}")
        except Exception as e:
            print(f"⚠️  删除负载容器失败: {e}")
        
        # 2. 停止并删除任务容器（task1:v1.0）
        print("\n2. 清理任务容器 (task1:v1.0)...")
        try:
            result = subprocess.run(
                ['bash', MANAGE_CONTAINERS_SCRIPT, 'stop', 'task1:v1.0'],
                cwd=os.path.dirname(MANAGE_CONTAINERS_SCRIPT),
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✓ 任务容器已停止")
            else:
                print(f"⚠️  停止任务容器时出现问题: {result.stderr}")
        except Exception as e:
            print(f"⚠️  停止任务容器失败: {e}")
        
        try:
            # 自动确认删除操作（输入 y）
            result = subprocess.run(
                ['bash', MANAGE_CONTAINERS_SCRIPT, 'rm', 'task1:v1.0'],
                cwd=os.path.dirname(MANAGE_CONTAINERS_SCRIPT),
                input='y\n',  # 自动输入 y 确认删除
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✓ 任务容器已删除")
            else:
                print(f"⚠️  删除任务容器时出现问题: {result.stderr}")
        except Exception as e:
            print(f"⚠️  删除任务容器失败: {e}")
        
        # 3. 停止agents
        print("\n3. 停止边缘设备agents...")
        try:
            result = subprocess.run(
                ['bash', RUN_AGENTS_SCRIPT, 'stop'],
                cwd=os.path.dirname(RUN_AGENTS_SCRIPT),
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✓ Agents已停止")
            else:
                print(f"⚠️  停止agents时出现问题: {result.stderr}")
        except Exception as e:
            print(f"⚠️  停止agents失败: {e}")
        
        print("\n✓ 清理完成")
        print("="*60)
    
    def run_single_experiment(self, dataset_id: int, experiment_id: int) -> Dict:
        """运行单次实验"""
        print(f"\n{'='*60}")
        print(f"开始实验 #{experiment_id} - 数据集 {dataset_id}")
        print(f"{'='*60}")
        
        experiment_result = {
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "load_balance_degree": None,
            "bandwidth_satisfaction": None,
            "composite_score": None,
            "error": None
        }
        
        try:
            # 1. 更新配置文件
            if not self.update_workload_config(dataset_id):
                experiment_result["error"] = "更新workload_config.json失败"
                return experiment_result
            
            if not self.update_task_links_config(dataset_id):
                experiment_result["error"] = "更新task_links_config.json失败"
                return experiment_result
            
            # 2. 启动agents
            if not self.start_agents():
                experiment_result["error"] = "启动agents失败"
                return experiment_result
            
            # 3. 启动test.py
            test_process = self.start_test_py()
            if test_process is None:
                experiment_result["error"] = "启动test.py失败"
                return experiment_result
            
            try:
                # 4. 等待服务就绪
                if not self.wait_for_service():
                    experiment_result["error"] = "服务启动超时"
                    return experiment_result
                
                # 5. 启动负载
                if not self.start_load():
                    experiment_result["error"] = "启动负载失败"
                    return experiment_result
                
                # 6. 获取调度结果
                scheduling_result = self.get_scheduling_result(task_id=1)
                if scheduling_result is None:
                    experiment_result["error"] = "获取调度结果失败"
                    return experiment_result
                
                # 7. 提取指标
                metrics = self.extract_metrics_from_logs(task_id=1, test_process=test_process)
                experiment_result.update({
                    "status": "success",
                    "load_balance_degree": metrics.get("load_balance_degree"),
                    "bandwidth_satisfaction": metrics.get("bandwidth_satisfaction"),
                    "composite_score": metrics.get("composite_score")
                })
                
                print(f"\n✓ 实验 #{experiment_id} 完成")
                print(f"  负载均衡度 (L): {experiment_result['load_balance_degree']}")
                print(f"  带宽满足度 (D_BW): {experiment_result['bandwidth_satisfaction']}")
                print(f"  综合指标: {experiment_result['composite_score']}")
                
            finally:
                # 清理：停止test.py
                if test_process:
                    test_process.terminate()
                    test_process.wait(timeout=10)
                
                # 清理容器和agents（每次实验后）
                self.cleanup_containers_and_agents()
        
        except Exception as e:
            experiment_result["error"] = str(e)
            print(f"❌ 实验 #{experiment_id} 失败: {e}")
        
        return experiment_result
    
    def save_results(self):
        """保存实验结果"""
        # 保存为CSV
        if self.results:
            with open(self.experiment_csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['experiment_id', 'dataset_id', 'timestamp', 'status',
                             'load_balance_degree', 'bandwidth_satisfaction', 'composite_score', 'error']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result)
            print(f"\n✓ 结果已保存到: {self.experiment_csv_file}")
        
        # 保存为JSON
        with open(self.experiment_json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✓ 结果已保存到: {self.experiment_json_file}")
    
    def run_experiments(self, dataset_ids: List[int], num_runs_per_dataset: int = 1):
        """运行多次实验"""
        print(f"\n开始批量实验")
        print(f"数据集: {dataset_ids}")
        print(f"每个数据集运行次数: {num_runs_per_dataset}")
        print(f"总实验次数: {len(dataset_ids) * num_runs_per_dataset}")
        
        experiment_id = 1
        for dataset_id in dataset_ids:
            for run in range(num_runs_per_dataset):
                result = self.run_single_experiment(dataset_id, experiment_id)
                self.results.append(result)
                experiment_id += 1
                
                # 每次实验后保存一次（防止数据丢失）
                self.save_results()
                
                # 实验间隔
                if experiment_id <= len(dataset_ids) * num_runs_per_dataset:
                    print(f"\n等待 {5} 秒后开始下一个实验...")
                    time.sleep(5)
        
        print(f"\n{'='*60}")
        print("所有实验完成！")
        print(f"{'='*60}")
        
        # 最终清理
        self.cleanup_containers_and_agents()
        
        # 保存结果
        self.save_results()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实验自动化脚本')
    parser.add_argument('--datasets', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                       help='要使用的数据集ID列表 (默认: 1 2 3 4 5)')
    parser.add_argument('--runs', type=int, default=1,
                       help='每个数据集运行的次数 (默认: 1)')
    parser.add_argument('--method', type=str, default='ppo',
                       choices=['ppo', 'heuristic', 'random', 'ppo_mapping', 'ppo_balance'],
                       help='调度算法类型: ppo / heuristic / random / ppo_mapping (默认: ppo)')
    
    args = parser.parse_args()
    
    # 根据命令行参数选择调度算法，传入 ExperimentAutomation
    automation = ExperimentAutomation(scheduler_method=args.method)
    automation.run_experiments(args.datasets, args.runs)


if __name__ == '__main__':
    main()
