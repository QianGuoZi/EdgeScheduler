#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
两阶段PPO网络调度器快速启动脚本
"""

import os
import sys
import torch
import numpy as np
from train_two_stage_ppo import TwoStagePPOTrainer

def main():
    """主函数"""
    print("🎯 两阶段PPO网络调度器快速启动")
    print("=" * 60)
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 创建训练器
    print("\n📋 创建训练器...")
    trainer = TwoStagePPOTrainer(
        # 节点数量范围
        num_physical_nodes_range=(5, 8),
        max_virtual_nodes_range=(3, 6),
        bandwidth_levels=10,
        
        # 物理资源范围
        physical_cpu_range=(50.0, 200.0),
        physical_memory_range=(100.0, 400.0),
        physical_bandwidth_range=(100.0, 1000.0),
        
        # 虚拟资源范围
        virtual_cpu_range=(10.0, 50.0),
        virtual_memory_range=(20.0, 100.0),
        virtual_bandwidth_range=(10.0, 200.0),
        
        # 网络连接概率
        physical_connectivity_prob=0.3,
        virtual_connectivity_prob=0.4,
        
        # 训练参数
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        
        # 文件管理
        model_dir="models",
        stats_dir="stats"
    )
    
    print("✅ 训练器创建成功")
    
    # 开始训练
    print("\n🚀 开始训练...")
    try:
        trainer.train(
            num_episodes=100,  # 快速训练100个episodes
            save_interval=50,   # 每50个episodes保存一次
            eval_interval=25    # 每25个episodes评估一次
        )
        print("✅ 训练完成")
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        return
    
    # 测试智能体
    print("\n🧪 测试智能体...")
    try:
        test_stats = trainer.test_agent(num_test_episodes=5)
        
        print(f"\n📊 测试结果:")
        print(f"   平均奖励: {np.mean(test_stats['rewards']):.3f}")
        print(f"   平均约束违反: {np.mean(test_stats['constraint_violations']):.2f}")
        print(f"   有效动作率: {test_stats['valid_actions']/test_stats['total_actions']:.2%}")
        
        print("✅ 测试完成")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return
    
    # 保存最终模型
    print("\n💾 保存最终模型...")
    try:
        trainer.save_model(episode=100)
        trainer.save_training_stats(episode=100)
        print("✅ 模型保存完成")
    except Exception as e:
        print(f"❌ 保存模型时出现错误: {e}")
        return
    
    print("\n🎉 快速启动完成！")
    print("\n📁 生成的文件:")
    print(f"   模型文件: {trainer.model_dir}/")
    print(f"   统计文件: {trainer.stats_dir}/")
    print(f"   训练曲线: {trainer.stats_dir}/training_curves.png")
    
    print("\n🔧 下一步:")
    print("   1. 查看训练曲线图了解训练效果")
    print("   2. 调整参数进行更长时间的训练")
    print("   3. 修改奖励函数权重优化性能")
    print("   4. 添加新的约束条件")

def quick_test():
    """快速测试函数"""
    print("🧪 快速测试两阶段PPO")
    print("=" * 40)
    
    # 导入测试模块
    try:
        from simple_two_stage_test import test_simple_two_stage
        test_simple_two_stage()
        print("✅ 简化测试通过")
    except Exception as e:
        print(f"❌ 简化测试失败: {e}")
        return False
    
    try:
        from two_stage_actor_design import test_two_stage_actor
        test_two_stage_actor()
        print("✅ 架构测试通过")
    except Exception as e:
        print(f"❌ 架构测试失败: {e}")
        return False
    
    try:
        from two_stage_environment import test_two_stage_environment
        test_two_stage_environment()
        print("✅ 环境测试通过")
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False
    
    print("🎉 所有测试通过！")
    return True

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 运行测试
        success = quick_test()
        if success:
            print("\n🚀 测试通过，可以开始训练！")
            print("运行: python quick_start.py")
        else:
            print("\n❌ 测试失败，请检查环境配置")
    else:
        # 运行快速启动
        main() 