#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_simple_two_stage():
    """简化的两阶段测试"""
    print("🧪 简化两阶段测试")
    print("=" * 60)
    
    # 测试参数
    physical_node_dim = 4
    virtual_node_dim = 3
    num_physical_nodes = 5
    max_virtual_nodes = 4
    bandwidth_levels = 10
    hidden_dim = 128
    
    print(f"✅ 参数设置完成")
    print(f"   物理节点维度: {physical_node_dim}")
    print(f"   虚拟节点维度: {virtual_node_dim}")
    print(f"   物理节点数: {num_physical_nodes}")
    print(f"   最大虚拟节点数: {max_virtual_nodes}")
    print(f"   带宽等级数: {bandwidth_levels}")
    print(f"   隐藏层维度: {hidden_dim}")
    
    # 创建测试数据
    physical_features = torch.randn(num_physical_nodes, physical_node_dim)
    virtual_features = torch.randn(max_virtual_nodes, virtual_node_dim)
    
    print(f"\n📊 测试数据创建完成")
    print(f"   物理特征: {physical_features.shape}")
    print(f"   虚拟特征: {virtual_features.shape}")
    
    # 模拟映射Actor的输出
    actual_virtual_nodes = virtual_features.size(0)
    mapping_logits = torch.randn(actual_virtual_nodes, num_physical_nodes)
    constraint_scores = torch.sigmoid(torch.randn(actual_virtual_nodes, num_physical_nodes))  # 确保在[0,1]范围内
    
    print(f"\n🎯 映射Actor输出:")
    print(f"   映射logits: {mapping_logits.shape}")
    print(f"   约束分数: {constraint_scores.shape}")
    
    # 应用约束
    combined_logits = mapping_logits + torch.log(constraint_scores + 1e-8)
    mapping_probs = F.softmax(combined_logits, dim=-1)
    
    print(f"   组合logits: {combined_logits.shape}")
    print(f"   映射概率: {mapping_probs.shape}")
    
    # 采样映射动作
    mapping_dist = torch.distributions.Categorical(mapping_probs)
    mapping_action = mapping_dist.sample()
    mapping_log_prob = mapping_dist.log_prob(mapping_action)
    
    print(f"\n📋 映射结果:")
    print(f"   映射动作: {mapping_action}")
    print(f"   映射log概率: {mapping_log_prob}")
    
    # 模拟带宽Actor的输出
    num_links = actual_virtual_nodes * (actual_virtual_nodes - 1) // 2
    bandwidth_logits = torch.randn(num_links, bandwidth_levels)
    bandwidth_constraint_scores = torch.sigmoid(torch.randn(num_links, bandwidth_levels))  # 确保在[0,1]范围内
    
    print(f"\n🎯 带宽Actor输出:")
    print(f"   链路数: {num_links}")
    print(f"   带宽logits: {bandwidth_logits.shape}")
    print(f"   带宽约束分数: {bandwidth_constraint_scores.shape}")
    
    # 应用带宽约束
    combined_bandwidth_logits = bandwidth_logits + torch.log(bandwidth_constraint_scores + 1e-8)
    bandwidth_probs = F.softmax(combined_bandwidth_logits, dim=-1)
    
    # 采样带宽动作
    bandwidth_dist = torch.distributions.Categorical(bandwidth_probs)
    bandwidth_action = bandwidth_dist.sample()
    bandwidth_log_prob = bandwidth_dist.log_prob(bandwidth_action)
    
    print(f"\n📋 带宽结果:")
    print(f"   带宽动作: {bandwidth_action}")
    print(f"   带宽log概率: {bandwidth_log_prob}")
    
    # 计算总log概率
    total_log_prob = mapping_log_prob.sum() + bandwidth_log_prob.sum()
    
    print(f"\n📊 总结:")
    print(f"   总log概率: {total_log_prob:.4f}")
    print(f"   映射动作: {mapping_action.tolist()}")
    print(f"   带宽动作: {bandwidth_action.tolist()}")
    
    print(f"\n🎯 简化两阶段测试完成！")

if __name__ == "__main__":
    test_simple_two_stage() 