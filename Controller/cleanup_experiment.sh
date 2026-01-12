#!/bin/bash
# -*- coding: utf-8 -*-
"""
实验清理脚本
用于清理实验后的容器和停止agents
"""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 脚本路径
MANAGE_CONTAINERS_SCRIPT="$PROJECT_ROOT/manage_containers.sh"
RUN_AGENTS_SCRIPT="$PROJECT_ROOT/run_agents.sh"

echo "========================================"
echo "实验清理脚本"
echo "========================================"
echo ""

# 检查脚本是否存在
if [ ! -f "$MANAGE_CONTAINERS_SCRIPT" ]; then
    echo -e "${RED}错误: manage_containers.sh 不存在${NC}"
    exit 1
fi

if [ ! -f "$RUN_AGENTS_SCRIPT" ]; then
    echo -e "${RED}错误: run_agents.sh 不存在${NC}"
    exit 1
fi

# 1. 停止并删除负载容器（stress:latest）
echo -e "${YELLOW}1. 清理负载容器 (stress:latest)...${NC}"
bash "$MANAGE_CONTAINERS_SCRIPT" stop stress:latest
echo "y" | bash "$MANAGE_CONTAINERS_SCRIPT" rm stress:latest
echo ""

# 2. 停止并删除任务容器（task1:v1.0）
echo -e "${YELLOW}2. 清理任务容器 (task1:v1.0)...${NC}"
bash "$MANAGE_CONTAINERS_SCRIPT" stop task1:v1.0
echo "y" | bash "$MANAGE_CONTAINERS_SCRIPT" rm task1:v1.0
echo ""

# 3. 停止agents
echo -e "${YELLOW}3. 停止边缘设备agents...${NC}"
bash "$RUN_AGENTS_SCRIPT" stop
echo ""

echo -e "${GREEN}✓ 清理完成！${NC}"
echo "========================================"
