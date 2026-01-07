#!/bin/bash

# 从 100.68.2.1 同步 agent.py 到其他所有设备
# 使用方法: ./sync_agent.sh

# 源设备
SOURCE_IP="100.68.2.1"
SOURCE_USER="nvidia"
SOURCE_PASS="nvidia"
SOURCE_PATH="/home/nvidia/qianguo/worker/agent.py"

# 设备组1: 100.68.1.x
GROUP1_IPS=("100.68.1.3" "100.68.1.4" "100.68.1.5" "100.68.1.6")
GROUP1_USER="nano"
GROUP1_PASS="123456a?"
GROUP1_PATH="/home/nano/qianguo/worker/agent.py"

# 设备组2: 100.68.2.x (不包括源设备)
GROUP2_IPS=("100.68.2.2" "100.68.2.3" "100.68.2.4" "100.68.2.5" "100.68.2.6")
GROUP2_USER="nvidia"
GROUP2_PASS="nvidia"
GROUP2_PATH="/home/nvidia/qianguo/worker/agent.py"

# 本地临时文件
LOCAL_TEMP="/tmp/agent.py"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查sshpass是否安装
check_sshpass() {
    if ! command -v sshpass &> /dev/null; then
        echo -e "${RED}错误: sshpass 未安装${NC}"
        echo "请先安装 sshpass: sudo apt install sshpass"
        exit 1
    fi
}

# 从源设备下载 agent.py
download_from_source() {
    echo -e "${YELLOW}从 ${SOURCE_IP} 下载 agent.py...${NC}"
    
    if sshpass -p "$SOURCE_PASS" scp -o StrictHostKeyChecking=no "${SOURCE_USER}@${SOURCE_IP}:${SOURCE_PATH}" "$LOCAL_TEMP" 2>/dev/null; then
        echo -e "${GREEN}下载成功${NC}"
        echo "文件大小: $(ls -lh $LOCAL_TEMP | awk '{print $5}')"
        return 0
    else
        echo -e "${RED}下载失败${NC}"
        return 1
    fi
}

# 上传到单个设备
upload_to_device() {
    local ip=$1
    local user=$2
    local pass=$3
    local remote_path=$4
    
    echo -e "${YELLOW}[${ip}]${NC} 正在同步..."
    
    # 确保目标目录存在
    local dir_path=$(dirname "$remote_path")
    sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "mkdir -p ${dir_path}" 2>/dev/null
    
    # 上传文件
    if sshpass -p "$pass" scp -o StrictHostKeyChecking=no "$LOCAL_TEMP" "${user}@${ip}:${remote_path}" 2>/dev/null; then
        echo -e "${GREEN}[${ip}]${NC} 同步成功"
    else
        echo -e "${RED}[${ip}]${NC} 同步失败"
    fi
}

# 主函数
main() {
    check_sshpass
    
    echo "========================================"
    echo "同步 agent.py 到所有设备"
    echo "源设备: ${SOURCE_IP}"
    echo "========================================"
    echo ""
    
    # 从源设备下载
    if ! download_from_source; then
        exit 1
    fi
    
    echo ""
    echo "--- 同步到设备组1 (100.68.1.x) ---"
    for ip in "${GROUP1_IPS[@]}"; do
        upload_to_device "$ip" "$GROUP1_USER" "$GROUP1_PASS" "$GROUP1_PATH"
    done
    
    echo ""
    echo "--- 同步到设备组2 (100.68.2.x) ---"
    for ip in "${GROUP2_IPS[@]}"; do
        upload_to_device "$ip" "$GROUP2_USER" "$GROUP2_PASS" "$GROUP2_PATH"
    done
    
    # 清理临时文件
    rm -f "$LOCAL_TEMP"
    
    echo ""
    echo "========================================"
    echo "同步完成!"
    echo "========================================"
}

main

