#!/bin/bash

# 批量更新所有emulator上的agent.py文件
# 使用方法: ./update_agent.sh

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 本地agent.py文件路径
LOCAL_AGENT_FILE="/home/qianguo/Edge-Scheduler/Worker/agent.py"

# 设备组1: 100.68.1.x (需要sudo运行docker)
GROUP1_IPS=("100.68.1.3" "100.68.1.4" "100.68.1.5" "100.68.1.6")
GROUP1_USER="nano"
GROUP1_PASS="123456a?"
GROUP1_REMOTE_PATH="/home/nano/qianguo/worker/agent.py"

# 设备组2: 100.68.2.x
GROUP2_IPS=("100.68.2.1" "100.68.2.2" "100.68.2.3" "100.68.2.4" "100.68.2.5" "100.68.2.6")
GROUP2_USER="nvidia"
GROUP2_PASS="nvidia"
GROUP2_REMOTE_PATH="/home/nvidia/qianguo/worker/agent.py"

# 检查本地文件是否存在
if [ ! -f "$LOCAL_AGENT_FILE" ]; then
    echo -e "${RED}错误: 本地agent.py文件不存在: $LOCAL_AGENT_FILE${NC}"
    exit 1
fi

echo "=========================================="
echo "开始批量更新agent.py文件"
echo "源文件: $LOCAL_AGENT_FILE"
echo "文件大小: $(ls -lh $LOCAL_AGENT_FILE | awk '{print $5}')"
echo "=========================================="

check_sshpass() {
    if ! command -v sshpass >/dev/null 2>&1; then
        echo -e "${RED}错误: sshpass 未安装${NC}"
        echo "请先安装: sudo apt install sshpass"
        exit 1
    fi
}

# 远程复制文件
remote_copy() {
    local src=$1
    local ip=$2
    local user=$3
    local pass=$4
    local dest=$5

    sshpass -p "$pass" scp -q -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$src" "${user}@${ip}:${dest}" 2>/dev/null
}

# 确保远程目录存在
ensure_remote_dir() {
    local ip=$1
    local user=$2
    local pass=$3
    local file_path=$4
    
    local dir_path=$(dirname "$file_path")
    sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "mkdir -p ${dir_path}" 2>/dev/null
}

# 更新单个设备
update_device() {
    local name=$1
    local ip=$2
    local user=$3
    local pass=$4
    local remote_path=$5

    echo ""
    echo -e "${YELLOW}正在更新: $name ($ip)${NC}"

    # 确保目标目录存在
    echo "  - 检查目标目录..."
    ensure_remote_dir "$ip" "$user" "$pass" "$remote_path"

    # 复制文件
    echo "  - 复制agent.py到远程机器..."
    if remote_copy "$LOCAL_AGENT_FILE" "$ip" "$user" "$pass" "$remote_path"; then
        echo -e "${GREEN}  - 更新成功: $name${NC}"
        return 0
    else
        echo -e "${RED}  - 更新失败: $name${NC}"
        return 1
    fi
}

check_sshpass

# 统计成功和失败数量
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_EMULATORS=""

# 处理设备组1
echo ""
echo "--- 设备组1 (100.68.1.x) ---"
for ip in "${GROUP1_IPS[@]}"; do
    name="emulator-${ip##*.}"
    if update_device "$name" "$ip" "$GROUP1_USER" "$GROUP1_PASS" "$GROUP1_REMOTE_PATH"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_EMULATORS="$FAILED_EMULATORS $name"
    fi
done

# 处理设备组2
echo ""
echo "--- 设备组2 (100.68.2.x) ---"
for ip in "${GROUP2_IPS[@]}"; do
    # 设备编号从5开始，与管理脚本一致
    last_octet=${ip##*.}
    name="emulator-$((4 + last_octet))"
    if update_device "$name" "$ip" "$GROUP2_USER" "$GROUP2_PASS" "$GROUP2_REMOTE_PATH"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_EMULATORS="$FAILED_EMULATORS $name"
    fi
done

echo ""
echo "=========================================="
echo "更新完成!"
echo -e "成功: ${GREEN}$SUCCESS_COUNT${NC} 台"
echo -e "失败: ${RED}$FAIL_COUNT${NC} 台"

if [ -n "$FAILED_EMULATORS" ]; then
    echo -e "${RED}失败的设备:$FAILED_EMULATORS${NC}"
fi
echo "=========================================="
