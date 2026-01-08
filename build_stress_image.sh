#!/bin/bash

# 批量为所有emulator构建stress镜像的脚本（支持账号/密码）
# 使用方法: ./build_stress_image.sh

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 设备组1: 100.68.1.x (需要sudo运行docker)
GROUP1_IPS=("100.68.1.3" "100.68.1.4" "100.68.1.5" "100.68.1.6")
GROUP1_USER="nano"
GROUP1_PASS="123456a?"
GROUP1_SUDO=true

# 设备组2: 100.68.2.x
GROUP2_IPS=("100.68.2.1" "100.68.2.2" "100.68.2.3" "100.68.2.4" "100.68.2.5" "100.68.2.6")
GROUP2_USER="nvidia"
GROUP2_PASS="nvidia"
GROUP2_SUDO=true

# Dockerfile所在目录
DOCKERFILE_DIR="/home/qianguo/Edge-Scheduler/Controller/stress_image"
IMAGE_NAME="stress:latest"

# 检查Dockerfile是否存在
if [ ! -f "$DOCKERFILE_DIR/Dockerfile" ]; then
    echo -e "${RED}错误: Dockerfile不存在: $DOCKERFILE_DIR/Dockerfile${NC}"
    exit 1
fi

echo "=========================================="
echo "开始批量构建stress镜像"
echo "镜像名称: $IMAGE_NAME"
echo "=========================================="

check_sshpass() {
    if ! command -v sshpass >/dev/null 2>&1; then
        echo -e "${RED}错误: sshpass 未安装${NC}"
        echo "请先安装: sudo apt install sshpass"
        exit 1
    fi
}

# 远程执行命令（支持sudo）
remote_exec() {
    local ip=$1
    local user=$2
    local pass=$3
    local cmd=$4
    local use_sudo=$5

    if [ "$use_sudo" == "true" ]; then
        cmd="echo '${pass}' | sudo -S bash -c \"${cmd}\""
    fi

    sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "$cmd" 2>/dev/null
}

# 远程复制文件
remote_copy() {
    local src=$1
    local ip=$2
    local user=$3
    local pass=$4
    local dest=$5

    sshpass -p "$pass" scp -q -o StrictHostKeyChecking=no "$src" "${user}@${ip}:${dest}"
}

build_on_device() {
    local name=$1
    local ip=$2
    local user=$3
    local pass=$4
    local use_sudo=$5

    echo ""
    echo -e "${YELLOW}正在处理: $name ($ip)${NC}"

    # 1. 创建临时目录
    echo "  - 创建构建目录..."
    remote_exec "$ip" "$user" "$pass" "mkdir -p /tmp/stress_build" false

    # 2. 复制Dockerfile
    echo "  - 复制Dockerfile到远程机器..."
    if ! remote_copy "$DOCKERFILE_DIR/Dockerfile" "$ip" "$user" "$pass" "/tmp/stress_build/Dockerfile"; then
        echo -e "${RED}  - 复制Dockerfile失败${NC}"
        return 1
    fi

    # 3. 构建镜像
    echo "  - 构建Docker镜像..."
    remote_exec "$ip" "$user" "$pass" "cd /tmp/stress_build && docker build -t $IMAGE_NAME ." "$use_sudo" | while read line; do
        echo "    $line"
    done

    # 4. 检查构建结果
    BUILD_RESULT=$(remote_exec "$ip" "$user" "$pass" "docker images -q $IMAGE_NAME" "$use_sudo")
    if [ -n "$BUILD_RESULT" ]; then
        echo -e "${GREEN}  - 构建成功: $name${NC}"
        return 0
    else
        echo -e "${RED}  - 构建失败: $name${NC}"
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
    if build_on_device "$name" "$ip" "$GROUP1_USER" "$GROUP1_PASS" "$GROUP1_SUDO"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_EMULATORS="$FAILED_EMULATORS $name"
    fi
    remote_exec "$ip" "$GROUP1_USER" "$GROUP1_PASS" "rm -rf /tmp/stress_build" false
done

# 处理设备组2
echo ""
echo "--- 设备组2 (100.68.2.x) ---"
for ip in "${GROUP2_IPS[@]}"; do
    # 设备编号从5开始，与管理脚本一致
    last_octet=${ip##*.}
    name="emulator-$((4 + last_octet))"
    if build_on_device "$name" "$ip" "$GROUP2_USER" "$GROUP2_PASS" "$GROUP2_SUDO"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_EMULATORS="$FAILED_EMULATORS $name"
    fi
    remote_exec "$ip" "$GROUP2_USER" "$GROUP2_PASS" "rm -rf /tmp/stress_build" false
done

echo ""
echo "=========================================="
echo "构建完成!"
echo -e "成功: ${GREEN}$SUCCESS_COUNT${NC} 台"
echo -e "失败: ${RED}$FAIL_COUNT${NC} 台"

if [ -n "$FAILED_EMULATORS" ]; then
    echo -e "${RED}失败的设备:$FAILED_EMULATORS${NC}"
fi
echo "=========================================="
