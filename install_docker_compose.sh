#!/bin/bash

# 批量在远程设备上安装 docker-compose (aarch64架构)
# 使用方法: ./install_docker_compose.sh

# 设备组1: 100.68.1.x
GROUP1_IPS=("100.68.1.3" "100.68.1.4" "100.68.1.5" "100.68.1.6")
GROUP1_USER="nano"
GROUP1_PASS="123456a?"

# 设备组2: 100.68.2.x
GROUP2_IPS=("100.68.2.1" "100.68.2.2" "100.68.2.3" "100.68.2.4" "100.68.2.5" "100.68.2.6")
GROUP2_USER="nvidia"
GROUP2_PASS="nvidia"

# docker-compose 版本和下载地址
COMPOSE_VERSION="v2.24.0"
COMPOSE_URL="https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-aarch64"
LOCAL_COMPOSE="/tmp/docker-compose-aarch64"

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

# 下载 docker-compose 到本地
download_compose() {
    if [ -f "$LOCAL_COMPOSE" ]; then
        echo -e "${GREEN}本地已有 docker-compose 文件${NC}"
        return 0
    fi
    
    echo "正在下载 docker-compose ${COMPOSE_VERSION} 到本地..."
    if curl -L "$COMPOSE_URL" -o "$LOCAL_COMPOSE" 2>/dev/null; then
        chmod +x "$LOCAL_COMPOSE"
        echo -e "${GREEN}下载成功${NC}"
        return 0
    else
        echo -e "${RED}下载失败，请检查网络连接${NC}"
        return 1
    fi
}

# 在单个设备上安装 docker-compose
install_on_device() {
    local ip=$1
    local user=$2
    local pass=$3
    
    echo -e "${YELLOW}[${ip}]${NC} 正在检查 docker-compose..."
    
    # 检查当前版本
    local check_cmd="docker-compose --version 2>/dev/null || /usr/local/bin/docker-compose --version 2>/dev/null"
    local current_version=$(sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "$check_cmd" 2>/dev/null)
    
    if [ -n "$current_version" ]; then
        # 检查是否是新版本 (v2.x)
        if echo "$current_version" | grep -q "version v2\|version 2\." ; then
            echo -e "${GREEN}[${ip}]${NC} 已是新版本: $current_version"
            return 0
        else
            echo -e "${YELLOW}[${ip}]${NC} 当前版本较旧: $current_version，正在更新..."
        fi
    fi
    
    # 通过 scp 上传 docker-compose 到远程设备
    echo -e "${YELLOW}[${ip}]${NC} 正在上传 docker-compose..."
    if ! sshpass -p "$pass" scp -o StrictHostKeyChecking=no "$LOCAL_COMPOSE" "${user}@${ip}:/tmp/docker-compose" 2>/dev/null; then
        echo -e "${RED}[${ip}]${NC} 上传失败"
        return 1
    fi
    
    # 安装到系统目录
    local install_cmd="echo '${pass}' | sudo -S bash -c '
        mv /tmp/docker-compose /usr/local/bin/docker-compose && \
        chmod +x /usr/local/bin/docker-compose && \
        ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose 2>/dev/null
    '"
    
    if sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 "${user}@${ip}" "$install_cmd" 2>/dev/null; then
        # 验证安装
        local verify=$(sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "docker-compose --version 2>/dev/null")
        if [ -n "$verify" ]; then
            echo -e "${GREEN}[${ip}]${NC} 安装成功: $verify"
        else
            echo -e "${RED}[${ip}]${NC} 安装可能失败，请手动检查"
        fi
    else
        echo -e "${RED}[${ip}]${NC} 安装失败"
    fi
}

# 主函数
main() {
    check_sshpass
    
    echo "========================================"
    echo "批量安装 docker-compose (aarch64)"
    echo "版本: ${COMPOSE_VERSION}"
    echo "========================================"
    echo ""
    
    # 先下载到本地
    if ! download_compose; then
        exit 1
    fi
    
    echo ""
    echo "--- 设备组1 (100.68.1.x) ---"
    for ip in "${GROUP1_IPS[@]}"; do
        install_on_device "$ip" "$GROUP1_USER" "$GROUP1_PASS"
    done
    
    echo ""
    echo "--- 设备组2 (100.68.2.x) ---"
    for ip in "${GROUP2_IPS[@]}"; do
        install_on_device "$ip" "$GROUP2_USER" "$GROUP2_PASS"
    done
    
    echo ""
    echo "========================================"
    echo "安装完成!"
    echo "========================================"
}

main
