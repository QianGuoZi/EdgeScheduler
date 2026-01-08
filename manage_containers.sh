#!/bin/bash

# 批量管理远程设备上的Docker容器（只管理指定镜像的容器）
# 使用方法: ./manage_containers.sh [stop|pause|unpause|rm|status] [镜像名称]
# 镜像名称可选: task1:v1.0 (默认) 或 stress:latest

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 解析参数
ACTION=${1:-status}
IMAGE_ARG=${2:-}

# 目标镜像名称（默认值）
DEFAULT_IMAGE="task1:v1.0"

# 设置目标镜像
if [ -n "$IMAGE_ARG" ]; then
    # 验证镜像名称是否支持
    if [[ "$IMAGE_ARG" == "task1:v1.0" || "$IMAGE_ARG" == "stress:latest" ]]; then
        TARGET_IMAGE="$IMAGE_ARG"
    else
        echo -e "${RED}错误: 不支持的镜像名称 '${IMAGE_ARG}'${NC}"
        echo "支持的镜像: task1:v1.0, stress:latest"
        exit 1
    fi
else
    TARGET_IMAGE="$DEFAULT_IMAGE"
fi

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

# 临时目录
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# 检查sshpass是否安装
check_sshpass() {
    if ! command -v sshpass &> /dev/null; then
        echo -e "${RED}错误: sshpass 未安装${NC}"
        echo "请先安装 sshpass: sudo apt install sshpass"
        exit 1
    fi
}

# 在远程设备上执行命令
remote_exec() {
    local ip=$1
    local user=$2
    local pass=$3
    local cmd=$4
    local use_sudo=$5
    
    if [ "$use_sudo" == "true" ]; then
        # 使用sudo执行命令
        cmd="echo '${pass}' | sudo -S bash -c '${cmd}'"
    fi
    
    sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "$cmd" 2>/dev/null
    return $?
}

# 停止容器
stop_containers() {
    local ip=$1
    local user=$2
    local pass=$3
    local use_sudo=$4
    
    echo -e "${YELLOW}[${ip}]${NC} 正在停止 ${TARGET_IMAGE} 容器..."
    
    local cmd="docker stop \$(docker ps -q --filter ancestor=${TARGET_IMAGE}) 2>/dev/null; echo done"
    local result=$(remote_exec "$ip" "$user" "$pass" "$cmd" "$use_sudo")
    
    if echo "$result" | grep -q "done"; then
        echo -e "${GREEN}[${ip}]${NC} 容器已停止"
    else
        echo -e "${YELLOW}[${ip}]${NC} 没有运行中的 ${TARGET_IMAGE} 容器"
    fi
}

# 暂停容器
pause_containers() {
    local ip=$1
    local user=$2
    local pass=$3
    local use_sudo=$4
    
    echo -e "${YELLOW}[${ip}]${NC} 正在暂停 ${TARGET_IMAGE} 容器..."
    
    local cmd="docker pause \$(docker ps -q --filter ancestor=${TARGET_IMAGE}) 2>/dev/null; echo done"
    local result=$(remote_exec "$ip" "$user" "$pass" "$cmd" "$use_sudo")
    
    if echo "$result" | grep -q "done"; then
        echo -e "${GREEN}[${ip}]${NC} 容器已暂停"
    else
        echo -e "${YELLOW}[${ip}]${NC} 没有运行中的 ${TARGET_IMAGE} 容器"
    fi
}

# 恢复暂停的容器
unpause_containers() {
    local ip=$1
    local user=$2
    local pass=$3
    local use_sudo=$4
    
    echo -e "${YELLOW}[${ip}]${NC} 正在恢复 ${TARGET_IMAGE} 容器..."
    
    local cmd="docker unpause \$(docker ps -aq --filter ancestor=${TARGET_IMAGE} --filter status=paused) 2>/dev/null; echo done"
    local result=$(remote_exec "$ip" "$user" "$pass" "$cmd" "$use_sudo")
    
    if echo "$result" | grep -q "done"; then
        echo -e "${GREEN}[${ip}]${NC} 容器已恢复"
    else
        echo -e "${YELLOW}[${ip}]${NC} 没有暂停的 ${TARGET_IMAGE} 容器"
    fi
}

# 删除容器
remove_containers() {
    local ip=$1
    local user=$2
    local pass=$3
    local use_sudo=$4
    
    echo -e "${YELLOW}[${ip}]${NC} 正在删除 ${TARGET_IMAGE} 容器..."
    
    # 先停止再删除
    local cmd="docker stop \$(docker ps -q --filter ancestor=${TARGET_IMAGE}) 2>/dev/null; docker rm \$(docker ps -aq --filter ancestor=${TARGET_IMAGE}) 2>/dev/null; echo done"
    local result=$(remote_exec "$ip" "$user" "$pass" "$cmd" "$use_sudo")
    
    if echo "$result" | grep -q "done"; then
        echo -e "${GREEN}[${ip}]${NC} 容器已删除"
    else
        echo -e "${YELLOW}[${ip}]${NC} 没有 ${TARGET_IMAGE} 容器"
    fi
}

# 查看容器状态
check_status() {
    local ip=$1
    local user=$2
    local pass=$3
    local use_sudo=$4
    
    echo -e "${BLUE}[${ip}]${NC} ${TARGET_IMAGE} 容器状态:"
    
    local cmd="docker ps -a --filter ancestor=${TARGET_IMAGE} --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
    local result=$(remote_exec "$ip" "$user" "$pass" "$cmd" "$use_sudo")
    
    if [ -n "$result" ] && ! echo "$result" | grep -q "^NAMES.*STATUS.*PORTS$"; then
        echo "$result"
    elif echo "$result" | grep -v "^NAMES" | grep -q .; then
        echo "$result"
    else
        echo "  没有 ${TARGET_IMAGE} 容器"
    fi
    echo ""
}

# 处理单个设备
process_device() {
    local ip=$1
    local user=$2
    local pass=$3
    local action=$4
    local use_sudo=$5
    
    case $action in
        stop)
            stop_containers "$ip" "$user" "$pass" "$use_sudo"
            ;;
        pause)
            pause_containers "$ip" "$user" "$pass" "$use_sudo"
            ;;
        unpause)
            unpause_containers "$ip" "$user" "$pass" "$use_sudo"
            ;;
        rm)
            remove_containers "$ip" "$user" "$pass" "$use_sudo"
            ;;
        status)
            check_status "$ip" "$user" "$pass" "$use_sudo"
            ;;
    esac
}

# 主函数
main() {
    check_sshpass
    
    echo "========================================"
    echo "批量管理Docker容器 (镜像: ${TARGET_IMAGE})"
    echo "操作: ${ACTION}"
    echo "========================================"
    echo ""
    
    # 如果是删除操作，需要确认
    if [ "$ACTION" == "rm" ]; then
        echo -e "${RED}警告: 此操作将删除所有设备上的所有容器！${NC}"
        read -p "是否继续？(y/n): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "已取消"
            exit 0
        fi
        echo ""
    fi
    
    echo "--- 设备组1 (100.68.1.x) ---"
    for ip in "${GROUP1_IPS[@]}"; do
        process_device "$ip" "$GROUP1_USER" "$GROUP1_PASS" "$ACTION" "$GROUP1_SUDO"
    done
    
    echo ""
    echo "--- 设备组2 (100.68.2.x) ---"
    for ip in "${GROUP2_IPS[@]}"; do
        process_device "$ip" "$GROUP2_USER" "$GROUP2_PASS" "$ACTION" "$GROUP2_SUDO"
    done
    
    echo ""
    echo "========================================"
    echo "操作完成!"
    echo "========================================"
}

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [操作] [镜像名称]"
    echo ""
    echo "操作:"
    echo "  status  - 查看所有设备的容器状态 (默认)"
    echo "  stop    - 停止所有设备上的容器"
    echo "  pause   - 暂停所有设备上的容器"
    echo "  unpause - 恢复所有设备上暂停的容器"
    echo "  rm      - 删除所有设备上的容器 (会先停止)"
    echo ""
    echo "镜像名称 (可选):"
    echo "  task1:v1.0  - 默认镜像"
    echo "  stress:latest - 压力测试镜像"
    echo ""
    echo "示例:"
    echo "  $0 status              # 查看 task1:v1.0 容器状态 (默认)"
    echo "  $0 status task1:v1.0   # 查看 task1:v1.0 容器状态"
    echo "  $0 status stress:latest # 查看 stress:latest 容器状态"
    echo "  $0 stop task1:v1.0     # 停止 task1:v1.0 容器"
    echo "  $0 stop stress:latest   # 停止 stress:latest 容器"
    echo "  $0 pause                # 暂停 task1:v1.0 容器 (默认)"
    echo "  $0 rm stress:latest     # 删除 stress:latest 容器"
}

# 入口
case $ACTION in
    stop|pause|unpause|rm|status)
        main
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo -e "${RED}错误: 未知操作 '${ACTION}'${NC}"
        show_help
        exit 1
        ;;
esac

