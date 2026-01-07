#!/bin/bash

# 批量配置远程设备sudo免密
# 使用方法: ./setup_sudoers.sh

# 设备组1: 100.68.1.x
GROUP1_IPS=("100.68.1.3" "100.68.1.4" "100.68.1.5" "100.68.1.6")
GROUP1_USER="nano"
GROUP1_PASS="123456a?"

# 设备组2: 100.68.2.x
GROUP2_IPS=("100.68.2.1" "100.68.2.2" "100.68.2.3" "100.68.2.4" "100.68.2.5" "100.68.2.6")
GROUP2_USER="nvidia"
GROUP2_PASS="nvidia"

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

# 配置单个设备的sudo免密
setup_sudoers() {
    local ip=$1
    local user=$2
    local pass=$3
    
    echo -e "${YELLOW}[${ip}]${NC} 正在配置sudo免密..."
    
    # 检查是否已经配置过
    local check_cmd="sudo -n true 2>/dev/null && echo 'already_configured' || echo 'need_setup'"
    local status=$(sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "$check_cmd" 2>/dev/null)
    
    if [ "$status" == "already_configured" ]; then
        echo -e "${GREEN}[${ip}]${NC} 已经配置过sudo免密，跳过"
        return 0
    fi
    
    # 添加sudoers配置
    # 使用echo和sudo tee来添加配置，需要输入密码
    local sudoers_line="${user} ALL=(ALL) NOPASSWD: ALL"
    local setup_cmd="echo '${pass}' | sudo -S bash -c 'echo \"${sudoers_line}\" > /etc/sudoers.d/${user}-nopasswd && chmod 440 /etc/sudoers.d/${user}-nopasswd'"
    
    if sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "$setup_cmd" 2>/dev/null; then
        # 验证配置是否成功
        local verify_cmd="sudo -n true 2>/dev/null && echo 'success' || echo 'failed'"
        local result=$(sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "$verify_cmd" 2>/dev/null)
        
        if [ "$result" == "success" ]; then
            echo -e "${GREEN}[${ip}]${NC} sudo免密配置成功"
        else
            echo -e "${RED}[${ip}]${NC} sudo免密配置失败，请手动检查"
        fi
    else
        echo -e "${RED}[${ip}]${NC} 配置失败，可能需要手动配置"
    fi
}

# 主函数
main() {
    check_sshpass
    
    echo "========================================"
    echo "批量配置sudo免密脚本"
    echo "========================================"
    echo ""
    echo -e "${YELLOW}警告: 此脚本将为远程用户配置sudo免密权限${NC}"
    echo ""
    read -p "是否继续？(y/n): " confirm
    
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
    
    echo ""
    echo "--- 设备组1 (100.68.1.x) - 用户: nano ---"
    for ip in "${GROUP1_IPS[@]}"; do
        setup_sudoers "$ip" "$GROUP1_USER" "$GROUP1_PASS"
    done
    
    echo ""
    echo "--- 设备组2 (100.68.2.x) - 用户: nvidia ---"
    for ip in "${GROUP2_IPS[@]}"; do
        setup_sudoers "$ip" "$GROUP2_USER" "$GROUP2_PASS"
    done
    
    echo ""
    echo "========================================"
    echo "配置完成!"
    echo "========================================"
}

main

