#!/bin/bash

# 批量登录远程设备并运行agent脚本（并行版本）
# 使用方法: ./run_agents.sh [start|stop|status|log <ip>]

ACTION=${1:-start}
TARGET_IP=${2:-""}

# 设备组1: 100.68.1.x
GROUP1_IPS=("100.68.1.3" "100.68.1.4" "100.68.1.5" "100.68.1.6")
GROUP1_USER="nano"
GROUP1_PASS="123456a?"
GROUP1_SCRIPT="/home/nano/qianguo/worker/agent.py"

# 设备组2: 100.68.2.x
GROUP2_IPS=("100.68.2.1" "100.68.2.2" "100.68.2.3" "100.68.2.4" "100.68.2.5" "100.68.2.6")
GROUP2_USER="nvidia"
GROUP2_PASS="nvidia"
GROUP2_SCRIPT="/home/nvidia/qianguo/worker/agent.py"

# 启动后等待时间（秒），等待进程完全启动
WAIT_TIME=3

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 临时目录用于存储并行执行结果
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

# 在远程设备上执行命令（后台运行，不等待返回）
remote_exec_background() {
    local ip=$1
    local user=$2
    local pass=$3
    local cmd=$4
    
    # 使用 & 让SSH在后台运行，不等待返回
    sshpass -p "$pass" ssh -f -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "$cmd" >/dev/null 2>&1 &
}

# 在远程设备上执行命令（等待返回）
remote_exec() {
    local ip=$1
    local user=$2
    local pass=$3
    local cmd=$4
    
    sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "$cmd" 2>/dev/null
    return $?
}

# 启动agent（单个设备）- 后台发送命令，不等待
start_agent_async() {
    local ip=$1
    local user=$2
    local pass=$3
    local script=$4
    
    # 使用nohup在后台运行，确保SSH断开后进程继续运行
    local cmd="source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null; conda activate py38 && nohup python ${script} > ~/agent.log 2>&1 &"
    
    # 后台发送启动命令，不等待
    remote_exec_background "$ip" "$user" "$pass" "$cmd"
}

# 停止agent（单个设备）
stop_agent() {
    local ip=$1
    local user=$2
    local pass=$3
    local result_file="$TEMP_DIR/result_${ip}"
    
    local cmd="pkill -f 'python.*agent.py' 2>/dev/null; exit 0"
    
    remote_exec "$ip" "$user" "$pass" "$cmd"
    echo "success" > "$result_file"
}

# 检查agent状态（单个设备）
check_status() {
    local ip=$1
    local user=$2
    local pass=$3
    local result_file="$TEMP_DIR/result_${ip}"
    
    local cmd="pgrep -f 'python.*agent.py' > /dev/null && echo 'running' || echo 'stopped'"
    local status=$(remote_exec "$ip" "$user" "$pass" "$cmd")
    
    echo "$status" > "$result_file"
}

# 并行处理单个设备
process_device() {
    local ip=$1
    local user=$2
    local pass=$3
    local script=$4
    local action=$5
    
    case $action in
        stop)
            stop_agent "$ip" "$user" "$pass"
            ;;
        status)
            check_status "$ip" "$user" "$pass"
            ;;
    esac
}

# 显示结果
show_results() {
    local action=$1
    shift
    local ips=("$@")
    
    for ip in "${ips[@]}"; do
        local result_file="$TEMP_DIR/result_${ip}"
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            case $action in
                start)
                    if [ "$result" == "running" ]; then
                        echo -e "${GREEN}[${ip}]${NC} agent启动成功"
                    else
                        echo -e "${RED}[${ip}]${NC} agent启动失败"
                    fi
                    ;;
                stop)
                    echo -e "${GREEN}[${ip}]${NC} agent已停止"
                    ;;
                status)
                    if [ "$result" == "running" ]; then
                        echo -e "${GREEN}[${ip}]${NC} agent正在运行"
                    else
                        echo -e "${RED}[${ip}]${NC} agent未运行"
                    fi
                    ;;
            esac
        else
            echo -e "${RED}[${ip}]${NC} 连接超时"
        fi
    done
}

# 执行start操作
do_start() {
    echo "正在发送启动命令..."
    echo ""
    
    # 并行发送启动命令（不等待）
    for ip in "${GROUP1_IPS[@]}"; do
        start_agent_async "$ip" "$GROUP1_USER" "$GROUP1_PASS" "$GROUP1_SCRIPT"
        echo -e "${YELLOW}[${ip}]${NC} 启动命令已发送"
    done
    
    for ip in "${GROUP2_IPS[@]}"; do
        start_agent_async "$ip" "$GROUP2_USER" "$GROUP2_PASS" "$GROUP2_SCRIPT"
        echo -e "${YELLOW}[${ip}]${NC} 启动命令已发送"
    done
    
    echo ""
    echo "等待 ${WAIT_TIME} 秒让进程启动..."
    sleep $WAIT_TIME
    
    echo ""
    echo "检查启动状态..."
    echo ""
    
    # 并行检查状态
    local pids=()
    
    for ip in "${GROUP1_IPS[@]}"; do
        check_status "$ip" "$GROUP1_USER" "$GROUP1_PASS" &
        pids+=($!)
    done
    
    for ip in "${GROUP2_IPS[@]}"; do
        check_status "$ip" "$GROUP2_USER" "$GROUP2_PASS" &
        pids+=($!)
    done
    
    # 等待所有状态检查完成
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    # 显示结果
    echo "--- 设备组1 (100.68.1.x) ---"
    show_results "start" "${GROUP1_IPS[@]}"
    
    echo ""
    echo "--- 设备组2 (100.68.2.x) ---"
    show_results "start" "${GROUP2_IPS[@]}"
}

# 执行stop/status操作
do_stop_or_status() {
    local action=$1
    
    echo "正在并行执行，请稍候..."
    echo ""
    
    # 并行执行
    local pids=()
    
    for ip in "${GROUP1_IPS[@]}"; do
        process_device "$ip" "$GROUP1_USER" "$GROUP1_PASS" "$GROUP1_SCRIPT" "$action" &
        pids+=($!)
    done
    
    for ip in "${GROUP2_IPS[@]}"; do
        process_device "$ip" "$GROUP2_USER" "$GROUP2_PASS" "$GROUP2_SCRIPT" "$action" &
        pids+=($!)
    done
    
    # 等待所有任务完成
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    # 显示结果
    echo "--- 设备组1 (100.68.1.x) ---"
    show_results "$action" "${GROUP1_IPS[@]}"
    
    echo ""
    echo "--- 设备组2 (100.68.2.x) ---"
    show_results "$action" "${GROUP2_IPS[@]}"
}

# 根据IP获取用户名和密码
get_credentials() {
    local ip=$1
    
    # 检查是否属于设备组1 (100.68.1.x)
    for check_ip in "${GROUP1_IPS[@]}"; do
        if [ "$ip" == "$check_ip" ]; then
            echo "$GROUP1_USER $GROUP1_PASS"
            return 0
        fi
    done
    
    # 检查是否属于设备组2 (100.68.2.x)
    for check_ip in "${GROUP2_IPS[@]}"; do
        if [ "$ip" == "$check_ip" ]; then
            echo "$GROUP2_USER $GROUP2_PASS"
            return 0
        fi
    done
    
    # 未找到
    return 1
}

# 查看指定设备的agent日志
do_log() {
    local ip=$1
    local lines=${2:-50}  # 默认显示最后50行
    
    if [ -z "$ip" ]; then
        echo -e "${RED}错误: 请指定设备IP地址${NC}"
        echo ""
        echo "使用方法: $0 log <ip> [行数]"
        echo ""
        echo "可用的设备IP:"
        echo "  设备组1: ${GROUP1_IPS[*]}"
        echo "  设备组2: ${GROUP2_IPS[*]}"
        exit 1
    fi
    
    # 获取凭据
    local credentials=$(get_credentials "$ip")
    if [ -z "$credentials" ]; then
        echo -e "${RED}错误: 未知的设备IP '${ip}'${NC}"
        echo ""
        echo "可用的设备IP:"
        echo "  设备组1: ${GROUP1_IPS[*]}"
        echo "  设备组2: ${GROUP2_IPS[*]}"
        exit 1
    fi
    
    local user=$(echo $credentials | cut -d' ' -f1)
    local pass=$(echo $credentials | cut -d' ' -f2)
    
    echo "========================================"
    echo "查看设备日志: ${ip}"
    echo "用户: ${user}"
    echo "显示最后 ${lines} 行"
    echo "========================================"
    echo ""
    
    # 获取日志
    local cmd="tail -n ${lines} ~/agent.log 2>/dev/null || echo '日志文件不存在或为空'"
    local log_content=$(remote_exec "$ip" "$user" "$pass" "$cmd")
    
    if [ -n "$log_content" ]; then
        echo "$log_content"
    else
        echo -e "${YELLOW}无法获取日志，请检查连接或日志文件是否存在${NC}"
    fi
}

# 实时查看日志（类似 tail -f）
do_log_follow() {
    local ip=$1
    
    if [ -z "$ip" ]; then
        echo -e "${RED}错误: 请指定设备IP地址${NC}"
        echo ""
        echo "使用方法: $0 logf <ip>"
        exit 1
    fi
    
    # 获取凭据
    local credentials=$(get_credentials "$ip")
    if [ -z "$credentials" ]; then
        echo -e "${RED}错误: 未知的设备IP '${ip}'${NC}"
        exit 1
    fi
    
    local user=$(echo $credentials | cut -d' ' -f1)
    local pass=$(echo $credentials | cut -d' ' -f2)
    
    echo "========================================"
    echo "实时查看设备日志: ${ip}"
    echo "按 Ctrl+C 退出"
    echo "========================================"
    echo ""
    
    # 实时查看日志
    sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "tail -f ~/agent.log"
}

# 主函数
main() {
    check_sshpass
    
    echo "========================================"
    echo "批量设备Agent管理脚本 (并行版本)"
    echo "操作: ${ACTION}"
    echo "========================================"
    echo ""
    
    case $ACTION in
        start)
            do_start
            ;;
        stop|status)
            do_stop_or_status "$ACTION"
            ;;
    esac
    
    echo ""
    echo "========================================"
    echo "操作完成!"
    echo "========================================"
}

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [start|stop|status|log|logf] [参数]"
    echo ""
    echo "  start        - 在所有设备上启动agent (默认)"
    echo "  stop         - 在所有设备上停止agent"
    echo "  status       - 检查所有设备上agent的运行状态"
    echo "  log <ip> [n] - 查看指定设备的日志 (默认最后50行，n可指定行数)"
    echo "  logf <ip>    - 实时查看指定设备的日志 (类似 tail -f)"
    echo ""
    echo "可用的设备IP:"
    echo "  设备组1: 100.68.1.3, 100.68.1.4, 100.68.1.5, 100.68.1.6"
    echo "  设备组2: 100.68.2.1, 100.68.2.2, 100.68.2.3, 100.68.2.4, 100.68.2.5, 100.68.2.6"
    echo ""
    echo "示例:"
    echo "  $0 start              # 启动所有agent"
    echo "  $0 stop               # 停止所有agent"
    echo "  $0 status             # 查看运行状态"
    echo "  $0 log 100.68.1.3     # 查看设备100.68.1.3的最后50行日志"
    echo "  $0 log 100.68.1.3 100 # 查看设备100.68.1.3的最后100行日志"
    echo "  $0 logf 100.68.2.1    # 实时查看设备100.68.2.1的日志"
}

# 入口
case $ACTION in
    start|stop|status)
        main
        ;;
    log)
        check_sshpass
        do_log "$TARGET_IP" "$3"
        ;;
    logf)
        check_sshpass
        do_log_follow "$TARGET_IP"
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
