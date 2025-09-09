#!/bin/bash

# 检查是否提供了起始和结束ID
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <start_id> <end_id>"
    exit 1
fi

START_ID=$1
END_ID=$2

# 固定变量
TASK_ZIP="/home/qianguo/Edge-Scheduler/task.zip"
BASE_DIR="/home/qianguo/Edge-Scheduler/Controller/dml_tool"
PORT_BASE=6000


# 对每个 id 执行操作
for (( id=START_ID; id<=END_ID; id++ ))
do
    echo "Processing ID: $id"

    # 上传 task.zip 并启动任务
    echo "Uploading task.zip..."
    curl -F "file=@$TASK_ZIP" http://localhost:3333/taskRequestFile

    echo "Starting task with ID= $id..."
    curl "http://localhost:3333/startupTask?taskId=$id"

    # 进入对应目录
    cd "${BASE_DIR}/${id}" || { echo "Directory ${BASE_DIR}/${id} not found"; exit 1; }

    # 等待10秒
    echo "Waiting 10 seconds before next step..."
    sleep 5
    
    # 运行 Python 配置脚本
    python3 dataset_conf.py -d gl_dataset.json -t "$id"
    python3 gl_structure_conf.py -s gl_structure.json -t "$id"

    # 提交配置
    curl "http://localhost:3333/conf/dataset?taskId=${id}"
    curl "http://localhost:3333/conf/structure?taskId=${id}"


    # 发送启动任务请求
    TARGET_PORT=$((PORT_BASE + id))
    curl -X POST -d "taskID=${id}" "http://localhost:${TARGET_PORT}/task/${id}/startTask"
    
    echo "Finished processing ID: $id"
done

echo "All tasks completed."