import abc
import os
import shutil
import threading
import time
from typing import Dict
import zipfile

from flask import request

from Controller.base.controller import Controller
from Controller.base.utils import read_json

dirName = '/home/qianguo/controller/'
class Manager(object):
    """
    负责和用户的通用交互
    """
    def __init__(self, controller: Controller):
        self.controller = controller
        self.__load_default_route()

        self.check_timer = threading.Timer(1.0, self._check_scheduled_tasks)
        self.check_timer.daemon = True
        self.check_timer.start()

    
    def __load_default_route(self):
        @self.testbed.flask.route('/taskRequestFile', methods=['POST'])
        def route_receive_request():
            """
            接收用户发送的任务文件，接收压缩包（包括links.json文件，dml_app,dml_tool,manager.py文件）
            task_file
                ├─ links.json
                ├─ manager.py
                ├─ dataset*
                    ├─ test_data
                    ├─ train_data
                ├─ dml_tool
                    ├─ dataset.json
                    ├─ structure.json
                    ├─ structure_conf.py
                    ├─ dataset_conf.py
                ├─ dml_app
                    ├─ nns
                    ├─ dml_req.txt
                    ├─ peer.py
                    ├─ Dockerfile
            """
            def analyse_file(taskId: int):
                """
                处理用户的文件，将压缩包解压并放到不同的文件夹中
                dml_app和dml_file需要和worker挂载，要不直接挂载一个大的文件夹，然后往里面更新算了（
                那挂载可以不动了，直接在下面为task创建对应的文件夹，文件路径改改
                worker_utils可以获得taskId，喜
                """
                zip_filename = f"{taskId}_taskFile.zip"
                zip_path = os.path.join(dirName, "task_file", zip_filename)
                
                current_directory = os.path.join(dirName, "task_file", str(taskId))
                os.makedirs(current_directory, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(current_directory)
                            
                # os.remove(zip_path)

                # 复制 links.json到task_links/{taskId}/links.json
                links_json_path = os.path.join(current_directory, 'links.json')
                target_links_json_path = os.path.join(dirName, "task_links", str(taskId))
                if not os.path.exists(target_links_json_path) :
                    os.makedirs(target_links_json_path)
                shutil.copy2(links_json_path, target_links_json_path)

                # 复制 manager.py到task_manager/{taskId}/ml_manager.py
                manager_path = os.path.join(current_directory, 'ml_manager.py')
                target_manager_path = os.path.join(dirName, "task_manager", str(taskId))
                if not os.path.exists(target_manager_path) :
                    os.makedirs(target_manager_path)
                shutil.copy2(manager_path, target_manager_path)
                
                # 创建 dml_tool 子目录
                target_dml_file_dir = os.path.join(dirName, "dml_tool", str(taskId))
                os.makedirs(target_dml_file_dir, exist_ok=True)
                
                # 复制 dml_file 文件夹到指定目录 /controller/dml_file/{taskId}/
                source_dml_file_dir = os.path.join(current_directory, 'dml_tool')
                if os.path.exists(source_dml_file_dir):
                    shutil.copytree(source_dml_file_dir, target_dml_file_dir, dirs_exist_ok=True)

                # 创建 dml_app 子目录
                target_dml_app_dir = os.path.join(dirName, "dml_app", str(taskId))
                os.makedirs(target_dml_app_dir, exist_ok=True)
                
                # 移动 dml_app 文件夹到指定目录 /controller/dml_app/{taskId}/
                source_dml_app_dir = os.path.join(current_directory, 'dml_app')
                if os.path.exists(source_dml_app_dir):
                    shutil.copytree(source_dml_app_dir, target_dml_app_dir, dirs_exist_ok=True)
                return
            if 'file' not in request.files:
                    return 'No file part', 400
                
            file = request.files['file']
                
            # 如果用户没有选择文件，浏览器可能会发送一个没有文件名的空文件
            if file.filename == '':
                return 'No selected file', 400
                
            if file:
                taskId = self.controller.next_task_id()
                filename = f"{taskId}_taskFile.zip"
                save_path = os.path.join(dirName,'task_file', filename)
                print(save_path)
                file.save(save_path)
                analyse_file(taskId)

                # 返回成功响应
                return 'File successfully uploaded.', 200
            
            return
    
        @self.testbed.flask.route('/startupTask', methods=['GET'])
        def route_startup_task():
            """
            用户信息已发送完毕，开始执行用户的任务
            现在改为异步方式:
            1. 将任务加入待调度队列
            2. 定期检查已调度队列
            3. 当发现已调度的任务时启动对应容器
            """
            task_id = request.args.get('taskId')
            # 将任务添加到待调度队列
            self.controller.add_pending_task(task_id)
            return 'Task submitted for scheduling'
        
        

        def task_schedule(taskId: int):
            """
            丢给Scheduler处理，得到gl_run.py里的配置
            """
            allocation = self.controller.scheduler.resource_schedule(taskId)
            for node, node_info in allocation.items(): 
                print(f"node name: {node}, node object: {node_info}")
            return allocation
            
    def task_start(self, taskId: int, allocation: Dict):
        """
        启动相应的容器
        """
            # 挂载
        nfsApp = self.testbed.nfs['dml_app']
        nfsDataset = self.testbed.nfs['dataset']
                
        # 添加节点
        for node_name, node_info in allocation.items():
            emu = self.testbed.emulator[node_info['emulator']]
            self.add_node_name(taskId, node_name)
            en = self.testbed.add_emulated_node (node_name, '/home/qianguo/worker/dml_app/'+str(taskId),
                ['python3', 'gl_peer.py'], 'dml:v1.0', cpu=node_info['cpu'], ram=node_info['ram'], unit='G', emulator=emu)
            en.mount_local_path ('./dml_file', '/home/qianguo/worker/dml_file')
            en.mount_nfs (nfsApp, '/home/qianguo/worker/dml_app')
            en.mount_nfs (nfsDataset, '/home/qianguo/worker/dataset')
                
            # 解析links
        links_json = read_json (os.path.join (dirName, "task_links", str(taskId),'links.json'))
        self.testbed.load_link_user (taskId, links_json)

            # 保存信息
        self.testbed.save_yml_user(taskId) # 保存yml文件到controller
        self.testbed.save_node_info() # 保存节点信息到testbed
        self.testbed.manager.load_node_info() # 保存节点信息到manager
        self.testbed.send_tc() # 将tc信息发送给worker，没有的添加，有的更新
        self.launch_all_emulated_user(taskId, dirName)

    def check_and_start_tasks(self):
        """检查已调度队列并启动任务"""
        scheduled = self.controller.get_scheduled_task()
        if scheduled:
            task_id, allocation = scheduled
            self.task_start(task_id, allocation)

    def _check_scheduled_tasks(self):
        """定期检查已调度的任务"""
        while True:
            self.check_and_start_tasks()
            time.sleep(1)