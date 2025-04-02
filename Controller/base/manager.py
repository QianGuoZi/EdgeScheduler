from concurrent.futures import wait, ALL_COMPLETED
import os
import shutil
import time
from typing import Dict, List
import zipfile

from flask import json, request

from Controller.base.controller import Controller
from Controller.base.utils import read_json, send_data

dirName = '/home/qianguo/controller/'
class Manager(object):
    """
    负责和用户的通用交互
    """
    def __init__(self, controller: Controller):
        self.controller = controller
        self.__load_default_route()


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
            taskID = int(request.args.get('taskId'))
            # 将任务添加到待调度队列
            self.controller.add_pending_task(taskID)
            return 'Task submitted for scheduling'
        
        # def task_schedule(taskId: int):
        #     """
        #     丢给Scheduler处理，得到gl_run.py里的配置
        #     """
        #     allocation = self.controller.scheduler.resource_schedule(taskId)
        #     for node, node_info in allocation.items(): 
        #         print(f"node name: {node}, node object: {node_info}")
        #     return allocation

        # TODO：暂时用不上
        @self.testbed.flask.route('/getTaskStatus', methods=['GET'])
        def route_get_task_status():
            """
            获取任务状态
            """
            taskID = int(request.args.get('taskId'))
            if taskID in self.controller.get_deployed_tasks():
                return 'Task deployed'
            elif any(t[0] == taskID for t in self.controller.scheduled_tasks.queue):
                return 'Task scheduled'
            elif taskID in self.controller.pending_tasks.queue:
                return 'Task pending'
            return 'Task not found'
        
        @self.testbed.flask.route('/print', methods=['POST'])
        def route_print():
            """
            listen message from worker/worker_utils.py, send_print ().
            it will print the ${msg}.
            """
            print(request.form['msg'])
            # self.testbed.executor.submit(self.__send_logs_to_backend, request.form['msg'])
            return ''
        
        @self.testbed.flask.route('/update/tc', methods=['GET'])
        def route_update_tc():
            """
            you can send a GET request to this /update/tc to update the
            tc settings of physical and emulated nodes.
            """

            def update_physical_tc(_physical, _agent_port: int):
                """
                send the tc settings to a physical node.
                this request can be received by worker/agent.py, route_physical_tc ().
                """
                _data = {
                    'NET_NODE_NIC': _physical.nic,
                    'NET_NODE_TC': _physical.tc,
                    'NET_NODE_TC_IP': _physical.tcIP,
                    'NET_NODE_TC_PORT': _physical.tcPort
                }
                print('update_physical_tc: send to ' + _physical.name)
                _res = send_data('POST', '/physical/tc', _physical.ip, _agent_port,
                                 data={'data': json.dumps(_data)})
                if _res == '':
                    print('physical node ' + _physical.name + ' update tc succeed')
                else:
                    print('physical node ' + _physical.name + ' update tc failed, err:')
                    print(_res)

            def update_emulated_tc(_data: Dict, _emulator_ip: str, _agent_port: int):
                """
                send the tc settings to an emulator.
                this request can be received by worker/agent.py, route_emulated_tc_update ().
                """
                print('update_emulated_tc: send to ' + ', '.join(_data.keys()))
                _res = send_data('POST', '/emulated/tc/update', _emulator_ip, _agent_port,
                                 data={'data': json.dumps(_data)})
                _ret = json.loads(_res)
                for _name in _ret:
                    if 'msg' in _ret[_name]:
                        print('emulated node ' + _name + ' update tc failed, err:')
                        print(_ret[_name]['msg'])
                    else:
                        print('emulated node ' + _name + ' update tc succeed')

            time_start = time.time()
            filename = request.args.get('file')
            if filename[0] != '/':
                filename = os.path.join(self.controller.dirName, filename)

            with open(filename, 'r') as f:
                all_nodes = []
                # emulator's ip to emulated nodes in this emulator.
                emulator_ip_to_node: Dict[str, List] = {}
                links_json = json.loads(f.read().replace('\'', '\"'))
                for name in links_json:
                    n = self.controller.name_to_node(name)
                    all_nodes.append(n)
                    n.tc.clear()
                    n.tcIP.clear()
                    n.tcPort.clear()
                self.controller.load_link(links_json)
                for node in all_nodes:
                    if node.name in self.controller.pNode:
                        self.controller.executor.submit(update_physical_tc, node,
                                                     self.controller.agentPort)
                    else:
                        emulator_ip = node.ip
                        emulator_ip_to_node.setdefault(emulator_ip, []).append(node)
                for emulator_ip in emulator_ip_to_node:
                    data = {}
                    for en in emulator_ip_to_node[emulator_ip]:
                        data[en.name] = {
                            'NET_NODE_NIC': en.nic,
                            'NET_NODE_TC': en.tc,
                            'NET_NODE_TC_IP': en.tcIP,
                            'NET_NODE_TC_PORT': en.tcPort
                        }
                    self.controller.executor.submit(update_emulated_tc, data, emulator_ip,
                                                 self.controller.agentPort)
            time_end = time.time()
            print('update tc time all cost', time_end - time_start, 's')
            return ''
        
        @self.testbed.flask.route('/emulated/stop', methods=['GET'])
        def route_emulated_stop():
            """
            send a stop message to emulators.
            stop emulated nodes without remove them.
            this request can be received by worker/agent.py, route_emulated_stop ().
            """
            taskID = int(request.args.get('taskId'))
            self.__stop_all_emulated(taskID)
            return ''
        
        @self.testbed.flask.route('/emulated/clear', methods=['GET'])
        def route_emulated_clear():
            """
            send a clear message to emulators.
            stop emulated nodes and remove them.
            this request can be received by worker/agent.py, route_emulated_clear ().
            """
            taskID = int(request.args.get('taskId'))
            self.__clear_all_emulated(taskID)
            return ''
        
        @self.testbed.flask.route('/emulated/reset', methods=['GET'])
        def route_emulated_reset():
            """
            send a reset message to emulators.
            remove emulated nodes, volumes and network bridges.
            this request can be received by worker/agent.py, route_emulated_reset ().
            """
            taskID = int(request.args.get('taskId'))
            self.__reset_all_emulated(taskID)
            return ''

        
    def __stop_all_emulated(self, taskID: int):
        def stop_emulated(_emulator_ip: str, _agent_port: int):
            send_data('GET', '/emulated/stop?taskID=' + str(taskID), _emulator_ip, _agent_port)

        tasks = []
        for s in self.controller.task[taskID].emulator.values():
            if s.eNode:
                tasks.append(self.controller.executor.submit(stop_emulated, s.ipW, self.controller.agentPort))
        wait(tasks, return_when=ALL_COMPLETED)

    def __clear_all_emulated(self, taskID: int):
        def clear_emulated(_emulator_ip: str, _agent_port: int):
            send_data('GET', '/emulated/clear?taskID=' + str(taskID), _emulator_ip, _agent_port)

        tasks = []
        for s in self.controller.task[taskID].emulator.values():
            if s.eNode:
                tasks.append(self.controller.executor.submit(clear_emulated, s.ipW, self.controller.agentPort))
        wait(tasks, return_when=ALL_COMPLETED)

    def __reset_all_emulated(self, taskID: int):
        def reset_emulated(_emulator_ip: str, _agent_port: int):
            send_data('GET', '/emulated/reset?taskID=' + str(taskID), _emulator_ip, _agent_port)

        tasks = []
        for s in self.controller.task[taskID].emulator.values():
            if s.eNode:
                tasks.append(self.controller.executor.submit(reset_emulated, s.ipW, self.controller.agentPort))
        wait(tasks, return_when=ALL_COMPLETED)