from concurrent.futures import wait, ALL_COMPLETED
import os
import shutil
import threading
import time
from typing import Dict, List
import zipfile
import json as json_module

from flask import json, request

from .utils import read_json, send_data
from .node import BackgroundLoadNode

dirName = '/home/qianguo/Edge-Scheduler/Controller'
class Manager(object):
    """
    负责和用户的通用交互
    """
    def __init__(self, controller):
        self.controller = controller
        self.__load_default_route()


    def __load_default_route(self):
        @self.controller.flask.route('/taskRequestFile', methods=['POST'])
        def route_receive_request():
            try:
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
                        info_list = zip_ref.infolist()
                        for info in info_list:
                            # 跳过空文件名或目录项
                            if not info.filename or info.filename.endswith('/'):
                                continue
                                
                            # 去掉可能存在的根目录名
                            if '/' in info.filename:
                                # 提取子路径
                                sub_path = info.filename.split('/', 1)[1]
                                if sub_path:  # 确保子路径不为空
                                    info.filename = sub_path
                                    try:
                                        zip_ref.extract(info, current_directory)
                                    except Exception as e:
                                        print(f"解压文件 {info.filename} 时出错: {str(e)}")
                            else:
                                # 直接解压没有路径的文件
                                try:
                                    zip_ref.extract(info, current_directory)
                                except Exception as e:
                                    print(f"解压文件 {info.filename} 时出错: {str(e)}")
                                
                    # os.remove(zip_path)

                    # 复制 links.json到task_links/{taskId}/links.json
                    links_json_path = os.path.join(current_directory, 'links.json')
                    target_links_json_path = os.path.join(dirName, "task_links", str(taskId))
                    if not os.path.exists(target_links_json_path) :
                        os.makedirs(target_links_json_path)
                    shutil.copy2(links_json_path, target_links_json_path)

                    # 复制 manager.py到task_manager/{taskId}/user_manager.py
                    manager_path = os.path.join(current_directory, 'task_manager.py')
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
                if file.filename == '' or not file.filename:
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
            except Exception as e:
                print(f"Error: {str(e)}")  # 打印错误信息
                return str(e), 500  # 返回具体错误信息
    
        @self.controller.flask.route('/startupTask', methods=['GET'])
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
            print(f"Task {taskID} is submitted for scheduling.")
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
        @self.controller.flask.route('/getTaskStatus', methods=['GET'])
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
        
        @self.controller.flask.route('/print', methods=['POST'])
        def route_print():
            """
            listen message from worker/worker_utils.py, send_print ().
            it will print the ${msg}.
            """
            print(request.form['msg'])
            # self.controller.executor.submit(self.__send_logs_to_backend, request.form['msg'])
            return ''
        
        @self.controller.flask.route('/update/tc', methods=['GET'])
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
        
        @self.controller.flask.route('/emulated/stop', methods=['GET'])
        def route_emulated_stop():
            """
            send a stop message to emulators.
            stop emulated nodes without remove them.
            this request can be received by worker/agent.py, route_emulated_stop ().
            """
            taskID = int(request.args.get('taskId'))
            self.__stop_all_emulated(taskID)
            return ''
        
        @self.controller.flask.route('/emulated/clear', methods=['GET'])
        def route_emulated_clear():
            """
            send a clear message to emulators.
            stop emulated nodes and remove them.
            this request can be received by worker/agent.py, route_emulated_clear ().
            """
            taskID = int(request.args.get('taskId'))
            self.__clear_all_emulated(taskID)
            return ''
        
        @self.controller.flask.route('/emulated/reset', methods=['GET'])
        def route_emulated_reset():
            """
            send a reset message to emulators.
            remove emulated nodes, volumes and network bridges.
            this request can be received by worker/agent.py, route_emulated_reset ().
            """
            taskID = int(request.args.get('taskId'))
            self.__reset_all_emulated(taskID)
            return ''  
        
        @self.controller.flask.route('/conf/dataset', methods=['GET'])
        def route_conf_dataset():
            """
            listen message from user, send dataset conf file to all nodes.
            """
            taskID = int(request.args.get('taskId'))
            self.__send_conf(taskID, 'dataset')
            return ''

        @self.controller.flask.route('/conf/structure', methods=['GET'])
        def route_conf_structure():
            """
            listen message from user, send structure conf file to all nodes.
            """
            taskID = int(request.args.get('taskId'))
            self.__send_conf(taskID, 'structure')
            return ''
        
        @self.controller.flask.route('/emulated/tc', methods=['POST'])
        def route_emulated_tc():
            """处理模拟节点的tc响应"""
            try:
                # 添加调试信息
                print(f"收到 /emulated/tc 请求")
                print(f"请求表单数据: {request.form}")
                
                # 检查必需的字段
                if 'taskID' not in request.form:
                    print("错误: 请求中缺少 taskID 字段")
                    return 'Missing taskID field', 400
                
                if 'data' not in request.form:
                    print("错误: 请求中缺少 data 字段")
                    return 'Missing data field', 400
                
                taskID = int(request.form['taskID'])
                data: Dict = json_module.loads(request.form['data'])
                
                print(f"处理任务 {taskID} 的 tc 响应")
                print(f"数据内容: {data}")
                
                for name, ret in data.items():
                    if 'msg' in ret:
                        print('emulated node ' + name + ' tc failed, err:')
                        print(ret['msg'])
                    elif 'number' in ret:
                        print('emulated node ' + name + ' tc succeed')
                        with self.controller.lock:
                            if taskID in self.controller.task:
                                self.controller.task[taskID].deployedCount += int(ret['number'])
                                if self.controller.task[taskID].deployedCount == self.controller.task[taskID].virtualLinkNumber:
                                    print('tc finish')
                            else:
                                print(f"警告: 任务 {taskID} 不存在于 self.controller.task 中")
                return '', 200
                
            except KeyError as e:
                print(f"KeyError: 缺少字段 {str(e)}")
                return f'Missing field: {str(e)}', 400
            except ValueError as e:
                print(f"ValueError: {str(e)}")
                return f'Invalid value: {str(e)}', 400
            except Exception as e:
                print(f"处理 /emulated/tc 请求时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return f'Internal error: {str(e)}', 500
        
        @self.controller.flask.route('/bgload/add', methods=['POST'])
        def route_bgload_add():
            """通过HTTP请求添加背景负载到指定的emulator或所有emulator，或从配置文件读取
            
            请求参数（JSON格式）:
            方式1 - 手动指定参数:
            {
                "emulator": "emulator-1",  # emulator名称（可选），如果不提供则为所有emulator添加
                "cpu": 2,                  # CPU核心数（必需，除非使用use_config）
                "ram": 4,                  # 内存大小（必需，除非使用use_config）
                "unit": "G",               # 内存单位，可选，默认"G"（"M"或"G"）
                "image": "stress:latest",   # Docker镜像，可选，默认"stress:latest"
                "auto_launch": false       # 是否自动启动，可选，默认false
            }
            
            方式2 - 从配置文件读取:
            {
                "use_config": true,        # 使用配置文件，必需
                "config_path": "workload_config.json",  # 配置文件路径，可选，默认使用workload_config.json
                "auto_launch": false       # 是否自动启动，可选，默认false
            }
            """
            try:
                if not request.is_json:
                    return json.dumps({'success': False, 'error': '请求必须是JSON格式'}), 400
                
                data = request.get_json()
                auto_launch = data.get('auto_launch', False)
                
                # 如果指定了use_config，则从配置文件读取
                if data.get('use_config', False):
                    # 获取配置文件路径
                    config_path = data.get('config_path')
                    if config_path is None:
                        config_path = os.path.join(self.controller.dirName, 'workload_config.json')
                    elif not os.path.isabs(config_path):
                        # 如果是相对路径，则相对于dirName
                        config_path = os.path.join(self.controller.dirName, config_path)
                    
                    print(f"从配置文件读取背景负载配置: {config_path}")
                    
                    try:
                        with open(config_path, 'r') as f:
                            config = json_module.load(f)
                    except FileNotFoundError:
                        return json.dumps({
                            'success': False,
                            'error': f'配置文件不存在: {config_path}'
                        }), 404
                    except json_module.JSONDecodeError as e:
                        return json.dumps({
                            'success': False,
                            'error': f'配置文件格式错误: {str(e)}'
                        }), 400
                    
                    workloads = config.get('workloads', {})
                    if not workloads:
                        return json.dumps({
                            'success': False,
                            'error': '配置文件中没有workloads配置'
                        }), 400
                    
                    results = {
                        'success': True,
                        'message': '从配置文件添加背景负载',
                        'config_path': config_path,
                        'total_workloads': len(workloads),
                        'succeeded': [],
                        'failed': []
                    }
                    
                    # 为配置文件中每个启用的emulator添加背景负载
                    for emulator_name, workload_info in workloads.items():
                        # 检查是否启用
                        if not workload_info.get('enabled', True):
                            print(f"跳过未启用的负载: {emulator_name}")
                            continue
                        
                        # 检查emulator是否存在
                        if emulator_name not in self.controller.emulator:
                            print(f"警告: emulator {emulator_name} 不存在，跳过")
                            results['failed'].append({
                                'emulator': emulator_name,
                                'error': f'emulator {emulator_name} 不存在'
                            })
                            continue
                        
                        emulator = self.controller.emulator[emulator_name]
                        
                        # 获取负载配置
                        cpu = float(workload_info.get('cpu', 1))
                        ram = int(workload_info.get('ram', 1))
                        unit = workload_info.get('unit', 'G')
                        image = workload_info.get('image', 'stress:latest')
                        
                        # 内存单位转换
                        if unit == 'M':
                            ram = ram // 1024
                        
                        try:
                            # 创建背景负载节点
                            bg_node_name = f"{emulator_name}_bgload_{int(time.time())}"
                            bg_node = BackgroundLoadNode(
                                name=bg_node_name,
                                emulator_name=emulator_name,
                                cpu=cpu,
                                ram=ram,
                                image=image
                            )
                            
                            # 添加到emulator
                            emulator.add_background_load(bg_node)
                            print(f"成功添加背景负载到 {emulator_name}: CPU={cpu}, RAM={ram}GB")
                            
                            result_item = {
                                'emulator': emulator_name,
                                'bgload_name': bg_node_name,
                                'cpu': cpu,
                                'ram': ram,
                                'unit': 'G',
                                'image': image
                            }
                            
                            # 如果设置了自动启动，则启动背景负载
                            if auto_launch:
                                try:
                                    yml_path = emulator.save_background_load_yml(self.controller.dirName)
                                    if yml_path:
                                        self.controller._Controller__launch_background_load(emulator, yml_path)
                                        bg_node.is_running = True
                                        result_item['launched'] = True
                                except Exception as e:
                                    result_item['launched'] = False
                                    result_item['launch_error'] = str(e)
                                    print(f"自动启动背景负载失败 ({emulator_name}): {e}")
                            
                            results['succeeded'].append(result_item)
                            
                        except Exception as e:
                            error_msg = f'为 {emulator_name} 添加背景负载失败: {str(e)}'
                            print(error_msg)
                            results['failed'].append({
                                'emulator': emulator_name,
                                'error': str(e)
                            })
                    
                    # 如果有失败的，整体成功状态取决于是否有成功的
                    if results['failed'] and not results['succeeded']:
                        results['success'] = False
                        results['message'] = '所有emulator添加背景负载都失败'
                    elif results['failed']:
                        results['message'] = f"部分emulator添加成功 ({len(results['succeeded'])}/{len(workloads)})"
                    
                    # 处理网络拓扑带宽信息
                    links = config.get('links', {})
                    if links:
                        print(f"正在加载网络拓扑带宽配置...")
                        links_results = {
                            'succeeded': [],
                            'failed': []
                        }
                        for source_emulator, connections in links.items():
                            # 检查源emulator是否存在
                            if source_emulator not in self.controller.emulator:
                                print(f"警告: emulator {source_emulator} 不存在，跳过其网络拓扑配置")
                                links_results['failed'].append({
                                    'source': source_emulator,
                                    'error': f'emulator {source_emulator} 不存在'
                                })
                                continue
                            
                            for connection in connections:
                                dest_emulator = connection.get('dest')
                                bw_str = connection.get('bw', '0mbps')
                                
                                # 检查目标emulator是否存在
                                if dest_emulator not in self.controller.emulator:
                                    print(f"警告: 目标emulator {dest_emulator} 不存在，跳过连接 {source_emulator} -> {dest_emulator}")
                                    links_results['failed'].append({
                                        'source': source_emulator,
                                        'dest': dest_emulator,
                                        'error': f'目标emulator {dest_emulator} 不存在'
                                    })
                                    continue
                                
                                # 解析带宽值（支持mbps格式）
                                try:
                                    bw = int(bw_str.replace('mbps', '').replace('Mbps', '').replace('MBPS', ''))
                                except (ValueError, AttributeError):
                                    print(f"警告: 无法解析带宽值 '{bw_str}'，跳过连接 {source_emulator} -> {dest_emulator}")
                                    links_results['failed'].append({
                                        'source': source_emulator,
                                        'dest': dest_emulator,
                                        'error': f"无法解析带宽值 '{bw_str}'"
                                    })
                                    continue
                                
                                # 更新带宽使用量（不创建tc链接，只更新已用带宽记录）
                                try:
                                    self.controller.add_emulator_bw_pre_map(source_emulator, dest_emulator, bw)
                                    print(f"成功更新带宽使用量: {source_emulator} -> {dest_emulator} = {bw}mbps")
                                    links_results['succeeded'].append({
                                        'source': source_emulator,
                                        'dest': dest_emulator,
                                        'bandwidth': f'{bw}mbps'
                                    })
                                except Exception as e:
                                    print(f"更新带宽使用量失败 {source_emulator} -> {dest_emulator}: {e}")
                                    links_results['failed'].append({
                                        'source': source_emulator,
                                        'dest': dest_emulator,
                                        'error': str(e)
                                    })
                        
                        # 将网络拓扑结果添加到返回结果中
                        results['links'] = links_results
                        if links_results['succeeded']:
                            results['message'] += f", 成功设置 {len(links_results['succeeded'])} 条网络链路"
                    
                    return json.dumps(results), 200
                
                # 手动指定参数的方式（原有逻辑）
                # 检查必需参数
                if 'cpu' not in data:
                    return json.dumps({'success': False, 'error': '缺少必需参数: cpu'}), 400
                if 'ram' not in data:
                    return json.dumps({'success': False, 'error': '缺少必需参数: ram'}), 400
                
                emulator_name = data.get('emulator')  # 改为可选参数
                cpu = float(data['cpu'])
                ram = int(data['ram'])
                unit = data.get('unit', 'G')
                image = data.get('image', 'stress:latest')
                
                # 内存单位转换
                if unit == 'M':
                    ram = ram // 1024
                
                # 如果指定了emulator，只为该emulator添加
                if emulator_name:
                    # 检查emulator是否存在
                    if emulator_name not in self.controller.emulator:
                        return json.dumps({
                            'success': False, 
                            'error': f'emulator {emulator_name} 不存在'
                        }), 404
                    
                    emulator = self.controller.emulator[emulator_name]
                    
                    # 创建背景负载节点
                    bg_node_name = f"{emulator_name}_bgload_{int(time.time())}"
                    bg_node = BackgroundLoadNode(
                        name=bg_node_name,
                        emulator_name=emulator_name,
                        cpu=cpu,
                        ram=ram,
                        image=image
                    )
                    
                    # 添加到emulator
                    try:
                        emulator.add_background_load(bg_node)
                        print(f"成功添加背景负载到 {emulator_name}: CPU={cpu}, RAM={ram}GB")
                        
                        result = {
                            'success': True,
                            'message': f'背景负载已添加到 {emulator_name}',
                            'bgload_name': bg_node_name,
                            'cpu': cpu,
                            'ram': ram,
                            'unit': 'G',
                            'image': image
                        }
                        
                        # 如果设置了自动启动，则启动背景负载
                        if auto_launch:
                            try:
                                # 只启动这个新添加的背景负载
                                yml_path = emulator.save_background_load_yml(self.controller.dirName)
                                if yml_path:
                                    self.controller._Controller__launch_background_load(emulator, yml_path)
                                    bg_node.is_running = True
                                    result['launched'] = True
                                    result['message'] += '，并已启动'
                            except Exception as e:
                                result['launched'] = False
                                result['launch_error'] = str(e)
                                print(f"自动启动背景负载失败: {e}")
                        
                        return json.dumps(result), 200
                        
                    except Exception as e:
                        return json.dumps({
                            'success': False,
                            'error': f'添加背景负载失败: {str(e)}'
                        }), 400
                else:
                    # 为所有emulator添加背景负载
                    if not self.controller.emulator:
                        return json.dumps({
                            'success': False,
                            'error': '没有可用的emulator'
                        }), 400
                    
                    results = {
                        'success': True,
                        'message': '为所有emulator添加背景负载',
                        'total_emulators': len(self.controller.emulator),
                        'succeeded': [],
                        'failed': []
                    }
                    
                    # 为每个emulator添加背景负载
                    for name, emulator in self.controller.emulator.items():
                        try:
                            # 创建背景负载节点
                            bg_node_name = f"{name}_bgload_{int(time.time())}"
                            bg_node = BackgroundLoadNode(
                                name=bg_node_name,
                                emulator_name=name,
                                cpu=cpu,
                                ram=ram,
                                image=image
                            )
                            
                            # 添加到emulator
                            emulator.add_background_load(bg_node)
                            print(f"成功添加背景负载到 {name}: CPU={cpu}, RAM={ram}GB")
                            
                            result_item = {
                                'emulator': name,
                                'bgload_name': bg_node_name,
                                'cpu': cpu,
                                'ram': ram,
                                'unit': 'G',
                                'image': image
                            }
                            
                            # 如果设置了自动启动，则启动背景负载
                            if auto_launch:
                                try:
                                    yml_path = emulator.save_background_load_yml(self.controller.dirName)
                                    if yml_path:
                                        self.controller._Controller__launch_background_load(emulator, yml_path)
                                        bg_node.is_running = True
                                        result_item['launched'] = True
                                except Exception as e:
                                    result_item['launched'] = False
                                    result_item['launch_error'] = str(e)
                                    print(f"自动启动背景负载失败 ({name}): {e}")
                            
                            results['succeeded'].append(result_item)
                            
                        except Exception as e:
                            error_msg = f'为 {name} 添加背景负载失败: {str(e)}'
                            print(error_msg)
                            results['failed'].append({
                                'emulator': name,
                                'error': str(e)
                            })
                    
                    # 如果有失败的，整体成功状态取决于是否有成功的
                    if results['failed'] and not results['succeeded']:
                        results['success'] = False
                        results['message'] = '所有emulator添加背景负载都失败'
                    elif results['failed']:
                        results['message'] = f"部分emulator添加成功 ({len(results['succeeded'])}/{len(self.controller.emulator)})"
                    
                    return json.dumps(results), 200
                    
            except Exception as e:
                print(f"处理添加背景负载请求时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return json.dumps({
                    'success': False,
                    'error': f'内部错误: {str(e)}'
                }), 500
        
        @self.controller.flask.route('/bgload/launch', methods=['POST'])
        def route_bgload_launch():
            """启动指定emulator上的背景负载
            
            请求参数（JSON格式）:
            {
                "emulator": "emulator-1"  # emulator名称，可选，如果不提供则启动所有
            }
            """
            try:
                if not request.is_json:
                    data = {}
                else:
                    data = request.get_json() or {}
                
                emulator_name = data.get('emulator')
                
                if emulator_name:
                    # 启动指定emulator的背景负载
                    if emulator_name not in self.controller.emulator:
                        return json.dumps({
                            'success': False,
                            'error': f'emulator {emulator_name} 不存在'
                        }), 404
                    
                    emulator = self.controller.emulator[emulator_name]
                    if not emulator.bgLoadNode:
                        return json.dumps({
                            'success': False,
                            'error': f'emulator {emulator_name} 没有背景负载'
                        }), 400
                    
                    try:
                        yml_path = emulator.save_background_load_yml(self.controller.dirName)
                        if yml_path:
                            self.controller._Controller__launch_background_load(emulator, yml_path)
                            for bg_node in emulator.bgLoadNode.values():
                                bg_node.is_running = True
                            return json.dumps({
                                'success': True,
                                'message': f'背景负载已在 {emulator_name} 上启动'
                            }), 200
                        else:
                            return json.dumps({
                                'success': False,
                                'error': '保存yml文件失败'
                            }), 500
                    except Exception as e:
                        return json.dumps({
                            'success': False,
                            'error': f'启动失败: {str(e)}'
                        }), 500
                else:
                    # 启动所有emulator的背景负载
                    success = self.controller.launch_background_loads()
                    return json.dumps({
                        'success': success,
                        'message': '所有背景负载已启动' if success else '部分背景负载启动失败'
                    }), 200
                    
            except Exception as e:
                print(f"处理启动背景负载请求时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return json.dumps({
                    'success': False,
                    'error': f'内部错误: {str(e)}'
                }), 500
        
        @self.controller.flask.route('/bgload/stop', methods=['POST'])
        def route_bgload_stop():
            """停止指定emulator上的背景负载
            
            请求参数（JSON格式）:
            {
                "emulator": "emulator-1"  # emulator名称，可选，如果不提供则停止所有
            }
            """
            try:
                if not request.is_json:
                    data = {}
                else:
                    data = request.get_json() or {}
                
                emulator_name = data.get('emulator')
                
                if emulator_name:
                    # 停止指定emulator的背景负载
                    if emulator_name not in self.controller.emulator:
                        return json.dumps({
                            'success': False,
                            'error': f'emulator {emulator_name} 不存在'
                        }), 404
                    
                    emulator = self.controller.emulator[emulator_name]
                    if not emulator.bgLoadNode:
                        return json.dumps({
                            'success': False,
                            'error': f'emulator {emulator_name} 没有背景负载'
                        }), 400
                    
                    try:
                        self.controller._Controller__stop_background_load(emulator)
                        for bg_node in emulator.bgLoadNode.values():
                            bg_node.is_running = False
                        return json.dumps({
                            'success': True,
                            'message': f'背景负载已在 {emulator_name} 上停止'
                        }), 200
                    except Exception as e:
                        return json.dumps({
                            'success': False,
                            'error': f'停止失败: {str(e)}'
                        }), 500
                else:
                    # 停止所有emulator的背景负载
                    success = self.controller.stop_background_loads()
                    return json.dumps({
                        'success': success,
                        'message': '所有背景负载已停止'
                    }), 200
                    
            except Exception as e:
                print(f"处理停止背景负载请求时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return json.dumps({
                    'success': False,
                    'error': f'内部错误: {str(e)}'
                }), 500
        
        @self.controller.flask.route('/bgload/clear', methods=['POST'])
        def route_bgload_clear():
            """清理指定emulator上的背景负载（停止并删除）
            
            请求参数（JSON格式）:
            {
                "emulator": "emulator-1"  # emulator名称，可选，如果不提供则清理所有
            }
            """
            try:
                if not request.is_json:
                    data = {}
                else:
                    data = request.get_json() or {}
                
                emulator_name = data.get('emulator')
                
                if emulator_name:
                    # 清理指定emulator的背景负载
                    if emulator_name not in self.controller.emulator:
                        return json.dumps({
                            'success': False,
                            'error': f'emulator {emulator_name} 不存在'
                        }), 404
                    
                    emulator = self.controller.emulator[emulator_name]
                    if not emulator.bgLoadNode:
                        return json.dumps({
                            'success': False,
                            'error': f'emulator {emulator_name} 没有背景负载'
                        }), 400
                    
                    try:
                        self.controller._Controller__clear_background_load(emulator)
                        # 清理背景负载信息
                        emulator.bgCpuUsed = 0.0
                        emulator.bgRamUsed = 0
                        emulator.bgLoadNode.clear()
                        return json.dumps({
                            'success': True,
                            'message': f'背景负载已在 {emulator_name} 上清理'
                        }), 200
                    except Exception as e:
                        return json.dumps({
                            'success': False,
                            'error': f'清理失败: {str(e)}'
                        }), 500
                else:
                    # 清理所有emulator的背景负载
                    success = self.controller.clear_background_loads()
                    return json.dumps({
                        'success': success,
                        'message': '所有背景负载已清理'
                    }), 200
                    
            except Exception as e:
                print(f"处理清理背景负载请求时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return json.dumps({
                    'success': False,
                    'error': f'内部错误: {str(e)}'
                }), 500
        
        @self.controller.flask.route('/bgload/status', methods=['GET'])
        def route_bgload_status():
            """获取背景负载的状态
            
            查询参数:
            - emulator: emulator名称，可选，如果不提供则返回所有emulator的状态
            """
            try:
                emulator_name = request.args.get('emulator')
                
                if emulator_name:
                    # 返回指定emulator的状态
                    if emulator_name not in self.controller.emulator:
                        return json.dumps({
                            'success': False,
                            'error': f'emulator {emulator_name} 不存在'
                        }), 404
                    
                    emulator = self.controller.emulator[emulator_name]
                    bgloads = []
                    for bg_node in emulator.bgLoadNode.values():
                        bgloads.append({
                            'name': bg_node.name,
                            'cpu': bg_node.cpu,
                            'ram': bg_node.ram,
                            'image': bg_node.image,
                            'is_running': bg_node.is_running
                        })
                    
                    return json.dumps({
                        'success': True,
                        'emulator': emulator_name,
                        'bgloads': bgloads,
                        'total_cpu_used': emulator.bgCpuUsed,
                        'total_ram_used': emulator.bgRamUsed
                    }), 200
                else:
                    # 返回所有emulator的状态
                    all_status = {}
                    for name, emulator in self.controller.emulator.items():
                        bgloads = []
                        for bg_node in emulator.bgLoadNode.values():
                            bgloads.append({
                                'name': bg_node.name,
                                'cpu': bg_node.cpu,
                                'ram': bg_node.ram,
                                'image': bg_node.image,
                                'is_running': bg_node.is_running
                            })
                        all_status[name] = {
                            'bgloads': bgloads,
                            'total_cpu_used': emulator.bgCpuUsed,
                            'total_ram_used': emulator.bgRamUsed
                        }
                    
                    return json.dumps({
                        'success': True,
                        'all_emulators': all_status
                    }), 200
                    
            except Exception as e:
                print(f"处理查询背景负载状态请求时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return json.dumps({
                    'success': False,
                    'error': f'内部错误: {str(e)}'
                }), 500
    
    def __send_conf(self, taskID: int, conf_type: str):
        dml_file_conf = os.path.join(self.controller.dirName, 'dml_file/conf', str(taskID))
        for pn in self.controller.task[taskID].pNode.values():
            file_path = os.path.join(dml_file_conf, pn.name + '_' + conf_type + '.conf')
            with open(file_path, 'r') as f:
                print('sent ' + conf_type + ' conf to ' + pn.name)
                send_data('POST', '/conf/' + conf_type, pn.ip, pn.hostPort, files={'conf': f})
        for en in self.controller.task[taskID].eNode.values():
            file_path = os.path.join(dml_file_conf, en.name + '_' + conf_type + '.conf')
            with open(file_path, 'r') as f:
                print('sent ' + conf_type + ' conf to ' + en.name + ' ip:' + en.ip + ' port:' + str(en.hostPort))
                send_data('POST', '/conf/' + conf_type, en.ip, en.hostPort, files={'conf': f})
        
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

    def __after_log(self):
        time.sleep(5)
        print('try to stop all emulated nodes')
        self.__stop_all_emulated()