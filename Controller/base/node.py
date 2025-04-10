from concurrent.futures import ThreadPoolExecutor
import ipaddress
import os
import threading
from typing import Dict, List, Type
from flask import Flask, request

from .nfs import Nfs


class Worker(object):
    """
    worker的基类
    """
    def __init__(self, ID: int, name: str, ip: str):
        self.idW: int = ID  # worker ID.
        self.nameW: str = name  # worker name.
        self.ipW: str = ip  # worker IP.
        self.connected: Dict[int, List[int]]  # dst worker ID to real link ID set. 假设worker间有多种连接方式，每种连接方式有一个ID


    def check_network_range(self, network: str):
        subnet = ipaddress.ip_network(network, strict=False)
        subnet_start, subnet_end = [x for x in subnet.hosts()][0], [x for x in subnet.hosts()][-1]
        assert subnet_start <= ipaddress.ip_address(self.ipW) <= subnet_end, Exception(
            self.ipW + ' is not in the subnet of ' + network)

class Node(object):
    """
    node的基类
    """
    def __init__(self, ID: int, name: str, taskID: int,ip: str, nic: str, working_dir: str,
                 cmd: List[str], node_port: int, host_port: int):
        self.id: int = ID
        self.name: str = name
        self.tid: int = taskID
        self.ip: str = ip
        self.nic: str = nic
        self.workingDir: str = working_dir
        self.cmd: List[str] = cmd
        self.nodePort: int = node_port  # application listens on $(node_port). 改为node
        self.hostPort: int = host_port  # network requests are sent to $(host_port) of worker.
        self.variable: Dict[str, str] = {}  # system environment variable.
        self.tc: Dict[str, str] = {}  # dst name to dst bw.
        self.tcIP: Dict[str, str] = {}  # dst name to dst ip.
        self.tcPort: Dict[str, int] = {}  # dst name to dst host port.

        self.add_var({
            'EDGE_TB_ID': str(ID),
            'NET_NODE_NAME': name,
            'NODE_PORT': str(node_port)
        })

    def add_var(self, var_dict: Dict[str, str]):
        self.variable.update(var_dict)

    def link_to(self, name: str, bw: str, ip: str, port: int):
        assert name not in self.tc, Exception(self.name + ' already has a link to ' + name)
        self.tc[name] = bw
        self.tcIP[name] = ip
        self.tcPort[name] = port

class PhysicalNode(Node, Worker):
    """
    物理节点既是worker又是node，可以理解为只有一个node的worker
    好像不一定会用上，新的边缘设备可以放置多个容器
    """
    def __init__(self):
        pass

class EmulatedNode(Node):
    """
    一个用容器实现的node，部署在emulator中
    """
    def __init__(self, ID: int, name: str, taskID: int,nic: str, working_dir: str, cmd: List[str], node_port: int,
                 base_host_port: int, image: str, cpu: int, ram: int):
        # host port is related to node's id, dml port maps to host port in emulator.
        super().__init__(ID, name, taskID,'', nic, working_dir, cmd, node_port, base_host_port + ID)
        self.image: str = image  # Docker image.
        self.cpu = cpu  # cpu thread.
        self.ram = ram  # MB of memory.
        self.volume: Dict[str, str] = {}  # host path or nfs tag to node path.

    def mount_local_path(self, local_path: str, node_path: str):
        assert node_path[0] == '/', Exception(node_path + ' is not an absolute path')
        self.volume[local_path] = node_path

    def mount_nfs(self, nfs: Nfs, node_path: str):
        assert node_path[0] == '/', Exception(node_path + ' is not an absolute path')
        self.volume[nfs.tag] = node_path + '/:ro'

class Emulator(Worker):
    """
    可以部署多个emulatedNode
    """
    def __init__(self, ID: int, name: str, ip: str, cpu: int, ram: int, ip_controller: str):
        super().__init__(ID, name, ip)
        self.cpu: int = cpu  # cpu thread.
        self.ram: int = ram  # MB of memory.
        self.ipController: str = ip_controller  # ip of the task controller.
        self.cpuPreMap: int = 0  # allocated cpu.
        self.ramPreMap: int = 0  # allocated ram.
        self.nfs: List[Nfs] = []  # mounted nfs.
        self.eNode: Dict[str, EmulatedNode] = {}  # emulated node's name to emulated node object.
        #self.curr_cpu: int = 0	# 服务器目前用到的cpuId

    def mount_nfs(self, nfs: Nfs):
        assert nfs not in self.nfs, Exception(nfs.tag + ' has been mounted')
        self.check_network_range(nfs.subnet)
        self.nfs.append(nfs)

    def check_resource(self, name: str, cpu: int, ram: int):
        assert self.cpu - self.cpuPreMap >= cpu and self.ram - self.ramPreMap >= ram, Exception(
            self.nameW + '\'s cpu or ram is not enough for ' + name)

    def add_node(self, en: EmulatedNode):
        assert en.name not in self.eNode, Exception(en.name + ' has been added')
        en.ip = self.ipW
        self.cpuPreMap += en.cpu
        self.ramPreMap += en.ram
        self.eNode[en.name] = en

    def delete_node(self, en: EmulatedNode):
        assert en.name in self.eNode, Exception(en.name + ' is not existed')
        self.cpuPreMap -= en.cpu
        self.ramPreMap -= en.ram
        del self.eNode[en.name]
    
    def save_yml(self, path: str, taskID: int):
        if not self.eNode:
            return
        str_yml = 'version: "2.1"\n'
        if self.nfs:
            str_yml += 'volumes:\n'
            for nfs in self.nfs:
                str_yml = str_yml \
                          + '  ' + nfs.tag + ':\n' \
                          + '    driver_opts:\n' \
                          + '      type: "nfs"\n' \
                          + '      o: "addr=' + self.ipController + ',ro"\n' \
                          + '      device: ":' + nfs.path + '"\n'
        self.curr_cpu = 0
        str_yml += 'services:\n'
        for en in self.eNode.values():
            if en.tid != taskID:
                str_yml = str_yml \
                        + '  ' + en.name + ':\n' \
                        + '    container_name: ' + en.name + '\n' \
                        + '    image: ' + en.image + '\n' \
                        + '    working_dir: ' + en.workingDir + '\n' \
                        + '    stdin_open: true\n' \
                        + '    tty: true\n' \
                        + '    cap_add:\n' \
                        + '      - NET_ADMIN\n' \
                        + '    cpuset: ' + str(self.curr_cpu) + '-' + str(self.curr_cpu + en.cpu - 1) + '\n' \
                        + '    mem_limit: ' + str(en.ram) + 'M\n'
                self.curr_cpu += en.cpu
                str_yml += '    environment:\n'
                for key in en.variable:
                    str_yml += '      - ' + key + '=' + en.variable[key] + '\n'
                str_yml = str_yml \
                        + '    healthcheck:\n' \
                        + '      test: curl -f http://localhost:' + str(en.nodePort) + '/hi\n'
                str_yml = str_yml \
                        + '    ports:\n' \
                        + '      - "' + str(en.hostPort) + ':' + str(en.nodePort) + '"\n'
                if en.volume:
                    str_yml += '    volumes:\n'
                    for v in en.volume:
                        str_yml += '      - ' + v + ':' + en.volume[v] + '\n'
                if en.cmd:
                    str_yml += '    command: ' + ' '.join(en.cmd) + '\n'

        # save as yml file
        yml_name = os.path.join(path, self.nameW + '_' + str(taskID) + '.yml')
        with open(yml_name, 'w') as f:
            f.writelines(str_yml)

