import argparse
import json
import os

from conf_utils import read_json, load_node_info


class Conf:
	def __init__ (self, name, sync, epoch):
		self.name = name
		self.sync = sync
		self.epoch = epoch
		self.connect = {}

	def __hash__ (self):
		return hash (self.name)

	def to_json (self):
		return {
			'sync': self.sync,
			'epoch': self.epoch,
			'connect': self.connect,
		}


def gen_conf (all_node, conf_json, link_json, output_path, args_taskid):
	node_conf_map = {}

	for node in conf_json ['node_list']:
		name = node ['name']
		assert name not in node_conf_map, Exception (
			'duplicate node: ' + name)
		conf = node_conf_map [name] = Conf (name, conf_json ['sync'], node ['epoch'])

		if name in link_json:
			link_list = link_json [name]
			for link in link_list:
				dest = link ['dest']
				destName = args_taskid + '_' + dest
				# print('destName:', destName)
				assert destName in all_node, Exception ('no such node called ' + destName)
				assert destName not in conf.connect, Exception (
					'duplicate link from ' + name + ' to ' + destName)
				conf.connect [destName] = all_node [destName].ip + ':' + str (all_node [destName].port)

	for name in node_conf_map:
		conf_path = os.path.join (output_path, args.taskid + '_' + name + '_structure.conf')
		with open (conf_path, 'w') as f:
			f.writelines (json.dumps (node_conf_map [name].to_json (), indent=2))


if __name__ == '__main__':
	dirname = os.path.abspath (os.path.dirname (__file__))
	controller_path = os.path.abspath(os.path.join(dirname, '../../'))
	parser = argparse.ArgumentParser ()
	parser.add_argument ('-s', '--structure', dest='structure', required=True, type=str,
		help='./relative/path/to/structure/json/file')
	parser.add_argument ('-l', '--link', dest='link', required=False, type=str,
		default='links.json', help='./relative/path/to/link/json/file, default = ../links.json')
	parser.add_argument ('-n', '--node', dest='node', required=False, type=str,
		default='../node_info.json', help='./relative/path/to/node/info/json/file, default = ../node_info.json')
	parser.add_argument ('-o', '--output', dest='output', required=False, type=str,
		default='/dml_file/conf', help='./relative/path/to/output/folder/, default = ../dml_file/conf/')
	parser.add_argument('-t', '--taskid', dest='taskid', required=False, type=str,
    	default='default', help='task id for distinguishing different tasks')
	args = parser.parse_args()

	def insert_taskid_to_path(path, taskid):
		# 将路径分割成目录和文件名
		dir_path, filename = os.path.split(path)
		# 在目录中插入taskid
		new_path = os.path.join(dir_path, taskid, filename)
		return new_path

	# 处理node路径
	if args.node[0] != '/':
		base_path = os.path.join(dirname, args.node)
	else:
		base_path = args.node
	pathNode = controller_path + '/node_info/node_info_' + args.taskid + '.json'
	# print('pathNode:', pathNode)
	_, _, allNode = load_node_info(pathNode)

	# 处理structure路径
	if args.structure[0] != '/':
		base_path = os.path.join(dirname, args.structure)
	else:
		base_path = args.structure
	pathStructure = controller_path + '/dml_tool/' + args.taskid + '/' + args.structure
	# print('pathStructure:', pathStructure)
	confJson = read_json(pathStructure)

	# 处理link路径
	if args.link[0] != '/':
		base_path = os.path.join(dirname, args.link)
	else:
		base_path = args.link
	pathLink = controller_path + '/task_links/' + args.taskid + '/' + args.link
	# print('pathLink:', pathLink)
	linkJson = read_json(pathLink)

	# 处理output路径
	if args.output[0] != '/':
		base_path = os.path.join(dirname, args.output)
	else:
		base_path = args.output
	pathOutput = controller_path + args.output + '/' + args.taskid
	# print('pathOutput:', pathOutput)
	gen_conf(allNode, confJson, linkJson, pathOutput, args.taskid)
