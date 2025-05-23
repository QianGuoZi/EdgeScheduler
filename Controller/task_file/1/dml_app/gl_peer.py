import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask, request

import dml_utils
import worker_utils
from nns.nn_fashion_mnist import nn  # configurable parameter, from nns.whatever import nn.

dirname = os.path.abspath (os.path.dirname (__file__))

dml_port = os.getenv ('NODE_PORT')
ctl_addr = os.getenv ('NET_CTL_ADDRESS')
agent_addr = os.getenv ('NET_AGENT_ADDRESS')
node_name = os.getenv ('NET_NODE_NAME')
task_id = os.getenv('NET_TASK_ID')

input_shape = nn.input_shape
controller_path = os.path.abspath (os.path.join(dirname, '../..'))
log_dir = os.path.abspath(os.path.join(controller_path, 'dml_file/log', task_id))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.abspath (os.path.join (controller_path, 'dml_file/log/', task_id, node_name + '.log'))
print ('log_file:', log_file)
worker_utils.set_log (log_file)
conf = {}
peer_list = []
# configurable parameter, specify the dataset path.
train_path = os.path.join (controller_path, 'dataset/FASHION_MNIST/train_data')
train_images: np.ndarray
train_labels: np.ndarray
# configurable parameter, specify the dataset path.
test_path = os.path.join (controller_path, 'dataset/FASHION_MNIST/test_data')
test_images: np.ndarray
test_labels: np.ndarray

app = Flask (__name__)
lock = threading.RLock ()
executor = ThreadPoolExecutor (1)


# if this is container, docker will send a GET to here every 30s
# this ability is defined in controller/base/node.py, Class Emulator, save_yml (), healthcheck.
@app.route ('/hi', methods=['GET'])
def route_hi ():
	# 给agent发送heartbeat，若为第一次发送，需要部署tc设置
	# send a heartbeat to the agent.
	# when the agent receives the heartbeat of a container for the first time,
	# it will deploy the container's tc settings.
	# please ensure that your app implements this function, i.e.,
	# receiving docker healthcheck and sending heartbeat to the agent.
	worker_utils.heartbeat (agent_addr, node_name)
	return 'this is node ' + node_name + '\n'


@app.route ('/conf/dataset', methods=['POST'])
def route_conf_d ():
	# 加载数据集（？）
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	print ('POST at /conf/dataset')
	print ('conf:', conf)

	global train_images, train_labels
	train_images, train_labels = dml_utils.load_data (train_path, conf ['train_start_index'],
		conf ['train_len'], input_shape)
	global test_images, test_labels
	test_images, test_labels = dml_utils.load_data (test_path, conf ['test_start_index'],
		conf ['test_len'], input_shape)
	
	conf_dir = os.path.join(controller_path, 'dml_file/conf', task_id)
	os.makedirs(conf_dir, exist_ok=True)

	filename = os.path.join(conf_dir, node_name + '_dataset.conf')
	
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))
	return ''


@app.route ('/conf/structure', methods=['POST'])
def route_conf_s ():
	#  节点结构（？）
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	print ('POST at /conf/structure')
	print ('conf:', conf)

	conf_dir = os.path.join(controller_path, 'dml_file/conf', task_id)
	os.makedirs(conf_dir, exist_ok=True)

	filename = os.path.join(conf_dir, node_name + '_structure.conf')

	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))

	conf ['current_round'] = 0
	peer_list.extend (list (conf ['connect'].keys ()))
	return ''


@app.route ('/log', methods=['GET'])
def route_log ():
	executor.submit (on_route_log)
	return ''


def on_route_log ():
	worker_utils.send_log (ctl_addr, log_file, node_name)


@app.route ('/start', methods=['GET'])
def route_start ():
	print ('GET at /start')
	executor.submit (on_route_start)
	return ''


def on_route_start ():
	_, init_acc = dml_utils.test_on_batch (nn.model, test_images, test_labels, conf ['batch_size'])
	msg = dml_utils.log_acc (init_acc, 0)
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)

	with lock:
		gossip ()


def gossip ():
	peer = dml_utils.random_selection (peer_list, 1)
	worker_utils.log ('gossip to ' + peer [0])
	dml_utils.send_weights (nn.model.get_weights (), '/gossip', peer, conf ['connect'])


@app.route ('/gossip', methods=['POST'])
def route_gossip ():
	print ('POST at /gossip')
	weights = dml_utils.parse_weights (request.files.get ('weights'))
	executor.submit (on_route_gossip, weights)
	return ''


def on_route_gossip (received_weights):
	with lock:
		new_weights = np.add (nn.model.get_weights (), received_weights) / 2
		dml_utils.assign_weights (nn.model, new_weights)

		conf ['current_round'] += 1
		loss_list = dml_utils.train (nn.model, train_images, train_labels, conf ['epoch'],
			conf ['batch_size'])
		last_epoch_loss = loss_list [-1]
		msg = dml_utils.log_loss (last_epoch_loss, conf ['current_round'])
		worker_utils.send_print (ctl_addr, node_name + ': ' + msg)

		_, acc = dml_utils.test_on_batch (nn.model, test_images, test_labels, conf ['batch_size'])
		msg = dml_utils.log_acc (acc, conf ['current_round'])
		worker_utils.send_print (ctl_addr, node_name + ': ' + msg)

		if conf ['current_round'] < conf ['sync']:
			gossip ()


app.run (host='0.0.0.0', port=dml_port, threaded=True)
