import os

from base.scheduler import Scheduler
from base import default_testbed
from base.manager import Manager

# path of this file.
# dirName = os.path.abspath (os.path.dirname (__file__))
dirName = '/home/qianguo/Edge-Scheduler/Controller'

# we made up the following physical hardware so this example is NOT runnable.
if __name__ == '__main__':
	controller = default_testbed (ip='222.201.187.50', dir_name=dirName, manager=Manager, scheduler=Scheduler)
	nfsApp = controller.add_nfs (tag='dml_app', path=os.path.join (dirName, 'dml_app'))
	nfsDataset = controller.add_nfs (tag='dataset', path=os.path.join (dirName, 'dataset'))
	controller.export_nfs()
	# 初始化模拟器
	emu1 = controller.add_emulator ('emulator-1', '222.201.187.51', cpu=128, ram=256, unit='G')
	emu2 = controller.add_emulator ('emulator-2', '222.201.187.52', cpu=128, ram=256, unit='G')
	controller.send_emulator_info() # 发送模拟器信息
	# 添加物理链路
	controller.add_emulator_bw('emulator-1', 'emulator-2', bw=1000)  
	controller.add_emulator_bw('emulator-2', 'emulator-1', bw=1000)
	
	controller.flask.run(host='0.0.0.0', port=controller.port, threaded=True)