import os

from base.scheduler import Scheduler
from base import default_testbed
from base.manager import Manager

# path of this file.
# dirName = os.path.abspath (os.path.dirname (__file__))
dirName = '/home/qianguo/Edge-Scheduler/Controller'

# we made up the following physical hardware so this example is NOT runnable.
if __name__ == '__main__':
	controller = default_testbed (ip='100.68.1.50', dir_name=dirName, manager=Manager, scheduler=Scheduler)
	# controller = default_testbed (ip='222.201.187.50', dir_name=dirName, manager=Manager, scheduler=Scheduler)
	nfsApp = controller.add_nfs (tag='dml_app', path=os.path.join (dirName, 'dml_app'))
	nfsDataset = controller.add_nfs (tag='dataset', path=os.path.join (dirName, 'dataset'))
	controller.export_nfs()
	# 初始化模拟器（10台设备）
	# 设备组1: 100.68.1.x
	emu1 = controller.add_emulator('emulator-1', '100.68.1.3', cpu=4, ram=256, unit='G')
	emu2 = controller.add_emulator('emulator-2', '100.68.1.4', cpu=4, ram=256, unit='G')
	emu3 = controller.add_emulator('emulator-3', '100.68.1.5', cpu=4, ram=256, unit='G')
	emu4 = controller.add_emulator('emulator-4', '100.68.1.6', cpu=4, ram=256, unit='G')
	# 设备组2: 100.68.2.x
	emu5 = controller.add_emulator('emulator-5', '100.68.2.1', cpu=4, ram=256, unit='G')
	emu6 = controller.add_emulator('emulator-6', '100.68.2.2', cpu=4, ram=256, unit='G')
	emu7 = controller.add_emulator('emulator-7', '100.68.2.3', cpu=4, ram=256, unit='G')
	emu8 = controller.add_emulator('emulator-8', '100.68.2.4', cpu=4, ram=256, unit='G')
	emu9 = controller.add_emulator('emulator-9', '100.68.2.5', cpu=4, ram=256, unit='G')
	emu10 = controller.add_emulator('emulator-10', '100.68.2.6', cpu=4, ram=256, unit='G')
	controller.send_emulator_info()  # 发送模拟器信息

	# 添加物理链路（两两之间双向带宽）
	emulators = ['emulator-1', 'emulator-2', 'emulator-3', 'emulator-4', 'emulator-5',
	             'emulator-6', 'emulator-7', 'emulator-8', 'emulator-9', 'emulator-10']
	for i, emu_a in enumerate(emulators):
		for j, emu_b in enumerate(emulators):
			if i != j:
				controller.add_emulator_bw(emu_a, emu_b, bw=1000)
	
	controller.flask.run(host='0.0.0.0', port=controller.port, threaded=True)