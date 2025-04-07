from typing import Type

from .manager import Manager
from .controller import Controller
from .scheduler import Scheduler


def default_testbed (ip: str, dir_name: str, manager: Manager, scheduler: Scheduler,
		host_port: int = 8000) -> Controller:
	"""
	Default settings suitable for most situations.

	:param ip: ip of the testbed controller.
	:param dir_name: yml file saved in $(dir_name) folder.
	:param manager_class: class of Manager.
	:param host_port: emulated node maps dml port to emulator's host port
	starting from $(host_port).
	"""
	return Controller (ip, host_port, dir_name, manager, scheduler)