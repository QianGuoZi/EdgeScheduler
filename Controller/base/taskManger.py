import abc


class Manager(metaclass=abc.ABCMeta):
    """
    负责任务的部署、资源调整、任务状态监控等
    """
    def __init__(self):
        pass