from .BasicModule import BasicModule
from torch.nn import DataParallel

class MulGPUDataParallel(DataParallel, BasicModule):
    pass
