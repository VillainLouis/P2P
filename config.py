import os
from typing import List
import paramiko
from scp import SCPClient
from torch.utils.tensorboard import SummaryWriter
from comm_utils import *


class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    def __init__(self, config, rank):
        #这个config就是后面的client_config
        self.config = config
        self.rank = rank

    async def send_data(self, data, comm, epoch_idx):
        await send_data(comm, data, self.rank, epoch_idx)    

    async def send_init_config(self, comm, epoch_idx):
        print("before send", self.rank, "tag:", epoch_idx)
        await send_data(comm, self.config, self.rank, epoch_idx)    

    async def get_data(self, comm, epoch_idx):
        self.config.worker_paras = await get_data(comm, self.rank, epoch_idx)

class CommonConfig:
    def __init__(self):
        self.model_type = None
        self.dataset_type = None
        self.batch_size = None
        self.data_pattern = None
        self.lr = None
        self.decay_rate = None
        self.min_lr = None
        self.epoch = None
        self.momentum=None
        self.weight_decay=None
        self.para = None
        self.data_path = None
        self.neighbor_paras = dict()
        self.neighbor_info = dict() # 存的收到的层拉取信息，即需要发送给对应邻居的层
        self.comm_neighbors = None # list() comm 通信域中的邻居list, [rank号]
        self.train_loss = None
        self.tag=None
        #这里用来存worker的
        # 下面的都是用于层选择的信息
        self.older_models=Older_Models(3)
        self.neighbor_bandwidth=dict()
        self.neighbor_distribution=dict() # 每个邻居或自己的分布，都是一个长度为class number的list，每个值代表了每类数据的百分比
        self.num_layers=None
        self.partition_sizes=None
        self.layer_names=list()

# 定义Older_Models类，用于记录之前的模型参数，不足窗口大小的时候不会滑动
class Older_Models():
    def __init__(self, window_size):
        self.models = list()
        self.window_size = window_size
        self.index = 0 # 指向最老的模型，(index - 1) % window_size就是最新的模型
        self.size = 0

    def add_model(self, model_dict):
        if self.size < self.window_size:
            self.models.append(model_dict)
            self.size += 1
        else:
            self.models[self.index] = model_dict
            self.index = (self.index + 1) % self.size # 这里和下面的window_size与size等价
    
    def get_last_model_dict(self):
        return self.models[(self.index - 1) % self.size]

class ClientConfig:
    def __init__(self,
                common_config,
                custom: dict = dict()
                ):
        self.para = None
        self.train_data_idxes = None
        self.common_config=common_config

        self.average_weight=0.1
        self.local_steps=20
        self.compre_ratio=1
        self.train_time=0
        self.send_time=0
        self.worker_paras=None
        self.comm_neighbors=list()