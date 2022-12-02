from mpi4py import MPI
import os
import time
import argparse
import asyncio
import torch
import torch.optim as optim
# from pulp import *
import random
from config import ClientConfig, CommonConfig, Older_Models
from comm_utils import *
from training_utils import train, test
import datasets, models

import logging
import copy
import numpy as np
import threading

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(rank)% 4 + 0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# init logger
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/clients/' # + now + '/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)
filename = RESULT_PATH + now + "_" + 'Client' + '_'+ str(int(rank)) +'.log' # os.path.basename(__file__).split('.')[0] + '_'+ str(int(rank)) +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
# end logger

MASTER_RANK=0

async def get_init_config(comm, MASTER_RANK, config):
    logger.info("\nGetting init configuration...")
    config_received = await get_data(comm, MASTER_RANK, 1)
    logger.info("Geting init configuration Complete.\n")
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)

def main():
    logger.info("Client Rank: {}".format(rank))
    client_config = ClientConfig(
        common_config=CommonConfig()
    )

    logger.info("\nTraining Start...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    task = asyncio.ensure_future(get_init_config(comm,MASTER_RANK,client_config))
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    common_config = CommonConfig()
    common_config.model_type = client_config.common_config.model_type
    common_config.dataset_type = client_config.common_config.dataset_type
    common_config.batch_size = client_config.common_config.batch_size
    common_config.data_pattern=client_config.common_config.data_pattern
    common_config.lr = client_config.common_config.lr
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr=client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.momentum = client_config.common_config.momentum
    common_config.weight_decay = client_config.common_config.weight_decay
    common_config.data_path = client_config.common_config.data_path
    common_config.para=client_config.para

    common_config.window_size = client_config.common_config.window_size
    common_config.strategy = client_config.common_config.strategy
    common_config.older_models = Older_Models(window_size=common_config.window_size)
    logger.info("Window Size: {}".format(common_config.older_models.window_size))
 

    # 数据分布信息
    common_config.partition_sizes=client_config.common_config.partition_sizes
    logger.info("The Overall Data Distribution:")
    for partition in common_config.partition_sizes:
        logger.info("\n{}".format(partition))

    

    common_config.tag = 1 # 就是epoch数

    # init config
    logger.info("Rank {} has {} data samples.".format(rank, str(len(client_config.train_data_idxes))))
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type, common_config.data_path)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, selected_idxs=client_config.train_data_idxes)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=16, shuffle=False)
    local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())

    common_config.para=local_model # common_config.para就是local_model

    # 根据模型的type，生成对应的层名信息, 记录层的数量,层的参数量
    _cnt = 0
    for layer_name, paras in common_config.para.named_parameters():
        common_config.layer_names.append(layer_name)
        common_config.layer_num_parameters[layer_name] = paras.view(-1).size()[0]
        _cnt += 1
    common_config.num_layers = _cnt
    logger.info("\nThe number of parameters of layers of Model {}:".format(common_config.model_type))
    for layer_name in common_config.layer_num_parameters.keys():
        logger.info("\t{} -- number of parameters: {}".format(layer_name, common_config.layer_num_parameters[layer_name]))
    

    logger.info("\nThe Content of common_config:")
    logger.info(str(common_config.__dict__))


    class get_Thread(threading.Thread):
        def __init__(self,comm, common_config, client_rank, type):
            threading.Thread.__init__(self)
            self.comm=comm
            self.common_config=common_config
            self.client_rank=client_rank
            self.type=type
        def run(self):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tasks = []
            if self.type == 'info': # 层拉取信息
                tasks.append(
                    asyncio.ensure_future(
                        get_info(self.comm,self.common_config,self.client_rank, epoch_idx=common_config.tag)
                    )
                )
            elif self.type == 'para': # 层参数信息
                tasks.append(
                    asyncio.ensure_future(
                        get_para(self.comm,self.common_config,self.client_rank)
                    )
                )
            loop.run_until_complete(asyncio.wait(tasks))

            loop.close()
    mutex = threading.Lock()

    async def get_info(comm, common_config, rank, epoch_idx):
        # print("get_data")
        # logger.info("get_data")
        logger.info("\tClient Rank {} Getting Layer Information from Client Rank {}".format(comm.Get_rank(), rank))
        received_para = await get_data(comm, rank, epoch_idx)
        mutex.acquire()
        common_config.neighbor_info[rank] = received_para
        common_config.len_neighbor_received += 1
        # common_config.neighbor_received_flag[rank] = True
        mutex.release()


    async def get_para(comm, common_config, rank):
    # print("get_data")
        # while True:
        logger.info("\tClient Rank {} Getting Parameters from Client Rank {}".format(comm.Get_rank(), rank))
        received_para = await get_data(comm, rank, common_config.tag)
        mutex.acquire()
        common_config.neighbor_paras[rank] = received_para
        common_config.len_neighbor_received += 1
        # common_config.neighbor_received_flag[rank] = True
        mutex.release()

    class send_Thread(threading.Thread):
        def __init__(self,comm, data, common_config, client_rank):
            threading.Thread.__init__(self)
            self.comm=comm
            self.data=data
            self.common_config=common_config
            self.client_rank=client_rank
        def run(self):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tasks = []
            tasks.append(
                asyncio.ensure_future(
                    send_para(self.comm,self.data,self.common_config,self.client_rank)
                )
            )
            loop.run_until_complete(asyncio.wait(tasks))

            loop.close()
    async def send_para(comm, data, common_config, rank):
    # print("get_data")
        # while True:
        # logger.info("send_data")
        # print("send rank {}, get rank {}".format(comm.Get_rank(), rank))
        logger.info("\tClient Rank {} Sending paras to Client Rank {}".format(comm.Get_rank(), rank))
        # print("get rank: ", rank)
        await send_data(comm, data, rank, common_config.tag)



    total_computing_timer = 0 # 记录训练时间
    total_communication_timer = 0 # 利用模拟的带宽计算，不是实际的网络情况
    total_communication_cost = 0
    total_aggregation_timer = 0

    while True:
        # Start local Training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        _timer = time.time()
        tasks.append(
            asyncio.ensure_future(
                local_training(comm, common_config, train_loader)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        _timer = time.time() - _timer
        total_computing_timer += _timer
        logger.info("\nLocal Training Complete. \nEpoch {}'s Training Time: {}s".format(common_config.tag, _timer)) # 记录当前epoch的训练是时间
        logger.info("\nCurrent total time (local training + communication + aggregation) is: {}".format(total_communication_timer + total_computing_timer + total_aggregation_timer)) # 当前总时间：本地训练加通信

        # 本地训练完成之后，更新存储older_models的滑动窗口
        # common_config.older_models.add_model(dict(common_config.para.named_parameters()))
        logger.info("\nCurrent older_models Windows: Size : {}; Index {}".format(common_config.older_models.size, common_config.older_models.index))

        
        logger.info("\nClient {}'s neighbors' rank:".format(rank)) # 打印一下邻居的rank号
        for neighbor_idx in common_config.comm_neighbors:
            logger.info("\t Neighbor Rank {}".format(neighbor_idx))

        # 模拟网络带宽，在一定范围内随机带宽
        logger.info("\nCurrent neighbors' Bandwidth information:")
        for neighbor_idx in common_config.comm_neighbors:
            _rand_bandwidth = random.gauss(10 + (rank%10), 5)
            _rand_bandwidth = max(_rand_bandwidth, -_rand_bandwidth)
            common_config.neighbor_bandwidth[neighbor_idx] = _rand_bandwidth # 模拟网络波动的范围是10到20内，用于选择peer
            logger.info("\tNeighbor Rank: {}: {}MBps".format(neighbor_idx, common_config.neighbor_bandwidth[neighbor_idx]))

        # 计算与邻居的数据分布差异 neighbor_distribution存的就是差异，直接用就可以
        # TODO 修改计算差异为cosine方式
        logger.info("\nCurrent client {}'s distribution discrepancies of its neighbors:".format(rank))
        for neighbor_idx in common_config.comm_neighbors:
            common_config.neighbor_distribution[neighbor_idx] = torch.norm(torch.from_numpy(common_config.partition_sizes[neighbor_idx - 1] - common_config.partition_sizes[rank - 1]))
            logger.info("\tNeighbor Rank {}: {}".format(neighbor_idx, common_config.neighbor_distribution[neighbor_idx]))

        # Generate Pulling (layers) information ： 算法的核心就是如何决定层的拉取
        if common_config.strategy == 'D-PSGD':
            # Pulling Whole Model from Every Neighbor
            layers_needed_dict = generate_layers_information(common_config=common_config, whole_model=True)
        elif common_config.strategy == 'LFPL':
            layers_needed_dict = generate_layers_information(common_config=common_config, whole_model=False)
        elif common_config.strategy == 'NetMax':
            layers_needed_dict = net_max(common_config=common_config)
        elif common_config.strategy == 'Rand':
            layers_needed_dict = rand_layer(common_config=common_config)
        # TODO 基准算法

        logger.info("\nLayer pulling infomation:")
        # logger.info("{}".format(layers_needed_dict))
        for neighbor_idx in layers_needed_dict.keys():
            logger.info("\tThe Layers pulling from Neighbor Rank {}: {}".format(neighbor_idx, layers_needed_dict[neighbor_idx]))
            
        # 记录拉取层的通信量和时间，根据带宽计算通信时间
        _communication_cost, _communication_time = communication_cost(common_config=common_config, layer_info_dict=layers_needed_dict)
        logger.info("\nCommunication Traffic (Epoch {}) for pulling layers from neighbors is {}MB.".format(common_config.tag,_communication_cost))
        logger.info("Communication Time (Epoch {}) for pulling layers from neighbors is {}s.".format(common_config.tag,_communication_time))
        total_communication_timer += _communication_time
        total_communication_cost += _communication_cost

        # # Tell neighbors: Layers needed from corresponding neighbors --> 存储在 common_config.neighbor_info=dict()
        # logger.info("\n")
        # logger.info("Sending/getting information to/from its neighbors")
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # tasks = []

        # print(layers_needed_dict) # 打印到终端，方便及时查看全局信息
        # for i in range(len(common_config.comm_neighbors)):
        #     nei_rank=common_config.comm_neighbors[i]
        #     data = layers_needed_dict[nei_rank] # 将layers_needed_dict中的层拉取信息发送给对应邻居，
        #                                             # 即layers_needed_dict[neighbor_idx]发送给rank=neighbor_idx的邻居

        #     if nei_rank > rank:
        #         task = asyncio.ensure_future(send_para(comm, data, nei_rank, common_config.tag))
        #         tasks.append(task)
        #         # print("worker send")
        #         task = asyncio.ensure_future(get_info(comm, common_config, nei_rank, common_config.tag))
        #         tasks.append(task)
        #     else:
        #         task = asyncio.ensure_future(get_info(comm, common_config, nei_rank, common_config.tag))
        #         tasks.append(task)
        #         # print("worker send")
        #         task = asyncio.ensure_future(send_para(comm, data, nei_rank, common_config.tag))
        #         tasks.append(task)
        # loop.run_until_complete(asyncio.wait(tasks))
        # loop.close()
        # logger.info("Sending/getting information complete")

        # Tell neighbors: Layers needed from corresponding neighbors --> 存储在 common_config.neighbor_info=dict()
        logger.info("\n")
        logger.info("Sending/getting information to/from its neighbors")
        print(layers_needed_dict) # 打印到终端，方便及时查看全局信息
        for i in range(len(common_config.comm_neighbors)):
            nei_rank=common_config.comm_neighbors[i]
            data = layers_needed_dict[nei_rank] # 将layers_needed_dict中的层拉取信息发送给对应邻居，
                                                    # 即layers_needed_dict[neighbor_idx]发送给rank=neighbor_idx的邻居
            send_T=send_Thread(comm, data,common_config, nei_rank)
            send_T.start()
            get_T=get_Thread(comm, common_config, nei_rank, type='info')
            get_T.start()
        
        while True:
            mutex.acquire()
            sss=common_config.len_neighbor_received
            mutex.release()
            if sss==len(common_config.comm_neighbors):
                break
        common_config.len_neighbor_received=0
        logger.info("Sending/getting information complete")




        
        # Sending/Get parameters: Get layers parameters from neighbors --> 根据common_config.neighbor_info=dict()，将本地模型的对应层发给邻居
        #     因为是模拟的带宽，所以不需要在这里记录通信量和通信时间信息
        local_model=common_config.para
        layers_sending_dict = dict() # 存储所有需要发送的层参数的信息
        for neighbor_idx in common_config.comm_neighbors:
            layers_sending_dict[neighbor_idx] = get_layers_dict(local_model, common_config.neighbor_info[neighbor_idx])
        # logger.info("layers_sending_dict: ")
        # for neighbor_idx in layers_sending_dict.keys():
        #     logger.info("\t{}".format(layers_sending_dict[neighbor_idx].keys()))


        # logger.info("\n")
        # logger.info("Sending/getting parameters to/from its neighbors")
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # tasks = []
        # for i in range(len(common_config.comm_neighbors)):
        #     nei_rank=common_config.comm_neighbors[i]
        #     data = layers_sending_dict[nei_rank]
        #     # logger.info("Client {}'s neighbors:{}".format(rank, common_config.comm_neighbors[i]))

        #     if nei_rank > rank:
        #         task = asyncio.ensure_future(send_para(comm, data, nei_rank, common_config.tag))
        #         tasks.append(task)
        #         # print("worker send")
        #         task = asyncio.ensure_future(get_para(comm, common_config, nei_rank, common_config.tag))
        #         tasks.append(task)
        #     else:
        #         task = asyncio.ensure_future(get_para(comm, common_config, nei_rank, common_config.tag))
        #         tasks.append(task)
        #         # print("worker send")
        #         task = asyncio.ensure_future(send_para(comm, data, nei_rank, common_config.tag))
        #         tasks.append(task)
        # loop.run_until_complete(asyncio.wait(tasks))
        # loop.close()
        # logger.info("Sending/getting parameters complete\n")

        
        logger.info("\nSending/getting parameters to/from its neighbors")
        for i in range(len(common_config.comm_neighbors)):
            nei_rank=common_config.comm_neighbors[i]
            data = layers_sending_dict[nei_rank] # 将layers_needed_dict中的层拉取信息发送给对应邻居，
                                                    # 即layers_needed_dict[neighbor_idx]发送给rank=neighbor_idx的邻居
            send_T=send_Thread(comm, data,common_config, nei_rank)
            send_T.start()
            get_T=get_Thread(comm, common_config, nei_rank, type='para')
            get_T.start()
        
        while True:
            mutex.acquire()
            sss=common_config.len_neighbor_received
            mutex.release()
            if sss==len(common_config.comm_neighbors):
                break
        common_config.len_neighbor_received=0
        logger.info("Sending/getting parameters complete\n")


        logger.info("\nReceived Neighbor's Layers:")
        for neighbor_idx in common_config.neighbor_paras.keys():
            logger.info("\t Neighbor Rank {}: {}".format(neighbor_idx,common_config.neighbor_paras[neighbor_idx].keys()))

        # Aggregate 
        # local_para = aggregate_model(local_para, common_config)
        # torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
        logger.info("\nLocal Model Aggregating...")
        _timer = time.time()
        local_model = aggregate_model_with_dict(local_model, common_config)
        _timer = time.time() - _timer
        logger.info("Current Epoch Aggregation Time: {}".format(_timer))
        total_aggregation_timer += _timer
        logger.info("Local Aggregation Complete.\n")

        # 滑动窗口存储模型，根据上一轮聚合之后的模型更具有指导意义，不会偏向与自己的数据分布
        tmp_model = copy.deepcopy(common_config.para)
        common_config.older_models.add_model(tmp_model.state_dict())

        common_config.para=local_model


        # Test
        logger.info("Testing...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training2(comm, common_config, test_loader)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        if common_config.tag==common_config.epoch+1:
            logger.info("The Training Time is {}".format(total_computing_timer))
            logger.info("THe Commnunication Time is {}".format(total_communication_timer))
            logger.info("The Training and Communicating Time is {}".format(total_computing_timer + total_communication_timer))
            logger.info("The Total Communication Cost is: {}MB".format(total_communication_cost))
            break

def rand_layer(common_config):
    # Random pull layers from neighbors to compose a whole model.
    
    # Init
    result = dict()
    for neighbor_idx in common_config.comm_neighbors:
        result[neighbor_idx] = list()

    # Generating result
    for layer in common_config.layer_names:
        _tmp_neighbor = random.choice(common_config.comm_neighbors)
        result[_tmp_neighbor].append(layer)
    return result

def net_max(common_config):
    # Find the neighbor with Maximum Bandwidth
    _max_bandwidth = 0
    _target_neighbor_idx = None
    for neighbor_idx in common_config.neighbor_bandwidth.keys():
        cur_neighbor_bandwidth = common_config.neighbor_bandwidth[neighbor_idx]
        if cur_neighbor_bandwidth > _max_bandwidth:
            _max_bandwidth = cur_neighbor_bandwidth
            _target_neighbor_idx = neighbor_idx
    
    # Generating pull information --> pull the whole model from the neighbor found above.
    result = dict()
    for neighbor_idx in common_config.comm_neighbors:
        if neighbor_idx == _target_neighbor_idx:
            result[neighbor_idx] = common_config.layer_names
        else:
            result[neighbor_idx] = list()
    return result
    
    




def communication_cost(common_config, layer_info_dict):
    _communication_cost = 0
    for neighbor_idx in layer_info_dict.keys():
        for layer_name in layer_info_dict[neighbor_idx]:
            _communication_cost += common_config.layer_num_parameters[layer_name] * 4 / 1024 / 1024
    _communication_time = 0.0
    for neighbor_idx in layer_info_dict.keys():
        _tmp_time = 0.0
        for layer_name in layer_info_dict[neighbor_idx]:
            _tmp_time += common_config.layer_num_parameters[layer_name] * 4 / 1024 / 1024 / common_config.neighbor_bandwidth[neighbor_idx]
        if _tmp_time > _communication_time:
            _communication_time = _tmp_time
    return _communication_cost, _communication_time



def generate_layers_information(common_config, whole_model=False):
    # 输入： (全局第一轮通信传输全模型，第二轮开始)
    #     1. 自己的本地模型
    #     2. 之前保存的几轮聚合之后的模型，用于计算差异和学习速度 -- 窗口大小确定保存的模型参数个数
    # 返回一个矩阵：（邻居数量 * 模型层数）
    # 
    # TODO 层的选择与邻居的选择，需要一起考虑，不是独立的.
    #                 2022/11/24：考虑先确定层，再确定client，受限考虑模型的收敛和是否学习到有用的特征，再去确定从那些客户端选择
    # 
    def layer_selector(common_config):
        # 实现想法：
        #       1. 不同阶段考虑不同的层，根据精度确定训练阶段 -- 暂不考虑
        #       2. 相比于上次更新，层的差异性
        #       3. 层的学习速度的计算
        #           4. SVCCA? -- 后面补充
        # 输入：1）之前保存的若干模型参数（设定滑动窗口的大小决定需要考虑之前的几轮更新的模型）-- 差异性、学习速度(与自己的模型做对比？)
        #       2）当前的模型测试精度，决定更新的阶段，来决定层的重要性 -- 暂不考虑
        #            3）层的SVCCA的值？ -- 后面补充
        # 返回：
        
        # 1) 差异性(与自己的模型做对比？后面可以尝试下与拉取的邻居层的对比)
        logger.info("\nIN layer_selector:")
        discrepancy = dict()
        last_model_dict = common_config.older_models.get_last_model_dict()
        local_model = common_config.para
        logger.info("Discrepency Information:")
        for layer, paras in local_model.named_parameters():
            # logger.info("para ?= last_model_dict[layer] -- {}".format(paras == last_model_dict[layer]))
            discrepancy[layer] = torch.norm(paras - last_model_dict[layer])# 利用二范数计算差异值
            logger.info("\tLayer: {} -- {}".format(layer, discrepancy[layer]))

        #    层学习速度计算，根据当前窗口，有多少个模型就计算多少个
        learning_speed = dict()
        _epsilon = 1e-6
        logger.info("Learning Speed:")
        for layer, paras in local_model.named_parameters():
            denominator = 0
            for i in range(common_config.older_models.size - 1):
                # logger.info("denominator -- {}".format(denominator))
                denominator += torch.norm(common_config.older_models.models[(common_config.older_models.index + i)%common_config.older_models.size][layer] - common_config.older_models.models[(common_config.older_models.index + i + 1)%common_config.older_models.size][layer])
            molecule = torch.norm(common_config.older_models.models[common_config.older_models.index][layer] - common_config.older_models.models[(common_config.older_models.index - 1)%common_config.older_models.size][layer])
            learning_speed[layer] = molecule / (_epsilon + denominator)
            # logger.info("\tmolecule: {}".format(molecule))
            # logger.info("\tdenominator: {}".format(denominator))
            logger.info("\tLayer: {} -- {}".format(layer, learning_speed[layer]))
        # 差异和学习速度各占0.5的权重
        priority = dict()
        for layer in discrepancy.keys():
            priority[layer] = discrepancy[layer] * 0.1 + learning_speed[layer] * 0.9
        logger.info("Exit layer_selector")
        return sorted_dict(priority)


    def peer_selector(common_config):
        # 实现想法：
        #           1. 考虑数据分布，计算分布的差异值（是否可以考虑首轮全局通信，每个client保存连通的邻居的分布信息）
        #           2. 网络。根据上一轮的网络情况，记录下每个邻居的带宽，计算并均一化一个最大值
        #           3. 邻居的选择个数？1<=个数<=邻居总数
        # 实现思路：让每个client进程本地维护：
        #           1）一个长度为自己邻居大小的dict -- 网络带宽（基于上一轮的收取时间，需要对所有的邻居做一次归一化；
        #           2）一个长度为自己邻居大小的dict，用于存储邻居的数据分布信息--数据分布的相似度
        #           3) 1）2）各占0.5的权重，加权的结果按顺序排列，选出指定个数的邻居，并返回指定个数的邻居
        #          返回一个字典
        priority = dict()
        temp_value_for_normalize = 0
        for neighbor_idx in common_config.comm_neighbors:
            # 首轮是从全部邻居拉取，所以都是有带宽数据的，需要自己每次接受层的时候更新common_config.neighbor_bandwidth字典
            # 现在实现的是模拟的网络情况
            temp_value_for_normalize += common_config.neighbor_bandwidth[neighbor_idx]
        for neighbor_idx in common_config.comm_neighbors:
            priority_bandwidth = common_config.neighbor_bandwidth[neighbor_idx] / temp_value_for_normalize # 带宽大的优先
            priority_distribution = common_config.neighbor_distribution[neighbor_idx] # 分布差别大的优先
            priority[neighbor_idx] = priority_bandwidth * 0.5 + priority_distribution * 0.5
        return sorted_dict(priority)

        
    def sorted_dict(dict_be_sorted, key=lambda x:x[1], reverse=True): # 默认按键值升序排列
        dict_be_sorted_by_key = sorted(dict_be_sorted.items(), key=key, reverse=reverse)
        return dict(dict_be_sorted_by_key)

    
    result = dict()
    if common_config.tag > 1 and whole_model == False:
        layers_info = layer_selector(common_config=common_config)
        logger.info("\nLayer Priority Information:")
        for layer in layers_info.keys():
            logger.info("\t{} -- Priority: {}".format(layer, layers_info[layer]))
        neighbor_info = peer_selector(common_config=common_config)
        logger.info("\nNeighbor Priority Information:")
        for neighbor_idx in neighbor_info.keys():
            logger.info("\tNeighbor Rank {} -- Priority: {}".format(neighbor_idx, neighbor_info[neighbor_idx]))
        
        # print(layers_info)
        # print(neighbor_info)

        # 对邻居做选择结果做归一化
        temp_value_for_normalize = 0
        for neighbor_idx in neighbor_info:
            temp_value_for_normalize += neighbor_info[neighbor_idx]
        for neighbor_idx in neighbor_info:
            neighbor_info[neighbor_idx] /= temp_value_for_normalize
            result[neighbor_idx] = list() # 顺便初始化
        
        # 根据layer_info和neighbor_info生成选择层的信息
        select = 2
        # 方案1：手动将优先级高的layer映射到优先级高的neighbor
        if select == 1:
            _start = 0
            _end = 0
            num_layers_selected = int(common_config.num_layers * 1) # 设置从所有邻居拿去层的总数
            for neighbor_name in neighbor_info.keys():
                # 从优先度高的peer拿优先度更高的层，百分比？向上取整 
                _end = _start + int(neighbor_info[neighbor_name] * num_layers_selected) # 邻居的权重越大，拿越重要，越多的层
                
                _idx = 0
                for layer in layers_info.keys(): # 根据[_start,_end)拿去对应重要性的层
                    if _idx >= _start and _idx < _end:
                        result[neighbor_name].append(layer)
                    _idx += 1

                _start = _end

        # 方案2：以邻居的权重为概率，将层按照概率映射到邻居
        elif select == 2:
            _neighbors_list = list()
            _neighbors_probability = list()
            for neighbor_idx in neighbor_info:
                _neighbors_list.append(neighbor_idx)
                _neighbors_probability.append(neighbor_info[neighbor_idx])
            # 让概率和为1
            _sum = sum(_neighbors_probability)
            _rand_neighb_idx = random.choice(list(range(len(_neighbors_probability))))
            _neighbors_probability[_rand_neighb_idx] = _neighbors_probability[_rand_neighb_idx] - (_sum - 1.0)
            
            _neighbors_probability = np.array(_neighbors_probability)
            
            _mapping_model_number = int(len(common_config.comm_neighbors) / 2)
            for layer in common_config.layer_names:
                for i in range(_mapping_model_number):
                    _sample_neighbor = np.random.choice(_neighbors_list, p=_neighbors_probability.ravel())
                    result[_sample_neighbor].append(layer)
    else:
        for neighbor_idx in common_config.comm_neighbors:
            result[neighbor_idx] = common_config.layer_names
    return result



async def local_training(comm, common_config, train_loader):
    comm_neighbors = await get_data(comm, MASTER_RANK, common_config.tag)
    # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    # torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    local_model = common_config.para
    local_model.to(device)
    epoch_lr = common_config.lr
    
    local_steps = 50 # 20
    if common_config.tag > 1 and common_config.tag % 1 == 0:
        epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        common_config.lr = epoch_lr
    

    logger.info("\n")    
    logger.info("*" * 150)
    logger.info("Starting Local Training (Epoch-{} lr: {})...".format(common_config.tag, epoch_lr))
    if common_config.momentum<0:
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
    else:
        optimizer = optim.SGD(local_model.parameters(),momentum=common_config.momentum, lr=epoch_lr, weight_decay=common_config.weight_decay)
    train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=common_config.model_type)
    # local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()

    common_config.comm_neighbors = comm_neighbors
    common_config.para = local_model
    common_config.train_loss = train_loss

async def local_training2(comm, common_config, test_loader):
    # 2号函数是测试函数
    # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    # torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    local_model = common_config.para
    local_model.to(device)
    # torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
    test_loss, acc = test(local_model, test_loader, device, model_type=common_config.model_type)
    logger.info("After aggregation, Epoch: {}, Train Loss: {}, Test Loss: {}, test accuracy: {}".format(common_config.tag, common_config.train_loss, test_loss, acc))
    logger.info("Sending Test information to server process.")

    data=(acc, test_loss, common_config.train_loss)
    # send_data_await(comm, local_paras, MASTER_RANK, common_config.tag)
    await send_data(comm, data, MASTER_RANK, common_config.tag)
    logger.info("Sending Complete.")
    # local_para = await get_data(comm, MASTER_RANK, common_config.tag)
    # common_config.para=local_para
    # common_config.tag = common_config.tag+1
    # logger.info("get end")
    common_config.tag = common_config.tag+1

async def send_para(comm, data, rank, epoch_idx):
    # print("send_data")
    # logger.info("send_data")
    # print("send rank {}, get rank {}".format(comm.Get_rank(), rank))
    logger.info("\tClient Rank {} Sending paras to Client Rank {}".format(comm.Get_rank(), rank))
    # print("get rank: ", rank)
    await send_data(comm, data, rank, epoch_idx)

async def get_info(comm, common_config, rank, epoch_idx):
    # print("get_data")
    # logger.info("get_data")
    logger.info("\tClient Rank {} Getting Layer Information from Client Rank {}".format(comm.Get_rank(), rank))
    common_config.neighbor_info[rank] = await get_data(comm, rank, epoch_idx)

async def get_para(comm, common_config, rank, epoch_idx):
    # print("get_data")
    # logger.info("get_data")
    logger.info("\tClient Rank {} Getting Parameters from Client Rank {}".format(comm.Get_rank(), rank))
    common_config.neighbor_paras[rank] = await get_data(comm, rank, epoch_idx)

# def aggregate_model(local_para, common_config):
#     with torch.no_grad():
#         weight=1.0/(len(common_config.comm_neighbors)+1)
#         para_delta = torch.zeros_like(local_para)
#         for neighbor_name in common_config.comm_neighbors:
#             logger.info("Update local model use information from neighbor idx: {},".format(neighbor_name))
#             model_delta = common_config.neighbor_paras[neighbor_name] - local_para
#             para_delta += weight * model_delta

#         local_para += para_delta
#     return local_para


def get_layers_dict(model, layers=list()):
    '''
        根据layers列表，获取model对应的层参数，最为{层名：层参数}的字典返回
    '''
    layers_dict = dict()
    for name, paras in model.named_parameters():
        if name in layers:
            layers_dict[name] = paras
    return layers_dict

def aggregate_model_with_dict(local_model, common_config):
    local_blocks_dicts = list()
    for neighbor_name in common_config.comm_neighbors:
        # print("neighbor name: {}".format(neighbor_name))
        local_blocks_dicts.append(common_config.neighbor_paras[neighbor_name])
            
    # logger.info("common_config.neighbors:")
    # for neighbor_name in common_config.comm_neighbors:
    #     first = True
    #     for layer_name in common_config.layer_names:
    #         # logger.info(common_config.neighbor_paras.keys())
    #         if layer_name in common_config.neighbor_paras[neighbor_name]:
    #             if first:
    #                 first = False
    #                 logger.info("\tNeighbor index: {}".format(neighbor_name))
    #             logger.info("\t\tLayer {}: -- {}".format(layer_name, common_config.neighbor_paras[neighbor_name][layer_name].view(-1).size()[0]))
    # logger.info("End common_config.neighbors.")


    # logger.info("local_block_dicts:")



    dict_keys = common_config.layer_names

    updated_para_dict = dict()
    # 利用空字典、邻居传来的block、本地模型 --> 模型字典 --> 构建新的模型
    local_model_dict = dict(local_model.named_parameters())
    with torch.no_grad(): # 直接平均的聚合方式
        for layer_name in dict_keys:
            # 按层一次更新模型字典参数
            # print(name)
            first = True
            #is_layer_aggregate = False # 是否利用邻居的层进行更新
            
            # 此时upadated_para_dict[name]内没有参数
            # 去local_blocks_dict去看看邻居中有没有该层的参数，如果有，就加和
            for local_blocks_dict in local_blocks_dicts: # 对于每一层参数都扫一边所有的邻居block参数字典
                if layer_name in local_blocks_dict.keys():
                    if first:
                        layer_para = local_blocks_dict[layer_name].clone().detach()
                        count = 1
                        first = False
                        #is_layer_aggregate = True
                    else:
                        layer_para += local_blocks_dict[layer_name]
                        count += 1
            if first == False:
                # 邻居中有该层，则需要再加上自己的参数进行平均
                layer_para += local_model_dict[layer_name]
                count += 1
                layer_para /= count # 样本均分，所以平局参数
                #layer_para = layer_para * 0.5 + local_model_dict[name] * 0.5 # 邻居参数占更新权重一半
                updated_para_dict[layer_name] = layer_para
                logger.info("\tLayer {}: Use {} Neighbor layer to update.".format(layer_name, count - 1))
                # print("updated_para_dict[{}] == local_model_dict[{}]:{}".format(name, name, updated_para_dict[name] == local_model_dict[name]))
            else:
                # 邻居中没有该层的参数，沿用自己之前的莫i选哪个参数
                updated_para_dict[layer_name] = local_model_dict[layer_name]
                logger.info("\tLayer {} Use its Own previous layer.".format(layer_name))
                # print("updated_para_dict[{}] == local_model_dict[{}]:{}".format(name, name, updated_para_dict[name] == local_model_dict[name]))
            # if is_layer_aggregate == False: # 没有利用邻居层更新的话，之间从本地模型拷贝参数
            #     updated_para_dict[name] = local_model_dict[name]
        # print(updated_para_dict.keys())

    # 更新后的模型字典参数载入模型
    #updated_model = copy.deepcopy(local_model)
    local_model.load_state_dict(updated_para_dict)
    return local_model



if __name__ == '__main__':
    main()
