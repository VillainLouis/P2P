import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from pulp import *
import random
from config import ClientConfig, CommonConfig
from comm_utils import *
from training_utils import train, test
import datasets, models
from mpi4py import MPI
import logging

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
RESULT_PATH = os.getcwd() + '/clients/' + now + '/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)


filename = RESULT_PATH + now + "_" +os.path.basename(__file__).split('.')[0] + '_'+ str(int(rank)) +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
# end logger

MASTER_RANK=0

async def get_init_config(comm, MASTER_RANK, config):
    logger.info("before init")
    config_received = await get_data(comm, MASTER_RANK, 1)
    logger.info("after init")
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)

def main():
    logger.info("client_rank:{}".format(rank))
    client_config = ClientConfig(
        common_config=CommonConfig()
    )

    logger.info("start")
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
    
    common_config.tag = 1
    # init config
    logger.info(str(common_config.__dict__))

    logger.info(str(len(client_config.train_data_idxes)))
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type, common_config.data_path)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, selected_idxs=client_config.train_data_idxes)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=16, shuffle=False)
    local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    common_config.para=local_model # common_config.para就是local_model

    while True:
        # Start local Training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training(comm, common_config, train_loader)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        # Generate Pulling (layers) information ： 算法的核心就是如何决定层的拉取
        layers_needed_dict = dict() # {neighbor_name : list()} 每个邻居名字：[层名字的list]
        for neighbor_idx in common_config.comm_neighbors:
            layers_needed_dict[neighbor_idx] = []
        
        f0 = ["features.0.weight", "features.0.bias"] 
        f3 = ["features.3.weight", "features.3.bias"]
        f6 = ["features.6.weight", "features.6.bias"]
        f8 = ["features.8.weight", "features.8.bias"]
        f10 = ["features.10.weight", "features.10.bias"]
        c0 = ["classifier.0.weight", "classifier.0.bias"]
        c2 = ["classifier.2.weight", "classifier.2.bias"]
        c4 = ["classifier.4.weight", "classifier.4.bias"] 
        whole_model = f0 + f3 + f6 + f8 + f10 + c0 + c2 + c4  # 针对AlexNet的模型层参数名
            
        for neighbor_idx in common_config.comm_neighbors:
            layers_needed_dict[neighbor_idx] = whole_model

        

        # Tell neighbors: Layers needed from corresponding neighbors --> 存储在 common_config.neighbor_info=dict()
                
        logger.info("Client {}'s neighbors' indices:".format(rank)) # 打印一下邻居的rank号
        for neighbor_idx in common_config.comm_neighbors:
            logger.info("\t{}".format(neighbor_idx))
        logger.info("\n")
        logger.info("Sending/getting information to/from its neighbors")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []

        for i in range(len(common_config.comm_neighbors)):
            nei_rank=common_config.comm_neighbors[i]
            data = layers_needed_dict[neighbor_idx] # 将layers_needed_dict中的层拉取信息发送给对应邻居，
                                                    # 即layers_needed_dict[neighbor_idx]发送给rank=neighbor_idx的邻居

            if nei_rank > rank:
                task = asyncio.ensure_future(send_para(comm, data, nei_rank, common_config.tag))
                tasks.append(task)
                # print("worker send")
                task = asyncio.ensure_future(get_info(comm, common_config, nei_rank, common_config.tag))
                tasks.append(task)
            else:
                task = asyncio.ensure_future(get_info(comm, common_config, nei_rank, common_config.tag))
                tasks.append(task)
                # print("worker send")
                task = asyncio.ensure_future(send_para(comm, data, nei_rank, common_config.tag))
                tasks.append(task)
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        logger.info("Sending/getting information complete")

        
        # Sending/Get parameters: Get layers parameters from neighbors --> 根据common_config.neighbor_info=dict()，将本地模型的对应层发给邻居
        local_model=common_config.para
        layers_sending_dict = dict() # 存储所有需要发送的层参数的信息
        for neighbor_idx in common_config.comm_neighbors:
            layers_sending_dict[neighbor_idx] = get_layers_dict(local_model, common_config.neighbor_info[neighbor_idx])

        logger.info("\n")
        logger.info("Sending/getting parameters to/from its neighbors")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        for i in range(len(common_config.comm_neighbors)):
            nei_rank=common_config.comm_neighbors[i]
            data = layers_sending_dict[neighbor_idx]
            logger.info("Client {}'s neighbors:{}".format(rank, common_config.comm_neighbors[i]))

            if nei_rank > rank:
                task = asyncio.ensure_future(send_para(comm, data, nei_rank, common_config.tag))
                tasks.append(task)
                # print("worker send")
                task = asyncio.ensure_future(get_para(comm, common_config, nei_rank, common_config.tag))
                tasks.append(task)
            else:
                task = asyncio.ensure_future(get_para(comm, common_config, nei_rank, common_config.tag))
                tasks.append(task)
                # print("worker send")
                task = asyncio.ensure_future(send_para(comm, data, nei_rank, common_config.tag))
                tasks.append(task)

        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        logger.info("Sending/getting parameters complete")

        # Aggregate 
        # local_para = aggregate_model(local_para, common_config)
        # torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
        local_model = aggregate_model_with_dict(local_model, common_config)

        common_config.para=local_model

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
            break


async def local_training(comm, common_config, train_loader):
    comm_neighbors = await get_data(comm, MASTER_RANK, common_config.tag)
    # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    # torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    local_model = common_config.para
    local_model.to(device)
    epoch_lr = common_config.lr
    
    local_steps = 20
    if common_config.tag > 1 and common_config.tag % 1 == 0:
        epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        common_config.lr = epoch_lr
    

    logger.info("\n")    
    logger.info("*" * 100)
    logger.info("epoch-{} lr: {}".format(common_config.tag, epoch_lr))
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
    # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    # torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    local_model = common_config.para
    local_model.to(device)
    # torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
    test_loss, acc = test(local_model, test_loader, device, model_type=common_config.model_type)
    logger.info("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(common_config.tag, common_config.train_loss, test_loss, acc))
    logger.info("send para")

    data=(acc, test_loss, common_config.train_loss)
    # send_data_await(comm, local_paras, MASTER_RANK, common_config.tag)
    await send_data(comm, data, MASTER_RANK, common_config.tag)
    logger.info("after send")
    # local_para = await get_data(comm, MASTER_RANK, common_config.tag)
    # common_config.para=local_para
    # common_config.tag = common_config.tag+1
    # logger.info("get end")
    common_config.tag = common_config.tag+1

async def send_para(comm, data, rank, epoch_idx):
    # print("send_data")
    logger.info("send_data")
    # print("send rank {}, get rank {}".format(comm.Get_rank(), rank))
    logger.info("send rank {}, get rank {}".format(comm.Get_rank(), rank))
    # print("get rank: ", rank)
    await send_data(comm, data, rank, epoch_idx)

async def get_info(comm, common_config, rank, epoch_idx):
    # print("get_data")
    logger.info("get_data")
    logger.info("get rank {}, send rank {}".format(comm.Get_rank(), rank))
    common_config.neighbor_info[rank] = await get_data(comm, rank, epoch_idx)

async def get_para(comm, common_config, rank, epoch_idx):
    # print("get_data")
    logger.info("get_data")
    logger.info("get rank {}, send rank {}".format(comm.Get_rank(), rank))
    common_config.neighbor_paras[rank] = await get_data(comm, rank, epoch_idx)

def aggregate_model(local_para, common_config):
    with torch.no_grad():
        weight=1.0/(len(common_config.comm_neighbors)+1)
        para_delta = torch.zeros_like(local_para)
        for neighbor_name in common_config.comm_neighbors:
            logger.info("Update local model use information from neighbor idx: {},".format(neighbor_name))
            model_delta = common_config.neighbor_paras[neighbor_name] - local_para
            para_delta += weight * model_delta

        local_para += para_delta
    return local_para

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
        local_blocks_dict = common_config.neighbor_paras[neighbor_name]
        local_blocks_dicts.append(local_blocks_dict)
    
    dict_keys = ['features.0.weight', 'features.0.bias', 'features.3.weight', 'features.3.bias', 'features.6.weight', 'features.6.bias', 'features.8.weight', 'features.8.bias', 'features.10.weight', 'features.10.bias',
                'classifier.0.weight', 'classifier.0.bias', 'classifier.2.weight', 'classifier.2.bias', 'classifier.4.weight', 'classifier.4.bias'
                ]
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
                logger.info("This layer use {} neighbor paras to update.".format(count - 1))
                logger.info("Layer {} Use neighbor layer to update.".format(layer_name))
                # print("updated_para_dict[{}] == local_model_dict[{}]:{}".format(name, name, updated_para_dict[name] == local_model_dict[name]))
            else:
                # 邻居中没有该层的参数，沿用自己之前的莫i选哪个参数
                updated_para_dict[layer_name] = local_model_dict[layer_name]
                logger.info("Layer {} Use its own previous layer to update.".format(layer_name))
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
