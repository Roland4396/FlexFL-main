#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
from collections import OrderedDict
from datetime import datetime

import torch
import wandb
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
from tqdm import tqdm

from arch_profiles import build_width_only_scale_list
from models.Fed import Aggregation_FedSlim, split_model
from getAPOZ import getNet, modelList
from models import ResNet18_cifar, MobileNetV2
from models.Fed import get_model_list, select_clients, summon_clients, FlexFL_select_clients
from models.vgg import vgg_16_bn
from utils.Clients import Clients
from utils.utils import save_result, my_save_result, get_final_acc
from models.test import test_img, test
from models.Update import DatasetSplit, LocalUpdate_FedAvg
from optimizer.Adabelief import AdaBelief
from wandbUtils import init_run, upload_data, endrun


def HeteroFL(args, dataset_train, dataset_test, dict_users):
    run = init_run(args, "Fed-Experiment")
    net_glob = getNet(args, [1] * 50)
    scaleList = calculateScaleHeteroFL(args, net_glob, np.array([0.5] * 20))
    net_glob_list, net_slim_info = modelList(args, scaleList)

    # training
    avg_acc = [0]
    clients_list = summon_clients(args)
    # 开始训练
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []

        m = max(int(args.frac * args.num_users), 1)
        models = np.random.choice(range(len(net_glob_list)), m, replace=True)  # 模型选择
        idx_users = FlexFL_select_clients(args, clients_list, models,True)

        print(f"this epoch choose: {idx_users}")
        print(f"this epoch models: {models}")
        print(f"hetero_proportion: \t{args.client_hetero_ration}")
        # 需要print 每个客户端的计算资源

        for id, (user_idx, model_idx) in enumerate(idx_users):
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[user_idx])
            w = local.train(round=iter,
                            net=copy.deepcopy(net_glob_list[model_idx]).to(args.device))  # 这里开始正式训练

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[user_idx]))

        w_glob = Aggregation_FedSlim(w_locals, lens, net_glob_list[-1].state_dict())
        accDict = {}
        for idx, net in enumerate(net_glob_list):
            net.load_state_dict(split_model(w_glob, net.state_dict()))
            print(net_slim_info[idx])
            accDict[f"{idx}-acc"] = (test(net, dataset_test, args))
        upload_data(args, run, iter, accDict, avg_acc, net_slim_info)
    endrun(run)


def calculateScaleHeteroFL(args, net_glob, APOZ):
    return build_width_only_scale_list(args)
