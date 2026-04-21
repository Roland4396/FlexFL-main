#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
"""
HeteroFL with 4 levels support
Parameters ratio: 1/8, 1/4, 1/2, 1
"""
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

from Algorithm.Training_FlexFL import calculateScale, modelList
from models.Fed import Aggregation_FedSlim, split_model
from getAPOZ import getNet
from models import ResNet18_cifar, MobileNetV2
from models.Fed import get_model_list, select_clients, summon_clients, FlexFL_select_clients
from models.vgg import vgg_16_bn
from utils.Clients import Clients
from utils.utils import save_result, my_save_result, get_final_acc
from models.test import test_img, test
from models.Update import DatasetSplit, LocalUpdate_FedAvg
from optimizer.Adabelief import AdaBelief
from wandbUtils import init_run, upload_data, endrun


def HeteroFL_4levels(args, dataset_train, dataset_test, dict_users):
    """HeteroFL with 4 model levels: 1/8, 1/4, 1/2, 1 of parameters"""
    run = init_run(args, "Fed-Experiment")

    # Create 4 levels of models based on parameter ratios
    scaleList = calculateScaleHeteroFL_4levels(args)
    net_glob_list, net_slim_info = modelList(args, scaleList)

    # Print model configuration
    print("\n" + "="*80)
    print("HeteroFL 4-Levels Configuration")
    print("="*80)
    print(f"Number of model levels: {len(net_glob_list)}")
    for idx, info in enumerate(net_slim_info):
        print(f"Level {idx+1}: {info}")
    print("="*80 + "\n")

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

        # 使用4个模型等级的客户端选择
        idx_users = FlexFL_select_clients(args, clients_list, models, len(net_glob_list) >= 3)

        print(f"this epoch choose: {idx_users}")
        print(f"this epoch models: {models}")
        print(f"hetero_proportion: \t{args.client_hetero_ration}")

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


def calculateScaleHeteroFL_4levels(args):
    """
    Calculate scale rates for 4 model levels
    Target parameter ratios: 1/8 (12.5%), 1/4 (25%), 1/2 (50%), 1 (100%)

    Since HeteroFL uses uniform scaling across all layers,
    we use scale values that achieve the target parameter ratios
    """

    # For HeteroFL, parameter ratio ≈ scale^2 (approximately)
    # So scale = sqrt(parameter_ratio)
    scale_values = [
        np.sqrt(0.125),  # sqrt(1/8) ≈ 0.354
        np.sqrt(0.25),   # sqrt(1/4) = 0.5
        np.sqrt(0.5),    # sqrt(1/2) ≈ 0.707
        1.0              # full model
    ]

    # Create rate arrays for each level
    # In HeteroFL/FlexFL, rate is an array for each layer
    # For uniform scaling, all layers use the same rate
    scaleList = []

    # Get a sample network to determine the number of layers
    net_glob = getNet(args, [1] * 50)  # Assuming 50 layers max
    num_layers = len([p for p in net_glob.features.parameters() if len(p.shape) == 4])  # Count conv layers

    print(f"\nDetected {num_layers} convolutional layers in the model")

    # Create rate arrays for each scale level
    for i, scale in enumerate(scale_values):
        # Create uniform rate array for all layers
        rate_array = np.ones(num_layers) * scale
        scaleList.append(torch.tensor(rate_array, dtype=torch.float32))

        # Calculate and print expected parameter ratio
        expected_params = scale ** 2
        print(f"Level {i+1}: scale={scale:.3f}, expected params ratio={expected_params:.3f}")

    return scaleList


def calculateScaleHeteroFL_4levels_adaptive(args, net_glob, APOZ):
    """
    Alternative method: Adaptive scaling based on APOZ scores
    This method tries to find scales that result in exactly 12.5%, 25%, 50% of parameters
    """
    ans = [np.ones(len(APOZ))]  # 100% model
    originFeatureParams = sum(p.numel() for p in net_glob.features.parameters())

    target_ratios = [0.125, 0.25, 0.5]  # 1/8, 1/4, 1/2
    found_scales = []

    for target_ratio in target_ratios:
        print(f"\nSearching for scale that gives {target_ratio*100:.1f}% parameters...")

        best_gamma = 1
        best_diff = float('inf')

        for gamma in range(1, 1000):
            temp = APOZ * gamma / 50
            temp = torch.tensor(temp)
            net = getNet(args, torch.clamp(temp, min=0.1, max=1))
            currentParams = sum(p.numel() for p in net.features.parameters())
            current_ratio = currentParams / originFeatureParams

            diff = abs(current_ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_gamma = gamma

            # If we're close enough, stop searching
            if diff < 0.02:  # Within 2% of target
                break

        # Use the best gamma found
        best_scale = APOZ * best_gamma / 50
        best_scale = torch.tensor(best_scale)
        best_scale = torch.clamp(best_scale, min=0.1, max=1)

        net = getNet(args, best_scale)
        actual_params = sum(p.numel() for p in net.features.parameters())
        actual_ratio = actual_params / originFeatureParams

        print(f"Found scale with {actual_ratio*100:.1f}% parameters (target: {target_ratio*100:.1f}%)")
        print(f"Scale values: {best_scale[:5]}... (showing first 5)")  # Show first few values

        found_scales.append(best_scale)

    # Add the found scales in order (smallest to largest)
    found_scales.reverse()  # Reverse to have smallest first
    found_scales.append(np.ones(len(APOZ)))  # Add 100% model

    return found_scales


# For backward compatibility, keep the original function name
def HeteroFL(args, dataset_train, dataset_test, dict_users):
    """Wrapper function that calls the 4-level version"""
    return HeteroFL_4levels(args, dataset_train, dataset_test, dict_users)