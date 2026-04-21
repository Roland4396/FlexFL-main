#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decoupled with Single Level - 只训练最小的模型 (1/8参数量)
本质上就是标准FedAvg，但使用缩小版模型
"""

import torch
import copy
import numpy as np
from tqdm import tqdm
from torch import nn

from getAPOZ import getNet
from models.Fed import Aggregation
from models.test import test
from models.Update import LocalUpdate_FedAvg
from wandbUtils import init_run, upload_data, endrun


def Decoupled_SingleLevel(args, dataset_train, dataset_test, dict_users):
    """
    只训练最小模型的Decoupled版本
    参数量为完整模型的1/8
    """

    # 创建最小的模型 (1/8参数量)
    # Scale = sqrt(1/8) ≈ 0.354
    scale_rate = torch.tensor([0.354] * 50)  # 所有层使用相同的缩放
    net_glob = getNet(args, scale_rate)

    # 计算模型参数量
    total_params = sum(p.numel() for p in net_glob.parameters())
    print("\n" + "="*80)
    print("Decoupled Single Level (最小模型)")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"参数量: {total_params:,} (约为完整模型的1/8)")
    print(f"Scale: 0.354")
    print("="*80 + "\n")

    net_glob.to(args.device)
    net_glob.train()

    # 初始化
    run = init_run(args, "Fed-Experiment")
    avg_acc = []

    # 开始训练 (标准FedAvg流程)
    for iter in tqdm(range(args.epochs)):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []

        # 选择客户端
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        print(f"Selected clients: {idxs_users}")

        # 本地训练
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

        # FedAvg聚合
        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)

        # 测试
        acc = test(net_glob, dataset_test, args)
        avg_acc.append(acc)

        print(f"Round {iter}: Test Accuracy = {acc:.2f}%")

        # 上传数据到wandb（如果启用）
        if args.log:
            accDict = {"small_model_acc": acc}
            upload_data(args, run, iter, accDict, avg_acc, ["1/8 model"])

    if args.log:
        endrun(run)

    print("\n" + "="*80)
    print(f"训练完成！最终准确率: {avg_acc[-1]:.2f}%")
    print("="*80)

    return net_glob, avg_acc