#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import numpy as np
from tqdm import tqdm

from arch_profiles import build_width_only_scale_list
from getAPOZ import modelList
from models.Fed import (
    Aggregation_FedRolex,
    FlexFL_select_clients,
    split_model,
    split_model_fedrolex,
    summon_clients,
)
from models.Update import LocalUpdate_FedAvg
from models.test import test
from utils.utils import save_model_checkpoints
from wandbUtils import endrun, init_run, upload_data


def FedRolex(args, dataset_train, dataset_test, dict_users):
    """FedRolex baseline with rolling submodel extraction.

    The model sizes are the same width-only four levels used by HeteroFL.
    The difference is parameter mapping: FedRolex rotates the active channel
    window across communication rounds, so uncovered global parameters are
    trained in later rounds instead of staying permanently inactive.
    """
    run = init_run(args, "Fed-Experiment")

    scale_list = build_width_only_scale_list(args)
    net_glob_list, net_slim_info = modelList(args, scale_list)
    clients_list = summon_clients(args)
    avg_acc = [0]

    print("\n" + "=" * 80)
    print("FedRolex Configuration")
    print("=" * 80)
    print(f"client_hetero_ration: {args.client_hetero_ration}")
    print("width_mode: rolling_width")
    print("metric: final global/largest model accuracy")
    for idx, info in enumerate(net_slim_info):
        print(f"Level {idx + 1}: {info}")
    print("=" * 80 + "\n")

    for iter in tqdm(range(args.epochs)):
        print("*" * 80)
        print("Round {:3d}".format(iter))

        w_locals = []
        lens = []

        m = max(int(args.frac * args.num_users), 1)
        models = np.random.choice(range(len(net_glob_list)), m, replace=True)
        idx_users = FlexFL_select_clients(args, clients_list, models, len(net_glob_list) >= 3)

        print(f"this epoch choose: {idx_users}")
        print(f"this epoch models: {models}")
        print(f"hetero_proportion: \t{args.client_hetero_ration}")
        print(f"fedrolex_round_offset: {iter}")

        global_state = net_glob_list[-1].state_dict()
        for user_idx, model_idx in idx_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[user_idx])
            local_net = copy.deepcopy(net_glob_list[model_idx]).to(args.device)
            local_net.load_state_dict(split_model_fedrolex(global_state, local_net.state_dict(), iter))
            w = local.train(round=iter, net=local_net)
            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[user_idx]))

        w_glob = Aggregation_FedRolex(w_locals, lens, net_glob_list[-1].state_dict(), iter)
        acc_dict = {}
        for idx, net in enumerate(net_glob_list):
            # Evaluation still uses the canonical prefix submodel for each
            # level; the paper result is the largest/global model.
            net.load_state_dict(split_model(w_glob, net.state_dict()))
            print(net_slim_info[idx])
            acc_dict[f"{idx}-acc"] = test(net, dataset_test, args)
        upload_data(args, run, iter, acc_dict, avg_acc, net_slim_info)

    save_model_checkpoints(
        args,
        net_glob_list,
        metadata={
            "scale_list": scale_list,
            "net_slim_info": net_slim_info,
            "width_mode": "rolling_width",
            "client_hetero_ration": args.client_hetero_ration,
            "metric": "final global/largest model accuracy",
            "kd": "disabled",
        },
    )
    endrun(run)
