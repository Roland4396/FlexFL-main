#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import json
import os

import torch


def save_result(data, ylabel, args):
    data = {'base': data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                    ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"))
    else:
        path += '/{}'.format(args.data_beta)
        file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                    ylabel, args.epochs, args.lr,
                                                    datetime.datetime.now().strftime(
                                                        "%Y_%m_%d_%H_%M_%S"))

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()


def my_save_result(data, label, ylabel, args):  # label 这一行对应的模型配置
    data = {label: data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:  # iid
        file = f'{args.dataset}_{args.algorithm}_{args.model}_{args.client_hetero_ration.replace(":", "")}_{ylabel}_{datetime.datetime.now().strftime("%m_%d")}.txt'
    else:  # Non-iid
        path += '/{}'.format(args.data_beta)
        file = f'{args.dataset}_{args.algorithm}_{args.model}_{args.client_hetero_ration.replace(":", "")}_{ylabel}_{datetime.datetime.now().strftime("%m_%d")}.txt'

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()

    return os.path.join(path, file)


def get_final_acc(file):
    idx = 0
    max = 0
    list = []
    with open(file, 'r') as f:
        for id, item in enumerate(f.readlines()):
            tmp = item.split()[5:]
            list.append(tmp)

    for i in range(len(list[0])):
        t = [float(row[i]) for row in list]

        # 计算第三列的平均值
        total = sum(t) / len(t) + t[-1]
        if  total > max :
            idx = i
            max = total

    print([row[idx] for row in list])


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _state_dict_to_cpu(state_dict):
    cpu_state = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def save_model_checkpoints(args, models, names=None, metadata=None):
    checkpoint_dir = getattr(args, "save_checkpoint_dir", "")
    if not checkpoint_dir:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    if names is None:
        names = ["level_{}".format(i + 1) for i in range(len(models))]
    if len(names) != len(models):
        raise ValueError("checkpoint names and models length mismatch")

    saved_files = []
    for name, model in zip(names, models):
        state_dict = model.state_dict() if hasattr(model, "state_dict") else model
        path = os.path.join(checkpoint_dir, "{}.pt".format(name))
        torch.save(_state_dict_to_cpu(state_dict), path)
        saved_files.append(path)

    info = {
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": getattr(args, "algorithm", None),
        "model": getattr(args, "model", None),
        "dataset": getattr(args, "dataset", None),
        "iid": getattr(args, "iid", None),
        "data_beta": getattr(args, "data_beta", None),
        "epochs": getattr(args, "epochs", None),
        "num_users": getattr(args, "num_users", None),
        "frac": getattr(args, "frac", None),
        "local_ep": getattr(args, "local_ep", None),
        "local_bs": getattr(args, "local_bs", None),
        "bs": getattr(args, "bs", None),
        "optimizer": getattr(args, "optimizer", None),
        "lr": getattr(args, "lr", None),
        "lr_decay": getattr(args, "lr_decay", None),
        "momentum": getattr(args, "momentum", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "seed": getattr(args, "seed", None),
        "client_hetero_ration": getattr(args, "client_hetero_ration", None),
        "client_chosen_mode": getattr(args, "client_chosen_mode", None),
        "checkpoint_files": saved_files,
    }
    if metadata:
        info.update(metadata)

    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(info), handle, ensure_ascii=False, indent=2)

    print("checkpoints saved to {}".format(checkpoint_dir))
    return checkpoint_dir

if __name__ == '__main__':
    get_final_acc(r'C:\Users\20626\Desktop\Code\fed_master\output\0\cifar10_FedSlim_vgg_433_acc_10_30.txt')
