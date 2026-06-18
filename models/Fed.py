#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
from collections import OrderedDict

import numpy as np
import torch

from .mobileNetV2 import MobileNetV2
from .resnet import ResNet18_cifar
from .vgg import vgg_16_bn


def _prefix_slices(tensor, target_shape):
    slices = tuple(slice(0, dim) for dim in target_shape)
    return tensor[slices]


def _is_qkv_weight(key, tensor):
    return key.endswith(".attn.in_proj_weight") and tensor.dim() == 2 and tensor.size(0) % 3 == 0


def _is_qkv_bias(key, tensor):
    return key.endswith(".attn.in_proj_bias") and tensor.dim() == 1 and tensor.size(0) % 3 == 0


def _slice_qkv_weight(source, target_shape):
    source_qkv_dim = source.size(0) // 3
    target_qkv_dim = target_shape[0] // 3
    target_in_dim = target_shape[1]
    pieces = []
    for offset in (0, source_qkv_dim, source_qkv_dim * 2):
        pieces.append(source[offset:offset + target_qkv_dim, :target_in_dim])
    return torch.cat(pieces, dim=0)


def _slice_qkv_bias(source, target_shape):
    source_qkv_dim = source.size(0) // 3
    target_qkv_dim = target_shape[0] // 3
    pieces = []
    for offset in (0, source_qkv_dim, source_qkv_dim * 2):
        pieces.append(source[offset:offset + target_qkv_dim])
    return torch.cat(pieces, dim=0)


def _slice_param(key, source, target_shape):
    if _is_qkv_weight(key, source):
        return _slice_qkv_weight(source, target_shape)
    if _is_qkv_bias(key, source):
        return _slice_qkv_bias(source, target_shape)
    return _prefix_slices(source, target_shape)


def _accumulate_qkv_weight(tmp_v, count_v, local_v, sample_count):
    global_qkv_dim = tmp_v.size(0) // 3
    local_qkv_dim = local_v.size(0) // 3
    local_in_dim = local_v.size(1)
    for local_offset, global_offset in zip(
        (0, local_qkv_dim, local_qkv_dim * 2),
        (0, global_qkv_dim, global_qkv_dim * 2),
    ):
        target = (slice(global_offset, global_offset + local_qkv_dim), slice(0, local_in_dim))
        source = (slice(local_offset, local_offset + local_qkv_dim), slice(0, local_in_dim))
        tmp_v[target] += local_v[source] * sample_count
        count_v[target] += sample_count


def _accumulate_qkv_bias(tmp_v, count_v, local_v, sample_count):
    global_qkv_dim = tmp_v.size(0) // 3
    local_qkv_dim = local_v.size(0) // 3
    for local_offset, global_offset in zip(
        (0, local_qkv_dim, local_qkv_dim * 2),
        (0, global_qkv_dim, global_qkv_dim * 2),
    ):
        target = slice(global_offset, global_offset + local_qkv_dim)
        source = slice(local_offset, local_offset + local_qkv_dim)
        tmp_v[target] += local_v[source] * sample_count
        count_v[target] += sample_count


def _accumulate_param(key, tmp_v, count_v, local_v, sample_count):
    if _is_qkv_weight(key, tmp_v):
        _accumulate_qkv_weight(tmp_v, count_v, local_v, sample_count)
        return
    if _is_qkv_bias(key, tmp_v):
        _accumulate_qkv_bias(tmp_v, count_v, local_v, sample_count)
        return
    slices = tuple(slice(0, dim) for dim in local_v.shape)
    tmp_v[slices] += local_v * sample_count
    count_v[slices] += sample_count


def Aggregation(w, lens):
    w_avg = None
    total_count = sum(lens)

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg


def split_model(global_param, slim_param):
    param = copy.deepcopy(slim_param)
    for k, v in param.items():  # 遍历所有层，每一层遍历
        # 检查该层是否在全局模型中存在
        if k not in global_param:
            # 如果不存在（例如小模型有shortcut但大模型没有），跳过
            continue
        if v.dim() == 0:
            param[k] = global_param[k]
            continue
        param[k] = _slice_param(k, global_param[k], v.shape)
    return param


def Aggregation_FedSlim(w, lens, global_model_param):
    w_avg = copy.deepcopy(global_model_param)  # largest model
    count = OrderedDict()
    for k, v in w_avg.items():  # 遍历所有层，每一层遍历
        parameter_type = k.split('.')[-1]

        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(w)):  # 遍历所有用户
            if k not in w[m]:
                continue
            if v.dim() == 0:
                tmp_v += w[m][k] * lens[m]
                count[k] += lens[m]
                continue
            _accumulate_param(k, tmp_v, count[k], w[m][k], lens[m])

        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        tmp_v[count[k] == 0] = global_model_param[k][count[k] == 0]
        w_avg[k] = tmp_v

    return w_avg


def Aggregation_ScaleFL(w, lens, grad_info, global_model_param):
    w_avg = copy.deepcopy(global_model_param)  # largest model
    count = OrderedDict()
    for idx, (k, v) in enumerate(w_avg.items()):  # 遍历所有层，每一层遍历
        parameter_type = k.split('.')[-1]

        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(w)):  # 遍历所有用户
            if isinstance(grad_info[m], dict):
                has_grad = grad_info[m].get(k, False)
            else:
                has_grad = idx < len(grad_info[m]) and grad_info[m][idx]

            if has_grad and k in w[m]:
                if v.dim() == 0:
                    tmp_v += w[m][k] * lens[m]
                    count[k] += lens[m]
                    continue
                _accumulate_param(k, tmp_v, count[k], w[m][k], lens[m])

        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        tmp_v[count[k] == 0] = global_model_param[k][count[k] == 0]
        w_avg[k] = tmp_v

    return w_avg


def _rolling_indices(global_dim, local_dim, round_idx, device):
    if local_dim > global_dim:
        raise ValueError(f"Local dim {local_dim} exceeds global dim {global_dim}")
    if local_dim == global_dim:
        return torch.arange(global_dim, device=device)
    start = int(round_idx) % int(global_dim)
    return (torch.arange(local_dim, device=device) + start) % global_dim


def _qkv_rolling_indexers(key, global_v, local_v, round_idx):
    if _is_qkv_weight(key, global_v):
        global_qkv_dim = global_v.size(0) // 3
        local_qkv_dim = local_v.size(0) // 3
        qkv_idx = _rolling_indices(global_qkv_dim, local_qkv_dim, round_idx, global_v.device)
        qkv_idx = torch.cat([
            qkv_idx,
            qkv_idx + global_qkv_dim,
            qkv_idx + global_qkv_dim * 2,
        ])
        input_idx = _rolling_indices(global_v.size(1), local_v.size(1), round_idx, global_v.device)
        return [qkv_idx, input_idx]
    if _is_qkv_bias(key, global_v):
        global_qkv_dim = global_v.size(0) // 3
        local_qkv_dim = local_v.size(0) // 3
        qkv_idx = _rolling_indices(global_qkv_dim, local_qkv_dim, round_idx, global_v.device)
        return [torch.cat([
            qkv_idx,
            qkv_idx + global_qkv_dim,
            qkv_idx + global_qkv_dim * 2,
        ])]
    return None


def _rolling_indexers(key, global_v, local_v, round_idx):
    qkv_indexers = _qkv_rolling_indexers(key, global_v, local_v, round_idx)
    if qkv_indexers is not None:
        return qkv_indexers
    return [
        _rolling_indices(global_dim, local_dim, round_idx, global_v.device)
        for global_dim, local_dim in zip(global_v.shape, local_v.shape)
    ]


def _select_by_indexers(source, indexers):
    selected = source
    for dim, idx in enumerate(indexers):
        if idx.numel() == selected.size(dim) and torch.equal(idx, torch.arange(selected.size(dim), device=idx.device)):
            continue
        selected = selected.index_select(dim, idx)
    return selected


def split_model_fedrolex(global_param, slim_param, round_idx):
    """Extract a rolling-width submodel from the global model."""
    param = copy.deepcopy(slim_param)
    for k, v in param.items():
        if k not in global_param:
            continue
        global_v = global_param[k]
        if v.shape == global_v.shape or v.dim() == 0:
            param[k] = global_v.detach().clone()
            continue
        indexers = _rolling_indexers(k, global_v, v, round_idx)
        param[k] = _select_by_indexers(global_v, indexers).detach().clone()
    return param


def _meshgrid(indexers):
    try:
        return torch.meshgrid(*indexers, indexing="ij")
    except TypeError:
        return torch.meshgrid(*indexers)


def _accumulate_rolling_param(key, tmp_v, count_v, local_v, sample_count, round_idx):
    if local_v.shape == tmp_v.shape:
        tmp_v += local_v.to(dtype=tmp_v.dtype) * sample_count
        count_v += sample_count
        return
    indexers = _rolling_indexers(key, tmp_v, local_v, round_idx)
    grid = _meshgrid(indexers)
    tmp_v[grid] += local_v.to(dtype=tmp_v.dtype) * sample_count
    count_v[grid] += sample_count


def Aggregation_FedRolex(w, lens, global_model_param, round_idx):
    """Aggregate rolling submodel updates back into the global model."""
    w_avg = copy.deepcopy(global_model_param)
    for k, v in w_avg.items():
        if not torch.is_floating_point(v):
            w_avg[k] = global_model_param[k]
            continue

        count_v = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for local_state, sample_count in zip(w, lens):
            if k not in local_state:
                continue
            local_v = local_state[k]
            if v.dim() == 0:
                tmp_v += local_v.to(dtype=tmp_v.dtype) * sample_count
                count_v += sample_count
                continue
            _accumulate_rolling_param(k, tmp_v, count_v, local_v, sample_count, round_idx)

        updated = count_v > 0
        if updated.any():
            tmp_v[updated] = tmp_v[updated].div_(count_v[updated])
            tmp_v[~updated] = global_model_param[k][~updated]
            w_avg[k] = tmp_v.to(dtype=v.dtype)
        else:
            w_avg[k] = global_model_param[k]
    return w_avg


def get_model_list(args):
    model_rate = args.width_ration
    depth_list = args.depth_saved

    net_glob_list = []
    net_slim_info = []
    for i in model_rate:
        for depth in depth_list:
            if args.model == 'vgg':
                net = vgg_16_bn(num_classes=args.num_classes, track_running_stats=False, num_channels=args.num_channels,
                                rate=[1] * depth + [i] * (15 - depth)).to(args.device)
            elif args.model == 'resnet':
                if args.dataset == 'widar':
                    pass
                    # net = ResNet18_widar(num_classes=args.num_classes, track_running_stats=False, slim_idx=depth, scale=i)
                else:
                    net = ResNet18_cifar(args.num_channels, args.num_classes, False, [1] * depth + [i] * (5 - depth))

            elif args.model == 'mobilenet':
                net = MobileNetV2(args.num_channels, args.num_classes, False, [1] * depth + [i] * (9 - depth))

            total = sum([param.nelement() for param in net.parameters()])
            net.to(args.device)
            net.train()
            print("==" * 50)
            print('【model config】  model_name:{}, width:{} , depth:{}, param:{}MB'.format(args.model, i, depth, total * 4 / 1e6))
            # print(net)
            net_glob_list.append(net)
            net_slim_info.append((i, depth, total / 1e6))  # 宽度 深度 参数量

            if i == 1.0:
                break
    return net_glob_list, net_slim_info


def select_clients(args, ration_users, net_glob_list_len):
    my_list = list(map(float, args.client_hetero_ration.split(':')))
    hetero_proportion = [round(x / sum(my_list), 2) for x in my_list]

    idx_users = []
    if net_glob_list_len == 7:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if int(model_type / 3) == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif int(model_type / 3) == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
                elif int(model_type / 3) == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if int(model_type / 3) == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif int(model_type / 3) == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
                elif int(model_type / 3) == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), int(args.num_users * sum(hetero_proportion[:3])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    elif net_glob_list_len == 5:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), args.num_users - 1))
                elif model_type == 3:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:3])), args.num_users - 1))
                elif model_type == 4:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:4])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), int(args.num_users * sum(hetero_proportion[:3])) - 1))
                elif model_type == 3:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:3])), int(args.num_users * sum(hetero_proportion[:4])) - 1))
                elif model_type == 4:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:4])), int(args.num_users * sum(hetero_proportion[:5])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    elif net_glob_list_len == 3:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), int(args.num_users * sum(hetero_proportion[:3])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    elif net_glob_list_len == 2:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    elif net_glob_list_len == 1:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))

    return idx_users


def summon_clients(args):
    clients = []  # Every client is a tuple, miu ,sigma
    client_hetero_ration = list(map(float, args.client_hetero_ration.split(':')))

    # Support both 3 and 4 resource levels
    num_levels = len(client_hetero_ration)
    total_ratio = sum(client_hetero_ration)

    if num_levels == 3:
        # Original 3-level mode (for backward compatibility)
        users25 = int(args.num_users * round(client_hetero_ration[0] / total_ratio, 2))
        users50 = int(args.num_users * round(client_hetero_ration[1] / total_ratio, 2))
        users100 = int(args.num_users * round(client_hetero_ration[2] / total_ratio, 2))

        if args.r == 0:
            for i in range(users25):
                clients.append((35, random.choice([5, 8, 10])))
            for i in range(users50):
                clients.append((60, random.choice([5, 8, 10])))
            for i in range(users100):
                clients.append((110, random.choice([5, 8, 10])))
        elif args.r == 1:
            for i in range(users25):
                clients.append((35, random.choice([0])))
            for i in range(users50):
                clients.append((60, random.choice([0])))
            for i in range(users100):
                clients.append((110, random.choice([0])))
        elif args.r == 2:
            for i in range(users25):
                clients.append((35, random.choice([10, 20, 30])))
            for i in range(users50):
                clients.append((60, random.choice([10, 20, 30])))
            for i in range(users100):
                clients.append((110, random.choice([10, 20, 30])))

    elif num_levels == 4:
        # New 4-level mode: supports 4 model sizes (1/8, 1/4, 1/2, 1)
        users12_5 = int(args.num_users * round(client_hetero_ration[0] / total_ratio, 2))
        users25 = int(args.num_users * round(client_hetero_ration[1] / total_ratio, 2))
        users50 = int(args.num_users * round(client_hetero_ration[2] / total_ratio, 2))
        users100 = int(args.num_users * round(client_hetero_ration[3] / total_ratio, 2))

        if args.r == 0:
            for i in range(users12_5):
                clients.append((20, random.choice([5, 8, 10])))  # Low resource for 1/8 model
            for i in range(users25):
                clients.append((40, random.choice([5, 8, 10])))  # Medium-low for 1/4 model
            for i in range(users50):
                clients.append((70, random.choice([5, 8, 10])))  # Medium-high for 1/2 model
            for i in range(users100):
                clients.append((110, random.choice([5, 8, 10])))  # High resource for 1 model
        elif args.r == 1:
            for i in range(users12_5):
                clients.append((20, random.choice([0])))
            for i in range(users25):
                clients.append((40, random.choice([0])))
            for i in range(users50):
                clients.append((70, random.choice([0])))
            for i in range(users100):
                clients.append((110, random.choice([0])))
        elif args.r == 2:
            for i in range(users12_5):
                clients.append((20, random.choice([10, 20, 30])))
            for i in range(users25):
                clients.append((40, random.choice([10, 20, 30])))
            for i in range(users50):
                clients.append((70, random.choice([10, 20, 30])))
            for i in range(users100):
                clients.append((110, random.choice([10, 20, 30])))

    return clients


def FlexFL_select_clients(args, clients, models, is3=False):
    selected_user = []
    current_user_list = list(map(float, args.client_hetero_ration.split(':')))
    if is3:
        for model in models:
            if model == 0:
                user_idx = random.randint(0, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model4(resource, 0)))
            elif model == 1:
                user_idx = random.randint(4, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model4(resource, 1)))
            elif model == 2:
                user_idx = random.randint(14, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model4(resource, 2)))
            elif model == 3:
                user_idx = random.randint(20, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model4(resource, 3)))
            else:
                raise Exception
    else:
        for model in models:
            if model == 0:
                user_idx = random.randint(0, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model(resource, 0)))
            elif model == 1:
                user_idx = random.randint(3, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model(resource, 1)))
            elif model == 2:
                user_idx = random.randint(7, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model(resource, 2)))
            elif model == 3:
                user_idx = random.randint(14, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model(resource, 3)))
            else:
                raise Exception

    return selected_user  # user_idx , model


def resource_to_model(resource, original_model):
    # Map resource to 4 model levels: 0 (1/8), 1 (1/4), 2 (1/2), 3 (1)
    if resource < 30:
        model = 0  # Model 0: 1/8
    elif resource < 50:
        model = 1  # Model 1: 1/4
    elif resource < 90:
        model = 2  # Model 2: 1/2
    else:
        model = 3  # Model 3: 1
    return min(original_model, model)


def resource_to_model3(resource, original_model):
    if resource < 50:
        model = 0
    elif resource < 100:
        model = 1
    else:
        model = 2
    return min(original_model, model)


def resource_to_model4(resource, original_model):
    """Support 4 model levels for ScaleFL"""
    if resource < 40:
        model = 0  # 1/8 参数量
    elif resource < 70:
        model = 1  # 1/4 参数量
    elif resource < 100:
        model = 2  # 1/2 参数量
    else:
        model = 3  # 完整模型
    return min(original_model, model)
