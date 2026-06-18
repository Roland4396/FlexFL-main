import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from arch_profiles import get_scalefl_profile
from models.Fed import split_model, select_clients, Aggregation_ScaleFL, summon_clients, FlexFL_select_clients
from models.mobileNetV2_scaleFL import MobileNetV2_scaleFL
from models.resnet_scaleFL import ResNet18_cifar_scaleFL
from models.resnet_smart_scaleFL import ResNet110_cifar_scaleFL
from models.vit_flexfl import vit_small_scalefl
from models.vgg_scaleFL import vgg_16_scaleFL
from utils.utils import my_save_result, get_final_acc, save_model_checkpoints
from models.Update import LocalUpdate_ScaleFL
from wandbUtils import upload_data, endrun, init_run


def ScaleFL(args, dataset_train, dataset_test, dict_users):
    profile = get_scalefl_profile(args) if args.model in {"vgg", "resnet_smart", "vit"} else None
    model_rate = list(profile["scales"]) if profile is not None else args.width_ration
    exit0, exit1, exit2 = profile["exits"] if profile is not None else (6, 8, 10)
    run = init_run(args, "Fed-Experiment")
    net_glob_list = []
    net_slim_info = []
    for i in model_rate:
        if args.model == 'vgg':
            net = vgg_16_scaleFL(
                num_classes=args.num_classes,
                track_running_stats=False,
                num_channels=args.num_channels,
                scale=i,
                exit0=exit0,
                exit1=exit1,
                exit2=exit2,
            )
        elif args.model == 'resnet':
            net = ResNet18_cifar_scaleFL(num_channels=args.num_channels, num_classes=args.num_classes, track_running_stats=False, scale=i)
        elif args.model == 'resnet_smart':
            net = ResNet110_cifar_scaleFL(
                num_channels=args.num_channels,
                num_classes=args.num_classes,
                track_running_stats=False,
                scale=i,
                exit0=exit0,
                exit1=exit1,
                exit2=exit2,
            )
        elif args.model == 'mobilenet':
            net = MobileNetV2_scaleFL(num_channels=args.num_channels, num_classes=args.num_classes, trs=False, scale=i)
        elif args.model == 'vit':
            net = vit_small_scalefl(
                num_classes=args.num_classes,
                num_channels=args.num_channels,
                image_size=args.image_size,
                scale=i,
                exits=(exit0, exit1, exit2),
            )

        total = sum([param.nelement() for param in net.parameters()])
        net.to(args.device)
        net.train()
        print("==" * 50)
        print('【model config】  model_name:{}, width:{} , param:{}MB'.format(args.model, i, total * 4 / 1e6))
        print(net)
        net_glob_list.append(net)
        net_slim_info.append((i, (exit0, exit1, exit2), total * 4 / 1e6))  # 宽度 退出点 参数量

    # training
    acc_list = [[] for _ in net_glob_list]

    # 开始训练
    avg_acc = [0]
    clients_list = summon_clients(args)
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        grad_info = []
        lens = []

        m = max(int(args.frac * args.num_users), 1)
        models = np.random.choice(range(len(net_glob_list)), m, replace=True)  # 模型选择
        # 当有4个模型时使用is3=True（实际上支持4个模型了）
        idx_users = FlexFL_select_clients(args, clients_list, models, len(net_glob_list) >= 3)

        print(f"this epoch choose: {idx_users}")
        print(f"this epoch models: {models}")
        print(f"hetero_proportion: \t{args.client_hetero_ration}")
        # 需要print 每个客户端的计算资源

        for id, (user_idx, model_idx) in enumerate(idx_users):
            local = LocalUpdate_ScaleFL(args=args, dataset=dataset_train, idxs=dict_users[user_idx])
            w, requires_grad = local.train(round=iter,
                                           net=copy.deepcopy(net_glob_list[model_idx]).to(args.device), ee=model_idx + 1)  # 这里开始正式训练

            w_locals.append(copy.deepcopy(w))
            grad_info.append(copy.deepcopy(requires_grad))
            lens.append(len(dict_users[user_idx]))

        w_glob = Aggregation_ScaleFL(w_locals, lens, grad_info, net_glob_list[-1].state_dict())
        accDict = {}
        for idx, net in enumerate(net_glob_list):
            net.load_state_dict(split_model(w_glob, net.state_dict()))
            print(net_slim_info[idx])
            accDict[f"{idx}-acc"] = test_scaleFL(net, dataset_test, args, ee=idx + 1)
        upload_data(args, run, iter, accDict, avg_acc, net_slim_info)
    save_model_checkpoints(
        args,
        net_glob_list,
        metadata={
            "profile": profile,
            "scales": model_rate,
            "exits": (exit0, exit1, exit2),
            "net_slim_info": net_slim_info,
            "kd_gamma": args.gamma,
            "kd_T": args.T,
            "kd_active_after_round": args.epochs * 0.25,
        },
    )
    endrun(run)


def test_scaleFL(net_glob, dataset_test, args, ee):
    # testing
    acc_test, loss_test = test_img_scaleFL(net_glob, dataset_test, args, ee)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()


def test_img_scaleFL(net_g, datatest, args, ee):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            if args.dataset == 'widar':
                target = target.long()
            log_probs = net_g(data, ee)[-1]['output']
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    print(sum(p.numel() for p in net_g.parameters()))
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
