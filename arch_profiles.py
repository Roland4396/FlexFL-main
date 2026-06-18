import math
from types import SimpleNamespace

import numpy as np
import torch

from getAPOZ import getNet
from models.vit_flexfl import (
    VIT_DEPTH,
    VIT_HIDDEN_WIDTH_OPTIONS,
    VIT_SEARCH_EXIT_LOCATIONS,
    VIT_WIDTH_OPTIONS,
    vit_active_flops,
    vit_active_params,
    vit_rate_uses_hidden_width,
    vit_stage_feature_param_vector,
    vit_total_flops,
    vit_total_params,
)


TARGET_PARAM_BUDGETS_M = {
    "vgg": (5.806, 9.834, 18.424, 33.647),
    "resnet_smart": (0.288, 0.506, 0.946, 1.731),
    "vit": (2.713, 5.426, 10.852, 21.704),
}

VIT_TARGET_PARAM_BUDGETS_M_BY_DATASET = {
    "cifar10": (2.709, 5.417, 10.835, 21.670),
    "cifar100": (2.713, 5.426, 10.852, 21.704),
    "TinyImagenet": (2.718, 5.436, 10.871, 21.743),
}

TARGET_FLOP_BUDGETS_M = {
    # SmartFL 27fc95a trains ViT with 224x224 crops. FLOPs are kept absolute
    # here so all baselines search against the same full-model upper bound.
    "vit": (574.8, 1149.6, 2299.3, 4598.5),
}

VIT_TARGET_FLOP_BUDGETS_M_BY_DATASET = {
    "cifar10": (574.8, 1149.6, 2299.3, 4598.5),
    "cifar100": (574.8, 1149.6, 2299.3, 4598.5),
    "TinyImagenet": (574.8, 1149.6, 2299.3, 4598.6),
}
TARGET_RESOURCE_RATIOS = (0.125, 0.25, 0.5, 1.0)

WIDTH_ONLY_SCALES = {
    "vgg": (0.421875, 0.546875, 0.734375, 1.0),
    "resnet_smart": (0.40625, 0.546875, 0.734375, 1.0),
    "vit": None,
}

RATE_LENGTHS = {
    "vgg": 15,
    "resnet_smart": 5,
    "vit": 5,
}

VIT_PROFILE_RATE_LENGTH = 4
VIT_WIDTH_ONLY_EXITS = (3, 6, 9, 12)

UNIFORM_SCALE_GRID = [i / 64.0 for i in range(13, 65)]
VIT_SCALE_GRID = list(VIT_WIDTH_OPTIONS)
FLEX_MULTIPLIER_GRID = [i / 64.0 for i in range(8, 513)]
SCALEFL_RATIO_WEIGHT = {
    "vit": 0.35,
}
SCALEFL_SCALE_CANDIDATES_PER_LEVEL = 6
VIT_WIDTH_ONLY_MAX_RELATIVE_ERROR = 0.35
VIT_MLP_ONLY_FIXED_PROFILES = (
    (0.125, 0.125, 0.125, 0.125, 3),
    (11.0 / 12.0, 11.0 / 12.0, 11.0 / 12.0, 11.0 / 12.0, 3),
    (0.75, 0.75, 0.75, 0.75, 7),
    (1.0, 1.0, 1.0, 1.0, 12),
)
VIT_HIDDEN_MODE_MARKER = 1.0

SCALEFL_DEPTH_TOTAL = {
    "vgg": 13,
    "resnet_smart": 54,
    "vit": 12,
}

SCALEFL_FIXED_PROFILES = {
    "vgg": {
        "exits": (6, 7, 10),
        "scales": (0.515625, 0.515625, 0.546875, 1.0),
    },
    "resnet_smart": {
        "exits": (38, 42, 47),
        "scales": (0.734375, 0.78125, 0.890625, 1.0),
    },
}

VGG16_CHANNELS = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512)
VGG16_POOL_AFTER_CONV = {2, 4, 7, 10, 13}
RESNET110_STAGE_BLOCKS = (18, 18, 18)

FLEXFL_APOZ_PRESETS = {
    ("resnet_smart", "cifar10"): [
        0.5686834259033203,
        0.652520519606769,
        0.6017078437805177,
        0.7603804524739582,
        0.9060739440917969,
    ],
    ("resnet_smart", "cifar100"): [
        0.4976362762451172,
        0.5279737923443317,
        0.5184498920440673,
        0.5904706649780272,
        0.6594305216471354,
    ],
    ("resnet_smart", "TinyImagenet"): [
        0.4521520824432373,
        0.5559043244421483,
        0.5453087773323059,
        0.5657702350616455,
        0.7452228037516276,
    ],
    ("vit", "cifar10"): [0.50, 0.55, 0.65, 0.80],
    ("vit", "cifar100"): [0.52, 0.58, 0.68, 0.82],
    ("vit", "TinyImagenet"): [0.48, 0.54, 0.62, 0.76],
    ("vgg", "cifar10"): [
        0.5064863739013672,
        0.5133746032714843,
        0.6503060302734375,
        0.6177527160644531,
        0.6347842559814453,
        0.5949074096679687,
        0.5648363647460938,
        0.6065169677734376,
        0.6035700073242187,
        0.5883703002929688,
        0.5890387573242187,
        0.8103564453125,
        0.96360400390625,
        0.9940224609375,
        0.8050889892578125,
    ],
    ("vgg", "cifar100"): [
        0.49952767181396485,
        0.517141845703125,
        0.601208812713623,
        0.57901123046875,
        0.6531083755493164,
        0.6345844421386718,
        0.5976306762695313,
        0.6096923217773438,
        0.5957570190429687,
        0.6082689208984375,
        0.6996626586914062,
        0.780442626953125,
        0.985472412109375,
        0.9742340087890625,
        0.88470068359375,
    ],
    ("vgg", "TinyImagenet"): [
        0.5647340869903564,
        0.5541139526367187,
        0.6484428253173828,
        0.6571991767883301,
        0.7164076919555664,
        0.7096979827880859,
        0.683664794921875,
        0.785942024230957,
        0.752685302734375,
        0.7385288543701172,
        0.8081832733154297,
        0.9174990844726563,
        0.972046142578125,
        0.98541162109375,
        0.947284912109375,
    ],
}

FLEXFL_FIXED_RATES = {
    ("vgg", "cifar10"): (
        [1.0, 0.8863976001739502, 0.7371810078620911, 0.7323241233825684, 0.681614339351654, 0.6887738108634949, 0.7203015685081482, 0.6434333920478821, 0.6137080192565918, 0.6313056945800781, 0.6305318474769592, 0.37429800629615784, 0.19687332212924957, 0.17079107463359833, 0.25582069158554077],
        [1.0, 1.0, 0.9653561115264893, 0.9589958786964417, 0.8925901651382446, 0.9019656777381897, 0.9432520270347595, 0.8425913453102112, 0.8036652207374573, 0.8267098665237427, 0.8256964683532715, 0.49015215039253235, 0.2578102946281433, 0.22365498542785645, 0.3350032866001129],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.940200924873352, 0.49452704191207886, 0.42901092767715454, 0.6425971984863281],
        [1.0] * 15,
    ),
    ("vgg", "cifar100"): (
        [1.0, 0.9358463883399963, 0.827082097530365, 0.8144655823707581, 0.7028909921646118, 0.6856970191001892, 0.7267470955848694, 0.6780216097831726, 0.6598222851753235, 0.64447420835495, 0.5323634147644043, 0.4332723021507263, 0.18176652491092682, 0.2050386667251587, 0.1603381484746933],
        [1.0, 1.0, 1.0, 1.0, 0.916127622127533, 0.8937174677848816, 0.9472209215164185, 0.8837136030197144, 0.8599931001663208, 0.8399888873100281, 0.6938669085502625, 0.5647144317626953, 0.2369091808795929, 0.2672414183616638, 0.20898005366325378],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4962838888168335, 0.5598246455192566, 0.43777719140052795],
        [1.0] * 15,
    ),
    ("vgg", "TinyImagenet"): (
        [1.0, 1.0, 1.0, 0.9766988754272461, 0.8435485363006592, 0.7984982132911682, 0.8368401527404785, 0.6258275508880615, 0.6195949912071228, 0.6426187753677368, 0.5293339490890503, 0.3515445291996002, 0.26283007860183716, 0.2538141906261444, 0.0971933975815773],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8432760834693909, 0.8348780274391174, 0.8659015893936157, 0.7132550477981567, 0.4736913740634918, 0.3541523814201355, 0.34200388193130493, 0.1309639811515808],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9741954803466797, 0.7283511161804199, 0.7033664584159851, 0.26934102177619934],
        [1.0] * 15,
    ),
    ("resnet_smart", "cifar10"): (
        [1.0, 1.0, 1.0, 0.6824899315834045, 0.2039956897497177],
        [1.0, 1.0, 1.0, 0.9721799492835999, 0.2905837893486023],
        [1.0, 1.0, 1.0, 1.0, 0.6251954436302185],
        [1.0] * 5,
    ),
    ("resnet_smart", "cifar100"): (
        [0.8443275094032288, 0.6480051279067993, 0.6557639241218567, 0.5039970874786377, 0.36717647314071655],
        [1.0, 0.8640068173408508, 0.8743518590927124, 0.6719960570335388, 0.48956865072250366],
        [1.0, 1.0, 1.0, 0.9276467561721802, 0.675817608833313],
        [1.0] * 5,
    ),
    ("resnet_smart", "TinyImagenet"): (
        [0.9911600351333618, 0.715867280960083, 0.7257500290870667, 0.6045376062393188, 0.31449058651924133],
        [1.0, 0.9424075484275818, 0.9554177522659302, 0.7958469986915588, 0.41401293873786926],
        [1.0, 1.0, 1.0, 1.0, 0.6250002980232239],
        [1.0] * 5,
    ),
}


def supports_aligned_profiles(args):
    return args.model in TARGET_PARAM_BUDGETS_M


def target_param_budgets(args):
    if args.model == "vit":
        return tuple(
            int(round(value * 1e6))
            for value in VIT_TARGET_PARAM_BUDGETS_M_BY_DATASET[args.dataset]
        )
    return tuple(int(round(value * 1e6)) for value in TARGET_PARAM_BUDGETS_M[args.model])


def target_flop_budgets(args):
    if args.model == "vit":
        return tuple(
            float(value) * 1e6
            for value in VIT_TARGET_FLOP_BUDGETS_M_BY_DATASET[args.dataset]
        )
    return tuple(float(value) * 1e6 for value in TARGET_FLOP_BUDGETS_M[args.model])


def _cpu_args(args):
    return SimpleNamespace(**{**vars(args), "device": torch.device("cpu")})


def _vit_rate(stage_rates, exit_loc, width_mode="mlp"):
    payload = [*stage_rates[:VIT_PROFILE_RATE_LENGTH], float(exit_loc)]
    if width_mode == "hidden":
        payload.append(VIT_HIDDEN_MODE_MARKER)
    return torch.tensor(payload, dtype=torch.float32)


def build_vit_mlp_only_scale_list(args):
    _ = args
    return [
        torch.tensor(profile, dtype=torch.float32)
        for profile in VIT_MLP_ONLY_FIXED_PROFILES
    ]


def _vit_standalone_active_params(stage_rates, exit_loc, image_size, num_classes, num_channels, width_mode="mlp"):
    if exit_loc >= 12:
        return vit_total_params(
            stage_rates[:VIT_PROFILE_RATE_LENGTH],
            image_size,
            num_classes,
            num_channels,
            width_mode=width_mode,
        )
    return vit_active_params(
        stage_rates[:VIT_PROFILE_RATE_LENGTH],
        (exit_loc,),
        1,
        image_size,
        num_classes,
        num_channels,
        width_mode=width_mode,
    )


def _vit_standalone_active_flops(stage_rates, exit_loc, image_size, num_classes, num_channels, width_mode="mlp"):
    if exit_loc >= 12:
        return vit_total_flops(
            stage_rates[:VIT_PROFILE_RATE_LENGTH],
            image_size,
            num_classes,
            num_channels,
            width_mode=width_mode,
        )
    return vit_active_flops(
        stage_rates[:VIT_PROFILE_RATE_LENGTH],
        (exit_loc,),
        1,
        image_size,
        num_classes,
        num_channels,
        width_mode=width_mode,
    )


def _param_count(args, rate):
    if args.model == "vit":
        image_size = _image_size(args)
        rate_values = [float(v) for v in rate]
        stage_rates = rate_values[:VIT_PROFILE_RATE_LENGTH]
        width_mode = "hidden" if vit_rate_uses_hidden_width(rate_values) else "mlp"
        if len(rate_values) >= RATE_LENGTHS["vit"]:
            exit_loc = int(round(rate_values[4]))
            return _vit_standalone_active_params(
                stage_rates,
                exit_loc,
                image_size,
                args.num_classes,
                args.num_channels,
                width_mode=width_mode,
            )
        return vit_total_params(stage_rates, image_size, args.num_classes, args.num_channels, width_mode=width_mode)
    return sum(param.numel() for param in getNet(_cpu_args(args), rate).parameters())


def _best_uniform_scale(args, target_params):
    best = None
    for scale in UNIFORM_SCALE_GRID:
        params = _param_count(args, [scale] * RATE_LENGTHS[args.model])
        diff = abs(params - target_params)
        if best is None or diff < best[0]:
            best = (diff, scale)
    return best[1]


def _image_size(args):
    return int(getattr(args, "image_size", 224 if args.model == "vit" else (64 if args.dataset == "TinyImagenet" else 32)))


def _vit_total_stats(args, scale, width_mode="mlp"):
    stage_rates = [scale] * VIT_PROFILE_RATE_LENGTH
    image_size = _image_size(args)
    params = vit_total_params(stage_rates, image_size, args.num_classes, args.num_channels, width_mode=width_mode)
    flops = vit_total_flops(stage_rates, image_size, args.num_classes, args.num_channels, width_mode=width_mode)
    return params, flops


def _vit_active_stats(args, stage_rates, exit_loc, width_mode="mlp"):
    image_size = _image_size(args)
    params = _vit_standalone_active_params(
        stage_rates,
        exit_loc,
        image_size,
        args.num_classes,
        args.num_channels,
        width_mode=width_mode,
    )
    flops = _vit_standalone_active_flops(
        stage_rates,
        exit_loc,
        image_size,
        args.num_classes,
        args.num_channels,
        width_mode=width_mode,
    )
    return params, flops


def _resource_score(params, flops, target_params, target_flops):
    return (
        abs(params - target_params) / target_params
        + abs(flops - target_flops) / target_flops
    )


def _resolve_vit_width_profiles(args):
    # HeteroFL fallback: full depth and width-only, but the width space is
    # expanded to hidden dim/QKV because MLP-only cannot hit the 1/8 budget.
    image_size = _image_size(args)
    targets = list(zip(target_param_budgets(args), target_flop_budgets(args)))
    profiles = []
    previous_scale = 0.0
    for level, (target_params, target_flops) in enumerate(targets):
        if level == len(targets) - 1:
            scale = 1.0
            best_score = 0.0
        else:
            best = None
            for scale_candidate in VIT_HIDDEN_WIDTH_OPTIONS:
                if scale_candidate < previous_scale:
                    continue
                params = vit_total_params(
                    [scale_candidate] * VIT_PROFILE_RATE_LENGTH,
                    image_size,
                    args.num_classes,
                    args.num_channels,
                    width_mode="hidden",
                )
                flops = vit_total_flops(
                    [scale_candidate] * VIT_PROFILE_RATE_LENGTH,
                    image_size,
                    args.num_classes,
                    args.num_channels,
                    width_mode="hidden",
                )
                score = _resource_score(params, flops, target_params, target_flops)
                if best is None or score < best[0]:
                    best = (score, scale_candidate)
            best_score, scale = best
            previous_scale = scale
        profiles.append(_vit_rate([scale] * VIT_PROFILE_RATE_LENGTH, VIT_DEPTH, width_mode="hidden"))
    return tuple(profiles)


def build_width_only_scale_list(args):
    rate_len = RATE_LENGTHS[args.model]
    if args.model == "vit" and WIDTH_ONLY_SCALES["vit"] is None:
        return list(_resolve_vit_width_profiles(args))
    else:
        scales = WIDTH_ONLY_SCALES[args.model]
    return [
        torch.tensor([scale] * rate_len, dtype=torch.float32)
        for scale in scales
    ]


def build_smallest_width_model(args):
    if args.model == "vit":
        return getNet(_cpu_args(args), build_vit_mlp_only_scale_list(args)[0])
    return getNet(_cpu_args(args), build_width_only_scale_list(args)[0])


def _conv2d_params(in_channels, out_channels, kernel_size=3, bias=True):
    params = out_channels * in_channels * kernel_size * kernel_size
    return params + (out_channels if bias else 0)


def _bn2d_params(channels):
    return 2 * channels


def _linear_params(in_features, out_features, bias=True):
    params = in_features * out_features
    return params + (out_features if bias else 0)


def _vgg_classifier_params(in_channels, hidden_dim, num_classes):
    return (
        _linear_params(in_channels, hidden_dim)
        + _linear_params(hidden_dim, hidden_dim)
        + _linear_params(hidden_dim, num_classes)
    )


def _vgg_scalefl_active_params(args, scale, exits, level):
    exit_conv = exits[level - 1]
    input_channel = args.num_channels
    feature_params = 0
    exit_channels = None
    for conv_idx, base_channels in enumerate(VGG16_CHANNELS, start=1):
        out_channels = int(base_channels * scale)
        feature_params += _conv2d_params(input_channel, out_channels)
        feature_params += _bn2d_params(out_channels)
        input_channel = out_channels
        if conv_idx == exit_conv:
            exit_channels = out_channels
        if conv_idx >= exit_conv:
            break

    dim = 4096 if args.num_channels == 3 else 256
    hidden_dim = int(dim * scale)
    classifier_params = 0
    for exit_loc in exits[:level]:
        classifier_in = int(VGG16_CHANNELS[exit_loc - 1] * scale)
        classifier_params += _vgg_classifier_params(classifier_in, hidden_dim, args.num_classes)
    _ = exit_channels
    return feature_params + classifier_params


def _vit_scalefl_active_params(args, scale, exits, level):
    image_size = _image_size(args)
    return vit_active_params(scale, exits, level, image_size, args.num_classes, args.num_channels)


def _vit_scalefl_active_flops(args, scale, exits, level):
    image_size = _image_size(args)
    return vit_active_flops(scale, exits, level, image_size, args.num_classes, args.num_channels)


def _resnet_exit_channels(channels, exit_block):
    if exit_block <= RESNET110_STAGE_BLOCKS[0]:
        return channels[0]
    if exit_block <= sum(RESNET110_STAGE_BLOCKS[:2]):
        return channels[1]
    return channels[2]


def _resnet_basic_block_params(in_planes, planes, stride):
    params = _conv2d_params(in_planes, planes, bias=False) + _bn2d_params(planes)
    params += _conv2d_params(planes, planes, bias=False) + _bn2d_params(planes)
    if stride != 1 or in_planes != planes:
        params += _conv2d_params(in_planes, planes, kernel_size=1, bias=False)
        params += _bn2d_params(planes)
    return params


def _resnet_scalefl_active_params(args, scale, exits, level):
    channels = [int(base * scale) for base in (16, 32, 64)]
    params = _conv2d_params(args.num_channels, channels[0], bias=False) + _bn2d_params(channels[0])
    in_channel = channels[0]
    block_limit = exits[level - 1]
    block_idx = 0
    for stage_idx, num_blocks in enumerate(RESNET110_STAGE_BLOCKS):
        for stage_block_idx in range(num_blocks):
            block_idx += 1
            stride = 2 if stage_idx > 0 and stage_block_idx == 0 else 1
            params += _resnet_basic_block_params(in_channel, channels[stage_idx], stride)
            in_channel = channels[stage_idx]
            if block_idx >= block_limit:
                break
        if block_idx >= block_limit:
            break

    classifier_params = 0
    for exit_loc in exits[:level]:
        classifier_params += _linear_params(_resnet_exit_channels(channels, exit_loc), args.num_classes)
    return params + classifier_params


def _vit_scalefl_stats(args, scale, exits, level, cache=None):
    rounded_scale = round(float(scale), 8)
    key = None
    if cache is not None:
        key = (tuple(exits), int(level), rounded_scale)
        if key in cache:
            return cache[key]
    params = _vit_scalefl_active_params(args, rounded_scale, exits, level)
    flops = _vit_scalefl_active_flops(args, rounded_scale, exits, level)
    if cache is not None:
        cache[key] = (params, flops)
    return params, flops


def _largest_vit_scalefl_width_under_param_budget(args, exits, level, preferred_scale, target_params, cache=None):
    preferred_scale = max(0.0, min(float(preferred_scale), 1.0))
    params, flops = _vit_scalefl_stats(args, preferred_scale, exits, level, cache=cache)
    if params <= target_params:
        return preferred_scale, params, flops

    lo, hi = 0.0, preferred_scale
    best_scale = 0.0
    best_params, best_flops = _vit_scalefl_stats(args, best_scale, exits, level, cache=cache)
    for _ in range(14):
        mid = (lo + hi) / 2.0
        params, flops = _vit_scalefl_stats(args, mid, exits, level, cache=cache)
        if params <= target_params:
            best_scale, best_params, best_flops = mid, params, flops
            lo = mid
        else:
            hi = mid
    return best_scale, best_params, best_flops


def _scalefl_exit_grid(model):
    if model == "vit":
        candidates = []
        for exit0 in VIT_SEARCH_EXIT_LOCATIONS:
            for exit1 in VIT_SEARCH_EXIT_LOCATIONS:
                if exit1 <= exit0:
                    continue
                for exit2 in VIT_SEARCH_EXIT_LOCATIONS:
                    if exit2 <= exit1 or exit2 >= VIT_DEPTH:
                        continue
                    candidates.append((exit0, exit1, exit2))
        return candidates

    total = SCALEFL_DEPTH_TOTAL[model]
    min_exit = {
        "vgg": 1,
        "resnet_smart": 4,
        "vit": 1,
    }[model]
    candidates = []
    for exit0 in range(min_exit, total - 2):
        for exit1 in range(exit0 + 1, total - 1):
            for exit2 in range(exit1 + 1, total):
                candidates.append((exit0, exit1, exit2))
    return candidates


def get_scalefl_profile(args):
    if args.model in SCALEFL_FIXED_PROFILES:
        profile = SCALEFL_FIXED_PROFILES[args.model]
        return {
            "exits": profile["exits"],
            "scales": profile["scales"],
            "active_params": (None, None, None, None),
        }

    depth_base = float(SCALEFL_DEPTH_TOTAL[args.model])
    targets = target_param_budgets(args)[:-1]
    flop_targets = target_flop_budgets(args)[:-1] if args.model == "vit" else None

    best = None
    vit_stat_cache = {} if args.model == "vit" else None
    for exits in _scalefl_exit_grid(args.model):
        depth_ratios = tuple(exit_loc / depth_base for exit_loc in exits)

        if args.model == "vit":
            scales = []
            params = []
            flops = []
            for level, depth_ratio in enumerate(depth_ratios, start=1):
                scale, level_params, level_flops = _largest_vit_scalefl_width_under_param_budget(
                    args,
                    exits,
                    level,
                    depth_ratio,
                    targets[level - 1],
                    cache=vit_stat_cache,
                )
                scales.append(scale)
                params.append(level_params)
                flops.append(level_flops)
            if not (scales[0] <= scales[1] <= scales[2]):
                continue
            ratio_err = sum(abs(scale - depth_ratio) for scale, depth_ratio in zip(scales, depth_ratios))
            budget_err = sum(
                abs(param - target) / target + abs(flop - flop_target) / flop_target
                for param, target, flop, flop_target in zip(params, targets, flops, flop_targets)
            )
            score = (ratio_err, budget_err)
            if best is None or score < best[0]:
                best = (score, exits, tuple(scales), tuple(params))
            continue

        level_candidates = []
        for level in range(1, 4):
            candidates = []
            scale_grid = UNIFORM_SCALE_GRID
            for scale in scale_grid:
                if args.model == "vgg":
                    params = _vgg_scalefl_active_params(args, scale, exits, level)
                    flops = None
                else:
                    params = _resnet_scalefl_active_params(args, scale, exits, level)
                    flops = None

                budget_err = abs(params - targets[level - 1]) / targets[level - 1]
                ratio_err = abs(scale - depth_ratios[level - 1])
                candidates.append((budget_err, ratio_err, scale, params, flops))

            candidates.sort(key=lambda item: (item[0], item[1]))
            level_candidates.append(candidates[:SCALEFL_SCALE_CANDIDATES_PER_LEVEL])

        if len(level_candidates) < 3:
            continue

        for item0 in level_candidates[0]:
            for item1 in level_candidates[1]:
                if item1[2] < item0[2]:
                    continue
                for item2 in level_candidates[2]:
                    if item2[2] < item1[2]:
                        continue
                    items = (item0, item1, item2)
                    budget_err = sum(item[0] for item in items)
                    ratio_err = sum(item[1] for item in items)
                    score = budget_err + SCALEFL_RATIO_WEIGHT.get(args.model, 0.0) * ratio_err
                    scales = tuple(item[2] for item in items)
                    params = tuple(item[3] for item in items)
                    if best is None or score < best[0]:
                        best = (score, exits, scales, params)

    if best is None:
        raise RuntimeError(f"No ScaleFL profile found for model={args.model}")

    return {
        "exits": best[1],
        "scales": (*best[2], 1.0),
        "active_params": (*best[3], None),
    }


def resolve_flexfl_apoz(args):
    try:
        return FLEXFL_APOZ_PRESETS[(args.model, args.dataset)]
    except KeyError as exc:
        raise KeyError(f"No aligned FlexFL profile for {(args.model, args.dataset)}") from exc


def _normalize_vit_apoz(apoz):
    values = np.array(apoz, dtype=np.float32).flatten()
    if len(values) == VIT_PROFILE_RATE_LENGTH:
        return values
    if len(values) == VIT_DEPTH:
        return values.reshape(VIT_PROFILE_RATE_LENGTH, VIT_DEPTH // VIT_PROFILE_RATE_LENGTH).mean(axis=1)
    if len(values) < VIT_PROFILE_RATE_LENGTH:
        return np.pad(values, (0, VIT_PROFILE_RATE_LENGTH - len(values)), mode="edge")
    return values[:VIT_PROFILE_RATE_LENGTH]


def _nearest_hidden_width_rate(value):
    return min(VIT_HIDDEN_WIDTH_OPTIONS, key=lambda candidate: abs(candidate - value))


def build_vit_flexfl_hidden_scale_list(args, apoz=None):
    if apoz is None:
        apoz = resolve_flexfl_apoz(args)

    image_size = _image_size(args)
    min_rate = min(VIT_HIDDEN_WIDTH_OPTIONS)
    stage_apoz = _normalize_vit_apoz(apoz)
    full_stage_params = np.array(
        vit_stage_feature_param_vector(
            [1.0] * VIT_PROFILE_RATE_LENGTH,
            image_size,
            args.num_channels,
            width_mode="hidden",
        ),
        dtype=np.float32,
    )
    layer_weight = np.log(full_stage_params) / math.log(float(full_stage_params.max()))
    base_rate = 1.0 - stage_apoz * layer_weight
    base_rate = np.clip(base_rate, min_rate, 1.0)

    candidates = []
    for multiplier in FLEX_MULTIPLIER_GRID:
        raw_rate = np.clip(base_rate * multiplier, min_rate, 1.0)
        stage_rates = [_nearest_hidden_width_rate(float(value)) for value in raw_rate]
        params = vit_total_params(
            stage_rates,
            image_size,
            args.num_classes,
            args.num_channels,
            width_mode="hidden",
        )
        flops = vit_total_flops(
            stage_rates,
            image_size,
            args.num_classes,
            args.num_channels,
            width_mode="hidden",
        )
        candidates.append((multiplier, stage_rates, params, flops))

    scale_list = []
    previous_index = 0
    for target_params, target_flops in zip(target_param_budgets(args)[:-1], target_flop_budgets(args)[:-1]):
        best = None
        for idx, (_multiplier, stage_rates, params, flops) in enumerate(candidates):
            if idx < previous_index:
                continue
            score = _resource_score(params, flops, target_params, target_flops)
            if best is None or score < best[0]:
                best = (score, idx, stage_rates)
        previous_index = best[1]
        scale_list.append(_vit_rate(best[2], VIT_DEPTH, width_mode="hidden"))

    scale_list.append(_vit_rate([1.0] * VIT_PROFILE_RATE_LENGTH, VIT_DEPTH, width_mode="hidden"))
    return torch.tensor(base_rate, dtype=torch.float32), scale_list


def build_flexfl_scale_list(args, apoz=None):
    fixed_key = (args.model, args.dataset)
    if fixed_key in FLEXFL_FIXED_RATES:
        return None, [torch.tensor(rate, dtype=torch.float32) for rate in FLEXFL_FIXED_RATES[fixed_key]]

    if args.model == "vit":
        return build_vit_flexfl_hidden_scale_list(args, apoz)

    if apoz is None:
        apoz = resolve_flexfl_apoz(args)

    cpu_args = _cpu_args(args)
    net_glob = getNet(cpu_args, [1.0] * RATE_LENGTHS[args.model])
    layer_params = np.array([sum(param.numel() for param in layer.parameters()) for layer in net_glob.features])
    apoz_tensor = torch.tensor(np.array(apoz), dtype=torch.float32)
    layer_weight = torch.tensor(np.log(layer_params), dtype=torch.float32) / math.log(layer_params.max())
    scale_rate = 1 - apoz_tensor * layer_weight
    scale_rate = torch.clamp(scale_rate, min=0.05, max=1.0)

    candidates = []
    for multiplier in FLEX_MULTIPLIER_GRID:
        rate = torch.clamp(scale_rate * multiplier, min=0.05, max=1.0)
        candidates.append((multiplier, rate, _param_count(args, rate)))

    scale_list = []
    previous_index = 0
    for target_params in target_param_budgets(args)[:-1]:
        best = None
        for idx, (multiplier, rate, params) in enumerate(candidates):
            if idx < previous_index:
                continue
            diff = abs(params - target_params)
            if best is None or diff < best[0]:
                best = (diff, idx, rate)
        previous_index = best[1]
        scale_list.append(best[2])

    scale_list.append(torch.ones(RATE_LENGTHS[args.model], dtype=torch.float32))
    return scale_rate, scale_list
