import torch
import torch.nn as nn


VIT_DEPTH = 12
VIT_BASE_EMBED_DIM = 384
VIT_BASE_NUM_HEADS = 6
VIT_HEAD_DIM = 64
VIT_MIN_EMBED_DIM = VIT_BASE_NUM_HEADS
VIT_BASE_MLP_RATIO = 4.0
VIT_BASE_MLP_DIM = int(VIT_BASE_EMBED_DIM * VIT_BASE_MLP_RATIO)
VIT_NUM_STAGES = 4
VIT_STAGE_DEPTHS = (3, 3, 3, 3)
VIT_DEFAULT_EXITS = (8, 10, 11)
VIT_SEARCH_EXIT_LOCATIONS = tuple(range(1, VIT_DEPTH + 1))
VIT_WIDTH_OPTIONS = (
    0.125,
    0.25,
    1.0 / 3.0,
    5.0 / 12.0,
    0.5,
    2.0 / 3.0,
    0.75,
    7.0 / 8.0,
    11.0 / 12.0,
    1.0,
)
VIT_HIDDEN_DIM_OPTIONS = tuple(range(VIT_MIN_EMBED_DIM, VIT_BASE_EMBED_DIM + 1, VIT_BASE_NUM_HEADS))
VIT_HIDDEN_WIDTH_OPTIONS = tuple(dim / VIT_BASE_EMBED_DIM for dim in VIT_HIDDEN_DIM_OPTIONS)


def _normalize_rate(rate):
    if rate is None:
        return [1.0] * VIT_NUM_STAGES
    if isinstance(rate, torch.Tensor):
        rate = rate.tolist()
    if len(rate) == 0:
        return [1.0] * VIT_NUM_STAGES
    if len(rate) == 1:
        return [float(rate[0])] * VIT_NUM_STAGES
    if len(rate) == VIT_NUM_STAGES:
        return [float(v) for v in rate]
    if len(rate) == 50:
        return [1.0] * VIT_NUM_STAGES
    if len(rate) == VIT_DEPTH:
        stage_rates = []
        offset = 0
        for depth in VIT_STAGE_DEPTHS:
            stage_rates.append(float(rate[offset]))
            offset += depth
        return stage_rates
    if len(rate) in (5, 6):
        return [float(v) for v in rate[:VIT_NUM_STAGES]]
    raise ValueError(f"Unsupported ViT rate length: {len(rate)}")


def _extract_exit_loc(rate, default=VIT_DEPTH):
    if rate is None:
        return default
    if isinstance(rate, torch.Tensor):
        rate = rate.tolist()
    if len(rate) in (5, 6):
        return int(round(float(rate[4])))
    return default


def _normalize_exit_locations(exit_locations, include_final=False):
    if exit_locations is None:
        normalized = list(VIT_DEFAULT_EXITS)
    else:
        normalized = sorted(set(int(v) for v in exit_locations))
    upper = VIT_DEPTH if include_final else VIT_DEPTH - 1
    for loc in normalized:
        if loc < 1 or loc > upper:
            raise ValueError(f"ViT exit location must be in [1, {upper}], got {loc}")
    return normalized


def _stage_index(block_idx):
    return min((block_idx - 1) // VIT_STAGE_DEPTHS[0], VIT_NUM_STAGES - 1)


def _canonical_stage_rates(stage_rates):
    if stage_rates is None:
        return [1.0] * VIT_NUM_STAGES
    rates = [float(v) for v in stage_rates[:VIT_NUM_STAGES]]
    if not rates:
        rates = [1.0] * VIT_NUM_STAGES
    if len(rates) < VIT_NUM_STAGES:
        rates.extend([rates[-1]] * (VIT_NUM_STAGES - len(rates)))
    return rates


def _resize_token_dim(tokens, dim):
    current_dim = tokens.size(-1)
    if current_dim == dim:
        return tokens
    if current_dim > dim:
        return tokens[..., :dim]
    pad = tokens.new_zeros(*tokens.shape[:-1], dim - current_dim)
    return torch.cat([tokens, pad], dim=-1)


def _scaled_embed_dim(scale, width_mode="mlp"):
    if width_mode == "mlp":
        _ = scale
        return VIT_BASE_EMBED_DIM
    if width_mode != "hidden":
        raise ValueError(f"Unsupported ViT width_mode: {width_mode}")
    scale = max(min(float(scale), 1.0), VIT_MIN_EMBED_DIM / VIT_BASE_EMBED_DIM)
    raw_dim = int(round(VIT_BASE_EMBED_DIM * scale))
    raw_dim = max(VIT_MIN_EMBED_DIM, min(VIT_BASE_EMBED_DIM, raw_dim))
    return min(VIT_HIDDEN_DIM_OPTIONS, key=lambda dim: abs(dim - raw_dim))


def _scaled_mlp_dim(scale, embed_dim=None, width_mode="mlp"):
    if width_mode == "hidden":
        embed_dim = VIT_BASE_EMBED_DIM if embed_dim is None else embed_dim
        return int(embed_dim * VIT_BASE_MLP_RATIO)
    if width_mode != "mlp":
        raise ValueError(f"Unsupported ViT width_mode: {width_mode}")
    scale = max(min(float(scale), 1.0), 1.0 / VIT_BASE_MLP_DIM)
    hidden_dim = int(round(VIT_BASE_MLP_DIM * scale))
    return max(1, min(VIT_BASE_MLP_DIM, hidden_dim))


def _scaled_num_heads(embed_dim, width_mode="mlp"):
    if width_mode == "mlp" and embed_dim != VIT_BASE_EMBED_DIM:
        raise ValueError("MLP-only ViT keeps the attention embed dimension fixed at 384")
    candidates = [heads for heads in (6, 4, 3, 2, 1) if embed_dim % heads == 0]
    if not candidates:
        raise ValueError(f"ViT hidden dim {embed_dim} is not divisible by any supported head count")
    return min(candidates, key=lambda heads: abs((embed_dim / heads) - VIT_HEAD_DIM))


def _canonical_width_mode(width_mode):
    if width_mode in (None, "mlp"):
        return "mlp"
    if width_mode == "hidden":
        return "hidden"
    raise ValueError(f"Unsupported ViT width_mode: {width_mode}")


def _hidden_space_enabled(args):
    return getattr(args, "algorithm", None) in {"HeteroFL", "FlexFL"}


def vit_width_mode_from_args(args):
    return "hidden" if _hidden_space_enabled(args) else "mlp"


def vit_rate_uses_hidden_width(rate):
    if isinstance(rate, torch.Tensor):
        rate = rate.tolist()
    if rate is None:
        return False
    return len(rate) == 6 and int(round(float(rate[5]))) == 1


def _extract_width_mode(rate, default="mlp"):
    return "hidden" if vit_rate_uses_hidden_width(rate) else default


def _rate_payload(rate):
    if isinstance(rate, torch.Tensor):
        rate = rate.tolist()
    return rate


def _trim_rate_payload(rate):
    rate = _rate_payload(rate)
    if rate is None:
        return None
    if len(rate) >= 6:
        return rate[:5]
    return rate


def _scaled_embed_dim_compat(scale):
    return VIT_BASE_EMBED_DIM


def vit_num_patches(image_size):
    return (image_size // 16) ** 2


def vit_stage_dims(stage_rates, width_mode="mlp"):
    stage_rates = _canonical_stage_rates(stage_rates)
    width_mode = _canonical_width_mode(width_mode)
    return [_scaled_embed_dim(scale, width_mode=width_mode) for scale in stage_rates]


def vit_stage_mlp_dims(stage_rates, width_mode="mlp"):
    stage_rates = _canonical_stage_rates(stage_rates)
    width_mode = _canonical_width_mode(width_mode)
    stage_dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    return [
        _scaled_mlp_dim(scale, embed_dim=embed_dim, width_mode=width_mode)
        for scale, embed_dim in zip(stage_rates, stage_dims)
    ]


def vit_exit_head_params(embed_dim, num_classes):
    return (2 * embed_dim) + (embed_dim * num_classes + num_classes)


def vit_patch_embed_params(embed_dim, num_channels=3):
    return embed_dim * num_channels * 16 * 16 + embed_dim


def vit_block_params(in_dim=None, out_dim=None, mlp_scale=1.0, width_mode="mlp"):
    width_mode = _canonical_width_mode(width_mode)
    embed_dim = _scaled_embed_dim(mlp_scale, width_mode=width_mode) if out_dim is None else int(out_dim)
    mlp_dim = _scaled_mlp_dim(mlp_scale, embed_dim=embed_dim, width_mode=width_mode)
    norm1 = 2 * embed_dim
    qkv = embed_dim * (3 * embed_dim) + 3 * embed_dim
    proj = embed_dim * embed_dim + embed_dim
    norm2 = 2 * embed_dim
    mlp = embed_dim * mlp_dim + mlp_dim + mlp_dim * embed_dim + embed_dim
    return norm1 + qkv + proj + norm2 + mlp


def vit_block_flops(in_dim=None, out_dim=None, num_tokens=0, mlp_scale=1.0, width_mode="mlp"):
    width_mode = _canonical_width_mode(width_mode)
    embed_dim = _scaled_embed_dim(mlp_scale, width_mode=width_mode) if out_dim is None else int(out_dim)
    mlp_dim = _scaled_mlp_dim(mlp_scale, embed_dim=embed_dim, width_mode=width_mode)
    resize = 0
    if in_dim is not None and int(in_dim) != embed_dim:
        resize = num_tokens * max(int(in_dim), embed_dim)
    qkv = 3 * num_tokens * embed_dim * embed_dim
    attn = 2 * num_tokens * num_tokens * embed_dim
    proj = num_tokens * embed_dim * embed_dim
    mlp = 2 * num_tokens * embed_dim * mlp_dim
    return resize + qkv + attn + proj + mlp


def vit_stage_feature_param_vector(stage_rates, image_size, num_channels=3, width_mode="mlp"):
    stage_rates = _canonical_stage_rates(stage_rates)
    width_mode = _canonical_width_mode(width_mode)
    dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    num_patches = vit_num_patches(image_size)
    params = []

    stage0 = vit_patch_embed_params(dims[0], num_channels)
    stage0 += dims[0]
    stage0 += (num_patches + 1) * dims[0]
    prev_dim = dims[0]
    stage0_blocks = 0
    for _ in range(VIT_STAGE_DEPTHS[0]):
        stage0_blocks += vit_block_params(in_dim=prev_dim, out_dim=dims[0], mlp_scale=stage_rates[0], width_mode=width_mode)
        prev_dim = dims[0]
    stage0 += stage0_blocks
    params.append(stage0)

    for idx in range(1, VIT_NUM_STAGES):
        stage_params = 0
        for _ in range(VIT_STAGE_DEPTHS[idx]):
            stage_params += vit_block_params(in_dim=prev_dim, out_dim=dims[idx], mlp_scale=stage_rates[idx], width_mode=width_mode)
            prev_dim = dims[idx]
        params.append(stage_params)

    return params


def vit_stage_feature_flop_vector(stage_rates, image_size, num_channels=3, width_mode="mlp"):
    stage_rates = _canonical_stage_rates(stage_rates)
    width_mode = _canonical_width_mode(width_mode)
    dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    num_patches = vit_num_patches(image_size)
    num_tokens = num_patches + 1
    flops = []

    stage0 = num_patches * (16 * 16 * num_channels) * dims[0]
    prev_dim = dims[0]
    stage0_blocks = 0
    for _ in range(VIT_STAGE_DEPTHS[0]):
        stage0_blocks += vit_block_flops(in_dim=prev_dim, out_dim=dims[0], num_tokens=num_tokens, mlp_scale=stage_rates[0], width_mode=width_mode)
        prev_dim = dims[0]
    stage0 += stage0_blocks
    flops.append(stage0)

    for idx in range(1, VIT_NUM_STAGES):
        stage_flops = 0
        for _ in range(VIT_STAGE_DEPTHS[idx]):
            stage_flops += vit_block_flops(in_dim=prev_dim, out_dim=dims[idx], num_tokens=num_tokens, mlp_scale=stage_rates[idx], width_mode=width_mode)
            prev_dim = dims[idx]
        flops.append(stage_flops)

    return flops


def vit_total_params(stage_rates, image_size, num_classes, num_channels=3, width_mode="mlp"):
    width_mode = _canonical_width_mode(width_mode)
    dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    feature_params = sum(vit_stage_feature_param_vector(stage_rates, image_size, num_channels, width_mode=width_mode))
    final_head = vit_exit_head_params(dims[-1], num_classes)
    return feature_params + final_head


def vit_total_flops(stage_rates, image_size, num_classes, num_channels=3, width_mode="mlp"):
    width_mode = _canonical_width_mode(width_mode)
    dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    feature_flops = sum(vit_stage_feature_flop_vector(stage_rates, image_size, num_channels, width_mode=width_mode))
    final_head = dims[-1] * num_classes
    return feature_flops + final_head


def _level_stage_count(exits, level):
    stage_prefix_map = []
    for exit_loc in exits:
        if exit_loc <= 3:
            stage_prefix_map.append(1)
        elif exit_loc <= 6:
            stage_prefix_map.append(2)
        elif exit_loc <= 9:
            stage_prefix_map.append(3)
        else:
            stage_prefix_map.append(4)
    stage_prefix_map.append(4)
    return stage_prefix_map[level - 1]


def _vit_block_param_sequence(stage_rates, image_size, num_channels=3, width_mode="mlp"):
    stage_rates = _canonical_stage_rates(stage_rates)
    width_mode = _canonical_width_mode(width_mode)
    dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    num_patches = vit_num_patches(image_size)
    prefix = vit_patch_embed_params(dims[0], num_channels) + dims[0] + (num_patches + 1) * dims[0]
    blocks = []
    prev_dim = dims[0]
    for block_idx in range(1, VIT_DEPTH + 1):
        stage_idx = _stage_index(block_idx)
        blocks.append(
            vit_block_params(
                in_dim=prev_dim,
                out_dim=dims[stage_idx],
                mlp_scale=stage_rates[stage_idx],
                width_mode=width_mode,
            )
        )
        prev_dim = dims[stage_idx]
    return prefix, blocks


def _vit_block_flop_sequence(stage_rates, image_size, num_channels=3, width_mode="mlp"):
    stage_rates = _canonical_stage_rates(stage_rates)
    width_mode = _canonical_width_mode(width_mode)
    dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    num_patches = vit_num_patches(image_size)
    num_tokens = num_patches + 1
    prefix = num_patches * (16 * 16 * num_channels) * dims[0]
    blocks = []
    prev_dim = dims[0]
    for block_idx in range(1, VIT_DEPTH + 1):
        stage_idx = _stage_index(block_idx)
        blocks.append(
            vit_block_flops(
                in_dim=prev_dim,
                out_dim=dims[stage_idx],
                num_tokens=num_tokens,
                mlp_scale=stage_rates[stage_idx],
                width_mode=width_mode,
            )
        )
        prev_dim = dims[stage_idx]
    return prefix, blocks


def vit_active_params(stage_scale, exits, level, image_size, num_classes, num_channels=3, width_mode="mlp"):
    if isinstance(stage_scale, (list, tuple)):
        stage_rates = [float(v) for v in stage_scale]
    else:
        stage_rates = [float(stage_scale)] * VIT_NUM_STAGES
    width_mode = _canonical_width_mode(width_mode)
    dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    prefix, block_params = _vit_block_param_sequence(stage_rates, image_size, num_channels, width_mode=width_mode)
    active = prefix
    if level < 4:
        exit_loc = exits[level - 1]
        active += sum(block_params[:exit_loc])
        active += vit_exit_head_params(dims[_stage_index(exit_loc)], num_classes)
        return active
    active += sum(block_params)
    active += vit_exit_head_params(dims[-1], num_classes)
    return active


def vit_active_flops(stage_scale, exits, level, image_size, num_classes, num_channels=3, width_mode="mlp"):
    if isinstance(stage_scale, (list, tuple)):
        stage_rates = [float(v) for v in stage_scale]
    else:
        stage_rates = [float(stage_scale)] * VIT_NUM_STAGES
    width_mode = _canonical_width_mode(width_mode)
    dims = vit_stage_dims(stage_rates, width_mode=width_mode)
    prefix, block_flops = _vit_block_flop_sequence(stage_rates, image_size, num_channels, width_mode=width_mode)
    active = prefix
    if level < 4:
        exit_loc = exits[level - 1]
        active += sum(block_flops[:exit_loc])
        active += dims[_stage_index(exit_loc)] * num_classes
        return active
    active += sum(block_flops)
    active += dims[-1] * num_classes
    return active


class ViTExitHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens):
        return self.fc(self.norm(tokens[:, 0]))


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=16, stride=16)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_scale=1.0, width_mode="mlp"):
        super().__init__()
        hidden_dim = _scaled_mlp_dim(mlp_scale, embed_dim=embed_dim, width_mode=width_mode)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_scale=1.0, width_mode="mlp"):
        super().__init__()
        self.proj_in = None
        self.embed_dim = embed_dim
        self.width_mode = _canonical_width_mode(width_mode)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            _scaled_num_heads(embed_dim, self.width_mode),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_scale, self.width_mode)

    def forward(self, x):
        x = _resize_token_dim(x, self.embed_dim)
        if self.proj_in is not None:
            x = self.proj_in(x)
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTFlexFL(nn.Module):
    def __init__(self, num_classes, num_channels=3, image_size=32, rate=None, exit_loc=VIT_DEPTH, width_mode="mlp"):
        super().__init__()
        if num_channels != 3:
            raise ValueError("ViT backbone only supports RGB inputs.")
        self.width_mode = _extract_width_mode(rate, width_mode)
        self.stage_rates = _normalize_rate(rate)
        self.stage_rates = _canonical_stage_rates(self.stage_rates)
        self.stage_dims = vit_stage_dims(self.stage_rates, width_mode=self.width_mode)
        self.input_dim = self.stage_dims[0]
        self.final_dim = self.stage_dims[-1]
        self.image_size = image_size
        self.exit_loc = int(exit_loc)
        if self.exit_loc < 1 or self.exit_loc > VIT_DEPTH:
            raise ValueError(f"ViT exit_loc must be in [1, {VIT_DEPTH}], got {self.exit_loc}")

        self.conv_proj = PatchEmbed(num_channels, self.input_dim)
        num_tokens = vit_num_patches(image_size) + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.input_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, self.input_dim))
        self.dropout = nn.Dropout(0.0)

        blocks = []
        for block_idx in range(1, VIT_DEPTH + 1):
            stage_idx = _stage_index(block_idx)
            blocks.append(TransformerBlock(self.stage_dims[stage_idx], self.stage_rates[stage_idx], self.width_mode))
        self.blocks = nn.ModuleList(blocks)

        self.exit_heads = nn.ModuleList([
            ViTExitHead(self.stage_dims[_stage_index(block_idx)], num_classes)
            for block_idx in range(1, VIT_DEPTH)
        ])
        self.final_norm = nn.LayerNorm(self.final_dim)
        self.classifier = nn.Linear(self.final_dim, num_classes)

    @property
    def features(self):
        return self.blocks

    def _process_input(self, x):
        n, c, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"Expected input size {self.image_size}, got {(h, w)}")
        x = self.conv_proj(x)
        cls = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embedding)
        return x

    def forward_tokens(self, x, block_count=VIT_DEPTH):
        x = self._process_input(x)
        for block in self.blocks[:block_count]:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_tokens(x, self.exit_loc)
        if self.exit_loc < VIT_DEPTH:
            logits = self.exit_heads[self.exit_loc - 1](x)
        else:
            x = _resize_token_dim(x, self.final_dim)
            x = self.final_norm(x)
            logits = self.classifier(x[:, 0])
        return {"representation": x, "output": logits}


class ViTScaleFL(nn.Module):
    def __init__(self, num_classes, num_channels=3, image_size=32, scale=1.0, exits=None, width_mode="mlp"):
        super().__init__()
        if num_channels != 3:
            raise ValueError("ViT backbone only supports RGB inputs.")
        self.width_mode = _canonical_width_mode(width_mode)
        self.exit_locations = _normalize_exit_locations(exits)
        self.stage_rates = _normalize_rate([scale] * VIT_NUM_STAGES)
        self.stage_rates = _canonical_stage_rates(self.stage_rates)
        self.stage_dims = vit_stage_dims(self.stage_rates, width_mode=self.width_mode)
        self.input_dim = self.stage_dims[0]
        self.final_dim = self.stage_dims[-1]
        self.image_size = image_size

        self.conv_proj = PatchEmbed(num_channels, self.input_dim)
        num_tokens = vit_num_patches(image_size) + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.input_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, self.input_dim))
        self.dropout = nn.Dropout(0.0)

        blocks = []
        for block_idx in range(1, VIT_DEPTH + 1):
            stage_idx = _stage_index(block_idx)
            blocks.append(TransformerBlock(self.stage_dims[stage_idx], self.stage_rates[stage_idx], self.width_mode))
        self.blocks = nn.ModuleList(blocks)

        self.classifiers = nn.ModuleList()
        for exit_loc in self.exit_locations:
            stage_idx = _stage_index(exit_loc)
            self.classifiers.append(ViTExitHead(self.stage_dims[stage_idx], num_classes))
        self.final_norm = nn.LayerNorm(self.final_dim)
        self.final_head = nn.Linear(self.final_dim, num_classes)

    @property
    def features(self):
        return self.blocks

    def _process_input(self, x):
        n, c, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"Expected input size {self.image_size}, got {(h, w)}")
        x = self.conv_proj(x)
        cls = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embedding)
        return x

    def _forward_final(self, tokens):
        tokens = _resize_token_dim(tokens, self.final_dim)
        return self.final_head(self.final_norm(tokens)[:, 0])

    def forward(self, x, ee=1):
        tokens = self._process_input(x)
        outputs = []
        classifier_idx = 0
        for block_idx, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if classifier_idx < len(self.exit_locations) and block_idx == self.exit_locations[classifier_idx]:
                outputs.append({"output": self.classifiers[classifier_idx](tokens)})
                classifier_idx += 1
                if len(outputs) == ee and ee <= len(self.exit_locations):
                    return outputs
        outputs.append({"output": self._forward_final(tokens)})
        return outputs[:ee]


def vit_small_flexfl(num_classes, track_running_stats=True, num_channels=3, rate=None, image_size=32, exit_loc=None, width_mode=None):
    _ = track_running_stats
    resolved_exit_loc = _extract_exit_loc(rate) if exit_loc is None else int(exit_loc)
    resolved_width_mode = _extract_width_mode(rate, "mlp" if width_mode is None else width_mode)
    return ViTFlexFL(
        num_classes=num_classes,
        num_channels=num_channels,
        image_size=image_size,
        rate=rate,
        exit_loc=resolved_exit_loc,
        width_mode=resolved_width_mode,
    )


def vit_small_scalefl(num_classes, num_channels=3, image_size=32, scale=1.0, exits=None, width_mode="mlp"):
    return ViTScaleFL(
        num_classes=num_classes,
        num_channels=num_channels,
        image_size=image_size,
        scale=scale,
        exits=exits,
        width_mode=width_mode,
    )
