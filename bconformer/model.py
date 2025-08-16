"""
This file was modified based on Conformer (2021)

Modifications include updated parameters and the final classifiers, mainly in the Conformer class.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm
from functools import partial
from torch.nn.init import trunc_normal_
from typing import Iterable, Optional
from timm.layers import DropPath
from timm.data import Mixup
from timm.utils import accuracy, ModelEma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm1d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv1d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv1d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv1d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool1d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [B, C, L]
        x = self.sample_pooling(x).transpose(1, 2)  # [B, L', C_embed]
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)  # [B, L'+1, C_embed]

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm1d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, target_length):
        B, L, C = x.shape # x: [batch, seq_len, embed_dim]
        # [B, L, C] -> [B, L-1, C] -> [B, C, L-1]
        x_r = x[:, 1:].transpose(1, 2)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=target_length, mode='linear', align_corners=False)


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling, adapted to 1D conv for sequences
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm1d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv1d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)  # 1D conv
        self.bn1 = norm_layer(med_planes)  # 1D BN
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv1d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)  # 1D conv
        self.bn2 = norm_layer(med_planes)  # 1D BN
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv1d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)  # 1D conv
        self.bn3 = norm_layer(inplanes)  # 1D BN
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    ConvTransformer basic module, adapted for 1D sequence data.
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4

        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, stride=stride,
                                   res_conv=res_conv, groups=groups,
                                   norm_layer=partial(nn.BatchNorm1d, eps=1e-6))

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=1,
                                         res_conv=True, groups=groups,
                                         norm_layer=partial(nn.BatchNorm1d, eps=1e-6))
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups,
                                         norm_layer=partial(nn.BatchNorm1d, eps=1e-6))

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups,
                                                    norm_layer=partial(nn.BatchNorm1d, eps=1e-6)))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride,
                                 norm_layer=partial(nn.BatchNorm1d, eps=1e-6))

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        # x shape: [B, C, L]
        x, x2 = self.cnn_block(x)  # x2 shape: [B, C_out, L_out]
        x_st = self.squeeze_block(x2, x_t)  # x_st shape: [B, L', embed_dim]

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, target_length=x.shape[-1])

        # print(x.shape, x_t_r.shape)

        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class Conformer(nn.Module):
    def __init__(self, patch_size=1, in_chans=1280, num_classes=2, base_channel=320, channel_ratio=2,
                 num_med_block=0, embed_dim=1536, depth=12, num_heads=12, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., fusion_alpha=0.5):
        super().__init__()
        assert depth % 3 == 0
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.fusion_alpha = fusion_alpha

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # classifier heads
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_token_head = nn.Linear(embed_dim, num_classes)
        self.conv_cls_head = nn.Conv1d(2560, num_classes, kernel_size=1)

        # stem
        self.conv1 = nn.Conv1d(in_chans, 320, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(320)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # stage 1
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = 1
        self.conv_1 = ConvBlock(inplanes=320, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv1d(320, embed_dim, kernel_size=5, stride=1, padding=2)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                             attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0])

        # stage 2~4
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module(f'conv_trans_{i}',
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        # stage 5~8
        stage_2_channel = stage_1_channel * 2
        for i in range(fin_stage, fin_stage + depth // 3):
            in_channel = stage_1_channel if i == fin_stage else stage_2_channel
            self.add_module(f'conv_trans_{i}',
                            ConvTransBlock(
                                in_channel, stage_2_channel, i == fin_stage, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        # stage 9~12
        stage_3_channel = stage_2_channel * 2
        for i in range(fin_stage + depth // 3, fin_stage + 2 * (depth // 3)):
            in_channel = stage_2_channel if i == fin_stage + depth // 3 else stage_3_channel
            self.add_module(f'conv_trans_{i}',
                            ConvTransBlock(
                                in_channel, stage_3_channel, i == fin_stage + depth // 3, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block,
                                last_fusion=(i == depth)
                            )
                            )

        self.fin_stage = fin_stage + 2 * (depth // 3)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x):
        B, _, L = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))  # [B, 320, L]
        x = self.conv_1(x_base, return_x_2=False)  # [B, C, L]

        x_t = self.trans_patch_conv(x_base).transpose(1, 2)  # [B, L, embed_dim]
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        for i in range(2, self.fin_stage):
            x, x_t = getattr(self, f'conv_trans_{i}')(x, x_t)

        conv_cls = self.conv_cls_head(x)  # [B, num_classes, L]
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_token_head(x_t[:, 1:, :]).transpose(1, 2)  # [B, num_classes, L]

        # fusion classfier
        final_cls = self.fusion_alpha * conv_cls + (1 - self.fusion_alpha) * tran_cls

        return final_cls
