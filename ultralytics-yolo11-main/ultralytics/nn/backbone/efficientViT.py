# --------------------------------------------------------
# EfficientViT Model Architecture for Downstream Tasks
# Copyright (c) 2022 Microsoft
# Written by: Xinyu Liu
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import itertools
import math
from timm.models.layers import SqueezeExcite
import torch.nn.init as init
import numpy as np
import itertools
from einops import rearrange

__all__ = ['EfficientViT_M0', 'EfficientViT_M1', 'EfficientViT_M2', 'EfficientViT_M3', 'EfficientViT_M4',
           'EfficientViT_M5']


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def switch_to_deploy(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        # 可学习卷积权重与偏置
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # 用于生成offset和mask
        out_channels_offset_mask = deformable_groups * 3 * kernel_size * kernel_size
        self.conv_offset_mask = nn.Conv2d(
            in_channels, out_channels_offset_mask,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

        self.bn = nn.BatchNorm2d(out_channels)
        self.reset_parameters()

    def forward(self, x, feat):
        # 生成offset和mask
        out = self.conv_offset_mask(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)

        # 调用 deform_conv2d 进行可变形卷积
        x = torch.ops.torchvision.deform_conv2d(
            x, self.weight, offset, mask, self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups, self.deformable_groups, True)

        x = self.bn(x)
        return x

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class SimpleAlign(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleAlign, self).__init__()
        self.conv_offset = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),  # 生成注意力图
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, nbr_feat, ref_feat):
        # 拼接邻近帧和参考帧
        diff_feat = torch.cat([nbr_feat, ref_feat], dim=1)
        offset_feat = self.conv_offset(diff_feat)  # 特征引导融合
        attn_map = self.attn(offset_feat)  # 注意力图，强调差异区域
        aligned_feat = nbr_feat * attn_map + ref_feat * (1 - attn_map)
        return self.fuse(aligned_feat)

class PCDAlignment(nn.Module):
    """PCD对齐模块：用于将相邻帧特征与参考帧特征对齐"""

    def __init__(self, num_feat=768, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # 用于金字塔对齐的一组卷积
        self.offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 金字塔对齐的可变形卷积模块
        # self.dcn_pack = DCNv2(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.align_module = SimpleAlign(num_feat, num_feat)
        # 融合层
        self.fusion = nn.Conv2d(num_feat, num_feat, 1, 1)

        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """
        输入：
            nbr_feat_l: 相邻帧特征列表，长度为3 (L1, L2, L3)，每个为(b, c, h, w)
            ref_feat_l: 参考帧特征列表，结构同上

        输出：
            对齐后的特征张量 (b, c, h, w)
        """
        # 此处仅使用 L1 层进行对齐，简化处理
        offset = torch.cat([nbr_feat_l[0], ref_feat_l[0]], dim=1)  # 拼接邻帧和参考帧特征
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))

        # 使用 deformable conv 对邻帧特征进行对齐
        # feat = self.dcn_pack(nbr_feat_l[0], offset)
        feat = self.align_module (nbr_feat_l[0], offset)
        feat = self.lrelu(feat)

        # 可选的融合操作
        feat = self.fusion(feat)
        return feat

# class DCNv2(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=1, dilation=1, groups=1, deformable_groups=1):
#         super(DCNv2, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = (stride, stride)
#         self.padding = (padding, padding)
#         self.dilation = (dilation, dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#
#         self.weight = nn.Parameter(
#             torch.empty(out_channels, in_channels, *self.kernel_size)
#         )
#         self.bias = nn.Parameter(torch.empty(out_channels))
#
#         out_channels_offset_mask = (self.deformable_groups * 3 *
#                                     self.kernel_size[0] * self.kernel_size[1])
#         self.conv_offset_mask = nn.Conv2d(
#             self.in_channels,
#             out_channels_offset_mask,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             bias=True,
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
#         # self.act = Conv.default_act
#         self.reset_parameters()
#
#     def forward(self, x, feat):
#         out = self.conv_offset_mask(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#         x = torch.ops.torchvision.deform_conv2d(
#             x,
#             self.weight,
#             offset,
#             mask,
#             self.bias,
#             self.stride[0], self.stride[1],
#             self.padding[0], self.padding[1],
#             self.dilation[0], self.dilation[1],
#             self.groups,
#             self.deformable_groups,
#             True
#         )
#         x = self.bn(x)
#         # x = self.act(x)
#         return x
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         std = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-std, std)
#         self.bias.data.zero_()
#         self.conv_offset_mask.weight.data.zero_()
#         self.conv_offset_mask.bias.data.zero_()
#
#
# class PCDAlignment(nn.Module):
#     """Alignment module using Pyramid, Cascading and Deformable convolution
#     (PCD). It is used in EDVR.
#
#     Ref:
#         EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
#
#     Args:
#         num_feat (int): Channel number of middle features. Default: 64.
#         deformable_groups (int): Deformable groups. Defaults: 8.
#     """
#
#     def __init__(self, num_feat=768, deformable_groups=8):
#         super(PCDAlignment, self).__init__()
#
#         # Pyramid has three levels:
#         # L3: level 3, 1/4 spatial size
#         # L2: level 2, 1/2 spatial size
#         # L1: level 1, original spatial size
#         self.offset_conv1 = nn.ModuleDict()
#         self.offset_conv2 = nn.ModuleDict()
#         self.offset_conv3 = nn.ModuleDict()
#         self.dcn_pack = nn.ModuleDict()
#         self.feat_conv = nn.ModuleDict()
#
#         # Pyramids
#         for i in range(1, 0, -1):
#             self.offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#             self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#
#             self.dcn_pack = DCNv2(
#                 num_feat,
#                 num_feat,
#                 3,
#                 padding=1,
#                 deformable_groups=deformable_groups)
#
#         # Cascading dcn
#         self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#         self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.fusion = nn.Conv2d(num_feat, num_feat, 1, 1)
#         self.cas_dcnpack = DCNv2(
#             num_feat,
#             num_feat,
#             3,
#             padding=1,
#             deformable_groups=deformable_groups)
#
#         self.upsample = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def forward(self, nbr_feat_l, ref_feat_l):
#         """Align neighboring frame features to the reference frame features.
#
#         Args:
#             nbr_feat_l (list[Tensor]): Neighboring feature list. It
#                 contains three pyramid levels (L1, L2, L3),
#                 each with shape (b, c, h, w).
#             ref_feat_l (list[Tensor]): Reference feature list. It
#                 contains three pyramid levels (L1, L2, L3),
#                 each with shape (b, c, h, w).
#
#         Returns:
#             Tensor: Aligned features.
#         """
#         # Pyramids
#         upsampled_offset, upsampled_feat = None, None
#         for i in range(1, 0, -1):
#             level = f'l{i}'
#             # print('nbr_fear_l', nbr_feat_l[0].shape)
#             offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
#             offset = self.lrelu(self.offset_conv1(offset))
#             offset = self.lrelu(self.offset_conv2(offset))
#             feat = self.dcn_pack(nbr_feat_l[i - 1], offset)
#             feat = self.lrelu(feat)
#             # print(feat.shape)
#
#         # Cascading
#         '''
#         offset = torch.cat([feat, ref_feat_l[0]], dim=1)
#         offset = self.lrelu(
#             self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
#         feat = self.lrelu(self.cas_dcnpack(feat, offset))
#         '''
#         feat = self.fusion(feat)
#         return feat


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim  # 768
        self.hidden_dim = hidden_dim  # 768

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        # self.padding = 2    # 膨胀卷积
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # print('h_cur', h_cur.shape, input_tensor.shape)
        # combined = torch.cat([input_tensor, h_cur], dim=1)
        h_cur = h_cur.to(input_tensor.device)
        # input_tensor = F.interpolate(input_tensor, size=(640, 640), mode='bilinear', align_corners=False)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # print(combined.shape)
        # combined = self.scconv(combined)  # scconv
        # print('combined', combined.shape)
        combined_conv = self.conv(combined)

        # print('combined_conv', combined_conv.shape)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 64, dim=1)
        # print('cc_i', cc_f.shape)
        # print('c_cur',c_cur.shape)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        device = next(self.parameters()).device  # 获取模型所在的设备
        f = f.to(device)
        c_cur = c_cur.to(device)
        i = i.to(device)
        g = g.to(device)

        c_next = f * c_cur + i * g
        # c_next = c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        # return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),  # train
        #         torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cpu(),  # test
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cpu())


class ConvLSTM_dcn(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM_dcn, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        # if not len(kernel_size) == len(hidden_dim) == num_layers:
        #     raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        #######################################################################
        self.pcd_align = PCDAlignment(
            num_feat=3, deformable_groups=1)
        #######################################################################

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            # print('input_dim, hidden_dim', i, input_dim, hidden_dim)
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor  # (b, t, c, h, w)
        # print('cur_layer_input', cur_layer_input.shape)
        ###################################################################################
        # print('cur_layer_input', cur_layer_input.shape)
        b, t, c, h, w = cur_layer_input.shape

        ref_feat_l = [
            cur_layer_input[:, t - 1, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t - 1):
            nbr_feat_l = []
            nbr_feat_l = [  # neighboring feature list
                cur_layer_input[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat.append(cur_layer_input[:, t - 1, :, :, :])
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)
        ####################################################################################

        for layer_idx in range(self.num_layers):
            # print('hidden_state', hidden_state)
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=aligned_feat[:, t, :, :, :],
                                                 cur_state=[h, c])
                # h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                #                                 cur_state=[h, c])

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            aligned_feat = layer_output

            layer_output = layer_output.permute(1, 0, 2, 3, 4)
            # print('layer_output', layer_output.shape)
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            # print(input_tensor.shape)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            # print('layer', layer_idx, hidden_state)
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):  # t
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output = layer_output.permute(1, 0, 2, 3, 4)
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN1(torch.nn.Module):
    """前馈网络(Feed Forward Network)模块

    参数:
        ed (int): 输入/输出维度(embedding dimension)
        h (int): 隐藏层维度(通常为ed的2-4倍)
        resolution (int): 输入分辨率(用于BN层参数初始化)
    """

    def __init__(self, ed, h, resolution):
        super().__init__()
        # 第一层点式卷积(扩展维度)
        self.pw1 = Conv2d_BN(ed, ed, resolution=resolution)
        self.pw2 = Conv2d_BN(ed, ed, resolution=resolution)
        self.pw3 = Conv2d_BN(ed, ed, bn_weight_init=0, resolution=resolution)
        # 激活函数
        self.act = torch.nn.ReLU()
        # 第二层点式卷积(恢复原始维度)，BN权重初始化为0

    def forward(self, x):
        """前向传播过程: pw1 -> ReLU -> pw2"""

        g1 = self.pw1(x)
        g2 = self.pw2(x)
        grad_update = 0.1 * (g2 - g1) - 0.1 * 0.1
        x = self.act(x + grad_update)
        x = self.pw3(x)
        return x


class FFN(torch.nn.Module):
    """前馈网络(Feed Forward Network)模块

    参数:
        ed (int): 输入/输出维度(embedding dimension)
        h (int): 隐藏层维度(通常为ed的2-4倍)
        resolution (int): 输入分辨率(用于BN层参数初始化)
    """

    def __init__(self, ed, h, resolution):
        super().__init__()
        # 第一层点式卷积(扩展维度)
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        # 激活函数
        self.act = torch.nn.ReLU()
        # 第二层点式卷积(恢复原始维度)，BN权重初始化为0
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        """前向传播过程: pw1 -> ReLU -> pw2"""
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" 级联分组注意力机制 (Cascaded Group Attention)

    参数说明:
        dim (int): 输入通道数
        key_dim (int): 查询(query)和键(key)的维度
        num_heads (int): 注意力头的数量
        attn_ratio (int): 用于计算值(value)维度的乘数
        resolution (int): 输入分辨率(对应窗口大小)
        kernels (List[int]): 应用于查询的深度可分离卷积(dw conv)的核大小列表
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5  # 注意力分数缩放因子
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)  # 值(value)的维度
        self.attn_ratio = attn_ratio

        # 初始化每个注意力头的qkv转换层和深度卷积(dw conv)层
        qkvs = []  # 存储各头的qkv转换层
        dws = []  # 存储各头的深度卷积层
        for i in range(num_heads):
            # 每个头处理dim//num_heads通道，输出key_dim*2(用于q,k) + d(用于v)
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            # 对查询(q)应用深度可分离卷积，不同头使用不同的卷积核大小
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)  # 各头的qkv转换层
        self.dws = torch.nn.ModuleList(dws)  # 各头的深度卷积层

        # 最终投影层(包含ReLU激活和卷积BN)
        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(),
            Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution)
        )

        # 生成所有可能的位置组合(用于相对位置编码)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}  # 存储不同位置偏移的索引
        idxs = []  # 存储每对位置的偏移索引

        # 计算所有位置对的相对偏移并建立索引
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))  # 计算相对偏移
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        # 可学习的注意力偏置参数(每个头有自己的一组偏置)
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))

        # 注册缓冲区存储注意力偏置索引(不参与训练)
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """ 训练/推理模式切换处理 """
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab  # 训练模式下删除缓存的注意力偏置(动态计算)
        else:
            # 推理模式下预先计算并缓存注意力偏置
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # 输入x的形状: (B,C,H,W)
        B, C, H, W = x.shape

        # 获取当前模式下的注意力偏置(训练时动态计算，推理时使用缓存)
        trainingab = self.attention_biases[:, self.attention_bias_idxs]

        # 将输入按通道数分成num_heads份
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []  # 存储各头的输出

        feat = feats_in[0]  # 初始使用第一份输入
        for i, qkv in enumerate(self.qkvs):
            # 级联结构: 从第二个头开始，将前一个头的输出加到当前输入
            if i > 0:
                feat = feat + feats_in[i]

            # 通过qkv转换层得到查询(q)、键(k)和值(v)
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)

            # 对查询应用深度可分离卷积(增强局部性)
            q = self.dws[i](q)

            # 展平空间维度: (B, C, H, W) -> (B, C, H*W)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

            # 计算注意力分数 = (Q*K^T)/sqrt(d_k) + 位置偏置
            attn = (
                    (q.transpose(-2, -1) @ k) * self.scale  # QK^T并缩放
                    +
                    (trainingab[i] if self.training else self.ab[i])  # 添加位置偏置
            )

            # 应用softmax得到注意力权重
            attn = attn.softmax(dim=-1)  # 形状: (B, N, N)

            # 用注意力权重加权值(V)并恢复空间结构
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  #
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class MSSA(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, attn_drop=0.):
        super().__init__()

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)

        self.num_heads = num_heads

        self.scale = head_dim ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(attn_drop)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # 将输入从 [B, C, H, W] 转换为 [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)  # [144, 49, 128]
        # 2. 计算QKV并重排为多头形式: [144, 128, (3*h*d)] -> [144, h, 128, d]
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.num_heads)
        # 3. 计算注意力分数
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        # 4. 计算注意力权重
        attn = self.attend(dots)
        attn = self.attn_drop(attn)
        # 5. 应用注意力权重
        out = torch.matmul(attn, w)
        # 6. 合并多头输出: [144, h, 128, d] -> [144, 128, h*d]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 7. 通过输出投影层
        out = self.to_out(out)
        # 8. 恢复空间维度: [144, 128, 49] -> [144, 128, 7, 7]
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


# class ISTA(nn.Module):
#     def __init__(self, dim, hidden_dim, attn_drop=0., step_size=0.1):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(dim, dim))
#         with torch.no_grad():
#             init.kaiming_uniform_(self.weight)
#         self.step_size = step_size
#         self.lambd = 0.1
#
#
#     def forward(self, x ):
#         B, C, H, W = x.shape
#         # 将输入从 [B, C, H, W] 转换为 [B, H*W, C]
#         x = x.flatten(2).transpose(1, 2)  # [144, 49, 128]
#         # compute D^T * D * x
#         # print('ista', x.shape)
#         x1 = F.linear(x, self.weight, bias=None)
#         grad_1 = F.linear(x1, self.weight.t(), bias=None)
#         # compute D^T * x
#         grad_2 = F.linear(x, self.weight.t(), bias=None)
#         # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
#         grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
#         output = F.relu(x + grad_update)
#         output = output.transpose(1, 2).view(B, C, H, W)
#         return output

class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=8, head_dim=64, attn_drop=0.,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        ####################################################################################################################################
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                           attn_ratio=attn_ratio,
                                           resolution=window_resolution,
                                           kernels=kernels, )
        # self.attn = MSSA(dim, num_heads=num_heads, head_dim=head_dim, attn_drop=attn_drop)

    ####################################################################################################################################
    def forward(self, x):
        B, C, H, W = x.shape

        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                                           C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.permute(0, 3, 1, 2)

        return x


######################################################################################################################################################################################################
class EfficientViTBlock1(torch.nn.Module):
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                                                       resolution=resolution, window_resolution=window_resolution,
                                                       kernels=kernels))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ista = Residual(FFN1(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ista(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


######################################################################################################################################################################################################


class EfficientViTBlock(torch.nn.Module):
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))
        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                                                       resolution=resolution, window_resolution=window_resolution,
                                                       kernels=kernels))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class Priori_Knowledge(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(Priori_Knowledge, self).__init__()
        # 处理前两个通道的卷积层
        self.fc1 = nn.Conv2d(in_channels=internal_neurons, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.conv = nn.Conv2d(in_channels=3, out_channels=internal_neurons, kernel_size=3, stride=1, bias=True)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=internal_neurons, kernel_size=3, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        # 将输入拆分为前两个通道和第三个通道
        x121 = inputs[:, 0:3, :, :]  # 第0到2通道（共3个通道）
        x31 = inputs[:, 3:6, :, :]  # 第3到5通道（共3个通道）
        ##################################################################
        x12 = self.conv(x121)
        x3 = self.conv(x31)
        ##################################################################
        # 对前两个通道进行自适应池化操作
        x1_avg = F.adaptive_avg_pool2d(x12, output_size=(1, 1))
        x1_avg = self.fc1(x1_avg)
        x1_avg = F.relu(x1_avg, inplace=True)
        x1_avg = self.fc2(x1_avg)
        x1_avg = torch.sigmoid(x1_avg)

        x1_max = F.adaptive_max_pool2d(x12, output_size=(1, 1))
        x1_max = self.fc1(x1_max)
        x1_max = F.relu(x1_max, inplace=True)
        x1_max = self.fc2(x1_max)
        x1_max = torch.sigmoid(x1_max)

        # 对第三个通道进行池化操作并扩张生成权重
        x3_avg = F.adaptive_avg_pool2d(x3, output_size=(1, 1))
        x3_avg = self.fc1(x3_avg)
        x3_avg = F.relu(x3_avg, inplace=True)
        x3_avg = self.fc2(x3_avg)
        x3_avg = torch.sigmoid(x3_avg)

        x3_max = F.adaptive_max_pool2d(x3, output_size=(1, 1))
        x3_max = self.fc1(x3_max)
        x3_max = F.relu(x3_max, inplace=True)
        x3_max = self.fc2(x3_max)
        x3_max = torch.sigmoid(x3_max)
        # 将第三个通道的权重扩展并加到前两个通道
        x12_att = x1_avg + x1_max  # 前两个通道的注意力权重
        x3_att = x3_avg + x3_max  # 第三个通道的注意力权重
        # 将第三个通道的注意力加到前两个通道的结果上
        x_att = x12_att + x3_att
        # 将注意力作用到前通道
        output = x121 * x_att  ######################################################
        return output


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=400,
                 patch_size=16,
                 frozen_stages=0,
                 in_chans=3,
                 out_chans1=3,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 pretrained=None,
                 distillation=False, ):
        super().__init__()
        self.ca = Priori_Knowledge(input_channels=in_chans, internal_neurons=out_chans1)

        self.convlstm = ConvLSTM(input_size=(224, 224),
                                 input_dim=3,
                                 hidden_dim=3,
                                 kernel_size=(3, 3),
                                 num_layers=1,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)

        self.convlstm_dcn1 = ConvLSTM_dcn(input_size=(224, 224),
                                          input_dim=3,
                                          hidden_dim=3,
                                          kernel_size=(3, 3),
                                          num_layers=1,
                                          batch_first=False,
                                          bias=True,
                                          return_all_layers=False)
        resolution = img_size
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution),
                                               torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1,
                                                         resolution=resolution // 2), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1,
                                                         resolution=resolution // 4), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 1, 1,
                                                         resolution=resolution // 8))

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                ##########################################################################################################################
                eval('self.blocks' + str(i + 1)).append(
                    EfficientViTBlock1(stg, ed, kd, nh, ar, resolution, wd, kernels))
                ##########################################################################################################################
            if do[0] == 'subsample':
                # ('Subsample' stride)
                blk = eval('self.blocks' + str(i + 2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                               Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)), ))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1],
                              resolution=resolution)),
                                               Residual(
                                                   FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)), ))

        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        ############################convlstm########################################
        # x1 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # data = [] #初始化一个空列表用于存储处理后的张量。
        # for item in x1:
        #     item = item.unsqueeze(1)
        #     data.append(item)
        # data = torch.stack(data, dim=0) #  ([1, 3, 1, 640, 640])
        # hidden = data.permute(0, 2, 1, 3, 4)
        # # print('hidden.shape', hidden.shape)
        # lstm, _ = self.convlstm(hidden)
        # x2 = lstm[0][-1, :, :, :, :]
        #  ############################convlstm_dcn1########################################
        x1 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # x_shift2 = torch.roll(x, shifts=2, dims=2)  # 向下平移2像素（循环填充）
        x_shift4 = torch.roll(x1, shifts=4, dims=2)  # 向下平移4像素（循环填充）
        data = torch.stack([x_shift4, x1], dim=0)  # [3, 1, 3, 512, 512]
        lstm, _ = self.convlstm_dcn1(data)
        x2 = lstm[0][-1, :, :, :, :]
        # ####################################################################
        x2 = torch.cat([x1, x2], dim=1)
        x3 = self.ca(x2)
        x3 = F.interpolate(x3, size=(640, 640), mode='bilinear', align_corners=False)
        x = x + x3
        # ####################################################################
        outs = []
        x = self.patch_embed(x)
        x = self.blocks1(x)
        outs.append(x)
        x = self.blocks2(x)
        outs.append(x)
        x = self.blocks3(x)
        outs.append(x)

        return outs


EfficientViT_m0 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [64, 128, 192],
    'depth': [1, 2, 3],
    'num_heads': [4, 4, 4],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m1 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 144, 192],
    'depth': [1, 2, 3],
    'num_heads': [2, 3, 3],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m2 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 192, 224],
    'depth': [1, 2, 3],
    'num_heads': [4, 3, 2],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m3 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 240, 320],
    'depth': [1, 2, 3],
    'num_heads': [4, 3, 4],
    'window_size': [7, 7, 7],
    'kernels': [5, 5, 5, 5],
}

EfficientViT_m4 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 256, 384],
    'depth': [1, 2, 3],
    'num_heads': [4, 4, 4],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m5 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [192, 288, 384],
    'depth': [1, 3, 4],
    'num_heads': [3, 3, 4],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}


def EfficientViT_M0(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None,
                    model_cfg=EfficientViT_m0):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if pretrained:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(pretrained)['model']))
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M1(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None,
                    model_cfg=EfficientViT_m1):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if pretrained:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(pretrained)['model']))
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M2(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None,
                    model_cfg=EfficientViT_m2):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if pretrained:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(pretrained)['model']))
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M3(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None,
                    model_cfg=EfficientViT_m3):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if pretrained:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(pretrained)['model']))
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M4(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None,
                    model_cfg=EfficientViT_m4):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if pretrained:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(pretrained)['model']))
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M5(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None,
                    model_cfg=EfficientViT_m5):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if pretrained:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(pretrained)['model']))
    if fuse:
        replace_batchnorm(model)
    return model


def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        # k = k[9:]
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict


if __name__ == '__main__':
    model = EfficientViT_M0('efficientvit_m0.pth')
    inputs = torch.randn((1, 3, 640, 640))
    res = model(inputs)
    for i in res:
        print(i.size())