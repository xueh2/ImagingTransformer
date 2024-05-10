
"""
Backbone model - HRNet architecture

This file implements a HRNet design for the imaging backbone.
The input to the model is [B, T, C_in, H, W]. The output of the model is [B, T, N*C, H, W].
N is the number of resolution levels. C is the number of channels at the original resolution.
For every resolution level, the image size will be reduced by x2, with the number of channels increasing by x2.

Besides the aggregated output tensor, this backbone model also outputs the per-resolution-level feature maps as a list.

Please ref to the project page for the network design.
"""

import os
import sys
import argparse
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
import torchvision

from pathlib import Path
from argparse import Namespace

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Current_DIR))

Model_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Model_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from imaging_attention import *
from cells import *
from blocks import *
from setup.setup_utils import get_device
from utils.status import model_info

from backbone_base import STCNNT_Base_Runtime, set_window_patch_sizes_keep_num_window, set_window_patch_sizes_keep_window_size, DownSample, UpSample

__all__ = ['STCNNT_HRnet_model', 'STCNNT_HRnet', 'DownSample', 'UpSample']

# -------------------------------------------------------------------------------------------------
def STCNNT_HRnet_model(config, pre_feature_channels):
    """
    Simple function to return STCCNT HRnet model.
    Additionally, function computes feature_channels, a list of ints containing the number of channels in each feature returned by the model.
    """
    C_in = config.no_in_channel
    config.no_in_channel = pre_feature_channels[-1]
    model = STCNNT_HRnet(config=config)
    config.no_in_channel = C_in
    feature_channels = [int(config.backbone_hrnet.C * sum([np.power(2, k) for k in range(config.backbone_hrnet.num_resolution_levels)]))]
    return model, feature_channels

#-------------------------------------------------------------------------------------
class STCNNT_HRnet(STCNNT_Base_Runtime):
    """
    This class implemented the stcnnt version of HRnet with maximal 5 levels.
    """

    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup

        @args (from config):
            ---------------------------------------------------------------
            model specific arguments
            ---------------------------------------------------------------

            - C (int): number of channels, when resolution is reduced by x2, number of channels will increase by x2
            - num_resolution_levels (int): number of resolution levels; each deeper level will reduce spatial size by x2

            - block_str (str | list of strings): order of attention types and mixer
                format is list of XYXYXYXY...
                - X is "L", "G" or "T" or "V" for attention type
                - Y is "0" or "1" for with or without mixer
                - requires len(att_types[i]) to be even
                
                This string is the "Block string" to define the attention layers in a block.
                If a list of string is given, each string defines the attention structure for a resolution level.

            - use_interpolation (bool): whether to use interpolation in downsample layer; if False, use stride convolution

            ---------------------------------------------------------------
            Shared arguments used in this model
            ---------------------------------------------------------------
            - C_in (int): number of input channels

            - height (int list): expected heights of the input
            - width (int list): expected widths of the input

            - a_type ("conv", "lin"): type of attention in spatial heads
            - cell_type ("sequential", "parallel"): type of attention cell
            - window_size (int): size of window for local and global att
            - patch_size (int): size of patch for local and global att
            - window_sizing_method (str): "mixed", "keep_window_size", "keep_num_window"
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - stride_t (int): special stride for temporal attention k,q matrices
            - normalize_Q_K (bool): whether to use layernorm to normalize Q and K, as in 22B ViT paper
            - att_dropout_p (float): probability of dropout for attention coefficients
            - dropout (float): probability of dropout
            - att_with_output_proj (bool): whether to add output projection in the attention layer
            - scale_ratio_in_mixer (float): channel scaling ratio in the mixer
            - norm_mode ("layer", "batch", "instance"):
                layer - norm along C, H, W; batch - norm along B*T; or instance norm along H, W for C
            - interp_align_c (bool):
                whether to align corner or not when interpolating
            - residual (bool):
                whether to add long skip residual connection or not

            - optim ("adamw", "sgd", "nadam"): choices for optimizer
            - scheduler ("ReduceOnPlateau", "StepLR", "OneCycleLR"):
                choices for learning rate schedulers
            - global_lr (float): global learning rate
            - beta1, beta2 (float): parameters for adam
            - weight_decay (float): parameter for regularization
            - all_w_decay (bool): whether to separate model params for regularization

            - losses (list of "ssim", "ssim3D", "l1", "mse"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether we are dealing with complex images or not

            - load_path (str): path to load the weights from
        """
        super().__init__(config)

        self.check_class_specific_parameters(config)

        C = config.backbone_hrnet.C
        num_resolution_levels = config.backbone_hrnet.num_resolution_levels
        block_str = config.backbone_hrnet.block_str
        use_interpolation = config.backbone_hrnet.use_interpolation

        # assert C >= config.no_in_channel, "Number of channels should be larger than C_in"
        assert num_resolution_levels <= 5 and num_resolution_levels>=2, "Maximal number of resolution levels is 5"

        self.C = C
        self.num_resolution_levels = num_resolution_levels

        if isinstance(block_str, list):
            self.block_str = block_str if len(block_str)>=self.num_resolution_levels else [block_str[0] for n in range(self.num_resolution_levels)] # with bridge
        else:
            self.block_str = [block_str for n in range(self.num_resolution_levels)]

        self.use_interpolation = use_interpolation

        c = copy.copy(config)

        # compute number of windows and patches
        self.num_wind = [c.height//c.window_size[0], c.width//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]

        kwargs = {
            "C_in":c.no_in_channel,
            "C_out":C,
            "H":c.height,
            "W":c.width,
            "a_type":c.a_type,
            "window_size": c.window_size,
            "patch_size": c.patch_size,
            "is_causal":c.is_causal,
            "dropout_p":c.dropout_p,
            "n_head":c.n_head,

            "kernel_size":(c.kernel_size, c.kernel_size),
            "stride":(c.stride, c.stride),
            "padding":(c.padding, c.padding),

            "stride_s": (c.stride_s, c.stride_s),
            "stride_t":(c.stride_t, c.stride_t),

            "separable_conv": c.separable_conv,

            "mixer_kernel_size":(c.mixer_kernel_size, c.mixer_kernel_size),
            "mixer_stride":(c.mixer_stride, c.mixer_stride),
            "mixer_padding":(c.mixer_padding, c.mixer_padding),

            "norm_mode":c.norm_mode,
            "interpolate":"none",
            "interp_align_c":c.interp_align_c,

            "cell_type": c.cell_type,
            "normalize_Q_K": c.normalize_Q_K, 
            "att_dropout_p": c.att_dropout_p,
            "att_with_output_proj": c.att_with_output_proj, 
            "scale_ratio_in_mixer": c.scale_ratio_in_mixer,
            "cosine_att": c.cosine_att,
            "att_with_relative_postion_bias": c.att_with_relative_postion_bias,
            "block_dense_connection": c.block_dense_connection,

            "num_wind": self.num_wind,
            "num_patch": self.num_patch,

            "mixer_type": c.mixer_type,
            "shuffle_in_window": c.shuffle_in_window,

            "use_einsum": c.use_einsum,
            "temporal_flash_attention": c.temporal_flash_attention, 

            "activation_func": c.activation_func
        }

        window_sizes = []
        patch_sizes = []

        if num_resolution_levels >= 1:
            # define B00
            kwargs["C_in"] = c.no_in_channel
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height
            kwargs["W"] = c.width

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="B00")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="B00")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="B00")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[0]
            self.B00 = STCNNT_Block(**kwargs)
            print(f"B00 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # output stage 0
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height
            kwargs["W"] = c.width

            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[-1], patch_sizes[-1], module_name="output_0")

            kwargs["att_types"] = self.block_str[0]
            self.output_B0 = STCNNT_Block(**kwargs)
            print(f"B0 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

        if num_resolution_levels >= 2:
            # define B01
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="B01")
            kwargs["att_types"] = self.block_str[0]
            self.B01 = STCNNT_Block(**kwargs)
            print(f"B01 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B11
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="B11")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="B11")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="B11")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[1]
            self.B11 = STCNNT_Block(**kwargs)
            print(f"B11 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define down sample
            self.down_00_11 = DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)

            # define output B1
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[-1], patch_sizes[-1], module_name="output_1")
            kwargs["att_types"] = self.block_str[1]
            self.output_B1 = STCNNT_Block(**kwargs)
            print(f"output_1 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

        if num_resolution_levels >= 3:
            # define B02
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="B02")
            kwargs["att_types"] = self.block_str[0]
            self.B02 = STCNNT_Block(**kwargs)
            print(f"B02 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B12
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[1], patch_sizes[1], module_name="B12")
            kwargs["att_types"] = self.block_str[1]
            self.B12 = STCNNT_Block(**kwargs)
            print(f"B12 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B22
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height // 4
            kwargs["W"] = c.width // 4
            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="B22")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[1], patch_sizes[1], module_name="B22")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], [v//2 for v in self.num_wind], self.num_patch, module_name="B22")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[2]
            self.B22 = STCNNT_Block(**kwargs)
            print(f"B22 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define down sample
            self.down_01_12 = DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_01_22 = DownSample(N=2, C_in=self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_11_22 = DownSample(N=1, C_in=2*self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)

            # define output B2
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height // 4
            kwargs["W"] = c.width // 4
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[-1], patch_sizes[-1], module_name="output_B2")
            kwargs["att_types"] = self.block_str[2]
            self.output_B2 = STCNNT_Block(**kwargs)
            print(f"output_B2 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

        if num_resolution_levels >= 4:
            # define B03
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="B03")
            kwargs["att_types"] = self.block_str[0]
            self.B03 = STCNNT_Block(**kwargs)
            print(f"B03 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B13
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[1], patch_sizes[1], module_name="B13")
            kwargs["att_types"] = self.block_str[1]
            self.B13 = STCNNT_Block(**kwargs)
            print(f"B13 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B23
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height // 4
            kwargs["W"] = c.width // 4
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="B23")
            kwargs["att_types"] = self.block_str[2]
            self.B23 = STCNNT_Block(**kwargs)
            print(f"B23 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B33
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height // 8
            kwargs["W"] = c.width // 8

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="B33")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="B33")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="B33")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[3]
            self.B33 = STCNNT_Block(**kwargs)
            print(f"B33 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define down sample
            self.down_02_13 = DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_02_23 = DownSample(N=2, C_in=self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_02_33 = DownSample(N=3, C_in=self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_12_23 = DownSample(N=1, C_in=2*self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_12_33 = DownSample(N=2, C_in=2*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_22_33 = DownSample(N=1, C_in=4*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)

            # define output B3
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height // 8
            kwargs["W"] = c.width // 8
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[-1], patch_sizes[-1], module_name="output_B3")
            kwargs["att_types"] = self.block_str[3]
            self.output_B3 = STCNNT_Block(**kwargs)
            print(f"output_B3 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

        if num_resolution_levels >= 5:
            # define B04
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="B04")
            kwargs["att_types"] = self.block_str[0]
            self.B04 = STCNNT_Block(**kwargs)
            print(f"B04 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B14
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[1], patch_sizes[1], module_name="B14")
            kwargs["att_types"] = self.block_str[1]
            self.B14 = STCNNT_Block(**kwargs)
            print(f"B14 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B24
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height // 4
            kwargs["W"] = c.width // 4
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="B24")
            kwargs["att_types"] = self.block_str[2]
            self.B24 = STCNNT_Block(**kwargs)
            print(f"B24 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B34
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height // 8
            kwargs["W"] = c.width // 8
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[3], patch_sizes[3], module_name="B34")
            kwargs["att_types"] = self.block_str[3]
            self.B34 = STCNNT_Block(**kwargs)
            print(f"B34 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define B44
            kwargs["C_in"] = 16*self.C
            kwargs["C_out"] = 16*self.C
            kwargs["H"] = c.height // 16
            kwargs["W"] = c.width // 16

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="B44")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="B44")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], [v//4 for v in self.num_wind], self.num_patch, module_name="B44")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[4]
            self.B44 = STCNNT_Block(**kwargs)
            print(f"B44 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

            # define down sample
            self.down_03_14 = DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_03_24 = DownSample(N=2, C_in=self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_03_34 = DownSample(N=3, C_in=self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_03_44 = DownSample(N=4, C_in=self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

            self.down_13_24 = DownSample(N=1, C_in=2*self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_13_34 = DownSample(N=2, C_in=2*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_13_44 = DownSample(N=3, C_in=2*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

            self.down_23_34 = DownSample(N=1, C_in=4*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_23_44 = DownSample(N=2, C_in=4*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

            self.down_33_44 = DownSample(N=1, C_in=8*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

            # define output B4
            kwargs["C_in"] = 16*self.C
            kwargs["C_out"] = 16*self.C
            kwargs["H"] = c.height // 16
            kwargs["W"] = c.width // 16
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[-1], patch_sizes[-1], module_name="output_B4")
            kwargs["att_types"] = self.block_str[4]
            self.output_B4 = STCNNT_Block(**kwargs)
            print(f"output_B4 -- {kwargs['C_in']} to {kwargs['C_out']} --- {kwargs['att_types']}")

        # fusion stage
        if num_resolution_levels >= 2: 
            self.down_0_1 = DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
        if num_resolution_levels >= 3:
            self.down_0_2 = DownSample(N=2, C_in=self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
        if num_resolution_levels >= 4:
            self.down_0_3 = DownSample(N=3, C_in=self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
        if num_resolution_levels >= 5:
            self.down_0_4 = DownSample(N=4, C_in=self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

        if num_resolution_levels >= 3: 
            self.down_1_2 = DownSample(N=1, C_in=2*self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
        if num_resolution_levels >= 4: 
            self.down_1_3 = DownSample(N=2, C_in=2*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
        if num_resolution_levels >= 5: 
            self.down_1_4 = DownSample(N=3, C_in=2*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

        if num_resolution_levels >= 4: 
            self.down_2_3 = DownSample(N=1, C_in=4*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
        if num_resolution_levels >= 5: 
            self.down_2_4 = DownSample(N=2, C_in=4*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

        if num_resolution_levels >= 5: 
            self.down_3_4 = DownSample(N=1, C_in=8*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

        if num_resolution_levels >= 2: 
            self.up_1_0 = UpSample(N=1, C_in=2*self.C, C_out=self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 3: 
            self.up_2_0 = UpSample(N=2, C_in=4*self.C, C_out=self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 4: 
            self.up_3_0 = UpSample(N=3, C_in=8*self.C, C_out=self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 5: 
            self.up_4_0 = UpSample(N=4, C_in=16*self.C, C_out=self.C, method=c.upsample_method, with_conv=True)

        if num_resolution_levels >= 3: 
            self.up_2_1 = UpSample(N=1, C_in=4*self.C, C_out=2*self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 4: 
            self.up_3_1 = UpSample(N=2, C_in=8*self.C, C_out=2*self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 5: 
            self.up_4_1 = UpSample(N=3, C_in=16*self.C, C_out=2*self.C, method=c.upsample_method, with_conv=True)

        if num_resolution_levels >= 4: 
            self.up_3_2 = UpSample(N=1, C_in=8*self.C, C_out=4*self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 5: 
            self.up_4_2 = UpSample(N=2, C_in=16*self.C, C_out=4*self.C, method=c.upsample_method, with_conv=True)

        if num_resolution_levels >= 5: 
            self.up_4_3 = UpSample(N=1, C_in=16*self.C, C_out=8*self.C, method=c.upsample_method, with_conv=True)

        if num_resolution_levels >= 2: 
            self.up_1 = UpSample(N=1, C_in=2*self.C, C_out=2*self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 3: 
            self.up_2 = UpSample(N=2, C_in=4*self.C, C_out=4*self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 4: 
            self.up_3 = UpSample(N=3, C_in=8*self.C, C_out=8*self.C, method=c.upsample_method, with_conv=True)
        if num_resolution_levels >= 5: 
            self.up_4 = UpSample(N=4, C_in=16*self.C, C_out=16*self.C, method=c.upsample_method, with_conv=True)

    def check_class_specific_parameters(self, config):
        if not "backbone_hrnet" in config:
            raise "backbone_hrnet namespace should exist in config"

        err_str = lambda x : f"{x} should exist in config.backbone_hrnet"

        para_list = ["C", "num_resolution_levels", "block_str", "use_interpolation"]
        for arg_name in para_list:
            if not arg_name in config.backbone_hrnet:
                raise ValueError(err_str(arg_name))


    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): the input image, [B, T, Cin, H, W]

        @rets:
            - y_hat (5D torch.Tensor): aggregated output tensor
            - y_level_outputs (Tuple): tuple of tensor for every resolution level
        """

        x = self.permute(x)

        B, T, Cin, H, W = x.shape

        y_hat = None
        y_level_outputs = None

        # compute the block outputs
        if self.num_resolution_levels >= 1:
            x_00, _ = self.B00(x)

        if self.num_resolution_levels >= 2:
            x_01, _ = self.B01(x_00)
            x_11, _ = self.B11(self.down_00_11(x_00))

        if self.num_resolution_levels >= 3:
            x_02, _ = self.B02(x_01+x_00)
            
            x_12, _ = self.B12(x_11 
                            + self.down_01_12(x_01)
                            )
            
            x_22, _ = self.B22(self.down_11_22(x_11) 
                            + self.down_01_22(x_01)
                            )

        if self.num_resolution_levels >= 4:
            x_03, _ = self.B03(x_02+x_00+x_01)
            
            x_13, _ = self.B13(x_12 + self.down_02_13(x_02) + x_11)
            
            x_23, _ = self.B23(x_22 
                            + self.down_12_23(x_12) 
                            + self.down_02_23(x_02)
                            )
            
            x_33, _ = self.B33(self.down_22_33(x_22)
                            + self.down_12_33(x_12)
                            + self.down_02_33(x_02)
                            )

        if self.num_resolution_levels >= 5:
            x_04, _ = self.B04(x_03+x_02+x_01+x_00)
            
            x_14, _ = self.B14(x_13 
                            + self.down_03_14(x_03) 
                            + x_12 
                            + x_11
                            )
            
            x_24, _ = self.B24(x_23 
                            + self.down_13_24(x_13) 
                            + self.down_03_24(x_03)
                            + x_22
                            )

            x_34, _ = self.B34(x_33
                            + self.down_23_34(x_23)
                            + self.down_13_34(x_13)
                            + self.down_03_34(x_03)
                            )

            x_44, _ = self.B44(self.down_33_44(x_33)
                            + self.down_23_44(x_23)
                            + self.down_13_44(x_13)
                            + self.down_03_44(x_03)
                            )

        if self.num_resolution_levels == 1:
            y_hat_0, _ = self.output_B0(x_00)
            y_hat = y_hat_0
            y_level_outputs = [y_hat_0]

        if self.num_resolution_levels == 2:
            y_hat_0 = x_01 + self.up_1_0(x_11)
            y_hat_1 = x_11 + self.down_0_1(x_01)

            y_hat_0, _ = self.output_B0(y_hat_0)
            y_hat_1, _ = self.output_B1(y_hat_1)

            y_hat = torch.cat((y_hat_0, self.up_1(y_hat_1)), dim=2)

            y_level_outputs = [y_hat_0, y_hat_1]

        if self.num_resolution_levels == 3:
            y_hat_0 = x_02 + self.up_1_0(x_12) + self.up_2_0(x_22)
            y_hat_1 = self.down_0_1(x_02) + x_12 + self.up_2_1(x_22)
            y_hat_2 = self.down_0_2(x_02) + self.down_1_2(x_12) + x_22

            y_hat_0, _ = self.output_B0(y_hat_0)
            y_hat_1, _ = self.output_B1(y_hat_1)
            y_hat_2, _ = self.output_B2(y_hat_2)

            y_hat = torch.cat((y_hat_0, self.up_1(y_hat_1), self.up_2(y_hat_2)), dim=2)
            y_level_outputs = [y_hat_0, y_hat_1, y_hat_2]

        if self.num_resolution_levels == 4:
            y_hat_0 = x_03 + self.up_1_0(x_13) + self.up_2_0(x_23) + self.up_3_0(x_33)
            y_hat_1 = self.down_0_1(x_03) + x_13 + self.up_2_1(x_23) + self.up_3_1(x_33)
            y_hat_2 = self.down_0_2(x_03) + self.down_1_2(x_13) + x_23 + self.up_3_2(x_33)
            y_hat_3 = self.down_0_3(x_03) + self.down_1_3(x_13) + self.down_2_3(x_23) + x_33

            y_hat_0, _ = self.output_B0(y_hat_0)
            y_hat_1, _ = self.output_B1(y_hat_1)
            y_hat_2, _ = self.output_B2(y_hat_2)
            y_hat_3, _ = self.output_B3(y_hat_3)

            y_hat = torch.cat(
                (
                    y_hat_0,
                    self.up_1(y_hat_1),
                    self.up_2(y_hat_2),
                    self.up_3(y_hat_3)
                 ), dim=2)

            y_level_outputs = [y_hat_0, y_hat_1, y_hat_2, y_hat_3]

        if self.num_resolution_levels == 5:
            y_hat_0 =               x_04    + self.up_1_0(x_14)         + self.up_2_0(x_24)         + self.up_3_0(x_34)         + self.up_4_0(x_44)
            y_hat_1 = self.down_0_1(x_04)   +               x_14        + self.up_2_1(x_24)         + self.up_3_1(x_34)         + self.up_4_1(x_44)
            y_hat_2 = self.down_0_2(x_04)   + self.down_1_2(x_14)       +               x_24        + self.up_3_2(x_34)         + self.up_4_2(x_44)
            y_hat_3 = self.down_0_3(x_04)   + self.down_1_3(x_14)       + self.down_2_3(x_24)       +             x_34          + self.up_4_3(x_44)
            y_hat_4 = self.down_0_4(x_04)   + self.down_1_4(x_14)       + self.down_2_4(x_24)       + self.down_3_4(x_34)       +             x_44

            y_hat_0, _ = self.output_B0(y_hat_0)
            y_hat_1, _ = self.output_B1(y_hat_1)
            y_hat_2, _ = self.output_B2(y_hat_2)
            y_hat_3, _ = self.output_B3(y_hat_3)
            y_hat_4, _ = self.output_B4(y_hat_4)

            y_hat = torch.cat(
                (
                    y_hat_0,
                    self.up_1(y_hat_1),
                    self.up_2(y_hat_2),
                    self.up_3(y_hat_3),
                    self.up_4(y_hat_4)
                 ), dim=2)

            y_level_outputs = [y_hat_0, y_hat_1, y_hat_2, y_hat_3, y_hat_4]

        y_hat = self.permute(y_hat)
        y_level_outputs = [self.permute(curr_y_hat) for curr_y_hat in y_level_outputs]
        
        return [y_hat, y_level_outputs]

    def __str__(self):
        return create_generic_class_str(obj=self, exclusion_list=[nn.Module, OrderedDict, STCNNT_Block, DownSample, UpSample])

# -------------------------------------------------------------------------------------------------

def tests():

    from setup.setup_base import parse_config
    
    from utils.benchmark import benchmark_all, benchmark_memory, pytorch_profiler
    from setup.setup_utils import set_seed
    from colorama import Fore, Style

    device = get_device()
    
    B,C,T,H,W = 1, 1, 12, 256, 256
    test_in = torch.rand(B,C,T,H,W, dtype=torch.float32, device=device)

    config = parse_config()

    # attention modules
    config.kernel_size = 3
    config.stride = 1
    config.padding = 1
    config.stride_t = 2
    config.dropout_p = 0.1
    config.no_in_channel = C
    config.C_out = C
    config.height = H
    config.width = W
    config.batch_size = B
    config.time = T
    config.norm_mode = "instance2d"
    config.a_type = "conv"
    config.is_causal = False
    config.n_head = 32
    config.interp_align_c = True
   
    config.window_size = [H//8, W//8]
    config.patch_size = [H//32, W//32]
    
    config.num_wind =[8, 8]
    config.num_patch =[2, 2]
    
    config.window_sizing_method = "mixed"
    
    # losses
    config.losses = ["mse"]
    config.loss_weights = [1.0]
    config.load_path = None

    # to be tested
    config.residual = True
    config.device = None
    config.channels = [16,32,64]
    config.all_w_decay = True
    config.optim = "adamw"
    config.scheduler = "StepLR"

    config.complex_i = False

    config.summary_depth = 4

    config.backbone_hrnet = Namespace()
    config.backbone_hrnet.C = 32
    config.backbone_hrnet.num_resolution_levels = 4

    config.backbone_hrnet.use_interpolation = True

    config.cell_type = "sequential"
    config.normalize_Q_K = True 
    config.att_dropout_p = 0.0
    config.att_with_output_proj = True 
    config.scale_ratio_in_mixer  = 1.0

    config.cosine_att = True
    config.att_with_relative_postion_bias = False

    config.block_dense_connection = True
    
    config.optim = "adamw"
    config.scheduler = "ReduceLROnPlateau"
    config.all_w_decay = True
    
    config.device = device

    config.with_timer = True

    config.stride_s = 1
    config.separable_conv = True
    config.use_einsum = False

    config.mixer_kernel_size = 3
    config.mixer_stride = 1
    config.mixer_padding = 1

    config.mixer_type = 'conv'
    config.shuffle_in_window = False
    config.temporal_flash_attention = False 
    config.activation_func = 'prelu'

    config.upsample_method = 'linear'

    # ---------------------------------------------------------------------

    config.dropout_p = 0.1

    config.backbone_hrnet.block_str = ["T1L1G1",
                        "T1L1G1",
                        "T1L1G1",
                        "T1L1G1",
                        "T1L1G1"]

    model = STCNNT_HRnet(config=config)
    model.to(device=device)

    model = torch.compile(model, dynamic=True, fullgraph=True)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(test_in)

    config.with_timer = False
    print(f"{Fore.GREEN}-------------> STCNNT_HRnet-einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in, grad=None, min_run_time=5, desc='STCNNT_HRnet', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in, desc='STCNNT_HRnet', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # ---------------------------------------------------------------------

    X = torch.permute(test_in, [0, 2, 1, 3, 4])

    model = STCNNT_HRnet(config=config)
    model.to(device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(X)

    print(f"{Fore.GREEN}-------------> STCNNT_HRnet-einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in, grad=None, min_run_time=5, desc='STCNNT_HRnet', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in, desc='STCNNT_HRnet', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # ---------------------------------------------------------------------

    config.backbone_hrnet.block_str = ["C2C2C2",
                        "C3C3C3",
                        "C2C2C2",
                        "C3C3C3",
                        "C2C2C2"]

    config.dropout_p = 0.0

    model = STCNNT_HRnet(config=config)
    model.to(device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(test_in)

    config.with_timer = False
    print(f"{Fore.GREEN}-------------> CONV_HRnet-einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in, grad=None, min_run_time=5, desc='CONV_HRnet', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in, desc='CONV_HRnet', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # ---------------------------------------------------------------------

    config.stride_s = 1
    config.separable_conv = False
    config.use_einsum = False

    model = STCNNT_HRnet(config=config)
    model.to(device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(test_in)
            
    print(f"{Fore.GREEN}-------------> STCNNT_HRnet-einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in.to(device=device), grad=None, min_run_time=5, desc='STCNNT_HRnet-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in.to(device=device), desc='STCNNT_HRnet-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # ---------------------------------------------------------------------

    model = STCNNT_HRnet(config=config)
    model.to(device=device)
    with torch.no_grad():
        model_summary = model_info(model, config)
    print(f"Configuration for this run:\n{config}")
    print(f"Model Summary:\n{str(model_summary)}")

    print("Passed all tests")

if __name__=="__main__":
    tests()
