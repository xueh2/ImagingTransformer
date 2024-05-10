"""
Spatio-Temporal Convolutional Neural Net Transformer (STCNNT)

Implement the ViT style spatial attention.

"""

import math
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

from attention_modules import *

# -------------------------------------------------------------------------------------------------
# CNN attention with the ViT style - an image is split into windows. Attention coefficients are computed among all windows.

class SpatialViTAttention(CnnAttentionBase):
    """
    Multi-head cnn attention model for ViT style. An image is spatially splited into windows. 
    Attention matrix is computed between all windows. Number of pixels in a window are [window_size, window_size].
    """
    def __init__(self, C_in, C_out=16, H=128, W=128, 
                 window_size=[32, 32], num_wind=None,
                 a_type="conv", 
                 n_head=8,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), stride_qk=(1, 1), 
                 separable_conv=False, 
                 att_dropout_p=0.0, 
                 cosine_att=False, 
                 normalize_Q_K=False, 
                 att_with_relative_postion_bias=True,
                 att_with_output_proj=True,
                 use_einsum=False,
                 with_timer=False):
        """
        Defines the layer for a cnn attention on spatial dimension with local windows

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H, W]

        Either window_size or num_wind should be supplied. 
        window_size is the number of pixels per window, along H and W.
        num_wind is the number of windows along H and W.
        if both are supplied, num_wind taks priority.

        Shared parameters are defined in base class.

        @args:
            - window_size (int): number of pixels in a window
        """
        super().__init__(C_in=C_in, 
                         C_out=C_out, 
                         H=H, W=W,
                         n_head=n_head, 
                         kernel_size=kernel_size, 
                         stride=stride, 
                         separable_conv=separable_conv,
                         padding=padding, 
                         stride_qk=stride_qk,
                         att_dropout_p=att_dropout_p, 
                         cosine_att=cosine_att,
                         normalize_Q_K=normalize_Q_K, 
                         att_with_relative_postion_bias=att_with_relative_postion_bias,
                         att_with_output_proj=att_with_output_proj, with_timer=with_timer)

        self.C_out = C_out
        self.a_type = a_type
        self.window_size = window_size
        self.num_wind = num_wind
        self.use_einsum = use_einsum

        self.set_and_check_wind()

        if a_type=="conv":
            # key, query, value projections convolution
            # Wk, Wq, Wv
            self.key = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_qk, padding=padding, bias=False, separable_conv=self.separable_conv)
            self.query = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_qk, padding=padding, bias=False, separable_conv=self.separable_conv)
            self.value = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, separable_conv=self.separable_conv)
        elif a_type=="lin":
            # linear projections
            num_pixel_win = self.window_size[0]*self.window_size[1]
            self.key = LinearGridExt(C_in*num_pixel_win, C_out*num_pixel_win, bias=False)
            self.query = LinearGridExt(C_in*num_pixel_win, C_out*num_pixel_win, bias=False)
            self.value = LinearGridExt(C_in*num_pixel_win, C_out*num_pixel_win, bias=False)
        else:
            raise NotImplementedError(f"Attention type not implemented: {a_type}")

        if self.att_with_relative_postion_bias:
            self.define_relative_position_bias_table(num_win_h=self.num_wind[0], num_win_w=self.num_wind[1])
            self.define_relative_position_index(num_win_h=self.num_wind[0], num_win_w=self.num_wind[1])

    def attention(self, k, q, v):
        B, T, num_win_h, num_win_w, wh, ww, C = k.shape
        _, _, _, _, wh_v, ww_v, _ = v.shape

        assert self.num_wind[0] == num_win_h
        assert self.num_wind[1] == num_win_w

        # format the window
        hc = torch.div(C*wh*ww, self.n_head, rounding_mode="floor")
        hc_v = torch.div(C*wh_v*ww_v, self.n_head, rounding_mode="floor")

        tm = start_timer(enable=self.with_timer)
        q = q.reshape((B, T, num_win_h*num_win_w, self.n_head, hc)).transpose(2, 3)
        k = k.reshape((B, T, num_win_h*num_win_w, self.n_head, hc)).transpose(2, 3)
        v = v.reshape((B, T, num_win_h*num_win_w, self.n_head, hc_v)).transpose(2, 3)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")
        
        tm = start_timer(enable=self.with_timer)
        # [B, T, num_heads, num_windows, hc] x [B, T, num_heads, hc, num_windows] -> (B, T, num_heads, num_windows, num_windows)
        if self.cosine_att:
            att = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        else:
            if self.normalize_Q_K:
                eps = torch.finfo(k.dtype).eps
                k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
                q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

            att = q @ k.transpose(-2, -1) * torch.tensor(1.0 / math.sqrt(hc))

        end_timer(enable=self.with_timer, t=tm, msg="att")

        tm = start_timer(enable=self.with_timer)
        att = F.softmax(att, dim=-1)
        end_timer(enable=self.with_timer, t=tm, msg="softmax")

        tm = start_timer(enable=self.with_timer)
        # add the relative positional bias
        if self.att_with_relative_postion_bias:
            relative_position_bias = self.get_relative_position_bias(num_win_h, num_win_w)
            att = att + relative_position_bias
        end_timer(enable=self.with_timer, t=tm, msg="att_with_relative_postion_bias")

        tm = start_timer(enable=self.with_timer)
        att = self.attn_drop(att)
        end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

        tm = start_timer(enable=self.with_timer)
        # (B, T, num_heads, num_windows, num_windows) * (B, T, num_heads, num_windows, hc)
        y = att @ v
        y = y.transpose(2, 3) # (B, T, num_windows, num_heads, hc)
        y = torch.reshape(y, (B, T, num_win_h, num_win_w, wh_v, ww_v, C))
        end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        tm = start_timer(enable=self.with_timer)
        y = self.grid2im(y)
        end_timer(enable=self.with_timer, t=tm, msg="grid2im")

        return y

    def einsum_attention(self, k, q, v):
        B, T, num_win_h, num_win_w, wh, ww, C = k.shape
        _, _, _, _, wh_v, ww_v, _ = v.shape
        
        assert self.num_wind[0] == num_win_h
        assert self.num_wind[1] == num_win_w

        # format the window
        hc = torch.div(C*wh*ww, self.n_head, rounding_mode="floor")
        hc_v = torch.div(C*wh_v*ww_v, self.n_head, rounding_mode="floor")

        if self.has_flash_attention and hc <= 256 and (not self.att_with_relative_postion_bias) and (hc_v == hc):
            tm = start_timer(enable=self.with_timer)
            k = k.reshape((B*T, num_win_h*num_win_w, self.n_head, hc))
            q = q.reshape((B*T, num_win_h*num_win_w, self.n_head, hc))
            v = v.reshape((B*T, num_win_h*num_win_w, self.n_head, hc_v))
            end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

            tm = start_timer(enable=self.with_timer)
            y = self.perform_flash_atten(k, q, v)
            end_timer(enable=self.with_timer, t=tm, msg="perform_flash_atten")
        else:
            tm = start_timer(enable=self.with_timer)
            k = k.reshape((B, T, num_win_h*num_win_w, self.n_head, hc))
            q = q.reshape((B, T, num_win_h*num_win_w, self.n_head, hc))
            v = v.reshape((B, T, num_win_h*num_win_w, self.n_head, hc_v))
            end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

            tm = start_timer(enable=self.with_timer)
            if self.cosine_att:
                att = torch.einsum("BTWND, BTSND -> BTNWS", F.normalize(q, dim=-1), F.normalize(k, dim=-1))
                end_timer(enable=self.with_timer, t=tm, msg="BTWND, BTSND -> BTNWS")
            else:
                if self.normalize_Q_K:
                    eps = torch.finfo(k.dtype).eps
                    k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
                    q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

                att = torch.einsum("BTWND, BTSND -> BTNWS", q, k) * torch.tensor(1.0 / math.sqrt(hc))
                end_timer(enable=self.with_timer, t=tm, msg="normalize_Q_K, BTWND, BTSND -> BTNWS")

            tm = start_timer(enable=self.with_timer)
            att = F.softmax(att, dim=-1)
            end_timer(enable=self.with_timer, t=tm, msg="softmax")

            # add the relative positional bias
            tm = start_timer(enable=self.with_timer)
            if self.att_with_relative_postion_bias:
                relative_position_bias = self.get_relative_position_bias(num_win_h, num_win_w)
                att = att + relative_position_bias
            end_timer(enable=self.with_timer, t=tm, msg="relative_position_bias")

            tm = start_timer(enable=self.with_timer)
            att = self.attn_drop(att)
            end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

            # (B, T, num_heads, num_windows, num_windows) * (B, T, num_windows, num_heads, hc)
            tm = start_timer(enable=self.with_timer)
            y = torch.einsum("BTNWS, BTSND -> BTWND", att, v)
            end_timer(enable=self.with_timer, t=tm, msg="BTNWS, BTSND -> BTWND")

        tm = start_timer(enable=self.with_timer)
        y = y.reshape(B, T, num_win_h, num_win_w, wh_v, ww_v, C)
        end_timer(enable=self.with_timer, t=tm, msg="y.reshape")

        tm = start_timer(enable=self.with_timer)
        y = self.grid2im(y)
        end_timer(enable=self.with_timer, t=tm, msg="grid2im")

        return y

    def forward(self, x):
        """
        @args:
            x ([B, T, C, H, W]): Input of a batch of time series

        @rets:
            y ([B, T, C_out, H', W']): output tensor
        """
        B, T, C, H, W = x.size()
        Ws = self.window_size

        if self.a_type == "lin" and self.att_with_relative_postion_bias:
            assert H==self.H and W==self.W, f"For lin a_type with relative position bias, input H and W have to be the same as the class declaration."

        assert C == self.C_in, f"Input channel {C} does not match expected input channel {self.C_in}"
        assert H % self.num_wind[0] == 0, f"Height {H} should be divisible by window num {self.num_wind[0]}"
        assert W % self.num_wind[1] == 0, f"Width {W} should be divisible by window num {self.num_wind[1]}"

        # if self.att_with_relative_postion_bias:
        #     assert H == self.H, f"Input height {H} should equal to the preset H {self.H}"
        #     assert W == self.W, f"Input height {W} should equal to the preset H {self.W}"

        if self.a_type=="conv":
            tm = start_timer(enable=self.with_timer)
            k = self.key(x) # (B, T, C, H_prime, W_prime)
            q = self.query(x)
            v = self.value(x)
            # kqv = self.kqv(x)
            # k = kqv[:, :, 0:self.C_out, :, :]
            # q = kqv[:, :, self.C_out:2*self.C_out, :, :]
            # v = kqv[:, :, 2*self.C_out:, :, :]
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")

            tm = start_timer(enable=self.with_timer)
            k = self.im2grid(k) # (B, T, num_win_h, num_win_w, C, wh, ww)
            q = self.im2grid(q)
            v = self.im2grid(v)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")
        else:
            x = self.im2grid(x) # (B, T, num_win_h, num_win_w, C_in, wh, ww)
            k = self.key(x) # (B, T, num_win_h, num_win_w, C, wh, ww)
            q = self.query(x)
            v = self.value(x)

        #y1 = self.attention(torch.clone(k), torch.clone(q), torch.clone(v))
        if self.use_einsum:
            y = self.einsum_attention(k, q, v)
        else:
            y = self.attention(k, q, v)

        #assert torch.allclose(y1, y)

        tm = start_timer(enable=self.with_timer)
        y = self.output_proj(y)
        end_timer(enable=self.with_timer, t=tm, msg="output_proj")

        return y

    def im2grid(self, x):
        """
        Reshape the input into windows of local areas
        """
        b, t, c, h, w = x.shape

        wind_view = rearrange(x, 'b t c (num_win_h win_size_h) (num_win_w win_size_w) -> b t num_win_h num_win_w win_size_h win_size_w c', 
                              num_win_h=self.num_wind[0], win_size_h=h//self.num_wind[0], num_win_w=self.num_wind[1], win_size_w=w//self.num_wind[1])

        return wind_view

    def grid2im(self, x):
        """
        Reshape the windows back into the complete image
        """
        # b, t, num_win_h, num_win_w, c, wh, ww = x.shape

        b, t, num_win_h, num_win_w, wh, ww, c = x.shape

        im_view = rearrange(x, 'b t num_win_h num_win_w win_size_h win_size_w c -> b t c (num_win_h win_size_h) (num_win_w win_size_w)', 
                              num_win_h=num_win_h, win_size_h=wh, num_win_w=num_win_w, win_size_w=ww)

        return im_view

# -------------------------------------------------------------------------------------------------

def tests():

    import time

    print("Begin Testing")

    t = np.arange(256)
    t = np.reshape(t, (16,16))

    w = 8

    t = torch.from_numpy(t).to(dtype=torch.float32)
    t = torch.cat((t[None, :], t[None, :]), dim=0)

    B, T, C, H, W = 2, 4, 2, 16, 16
    C_out = 8
    test_in = t.repeat(B, T, 1, 1, 1)
    print(test_in.shape)
           
    spacial_vit = SpatialViTAttention(window_size=[w, w], num_wind=None, a_type="conv", 
                                      C_in=C, C_out=C_out, H=H, W=W, 
                                      stride_qk=(2,2),
                                      cosine_att=True, 
                                      normalize_Q_K=True, 
                                      att_with_relative_postion_bias=True,
                                      att_with_output_proj=True)
            
    a = spacial_vit.im2grid(test_in)  
    b = spacial_vit.grid2im(a)

    assert torch.allclose(test_in, b)
       
    if torch.norm(b-test_in)<1e-3:   
        print("Passed im2grid test")
    else:
        raise "im2grid test failed"
       
    a_types = ["conv", "lin"]
    normalize_Q_Ks = [True, False]
    cosine_atts = [True, False]
    att_with_relative_postion_biases = [True, False]
    att_with_output_projs = [True, False]
    stride_qks = [[1,1], [2,2]]

    device = get_device()

    B, T, C, H1, W1 = 2, 4, 2, 64, 64
    C_out = 8
    test_in = torch.rand(B, T, C, H1, W1).to(device=device)
    print(test_in.shape)
    
    B, T, C, H2, W2 = 2, 4, 2, 128, 128
    C_out = 8
    test_in2 = torch.rand(B, T, C, H2, W2).to(device=device)
    print(test_in2.shape)

    for a_type in a_types:
        for normalize_Q_K in normalize_Q_Ks:
            for att_with_output_proj in att_with_output_projs:
                for cosine_att in cosine_atts:
                    for att_with_relative_postion_bias in att_with_relative_postion_biases:
                        for stride_qk in stride_qks:
                            spacial_vit = SpatialViTAttention(window_size=None, num_wind=[8, 8],
                                                            a_type=a_type, 
                                                            C_in=C, C_out=C_out, 
                                                            H=H1, W=W1, 
                                                            stride_qk=stride_qk,
                                                            cosine_att=cosine_att, 
                                                            normalize_Q_K=normalize_Q_K, 
                                                            att_with_relative_postion_bias=att_with_relative_postion_bias,
                                                            att_with_output_proj=att_with_output_proj)
                            spacial_vit.to(device=device)

                            t0 = start_timer(enable=self.with_timer)
                            test_out = spacial_vit(test_in)
                            end_timer(enable=self.with_timer, t=t0, msg="forward pass - stride_qk {stride_qk}")

                            Bo, To, Co, Ho, Wo = test_out.shape
                            assert B==Bo and T==To and Co==C_out and H1==Ho and W1==Wo

                            loss = nn.MSELoss()
                            t0 = start_timer(enable=self.with_timer)
                            mse = loss(test_in, test_out[:,:,:C,:,:])
                            mse.backward()
                            end_timer(enable=self.with_timer, t=t0, msg="backward pass - stride_qk {stride_qk}")

                            if a_type == "conv":
                                test_out = spacial_vit(test_in2)

                                Bo, To, Co, Ho, Wo = test_out.shape
                                assert B==Bo and T==To and Co==C_out and H2==Ho and W2==Wo

    print("Passed SpatialViTAttention tests")
    
    print("Passed all tests")

# -------------------------------------------------------------------------------------------------

def benchmark():

    from utils.benchmark import benchmark_all, benchmark_memory, pytorch_profiler
    from setup.setup_utils import set_seed
    from colorama import Fore, Style

    set_seed(seed=53)

    device = get_device()

    min_run_time = 5

    B, T, C, H, W = 16, 12, 32, 256, 256
    C_out = 32
    n_head = 32
    test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)

    print(test_in[6:9,2:6, 2, 54, 34])
    print(test_in[11:,7:, 2, 54, 34])

    import torch.utils.benchmark as benchmark

    print(f"{Fore.GREEN}-------------> Vit attention <----------------------{Style.RESET_ALL}")

    m = SpatialViTAttention(window_size=None, num_wind=[8, 8],
                                        a_type="conv", 
                                        C_in=C, C_out=C_out, 
                                        H=H, W=W, 
                                        stride_qk=(2,2),
                                        separable_conv=True,
                                        n_head=n_head,
                                        cosine_att=False, 
                                        normalize_Q_K=False, 
                                        att_with_relative_postion_bias=True,
                                        att_with_output_proj=False,
                                        use_einsum=True)

    m.to(device=device)

    m.with_timer = True
    for _ in range(10):
        t0 = start_timer(enable=True)
        y = m(test_in)
        end_timer(enable=True, t=t0, msg=f"{Fore.RED} --- > forward, {m.stride_qk}, {m.separable_conv} < --- {Style.RESET_ALL}")

        t0 = start_timer(enable=True)
        loss = nn.MSELoss()
        mse = loss(test_in, y[:,:,:C,:,:])
        mse.backward()
        end_timer(enable=True, t=t0, msg=f"{Fore.RED} --- > backward, {m.stride_qk}, {m.separable_conv} < --- {Style.RESET_ALL}")

    m = SpatialViTAttention(window_size=None, num_wind=[8, 8],
                                        a_type="conv", 
                                        C_in=C, C_out=C_out, 
                                        H=H, W=W, 
                                        stride_qk=(1,1),
                                        separable_conv=False,
                                        n_head=n_head,
                                        cosine_att=False, 
                                        normalize_Q_K=False, 
                                        att_with_relative_postion_bias=True,
                                        att_with_output_proj=False,
                                        use_einsum=True,
                                        with_timer=True)
    m.to(device=device)
    for _ in range(10):
        t0 = start_timer(enable=True)
        y = m(test_in)
        end_timer(enable=True, t=t0, msg=f"{Fore.RED} --- > forward, {m.stride_qk}, {m.separable_conv} < --- {Style.RESET_ALL}")

        t0 = start_timer(enable=True)
        loss = nn.MSELoss()
        mse = loss(test_in, y[:,:,:C,:,:])
        mse.backward()
        end_timer(enable=True, t=t0, msg=f"{Fore.RED} --- > backward, {m.stride_qk}, {m.separable_conv} < --- {Style.RESET_ALL}")

    time.sleep(3.0)

    # -------------------------------------------------------

    m = SpatialViTAttention(window_size=None, num_wind=[8, 8],
                                    a_type="conv", 
                                    C_in=C, C_out=C_out, 
                                    H=H, W=W, 
                                    stride_qk=(2,2),
                                    separable_conv=True,
                                    n_head=n_head,
                                    cosine_att=False, 
                                    normalize_Q_K=False, 
                                    att_with_relative_postion_bias=True,
                                    att_with_output_proj=False,
                                    use_einsum=True)

    m.to(device=device)
    benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialViTAttention-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    benchmark_memory(m, test_in, desc='SpatialViTAttention-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # -------------------------------------------------------

    m = SpatialViTAttention(window_size=None, num_wind=[8, 8],
                                        a_type="conv", 
                                        C_in=C, C_out=C_out, 
                                        H=H, W=W, 
                                        stride_qk=(2,2),
                                        separable_conv=True,
                                        n_head=n_head,
                                        cosine_att=False, 
                                        normalize_Q_K=False, 
                                        att_with_relative_postion_bias=True,
                                        att_with_output_proj=False,
                                        use_einsum=False)

    m.to(device=device)

    m.with_timer = True
    t0 = start_timer(enable=True)
    with torch.inference_mode():
        y = m(test_in)
    end_timer(enable=True, t=t0, msg="forward")

    time.sleep(3.0)

    m.with_timer = False
    benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialViTAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    benchmark_memory(m, test_in, desc='SpatialViTAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # def loss(model, x):
    #     y = model(x)
    #     l = torch.sum(y)
    #     return l

    # pytorch_profiler(loss, m, test_in, trace_filename='/export/Lab-Xue/projects/mri/profiling/SpatialViTAttention.json', backward=True, amp=True, amp_dtype=torch.bfloat16, cpu=False, verbose=True)

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":
    #tests()
    benchmark()
