import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
import torch_dct

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class MSF(nn.Module):
#     def __init__(self, dim, bias):
#         super(MSF, self).__init__()
#
#         # hidden_features = int(dim * ffn_expansion_factor)
#         hidden_features = dim
#
#         self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
#
#         self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
#         self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
#         self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features, bias=bias)
#
#         self.relu3 = nn.ReLU()
#         self.relu5 = nn.ReLU()
#         self.relu7 = nn.ReLU()
#
#         self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#         self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)
#         self.dwconv7x7_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features , bias=bias)
#
#         self.relu3_1 = nn.ReLU()
#         self.relu5_1 = nn.ReLU()
#         self.relu7_1 = nn.ReLU()
#
#         self.project_out = nn.Conv2d(hidden_features * 3, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#
#         x1_3, x2_3, x3_3 = self.relu3(self.dwconv3x3(x)).chunk(3, dim=1)
#         x1_5, x2_5, x3_5 = self.relu5(self.dwconv5x5(x)).chunk(3, dim=1)
#         x1_7, x2_7, x3_7 = self.relu7(self.dwconv7x7(x)).chunk(3, dim=1)
#
#         x1 = torch.cat([x1_3, x1_5, x1_7], dim=1)
#         x2 = torch.cat([x2_3, x2_5, x2_7], dim=1)
#         x3 = torch.cat([x3_3, x3_5, x3_7], dim=1)  #
#
#         x1 = self.relu3_1(self.dwconv3x3_1(x1))
#         x2 = self.relu5_1(self.dwconv5x5_1(x2))
#         x3 = self.relu7_1(self.dwconv7x7_1(x3))
#
#         x = torch.cat([x1, x2, x3], dim=1)
#
#         x = self.project_out(x)
#
#         return x
import torch
import torch.nn as nn

class MSF(nn.Module):
    def __init__(self, dim, bias):
        super(MSF, self).__init__()

        # dim 是 4 的倍数
        hidden_features = dim

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        # --- 第一层：特征提取 ---
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features, bias=bias)
        # 第4分支：最大池化（不改变通道数）
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu_pool = nn.ReLU()

        # --- 第二层：深度演化 ---
        # 注意：这里每一路的输入通道数变为 hidden_features (通过 4 个 chunk 拼接得到)
        self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features, bias=bias)
        self.dwconv_pool_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.relu7_1 = nn.ReLU()
        self.relu_pool_1 = nn.ReLU()

        # 最后拼接 4 个分支，总通道数为 hidden_features * 4
        self.project_out = nn.Conv2d(hidden_features * 4, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)

        # 1. 每一路都经过变换并切分为 4 份
        x1_3, x2_3, x3_3, x4_3 = self.relu3(self.dwconv3x3(x)).chunk(4, dim=1)
        x1_5, x2_5, x3_5, x4_5 = self.relu5(self.dwconv5x5(x)).chunk(4, dim=1)
        x1_7, x2_7, x3_7, x4_7 = self.relu7(self.dwconv7x7(x)).chunk(4, dim=1)
        x1_p, x2_p, x3_p, x4_p = self.relu_pool(self.maxpool(x)).chunk(4, dim=1)

        # 2. 跨尺度特征重组 (Shuffle/Exchange)
        x1 = torch.cat([x1_3, x1_5, x1_7, x1_p], dim=1)
        x2 = torch.cat([x2_3, x2_5, x2_7, x2_p], dim=1)
        x3 = torch.cat([x3_3, x3_5, x3_7, x3_p], dim=1)
        x4 = torch.cat([x4_3, x4_5, x4_7, x4_p], dim=1)

        # 3. 各分支独立卷积
        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))
        x3 = self.relu7_1(self.dwconv7x7_1(x3))
        x4 = self.relu_pool_1(self.dwconv_pool_1(x4))

        # 4. 最终拼接
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.project_out(x)

        return x
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Wave2D(nn.Module):
    r"""
    Wave equation operator:
    d2u/dt2 - c2(d2u/dx2 + d2u/dy2) + αdu/dt = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by) * (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }

    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    =>
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt))
    """

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        # 用于将频率嵌入转换为时间步长 t 的小型网络
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        # Add wave speed parameter
        self.c = nn.Parameter(torch.ones(1) * 1)
        # Add damping parameter
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        # self.alpha = self.alpha.to('cuda')

    def infer_init_wave2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        # Quadratic term for wave equation
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            x = x.transpose(1, 2).reshape(B, C, H, W)
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # B,H,W,2C
        x, z = x.chunk(chunks=2, dim=-1)  # B,H,W,C
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        z = z.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (weight_cosn is not None) and (
                weight_cosn.device == x.device):
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        def dct2d(x):
            # 使用导入的 torch_dct 库
            return torch_dct.dct_2d(x, norm='ortho')

        def idct2d(x):
            # 使用导入的 torch_dct 库
            return torch_dct.idct_2d(x, norm='ortho')

        x_u0 = dct2d(x)
        x_v0 = dct2d(x)

        # 时间步长计算：根据频率嵌入动态调整每个 Token 的传播时间 t
        # freq_embed: (H, W, C) -> (B, H, W, C)，意味着每个频率分量有自己的传播时间
        if freq_embed is not None:
            t = self.to_k(freq_embed.unsqueeze(0).expand(B, -1, -1, -1).contiguous())
        else:
            t = torch.zeros((B, H, W, C), device=x.device, dtype=x.dtype)
        cos_term = torch.cos(self.c * t).permute(0, 3, 1, 2).contiguous()
        sin_term = torch.sin(self.c * t).permute(0, 3, 1, 2).contiguous() / self.c

        # print(cos_term.shape, x_u0.shape, sin_term.shape)
        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        final_term = wave_term + velocity_term

        x_final = idct2d(final_term)
        # x_final: (B, C, H, W)
        x = self.out_norm(x_final.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x * F.silu(z)
        x = self.out_linear(x.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class dwconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super(dwconv, self).__init__()
        self.dw1 = torch.nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_ch)
        self.pw = torch.nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        out1 = self.dw1(x)
        out = self.pw(out1)
        return out

class Att(nn.Module):
    def __init__(self, inch, outch):
        super(Att,self).__init__()
        self.conv1 = dwconv(inch,outch)
        self.conv2 = dwconv(inch,outch)
        self.conv3 = dwconv(inch,outch)
        self.conv4 = dwconv(inch,outch)
        self.conv5 = dwconv(inch,outch)
        self.conv6 = dwconv(inch,outch)
        self.conv7 = dwconv(inch,outch)
        self.conv8 = dwconv(inch,outch)
        self.conv9 = dwconv(inch,outch)
        self.conv10 = dwconv(inch,outch)
        self.conv11 = dwconv(inch,outch)
        self.act = nn.GELU()
        self.conv_down_ch = nn.Conv2d(inch*3, outch, 1, padding=0)

    def forward(self, x):
        h, w = x.size()[-2:]

        out1 = self.act(self.conv1(x))

        out_down1 = F.adaptive_avg_pool2d(out1, output_size=(h // 2, w // 2))
        out2 = self.act(self.conv2(out_down1))
        out2 = F.interpolate(out2, size=(h // 2, w // 2), mode='nearest')

        out_down2 = F.adaptive_avg_pool2d(out2, output_size=(h // 4, w // 4))
        out3 = self.act(self.conv3(out_down2))
        out3 = F.interpolate(out3, size=(h // 4, w // 4), mode='nearest')

        out_down3 = F.adaptive_avg_pool2d(out3, output_size=(h // 8, w // 8))
        out4 = self.act(self.conv4(out_down3))
        out4 = F.interpolate(out4, size=(h // 8, w // 8), mode='nearest')

        out_up4 = F.interpolate(out4, size=(h // 4, w // 4), mode='nearest')
        out5 = self.act(self.conv5(out_up4))

        out_up5 = F.interpolate(out5, size=(h // 2, w // 2), mode='nearest')
        out6 = self.act(self.conv6(out_up5))

        out_up6 = F.interpolate(out6, size=(h, w), mode='nearest')
        out7 = self.act(self.conv7(out_up6))
        #hw
        out_17 = out1 - out7
        out_17 = self.conv11(out_17)
        #hw/2 -hw
        out_26 = out2 - out6
        out_26 = self.conv10(out_26)
        out_26 = F.interpolate(out_26, size=(h, w), mode='nearest')
        #hw/4
        out_35 = out3 - out5
        out_35 = self.conv8(out_35)
        #hw/4 - hw/2 - hw
        out_35 = F.interpolate(out5, size=(h // 2, w // 2), mode='nearest')
        out_35 = self.conv9(out_35)
        out_35 = F.interpolate(out_35, size=(h, w), mode='nearest')

        out_3526 = torch.cat((out_26,out_35),1)
        out_352617 = torch.cat((out_3526,out_17),1)
        out_352617 = self.act(self.conv_down_ch(out_352617))
        out = out_352617 * x

        return out
class GCA(nn.Module):
    """
    Tips:
        Mainly borrows from SKNet (https://github.com/implus/SKNet)
    """
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  #
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2) # K=9, 64-9+8 + 1


        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x, x_size=None, params=None):

        H, W = x_size
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        out = x * attn
        out = out.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        # print(out.shape)
        return out


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                     # mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0.,freq_embed=None,
                     act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.freq_embed = freq_embed

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # --- 第一阶段：并行双分支 (Wave2D & Att) ---
        self.norm1 = norm_layer(dim)
        self.attn = Wave2D(res=14, dim=dim, hidden_dim=dim, freq_embed=freq_embed, infer_mode=False)
        self.han = Att(dim, dim)

        # 融合分支的 1x1 卷积（将 concat 的 2C 降回 C）
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        # --- 第二阶段：多尺度融合 (MSF) ---
        self.norm2 = norm_layer(dim)
        self.mlp = MSF(dim=dim, bias=False)  # 您之前定义的 4 分支 MSF

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x

        # -------- 并行阶段：归一化与准备 --------
        x_norm = self.norm1(x)
        x_img = x_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # 分支 1: Wave2D (基于窗口)
        if self.shift_size > 0:
            shifted_x = torch.roll(x_img, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x_img

        # 窗口切分处理
        # 注意：此处需根据你的 window_partition 输入维度调整，假设其处理 (B, H, W, C)
        x_windows = window_partition(shifted_x.permute(0, 2, 3, 1), self.window_size)
        x_windows = x_windows.permute(0, 3, 1, 2).contiguous()

        attn_windows = self.attn(x_windows, freq_embed=self.freq_embed)

        attn_windows = attn_windows.permute(0, 2, 3, 1).contiguous()
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        if self.shift_size > 0:
            out_w2d = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            out_w2d = shifted_x
        out_w2d = out_w2d.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # 分支 2: Att
        out_att = self.han(x_img)  # (B, C, H, W)

        # -------- 融合阶段 --------
        combined = torch.cat([out_w2d, out_att], dim=1)
        x_fused = self.fusion(combined)

        # 第一次残差连接
        x = shortcut + self.drop_path(x_fused.permute(0, 2, 3, 1).reshape(B, L, C))

        # -------- MSF 阶段 (类似于 FFN) --------
        res_msf = x
        x = self.norm2(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = self.mlp(x)  # 4 分支多尺度融合
        x = self.act(x)

        # 第二次残差连接
        x = res_msf + self.drop_path(x.permute(0, 2, 3, 1).reshape(B, L, C))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size} "#mlp_ratio={self.mlp_ratio}


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, res, window_size,
                 # mlp_ratio=4.,
                 # qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 freq_embed=None,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.freq_embed = freq_embed
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 # mlp_ratio=mlp_ratio,
                                 # qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 # drop=drop, attn_drop=attn_drop,
                                 freq_embed=freq_embed,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, res, window_size,freq_embed,
                 # mlp_ratio=4.,
                 # qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         res = res,
                                         window_size=window_size,
                                         freq_embed=freq_embed,
                                         # mlp_ratio=mlp_ratio,
                                         # qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         # drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class WaveFormerSR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `DeSeNet: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=132, depths=[4, 4, 4, 4, 4, 4], num_heads=[4, 4, 4, 4, 4, 4],
                  window_size=8,
                 # mlp_ratio=2.,
                 # qkv_bias=True, qk_scale=None,
                 freq_embed=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='pixelshuffle', resi_connection='1conv',
                 **kwargs):
        super(WaveFormerSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size


        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        # self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.res = window_size
        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            self.freq_embed.append(
                nn.Parameter(torch.zeros(self.res, self.res, self.embed_dim), requires_grad=True))
            trunc_normal_(self.freq_embed[i], std=.02)

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         res= self.res,
                         window_size=window_size,
                         freq_embed=self.freq_embed[i_layer],
                         # mlp_ratio=self.mlp_ratio,
                         # qkv_bias=qkv_bias, qk_scale=qk_scale,
                         # drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection,

                         )
            self.layers.append(layer)
            # # ========== Optinal GCA ====================
            self.gca = GCA(dim=embed_dim)
            self.layers.append(self.gca)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        # x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    input = torch.rand(1, 3, 128, 128).cuda()  # B C H W
    model = WaveFormerSR().cuda()
    flops, params = profile(model, inputs=(input,))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))

    output = model(input)
    print(output.size())
