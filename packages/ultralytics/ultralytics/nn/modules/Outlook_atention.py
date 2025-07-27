import torch
import torch.nn as nn
import torch.nn.functional as F
# from conv import Conv
from .conv import Conv

import math


class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        # 计算每个注意力头的维度
        head_dim = dim // num_heads
        self.num_heads = num_heads  # 注意力头的数量
        self.kernel_size = kernel_size  # 卷积核大小
        self.padding = padding  # 卷积填充
        self.stride = stride  # 卷积步幅

        # QK 的缩放因子，默认是头维度的倒数平方根
        self.scale = qk_scale or head_dim ** -0.5

        # 定义线性层，用于计算值 V
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # 定义线性层，用于计算注意力权重
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        # 定义丢弃层，用于注意力计算的丢弃
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义输出投影层
        self.proj = nn.Linear(dim, dim)
        # 定义输出的丢弃层
        self.proj_drop = nn.Dropout(proj_drop)

        # 定义展开操作，将输入特征图转化为局部窗口
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        # 定义平均池化操作，用于生成上下文信息
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        # 调整输入维度，从 (B, C, H, W) 转为 (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        B, H, W, C = x.shape  # 解包输入特征图的维度

        # 计算值 V，并调整维度为 (B, C, H, W)
        v = self.v(x).permute(0, 3, 1, 2)

        # 计算经过步幅处理后的特征图高度和宽度
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        # 将值 V 展开为局部窗口，调整形状为 (B, H, N, kxk, C/H)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        # 对输入特征图进行平均池化，生成上下文信息
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 计算注意力权重并调整形状为 (B, H, N, kxk, kxk)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)

        # 缩放注意力权重并进行 softmax 归一化
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 应用丢弃

        # 使用注意力权重对值 V 进行加权求和
        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)

        # 将特征图重构为原始尺寸
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        # 通过线性层进行输出投影
        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)  # 应用丢弃

        # 将输出维度调整回 (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x  # 返回处理后的特征图


class Bottleneck_OAtention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = OutlookAttention( c2, 4)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C2f_OAtention(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_OAtention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



if __name__ =='__main__':
    stars_Block =OutlookAttention(256)
    #创建一个输入张量，形状为(batch_size, H*W,C)
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =stars_Block(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)


