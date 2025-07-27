import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class MultiscaleDepthConvModule(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiscaleDepthConvModule, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class MultiscaleConvModule(nn.Module):
    """
    Multi-scale convolution block (MultiscaleConvModule)
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MultiscaleConvModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.multiscale_depthconv = MultiscaleDepthConvModule(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        multiscale_depthconv_outs = self.multiscale_depthconv(pout1)
        if self.add == True:
            dout = 0
            for dwout in multiscale_depthconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(multiscale_depthconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


#   Multi-scale convolution block (MultiscaleConvModule)
def MultiscaleConvModuleLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    multiscale_conv_module = MultiscaleConvModule(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(multiscale_conv_module)
    if n > 1:
        for i in range(1, n):
            multiscale_conv_module = MultiscaleConvModule(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(multiscale_conv_module)
    conv = nn.Sequential(*convs)
    return conv

class SpaceAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpaceAttentionModule, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        x = self.conv(x)
        x = self.sigmoid(x)
        return x * 0.5 + 0.5  # 保持权重输出在 [0.5, 1.0]


class MultiscaleConvAttentionModule(nn.Module):
    def __init__(self, channels, kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6'):
        super(MultiscaleConvAttentionModule, self).__init__()

        self.multiscale_conv_module = MultiscaleConvModuleLayer(channels, channels, n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.space_attention = SpaceAttentionModule()

    def forward(self, x):
        
        space_attention = self.space_attention(x) * x
        multiscale_conv_module = self.multiscale_conv_module(space_attention)

        return multiscale_conv_module

if __name__ =='__main__':
    stars_Block =MultiscaleConvAttentionModule(256)
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    output_tensor =stars_Block(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)
