import torch
import torch.nn as nn

# 通用卷积模块，用于减少重复定义
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True, activation=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DenseCat(nn.Module):
    def __init__(self, in_chn, out_chn, mode="add"):
        super(DenseCat, self).__init__()
        self.conv1 = ConvBlock(in_chn, in_chn)
        self.conv2 = ConvBlock(in_chn, in_chn)
        self.conv_out = ConvBlock(in_chn, out_chn, kernel_size=1, padding=0)
        self.mode = mode

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x + x1)
        y1 = self.conv1(y)
        y2 = self.conv2(y + y1)

        if self.mode == "add":
            out = x1 + x2 + y1 + y2
        elif self.mode == "diff":
            out = torch.abs(x1 + x2 - y1 - y2)
        else:
            raise ValueError("Invalid mode. Use 'add' or 'diff'.")

        return self.conv_out(out)

class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        self.reduction = ConvBlock(dim_in, dim_in // 2, kernel_size=1, padding=0) if reduction else None
        self.cat_add = DenseCat(dim_in // 2 if reduction else dim_in, dim_out, mode="add")
        self.cat_diff = DenseCat(dim_in // 2 if reduction else dim_in, dim_out, mode="diff")
        self.conv_final = ConvBlock(dim_out, dim_out)

    def forward(self, x1, x2):
        if self.reduction:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat_add(x1, x2)
        x_diff = self.cat_diff(x1, x2)
        return self.conv_final(x_diff) + x_add

if __name__ == "__main__":
    from thop import profile

    model = DF_Module(64, 64, True)
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    y = model(x1, x2)
    flops, params = profile(model, inputs=(x1, x2))
    print(f"FLOPs: {flops}, Params: {params}")
