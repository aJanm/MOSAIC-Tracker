import torch
import torch.nn as nn
 
 
class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()
 
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        # self.conv3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
        #     torch.nn.ReLU(inplace=True),
        # )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )
 
    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x+x1)
        # x3 = self.conv3(x2+x1)
 
        y1 = self.conv1(y)
        y2 = self.conv2(y+y1)
        # y3 = self.conv3(y2+y1)
 
        return self.conv_out(x1 + x2 + y1 + y2)
 
class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )
 
    # def forward(self, x, y):
    # #  x对应第一帧，y对应前一帧
    #     x1 = self.conv1(x).to(3)
    #     x2 = self.conv2(x+x1).to(3)
    #     # x3 = self.conv3(x2+x1)
 
    #     y1 = self.conv1(y).to(3)
    #     y2 = self.conv2(y+y1).to(3)
    #     # y3 = self.conv3(y2+y1)
    #     out = self.conv_out(torch.abs(x1 + x2  - y1 - y2 )).to(3)
    #     out = out + x
    #     return out
    def forward(self, x, y):
    #  x对应第一帧，y对应前一帧
        # print("多帧测试，FramesModule.py")
        x1 = self.conv1(x)
        x2 = self.conv2(x+x1)
        # x3 = self.conv3(x2+x1)
 
        y1 = self.conv1(y)
        y2 = self.conv2(y+y1)
        # y3 = self.conv3(y2+y1)
        out = self.conv_out(torch.abs(x1 + x2  - y1 - y2 ))
        # out = out + x  # 如果跑addModel3则需要把这行注释 2024年12月22日 否则报错
        return out
 
 
class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in//2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y
 
if __name__ =='__main__':
    from thop import profile
    model = DF_Module(64, 64, True)
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    y = model(x1, x2)
    flops, params = profile(model, inputs=(x1, x2))
    print(f"FLOPs: {flops}, Params: {params}")