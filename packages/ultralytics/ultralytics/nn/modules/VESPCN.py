import torch
import torch.nn as nn
import torch.nn.functional as F

# class MotionEstimation(nn.Module):
#     def __init__(self):
#         super(MotionEstimation, self).__init__()
#         # Example of a small convolutional network for motion estimation
#         self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)  # Output flow (u, v)

#     def forward(self, img1, img2):
#         # Concatenate two frames along channel dimension
#         x = torch.cat([img1, img2], dim=1)
#         flow = self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
#         return flow

# class MotionEstimation(nn.Module):
#     def __init__(self):
#         super(MotionEstimation, self).__init__()
#         self.conv1 = nn.Conv2d(1024, 64, kernel_size=3, padding=1)  # 修改为1024输入通道
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 2, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         flow = self.conv3(x)
#         return flow

class MotionEstimation(nn.Module):
    def __init__(self):
        super(MotionEstimation, self).__init__()
        self.conv1 = nn.Conv3d(512, 64, kernel_size=3, padding=1)  # 输入通道改为 512
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        flow = self.conv3(x)
        return flow



class Warp(nn.Module):
    def __init__(self):
        super(Warp, self).__init__()

    def forward(self, img, flow):
        # Warp the image based on the flow
        B, C, H, W = img.size()
        grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))
        grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1)

        if img.is_cuda:
            grid_x, grid_y = grid_x.cuda(), grid_y.cuda()

        flow_x, flow_y = flow[:, 0, :, :], flow[:, 1, :, :]
        grid_x = grid_x + flow_x
        grid_y = grid_y + flow_y

        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=3)

        return F.grid_sample(img, grid, align_corners=True)

class SpatioTemporalESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(SpatioTemporalESPCN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(32, upscale_factor ** 2, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

class VESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(VESPCN, self).__init__()
        self.motion_estimation = MotionEstimation()
        self.warp = Warp()
        self.spatio_temporal_espcn = SpatioTemporalESPCN(upscale_factor)

    def forward(self, frames):
        # Assume frames: [I_LR_t-1, I_LR_t, I_LR_t+1]
        frame_t_minus_1, frame_t, frame_t_plus_1 = frames

        # Motion Estimation and Compensation
        x = torch.cat([frame_t, frame_t_minus_1], dim=1)  # 按通道维度拼接
        flow_t_minus_1 = self.motion_estimation(x)  # 调用 motion_estimation
        # flow_t_minus_1 = self.motion_estimation(frame_t, frame_t_minus_1)
        x = torch.cat([frame_t, frame_t_plus_1], dim=1)  # 按通道维度拼接
        flow_t_plus_1 = self.motion_estimation(x)  # 调用 motion_estimation
        # flow_t_plus_1 = self.motion_estimation(frame_t, frame_t_plus_1)

        warped_t_minus_1 = self.warp(frame_t_minus_1, flow_t_minus_1)
        warped_t_plus_1 = self.warp(frame_t_plus_1, flow_t_plus_1)

        # Stack frames along the temporal dimension
        input_frames = torch.stack([warped_t_minus_1, frame_t, warped_t_plus_1], dim=2)

        # Spatio-temporal ESPCN
        sr_frame = self.spatio_temporal_espcn(input_frames)

        return sr_frame
