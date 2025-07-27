import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(TemporalAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        batch_size, channels, height, width = x1.size()

        # Compute query, key, value
        query = self.query_conv(x1).view(batch_size, -1, height * width)  # (B, C', H*W)
        key = self.key_conv(x2).view(batch_size, -1, height * width)      # (B, C', H*W)
        value = self.value_conv(x2).view(batch_size, -1, height * width)  # (B, C, H*W)

        # Attention map
        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=-1)  # (B, H*W, H*W)

        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, channels, height, width)  # Reshape back to (B, C, H, W)

        return self.gamma * out + x1


class TemporalFeatureEnhancement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalFeatureEnhancement, self).__init__()
        self.temporal_attention = TemporalAttention(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x1, x2):
        common = torch.add(x1, x2)
        # Temporal Attention
        attn_out = self.temporal_attention(x1, x2)
        
        # Temporal Difference and Fusion
        diff = torch.abs(x1 - x2)
        diff = F.relu(self.bn1(self.conv1(diff)))
        
        out = diff + attn_out
        out = F.relu(self.bn2(self.conv2(out)))
        # out = self.dropout(out)
        out = out + common # 2025年1月5日
        return out


if __name__ == "__main__":
    from thop import profile
    model = TemporalFeatureEnhancement(64, 64)
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    y = model(x1, x2)
    flops, params = profile(model, inputs=(x1, x2))
    print(f"FLOPs: {flops}, Params: {params}")
