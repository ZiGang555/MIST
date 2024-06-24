import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionFusionModule, self).__init__()
        self.gaussian_blur = GaussianBlur(3, 0.5)
        # 通道注意力模块
        self.channel_attention = ChannelAttention(channels)
        # 空间注意力模块
        self.spatial_attention = SpatialAttention()
        # 融合后的特征处理
        self.conv_fusion = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, ct_features, mask_features):
        # 计算注意力权重
        ct_channel_attention = self.channel_attention(ct_features)
        mask_channel_attention = self.channel_attention(mask_features)
        
        ct_spatial_attention = self.spatial_attention(ct_features)
        mask_spatial_attention = self.spatial_attention(mask_features)
        
        # 应用注意力权重
        ct_weighted = ct_features * ct_channel_attention * ct_spatial_attention
        mask_weighted = mask_features * mask_channel_attention * mask_spatial_attention
        
        # 融合特征
        fused_features = torch.cat([ct_weighted, mask_weighted], dim=1)
        
        # 后处理
        output = self.conv_fusion(fused_features)
        
        return output

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=1):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.weight = self.get_gaussian_kernel(kernel_size, sigma)

    def get_gaussian_kernel(self, kernel_size, sigma):
        # 生成高斯核
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / 
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x):
        # 应用高斯模糊
        gaussian_kernel = self.weight.expand(x.size(1), 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, gaussian_kernel, padding=self.padding, groups=x.size(1))
        return x
