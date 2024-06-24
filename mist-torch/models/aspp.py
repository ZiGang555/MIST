import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomASPPModule(nn.Module):
    def __init__(self, in_dim=2, edge_dim=2, output_dim=2, rates=[6, 12, 18]):
        super(CustomASPPModule, self).__init__()

        self.features = []
        # 可以考虑使用和输出维度相同的reduction_dim，这里直接使用输出维度作为中间处理的通道数
        reduction_dim = output_dim

        # 1x1卷积，处理主输入特征
        self.features.append(
            nn.Sequential(nn.Conv3d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.BatchNorm3d(reduction_dim), nn.ReLU(inplace=True)))

        # 不同膨胀率的3x3卷积，处理主输入特征
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv3d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False),
                # nn.BatchNorm3d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # 图像级特征
        self.img_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.img_conv = nn.Sequential(
            nn.Conv3d(in_dim, reduction_dim, kernel_size=1, bias=False),
            # nn.BatchNorm3d(reduction_dim), nn.ReLU(inplace=True)
            )

        # 边缘特征处理
        self.edge_conv = nn.Sequential(
            nn.Conv3d(edge_dim, reduction_dim, kernel_size=1, bias=False),
            # nn.BatchNorm3d(reduction_dim), nn.ReLU(inplace=True)
            )

        # 输出层不需要额外的卷积层来减少通道数，因为我们已经确保了所有特征的通道数与输出维度一致
        self.final_conv = nn.Conv3d(6 * reduction_dim, output_dim, kernel_size=1, bias=False)

    def forward(self, x, edge):
        x_size = x.size()

        # 图像级特征
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:], mode='trilinear', align_corners=True)

        # 边缘特征
        edge = self.img_pooling(edge)
        edge_features = self.edge_conv(edge)
        edge_features = F.interpolate(edge_features, x_size[2:], mode='trilinear', align_corners=True)
        

        # 合并所有特征
        out = img_features
        out = torch.cat((out, edge_features), 1)
        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)

        #
        out = self.final_conv(out)
        return out
