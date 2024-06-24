import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Convolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Curvature(torch.nn.Module):
    def __init__(self, ratio):
        super(Curvature, self).__init__()
        weights = torch.tensor([[[[-1/16, 5/16, -1/16], [5/16, -1, 5/16], [-1/16, 5/16, -1/16]]]])
        self.weight = torch.nn.Parameter(weights).cuda()
        self.ratio = ratio
 
    def forward(self, x):
        B, C, H, W = x.size()
        x_origin = x
        x = x.reshape(B*C,1,H,W)
        out = F.conv2d(x, self.weight)
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p=p.reshape(B, C)

        _, index = torch.topk(p, int(self.ratio*C), dim=1)
        selected = []
        for i in range(x_origin.shape[0]):
            selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)
        
        return selected
    

class Network(nn.Module):
    def __init__(self, in_ch=3, mode='ori', ratio=None):
        super(Network, self).__init__()
        self.mode = mode
        if self.mode == 'ori':
            self.ratio = [0,0]
        self.ratio = ratio
        self.ife1 = Curvature(self.ratio[0])
        self.ife2 = Curvature(self.ratio[1])

        # ---- U-Net ----
        self.conv1 = Convolution(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)  # feature map = shape(m/2,n/2,64)
        self.conv2 = Convolution(64, 128)
        self.pool2 = nn.MaxPool2d(2)  # feature map = shapem/4,n/4,128)
        self.conv3 = Convolution(128, 256)
        self.pool3 = nn.MaxPool2d(2)  # feature map = shape(m/8,n/8,256)
        self.conv4 = Convolution(256, 512)
        self.pool4 = nn.MaxPool2d(2)  # feature map = shape(m/16,n/16,512)

        self.conv5 = Convolution(512, 1024)  # feature map = shape(m/16,n/16,1024)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, output_padding=0)  
        self.conv6 = Convolution(1024, 512)  # feature map = shape(m/8,n/8,512)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, 2, 0, 0)
        self.conv7 = Convolution(int(256*(2+self.ratio[1])), 256)  # feature map = shape(m/4,n/4,256）
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, 2, 0, 0)
        self.conv8 = Convolution(int(128*(2+self.ratio[0])), 128)  # feature map = shape(m/2,n/2,128）
        self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, 2, 0, 0)
        self.conv9 = Convolution(128, 64)  # feature map = shape(m,n,64)

        self.out_conv1 = nn.Conv2d(64, 1, 1, 1, 0) 

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        if self.mode != 'ori':
            c2 = torch.cat([c2, self.ife1(c2)])
            c3 = torch.cat([c3, self.ife2(c3)])

        up1 = self.up_conv1(c5)
        merge1 = torch.cat([up1, c4], dim=1)
        c6 = self.conv6(merge1)
        up2 = self.up_conv2(c6)
        merge2 = torch.cat([up2, c3], dim=1)
        c7 = self.conv7(merge2)
        up3 = self.up_conv3(c7)
        merge3 = torch.cat([up3, c2], dim=1)
        c8 = self.conv8(merge3)
        up4 = self.up_conv4(c8)
        merge4 = torch.cat([up4, c1], dim=1)
        c9 = self.conv9(merge4)

        S_g_pred = self.out_conv1(c9) 

        return S_g_pred