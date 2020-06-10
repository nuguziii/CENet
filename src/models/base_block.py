import torch.nn as nn
from torch.nn import functional as F
import torch

class DBlock(nn.Module): # DBlock in base network
    def __init__(self, in_channels, n=4, k=16):
        super(DBlock, self).__init__()

        self.conv1 = nn.Sequential(SeparableConv(in_channels, n, k),
                                   nn.BatchNorm3d(k),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(SeparableConv(k, n, k),
                                   nn.BatchNorm3d(k),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(SeparableConv(2*k, n, k),
                                   nn.BatchNorm3d(k),
                                   nn.ReLU())

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(torch.cat((out1, out2), 1))
        out = torch.cat((out1, out2, out3), 1)
        return out

class SeparableConv(nn.Module): # separable convolution in DBlock
    def __init__(self, in_channels, n, k):
        super(SeparableConv, self).__init__()
        if in_channels <= 1:
            self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        else:
            self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                                   groups=int(in_channels / n), bias=True)
        self.pointwise = nn.Conv3d(in_channels, k, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class DTLayer(nn.Module): # deep transition layer (shape and contour)
    def __init__(self, in_channels):
        super(DTLayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 10, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm3d(10),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(20, 20, kernel_size=5, stride=1, padding=2, dilation=1),
                                   nn.BatchNorm3d(20),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(20, 20, kernel_size=5, stride=1, padding=4, dilation=2),
                                   nn.BatchNorm3d(20),
                                   nn.ReLU())
        self.downtransition = DownTransition(10, 20)
        self.uptransition = UpTransition(40, 10)
        self.conv4 = nn.Conv3d(20, 2, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out1_down = self.downtransition(out1)
        out2 = self.conv2(out1_down)
        out3 = self.conv3(out2)
        out4 = torch.cat((out2, out3), 1)
        out4_up = self.uptransition(out4)
        out5 = torch.cat((out1, out4_up), 1)
        out = self.conv4(out5)
        return out

class OTLayer(nn.Module): # out-transition layer
    def __init__(self, in_channels, out_channels):
        super(OTLayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm3d(in_channels//2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels//2, in_channels//2, kernel_size=5, stride=1, padding=2, dilation=1),
                                   nn.BatchNorm3d(in_channels//2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(in_channels//2, in_channels//2, kernel_size=5, stride=1, padding=4, dilation=2),
                                   nn.BatchNorm3d(in_channels//2),
                                   nn.ReLU())
        self.conv4 = nn.Conv3d(3*(in_channels//2), out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = torch.cat((out1, out2, out3), 1)
        out = self.conv4(out4)
        return out

class DownTransition(nn.Module): # Down Transition layer in base network
    def __init__(self, in_channels, out_channels):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
                                       nn.BatchNorm3d(out_channels),
                                       nn.ReLU())

    def forward(self, x):
        return self.down_conv(x)

class UpTransition(nn.Module): # Up Transition layer in base network
    def __init__(self, in_channels, out_channels):
        super(UpTransition, self).__init__()
        self.up_conv = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
                                     nn.BatchNorm3d(out_channels),
                                     nn.ReLU())

    def forward(self, x):
        return self.up_conv(x)

class PPM(nn.Module):
    def __init__(self, in_channels):
        super(PPM, self).__init__()
        ppms = []
        for ii in [1, 3, 5]:
            ppms.append(
                nn.Sequential(nn.AdaptiveAvgPool3d(ii),
                              nn.Conv3d(in_channels, in_channels, 1, 1, bias=False),
                              nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)
        self.ppm_cat = nn.Sequential(nn.Conv3d(in_channels * 4, in_channels, 3, 1, 1, bias=False),
                                     nn.ReLU(inplace=True))
    def forward(self, x):
        xls = [x]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](x), x.size()[2:], mode='trilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))
        return xls

class FAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FAM, self).__init__()
        self.pools_sizes = [2, 4, 8]
        pools, convs = [], []
        for i in self.pools_sizes:
            pools.append(nn.AvgPool3d(kernel_size=i, stride=i))
            convs.append(nn.Conv3d(in_channels, in_channels, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='trilinear', align_corners=True))
        resl = self.relu(resl)
        resl = self.conv_sum(resl)
        return resl

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(),
                                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                                nn.ReLU(),
                                nn.Linear(gate_channels // reduction_ratio, gate_channels))
    def forward(self, x):
        avg_pool = F.avg_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
        channel_att_raw_avg = self.mlp( avg_pool )

        max_pool = F.max_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
        channel_att_raw_max = self.mlp( max_pool )

        channel_att_sum = channel_att_raw_avg + channel_att_raw_max

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

class MPRAttention(nn.Module):
    def __init__(self, c, d, h, w):
        super(MPRAttention, self).__init__()
        self.ChannelGate = ChannelGate(c)
        self.D_conv = nn.Sequential(nn.Conv2d(in_channels=c * d, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(c),
                                    nn.ReLU())
        self.H_conv = nn.Sequential(nn.Conv2d(in_channels=c * h, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(c),
                                    nn.ReLU())
        self.W_conv = nn.Sequential(nn.Conv2d(in_channels=c * w, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(c),
                                    nn.ReLU())

        self.compress = ChannelPool()
        self.conv = nn.Sequential(nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False),
                                  nn.BatchNorm3d(1),
                                  nn.ReLU())

    def forward(self, x):
        b, c, d, w, h = x.size()
        x = self.ChannelGate(x)

        d_out = self.D_conv(x.reshape(-1, c * d, w, h)).reshape(-1, c, 1, w, h)
        h_out = self.H_conv(torch.transpose(x, 2, 4).reshape(-1, c * h, w, d)).transpose(2, 3).reshape(-1, c, d, w, 1)
        w_out = self.W_conv(torch.transpose(x, 2, 3).reshape(-1, c * w, d, h)).reshape(-1, c, d, 1, h)
        out = d_out + h_out + w_out

        out = self.compress(out)
        out = F.sigmoid(self.conv(out))
        return out * x

class OriginalAttention(nn.Module):
    def __init__(self, c, d, h, w):
        super(OriginalAttention, self).__init__()
        self.ChannelGate = ChannelGate(c)
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm3d(c),
                                   nn.ReLU())
        self.compress = ChannelPool()
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False),
                                  nn.BatchNorm3d(1),
                                  nn.ReLU())

    def forward(self, x):
        x = self.ChannelGate(x)
        out = self.compress(self.conv1(x))
        out = F.sigmoid(self.conv2(out))
        return out * x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )