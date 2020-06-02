import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, in_channels, n_classes=2):
        super(Network, self).__init__()
        # Base Network
        self.dblock1 = DBlock(in_channels)
        self.downtransition1 = DownTransition(48, 48)
        self.dblock2 = DBlock(48)
        self.downtransition2 = DownTransition(48, 48)
        self.dblock3 = DBlock(48)
        self.downtransition3 = DownTransition(48, 48)
        self.dblock4 = DBlock(48)
        self.fam1 = FAM(48, 48)
        self.uptransition1 = UpTransition(48, 48)
        self.fam2 = FAM(48, 48)
        self.uptransition2 = UpTransition(48, 48)
        self.fam3 = FAM(48, 48)
        self.uptransition3 = UpTransition(48, 48)
        self.fam4 = FAM(48, 48)

        self.ppm = PPM(48)

        self.F1 = nn.Sequential(nn.Conv3d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm3d(48),
                                nn.ReLU(inplace=True))
        self.F2 = nn.Sequential(nn.Conv3d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm3d(48),
                                nn.ReLU(inplace=True))
        self.F3 = nn.Sequential(nn.Conv3d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm3d(48),
                                nn.ReLU(inplace=True))
        self.F4 = nn.Sequential(nn.Conv3d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm3d(48),
                                nn.ReLU(inplace=True))

        #self.upsample_x2 = nn.Upsample(scale_factor=2, mode='trilinear')
        #self.upsample_x4 = nn.Upsample(scale_factor=4, mode='trilinear')
        #self.upsample_x8 = nn.Upsample(scale_factor=8, mode='trilinear')

        # Contour Transition
        self.contour = DTLayer(48)

        # Shape Transition
        self.shape1 = DTLayer(48)
        self.shape2 = DTLayer(48)

        # out transition layer
        self.out = OTLayer(50, n_classes)

    def forward(self, x):
        base_output, contour_input, shape_input = self.BaseNet(x)

        contour_size = contour_input[-1].size()
        contour_outputs = []
        for i in range(len(contour_input)):
            contour_outputs.append(
                F.softmax(self.contour(F.interpolate(contour_input[i], contour_size[2:], mode='trilinear', align_corners=True))))

        shape_output1 = self.shape1(shape_input)
        shape_output2 = self.shape2(shape_input)
        shape_output = shape_output1 - shape_output2

        out = self.out(torch.cat((base_output, F.interpolate(shape_output, base_output.size()[2:], mode='trilinear', align_corners=True)), 1))

        return F.softmax(out), contour_outputs, F.softmax(shape_output)

    def BaseNet(self, x):
        o1 = self.dblock1(x)
        o1_down = self.downtransition1(o1)
        o2 = self.dblock2(o1_down)
        o2_down = self.downtransition2(o2)
        o3 = self.dblock3(o2_down)
        o3_down = self.downtransition3(o3)
        o4 = self.dblock4(o3_down)

        ppm = self.ppm(o4)

        o5 = self.fam1(o4)
        o5_F = self.F1(self.uptransition1(o5) + F.interpolate(ppm, o3.size()[2:], mode='trilinear', align_corners=True) + o3)
        o6 = self.fam2(o5_F)
        o6_F = self.F2(self.uptransition2(o6) + F.interpolate(ppm, o2.size()[2:], mode='trilinear', align_corners=True) + o2)
        o7 = self.fam3(o6_F)
        o7_F = self.F3(self.uptransition3(o7) + F.interpolate(ppm, o1.size()[2:], mode='trilinear', align_corners=True) + o1)
        o8 = self.fam4(o7_F)
        o8_F = self.F4(o8)

        return o8_F, [o6, o7, o8], o7 # output, contour, shape

class DBlock(nn.Module): # DBlock in base network
    def __init__(self, in_channels, n=4, k=16):
        super(DBlock, self).__init__()

        self.conv1 = nn.Sequential(SeparableConv(in_channels, n, k),
                                   nn.BatchNorm3d(k),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(SeparableConv(k, n, k),
                                   nn.BatchNorm3d(k),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(SeparableConv(2*k, n, k),
                                   nn.BatchNorm3d(k),
                                   nn.ReLU(inplace=True))

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
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(20, 20, kernel_size=5, stride=1, padding=2, dilation=1),
                                   nn.BatchNorm3d(20),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(20, 20, kernel_size=5, stride=1, padding=4, dilation=2),
                                   nn.BatchNorm3d(20),
                                   nn.ReLU(inplace=True))
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
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels//2, in_channels//2, kernel_size=5, stride=1, padding=2, dilation=1),
                                   nn.BatchNorm3d(in_channels//2),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(in_channels//2, in_channels//2, kernel_size=5, stride=1, padding=4, dilation=2),
                                   nn.BatchNorm3d(in_channels//2),
                                   nn.ReLU(inplace=True))
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
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        return self.down_conv(x)

class UpTransition(nn.Module): # Up Transition layer in base network
    def __init__(self, in_channels, out_channels):
        super(UpTransition, self).__init__()
        self.up_conv = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
                                     nn.BatchNorm3d(out_channels),
                                     nn.ReLU(inplace=True))

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

if __name__=="__main__":
    from torchsummary import summary
    device = "cuda"
    model = Network(1)
    model.to(device)
    summary(model, input_size=(1, 64, 128, 128))