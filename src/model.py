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
        self.uptransition1 = UpTransition(48, 48)
        self.dblock5 = DBlock(48)
        self.uptransition2 = UpTransition(48, 48)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.dblock6 = DBlock(48)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.dblock7 = DBlock(48)
        self.dblock8 = DBlock(144)

        # Contour Transition
        self.contour = DTLayer(48)

        # Shape Transition
        self.shape1 = DTLayer(48)
        self.shape2 = DTLayer(48)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear')

        # out transition layer
        self.out1 = OTLayer(48, 48)
        self.out2 = OTLayer(50, n_classes)

    def forward(self, x):
        base_output, contour_input, shape_input = self.BaseNet(x)
        contour_output = self.contour(contour_input)
        shape_output1 = self.shape1(shape_input)
        shape_output2 = self.shape2(shape_input)
        shape_output = shape_output1 - shape_output2
        out1 = self.out1(base_output)
        out = self.out2(torch.cat((out1, self.upsample3(shape_output)), 1))

        return F.softmax(out), F.softmax(contour_output), F.softmax(shape_output)

    def BaseNet(self, x):
        o1 = self.dblock1(x)
        o1_down = self.downtransition1(o1)
        o2 = self.dblock2(o1_down)
        o2_down = self.downtransition2(o2)
        o3 = self.dblock3(o2_down)
        o3_down = self.downtransition3(o3)
        o4 = self.dblock4(o3_down)
        o4_up = self.uptransition1(o4)
        o5 = self.dblock5(o3+o4_up)
        o5_up = self.uptransition2(o5)
        o6 = self.dblock6(o2+o5_up)
        o7 = self.dblock7(o1)
        o_cat = torch.cat((self.upsample1(o5), self.upsample2(o6), o7), 1)
        o8 = self.dblock8(o_cat)
        return o8, o1, o6 # output, contour, shape

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

if __name__=="__main__":
    from torchsummary import summary
    device = "cuda"
    model = Network(1)
    model.to(device)
    summary(model, input_size=(1, 64, 128, 128))