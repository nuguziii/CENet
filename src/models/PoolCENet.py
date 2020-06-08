from .base_block import *

class PoolCENet(nn.Module):
    def __init__(self, in_channels, n_classes=2, attention=None):
        super(PoolCENet, self).__init__()
        self.attention = attention
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

        # Attention
        if attention == 'MPR':
            self.a1 = MPRAttention(48, 32, 64, 64)
            self.a2 = MPRAttention(48, 16, 32, 32)
        elif attention == 'Original':
            self.a1 = OriginalAttention(48, 32, 64, 64)
            self.a2 = OriginalAttention(48, 16, 32, 32)

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

        #contour_size = contour_input[-1].size()
        #contour_outputs = []
        #for i in range(len(contour_input)):
        #    contour_outputs.append(
        #        F.softmax(self.contour(F.interpolate(contour_input[i], contour_size[2:], mode='trilinear', align_corners=True))))

        shape_output1 = self.shape1(shape_input)
        shape_output2 = self.shape2(shape_input)
        shape_output = shape_output1 - shape_output2

        out = self.out(torch.cat((base_output, F.interpolate(shape_output, base_output.size()[2:], mode='trilinear', align_corners=True)), 1))

        return F.softmax(out), F.softmax(shape_output)

    def BaseNet(self, x):
        o1 = self.dblock1(x)
        o1_down = self.downtransition1(o1)
        o2 = self.dblock2(o1_down)
        o2_down = self.downtransition2(o2)
        o3 = self.dblock3(o2_down)
        o3_down = self.downtransition3(o3)
        o4 = self.dblock4(o3_down)

        ppm = self.ppm(o4)

        # attention
        if self.attention is not None:
            o3_att = self.a2(o3)
            o2_att = self.a1(o2)
            o3 = o3_att + o3
            o2 = o2_att + o2

        o5 = self.fam1(o4)
        o5_F = self.F1(self.uptransition1(o5) + F.interpolate(ppm, o3.size()[2:], mode='trilinear', align_corners=True) + o3)
        o6 = self.fam2(o5_F)
        o6_F = self.F2(self.uptransition2(o6) + F.interpolate(ppm, o2.size()[2:], mode='trilinear', align_corners=True) + o2)
        o7 = self.fam3(o6_F)
        o7_F = self.F3(self.uptransition3(o7) + F.interpolate(ppm, o1.size()[2:], mode='trilinear', align_corners=True) + o1)
        o8 = self.fam4(o7_F)
        o8_F = self.F4(o8)

        return o8_F, [o6, o7, o8], o7 # output, contour, shape

if __name__=="__main__":
    from torchsummary import summary
    device = "cuda"
    model = PoolCENet(1, attention='MPR')
    model.to(device)
    summary(model, input_size=(1, 64, 128, 128))