from .base_block import *

class CENet(nn.Module):
    def __init__(self, in_channels, n_classes=2, attention=None):
        super(CENet, self).__init__()
        self.attention = attention

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

        # Attention
        if attention=='MPR':
            self.a1 = MPRAttention(48, 32, 64, 64)
            self.a2 = MPRAttention(48, 16, 32, 32)
        elif attention=='Original':
            self.a1 = OriginalAttention(48, 32, 64, 64)
            self.a2 = OriginalAttention(48, 16, 32, 32)

        # Contour Transition
        #self.contour = DTLayer(48)

        # Shape Transition
        self.shape1 = DTLayer(48)
        self.shape2 = DTLayer(48)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear')

        # out transition layer
        #self.out1 = OTLayer(50, 48)
        self.out2 = OTLayer(50, n_classes)

    def forward(self, x):
        base_output, contour_input, shape_input = self.BaseNet(x)
        #contour_output = self.contour(contour_input)
        shape_output1 = self.shape1(shape_input)
        shape_output2 = self.shape2(shape_input)
        shape_output = shape_output1 - shape_output2
        #out1 = self.out1(torch.cat((base_output, contour_output), 1))
        out = self.out2(torch.cat((base_output, self.upsample3(shape_output)), 1))

        return F.softmax(out), F.softmax(shape_output)

    def BaseNet(self, x):
        o1 = self.dblock1(x)
        o1_down = self.downtransition1(o1)
        o2 = self.dblock2(o1_down)
        o2_down = self.downtransition2(o2)
        o3 = self.dblock3(o2_down)
        o3_down = self.downtransition3(o3)
        o4 = self.dblock4(o3_down)

        # attention
        if self.attention is not None:
            o3_att = self.a2(o4)
            o2_att = self.a1(o2)
            o3 = o3_att + o3
            o2 = o2_att + o2

        o4_up = self.uptransition1(o4)
        o5 = self.dblock5(o3 + o4_up)
        o5_up = self.uptransition2(o5)
        o6 = self.dblock6(o2 + o5_up)
        o7 = self.dblock7(o1)
        o_cat = torch.cat((self.upsample1(o5), self.upsample2(o6), o7), 1)
        o8 = self.dblock8(o_cat)
        return o8, o1, o6 # output, contour, shape

if __name__=="__main__":
    from torchsummary import summary
    device = "cuda"
    model = CENet(1)
    model.to(device)
    summary(model, input_size=(1, 64, 128, 128))