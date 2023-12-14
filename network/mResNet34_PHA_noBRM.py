"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from network import resnet
from network import mresnet

nonlinearity = partial(F.relu, inplace=True)

class mResNet34_PHA_noBRM(nn.Module):
    def __init__(self, num_classes=2):
        super(mResNet34_PHA_noBRM, self).__init__()

        backbone = mresnet.mresnet34(pretrained=True)

        self.firstconv = backbone.conv1_1
        self.firstbn = backbone.bn1
        self.firstrelu = backbone.relu
        self.firstmaxpool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.aspp = _ASPP(512, 512, [6,12,18])

        self.decoder2 = nn.Sequential(nn.Conv2d(640, 512, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.decoder1 = nn.Sequential(nn.Conv2d(320, 256, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 64, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.decoder0 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 16, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 16, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True))


        self.segmenter =  nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder,input shape is 1*3*320*320
        b, c, h, w = x.size()[0], x.size()[1], x.size()[2], x.size()[3]
        x = self.firstconv(x)           # 64,  160, 160
        x = self.firstbn(x)             # 64,  160, 160
        x = self.firstrelu(x)           # 64,  160, 160
        x_max = self.firstmaxpool(x)    # 64,   80,  80
        e1 = self.encoder1(x_max)       # 64,   80,  80
        e2 = self.encoder2(e1)          # 128,  40,  40
        e3 = self.encoder3(e2)          # 256,  40,  40
        e4 = self.encoder4(e3)          # 512,  40,  40

        # Center
        center = self.aspp(e4)          # 512,  40,  40

        # Decoder
        c_up  = F.interpolate(center, size=(int(h/8), int(w/8)), mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([c_up, e2], dim=1))     # 256,  40,  40
        d2_up = F.interpolate(d2, size=(int(h/4), int(w/4)), mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2_up, e1], dim=1))    #  64,  80,  80
        d1_up = F.interpolate(d1, size=(int(h/2), int(w/2)), mode='bilinear', align_corners=False)
        d0 = self.decoder0(torch.cat([d1_up, x], dim=1) )    #  16, 160, 160
        d0_up = F.interpolate(d0,size=(h, w), mode='bilinear', align_corners=False) # 1,320,320

        segment_oup = torch.sigmoid(self.segmenter(d0_up))        # 1,320,320

        return segment_oup


class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(_ASPP, self).__init__()

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3)
        self.b4 = _AsppPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))


if __name__=="__main__":
    input=torch.ones(16,4,320,320)
    net=mResNet34_PHA_noBRM()
    print(net)
    segment_oup, boundary_oup=net.forward(input)
    print(segment_oup.size())
