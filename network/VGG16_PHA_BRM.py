"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class VGG16_PHA_BRM_GF1(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16_PHA_BRM_GF1, self).__init__()

        self.stage1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))

        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))

        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))

        self.stage4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.stage5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.aspp = _ASPP(512, 512, [6,12,18])

        self.decoder2 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.decoder1 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1),
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

        self.bd_extract1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.bd_extract2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.bd_extract3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True))

        self.score_dsn1 = nn.Conv2d(128, 1, kernel_size=1)
        self.score_dsn2 = nn.Conv2d(256, 1, kernel_size=1)
        self.score_dsn3 = nn.Conv2d(512, 1, kernel_size=1)

        self.score_boundary_fusion = nn.Conv2d(3, 1, kernel_size=1)

        self.concater = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True))

        self.segmenter =  nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder,input shape is 1*3*320*320
        b, c, h, w = x.size()[0], x.size()[1], x.size()[2], x.size()[3]

        x = self.stage1(x)       # 64,  320, 320
        x = self.Maxpool(x)      # 64,  160, 160

        e1 = self.stage2(x)      # 128, 160, 160
        e1 = self.Maxpool(e1)    # 128, 80,  80

        e2 = self.stage3(e1)     # 256, 80,  80
        e2 = self.Maxpool(e2)    # 256, 40,  40

        e3 = self.stage4(e2)     # 512, 40,  40
        e4 = self.stage5(e3)     # 512, 40,  40

        # Center
        center = self.aspp(e4)   # 512, 40,  40

        # Decoder
        c_up  = F.interpolate(center, size=(int(h/8), int(w/8)), mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([c_up, e2], dim=1))     # 256,  40,  40
        d2_up = F.interpolate(d2, size=(int(h/4), int(w/4)), mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2_up, e1], dim=1))    #  64,  80,  80
        d1_up = F.interpolate(d1, size=(int(h/2), int(w/2)), mode='bilinear', align_corners=False)
        d0 = self.decoder0(torch.cat([d1_up, x], dim=1) )    #  16, 160, 160


        # EdgeNet Block ==> 2x
        bd_ex1 = torch.cat([self.bd_extract1(x),  x], dim=1)     # 128, 160, 160
        bd_ex2 = torch.cat([self.bd_extract2(e1), e1], dim=1)    # 256,  80, 80
        bd_ex3 = torch.cat([self.bd_extract3(e2), e2], dim=1)    # 512, 40, 40
        so1 = self.score_dsn1(bd_ex1)                            # 1, 160, 160
        so2 = self.score_dsn2(bd_ex2)                            # 1,  80,  80
        so3 = self.score_dsn3(bd_ex3)                            # 1,  40,  40
        so2 = F.interpolate(so2, size=(so1.size()[2], so1.size()[3]), mode='bilinear', align_corners=False) # 1,160,160
        so3 = F.interpolate(so3, size=(so1.size()[2], so1.size()[3]), mode='bilinear', align_corners=False) # 1,160,160

        boundary_fusion = self.score_boundary_fusion(torch.cat((so1, so2, so3), dim=1))     # 1,  160, 160

        concat = self.concater(d0 + boundary_fusion.repeat(1, d0.size()[1], 1, 1))          # 16, 160, 160

        boundary_up = F.interpolate(boundary_fusion, size=(h, w), mode='bilinear', align_corners=False)   # 1,320,320
        boundary_oup = torch.sigmoid(boundary_up)      # 1,320,320

        segment_up = F.interpolate(self.segmenter(concat),
                                   size=(h, w), mode='bilinear', align_corners=False) # 1,320,320
        segment_oup = torch.sigmoid(segment_up)        # 1,320,320

        return segment_oup, boundary_oup


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
    net=VGG16_PHA_BRM_GF1()
    print(net)
    segment_oup, boundary_oup=net.forward(input)
    print(segment_oup.size())
