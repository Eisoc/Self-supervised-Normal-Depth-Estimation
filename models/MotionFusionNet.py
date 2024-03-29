import torch
import torch.nn as nn
import torch.nn.init as init

from .MotionFeatureNet import MotionFeatureNet
from .SemanticFeatureNet import SemanticFeatureNet
from utils.motion_split.build_unit import build_units


class MotionFusionNet(nn.Module):
    def __init__(self):
        super(MotionFusionNet, self).__init__()

        self.motionFeatureNet = MotionFeatureNet() 
        self.semanticFeatureNet = SemanticFeatureNet() 

        # 前两个网络输出的通道数
        C1 = 512 
        C2 = 2048

        options = [
            ('2', [C1 + C2, 128, 512, 1]), ('1', [512, 128, 512]),
            ('1', [512, 128, 512]), ('4', [512, 256, 1024, 128, 1, 2]),
            ('3', [1024, 256, 1024, 128, 1, 4]), ('3', [1024, 256, 1024, 128, 1, 8]),
            ('4', [1024, 512, 2048, 256, 4, 16]), ('3', [2048, 512, 2048, 256, 4, 16])
        ]

        self.layer1 = nn.Sequential(*build_units(options))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=3, kernel_size=(16, 16), stride=(16, 16)),
            nn.BatchNorm2d(3)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
 
    def forward(self, image, flow):
        out_motion = self.motionFeatureNet(flow)
        out_semantic = self.semanticFeatureNet(image)

        fusion = torch.cat((out_motion, out_semantic), dim=1)
        fusion = self.layer1(fusion)
        fusion = self.layer2(fusion)
        fusion = self.layer3(fusion)
        return fusion