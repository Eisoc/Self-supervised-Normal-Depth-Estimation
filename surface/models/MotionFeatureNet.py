import torch.nn as nn
import torch.nn.init as init

from utils.motion_split.build_unit import build_units


class MotionFeatureNet(nn.Module):
    def __init__(self):
        super(MotionFeatureNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  

        options = [
            ('2', [64, 64, 256, 1]),
            ('1', [256, 64, 256]),
            ('1', [256, 64, 256]),
            ('2', [256, 256, 512, 2]),
            ('1', [512, 128, 512]),
            ('2', [512, 256, 512, 2])
        ]

        self.layer3 = nn.Sequential(*build_units(options))
        self.layers = nn.ModuleList([
            self.layer1, self.layer2, self.layer3
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, flow): 
        motion_feature = self.layer1(flow)
        motion_feature = self.layer2(motion_feature)
        motion_feature = self.layer3(motion_feature)

        return motion_feature