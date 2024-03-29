import torch.nn as nn
import torch.nn.init as init

from utils.motion_split.build_unit import build_units


class SemanticFeatureNet(nn.Module):
    def __init__(self):
        super(SemanticFeatureNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        options = [
            ('2', [64, 64, 256, 1]), ('1', [256, 64, 256]),
            ('1', [256, 64, 256]), ('2', [256, 256, 512, 2]),
            ('1', [512, 128, 512]), ('1', [512, 128, 512]),
            ('3', [512, 128, 512, 64, 1, 2]), ('2', [512, 768, 1024, 2]),
            ('1', [1024, 256, 1024]), ('3', [1024, 256, 1024, 256, 1, 2]),
            ('3', [1024, 256, 1024, 256, 1, 4]), ('3', [1024, 256, 1024, 256, 1, 8]),
            ('3', [1024, 256, 1024, 256, 1, 16]), ('4', [1024, 512, 2048, 256, 2, 4]),
            ('3', [2048, 512, 2048, 512, 2, 8]), ('3', [2048, 512, 2048, 512, 2, 16])
        ]

        self.layer3 = nn.Sequential(*build_units(options))

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([
            self.layer1, self.layer2, self.layer3, self.layer4
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

    def forward(self, image1):
        semantic_feature = self.layer1(image1)
        semantic_feature = self.layer2(semantic_feature)
        semantic_feature = self.layer3(semantic_feature)    
        return semantic_feature