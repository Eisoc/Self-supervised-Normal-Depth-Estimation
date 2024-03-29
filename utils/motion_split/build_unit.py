from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet1(nn.Module):
    def __init__(self, options: List[int]):
        super(ResNet1, self).__init__()

        d0, d1, d2 = options

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=d0, out_channels=d1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        out_1 = self.layer_1(x)
        out_2 = x
        return F.relu(out_1 + out_2)


class ResNet2(nn.Module):
    def __init__(self, options: List[int]):
        super(ResNet2, self).__init__()

        d0, d1, d2, s = options

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=d0, out_channels=d1, kernel_size=1, stride=s, padding=0),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d2),
            nn.ReLU(inplace=True)
        )
        
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=d0, out_channels=d2, kernel_size=1, stride=s, padding=0),
            nn.BatchNorm2d(d2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_1 = self.layer_1(x)
        out_2 = self.layer_2(x)
        return F.relu(out_1 + out_2)

class ResNet3(nn.Module):
    def __init__(self, options: List[int]):
        super(ResNet3, self).__init__()

        d0, d1, d2, d3, p, d = options

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=d0, out_channels=d1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d3//2, kernel_size=3, stride=1, padding=p, dilation=p),
            nn.BatchNorm2d(d3//2),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d3//2, kernel_size=3, stride=1, padding=d, dilation=d),
            nn.BatchNorm2d(d3//2),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=d3, out_channels=d2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.initial_conv(x)

        out1 = self.branch1(x1)
        out2 = self.branch2(x1)

        concatenated = torch.cat((out1, out2), dim=1)

        up_out = self.final_conv(concatenated)
        down_out = x

        return F.relu(up_out + down_out)
    
class ResNet4(nn.Module):
    def __init__(self, options: List[int]):
        super(ResNet4, self).__init__()

        d0, d1, d2, d3, p, d = options

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=d0, out_channels=d1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=True)
        )

        # 上层的两个分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d3//2, kernel_size=3, stride=1, padding=p, dilation=p),
            nn.BatchNorm2d(d3//2),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d3//2, kernel_size=3, stride=1, padding=d, dilation=d),
            nn.BatchNorm2d(d3//2),
            nn.ReLU(inplace=True)
        )

        # 上层的最后的1x1卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=d3, out_channels=d2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d2),
            nn.ReLU(inplace=True)
        )
        
        # 下层
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels=d0, out_channels=d2, kernel_size=1, stride=1),
            nn.BatchNorm2d(d2)
        )
    def forward(self, x):
        x1 = self.initial_conv(x)

        # 上层的两个分支
        out1 = self.branch1(x1)
        out2 = self.branch2(x1)

        # 将两个分支的结果拼接起来
        concatenated = torch.cat((out1, out2), dim=1)

        up_out = self.final_conv(concatenated)
        
        # 下层不进行任何操作，直接使用输入x
        down_out = self.down_conv(x)

        return F.relu(up_out + down_out)

def build_resnet(type: str, options: List[int]) -> nn.Module:
    if type == '1':
        return ResNet1(options)
    elif type == '2':
        return ResNet2(options)
    elif type == '3':
        return ResNet3(options)
    elif type == '4':
        return ResNet4(options)
    else:
        raise NameError()

def build_units(options: List[Tuple[str, List[int]]]) -> List[nn.Module]:
    result = []
    for (type, opt) in options:
        result.append(build_resnet(type, opt))
    return result