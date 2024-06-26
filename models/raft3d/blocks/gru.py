import torch
import torch.nn as nn
import torch.nn.functional as F

# 1和2是两个方向？

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128, dilation=4):
        super(ConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.convz1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convz2 = nn.Conv2d(hidden_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

        self.convr1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convr2 = nn.Conv2d(hidden_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

        self.convq1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convq2 = nn.Conv2d(hidden_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

    def forward(self, h, *inputs): # h是隐藏状态，多个输入
        iz, ir, iq = 0, 0, 0
        for inp in inputs:
            inp = inp.split([self.hidden_dim]*3, dim=1) # 沿dim=1将inp分割成三个部分，每个部分大小为hidden_dim，结果是inp是一个包含三个张量的列表
            iz = iz + inp[0]
            ir = ir + inp[1]
            iq = iq + inp[2]

        z = torch.sigmoid(self.convz1(h) + self.convz2(h) + iz)
        r = torch.sigmoid(self.convr1(h) + self.convr2(h) + ir)
        q = torch.tanh(self.convq1(r*h) + self.convq2(r*h) + iq)

        h = (1-z) * h + z * q
        return h
