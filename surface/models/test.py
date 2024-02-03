
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
import DispNetS

# 插入您之前定义的辅助函数（downsample_conv, predict_disp, conv, upconv, crop_like）和 DispNetS 类定义

# 初始化 DispNetS 网络
disp_net = DispNetS.DispNetS()

# 将网络切换到评估模式
disp_net.eval()

# 创建一个测试张量，模拟您的输入数据
# 假设输入尺寸为 [4, 3, 128, 416]
test_input = torch.randn(4, 3, 128, 416)

# 将测试张量传递通过网络
test_output = disp_net(test_input)

# 打印输出尺寸
print("Output size:", test_output.size())