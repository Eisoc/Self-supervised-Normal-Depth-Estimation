import torch

# 定义一个三分类交叉熵损失函数
def loss_func(y_pred, y, train=True):
    if train:
        weights = torch.tensor([0, 1, 1])
    else:
        weights = torch.ones(3) # 均匀权重
    weights = weights.view(1, 3, 1, 1).to('cuda')
    return torch.mean(-torch.sum(torch.mul(y * torch.log(y_pred + 1e-10), weights), dim=3)) 