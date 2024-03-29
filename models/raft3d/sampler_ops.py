import torch
import torch.nn.functional as F

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def depth_sampler(depths, coords):
    # depths 是一个包含深度值的张量，维度为 [batch_size, height, width]
    # coords 是一个包含坐标信息的张量，维度为 [batch_size, height, width, 2]，坐标信息是在图像 I2 中的像素坐标
    depths_proj, valid = bilinear_sampler(depths[:,None], coords, mask=True) # 双线性插值提取深度
    return depths_proj.squeeze(dim=1), valid 
    # depths_proj 是从depths中采样得到的深度值，维度为[batch_size, height, width]，通过 squeeze 函数去除了维度为1的维度