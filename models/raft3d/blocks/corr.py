import math
import torch
import torch.nn.functional as F

import lietorch_extras


class CorrSampler(torch.autograd.Function):
    """ Index from correlation pyramid """
    @staticmethod # 静态方法
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords) 
        # Volume: (b, h1, w1, h2, w2)表示I_1的每个像素和I_2的每个像素的相关性
        # Coords: (b, 2, h1, w1)表示I_1的每个像素，对应到I_2上的坐标
        ctx.radius = radius
        # Radius: 查询时的像素半径
        corr, = lietorch_extras.corr_index_forward(volume, coords, radius)
        return corr
        # 输出corr: (b, 2*radius+1, 2*radius+1, h1, w1)表示I_1的每个像素（h1, w1），与其对应到I_2上的坐标（coords）的一个邻域范围内（2*radius+1, 2*radius+1)的像素点的相关性

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = lietorch_extras.corr_index_backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2) # 对两图特征使用矩阵乘法得到相关性表

        batch, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, h2, w2) # (b,h,w,1,h,w) --> (bhw,1,h,w)
        
        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch, h1, w1, h2//2**i, w2//2**i)) 
            corr = F.avg_pool2d(corr, 2, stride=2) # 使用平均池化的方式获得多尺度查找表
            
    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i], coords/2**i, self.radius)
            out_pyramid.append(corr.view(bz, -1, ht, wd))

        return torch.cat(out_pyramid, dim=1)
        # 四个分辨率得到四个结果，拼在一起(b, 4* (2*randius+1) * (2*randius+1), h1, w1)

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd) / 4.0 # 第一帧图特征 (b,c,g,w) --> (bc,h,w)
        fmap2 = fmap2.view(batch, dim, ht*wd) / 4.0 # 第二帧图特征 (b,c,g,w) --> (bc,h,w)
        corr = torch.matmul(fmap1.transpose(1,2), fmap2) # (b,hw,c)*(b,c,hw) --> (b,hw,hw)后两维使用矩阵乘法，第一位由广播得到
        return corr.view(batch, ht, wd, ht, wd) # (b,hw,hw) --> (b,h,w,1,h,w)

