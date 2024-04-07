import torch
import torch.nn.functional as F

from .sampler_ops import *

MIN_DEPTH = 0.05

def project(Xs, intrinsics):
    """ Pinhole camera projection
            针孔相机投影 """
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:,None,None].unbind(dim=-1)
    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = torch.stack([x, y, d], dim=-1)
    return coords

def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection 
            针孔相机反投影 """

    ht, wd = depths.shape[-2:]
    
    fx, fy, cx, cy = \
        intrinsics[:,None,None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(), 
        torch.arange(wd).to(depths.device).float())

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return torch.stack([X, Y, Z], dim=-1)

def projective_transform(Ts, depth, intrinsics): # Ts是一个表示变换矩阵的张量，用于i将点从I_1变换到I_2坐标系
    """ Project points from I1 to I2 
            将I_1上的点投影到I_2上 """
    
    X0 = inv_project(depth, intrinsics) # 用深度信息和内参，将点从I_1的像素坐标转换为三维坐标保存在X0
    X1 = Ts * X0 # 左乘T矩阵得到在图像I_2中的三维坐标
    x1 = project(X1, intrinsics) # 将三维坐标投影到图像I_2的像素坐标

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH) # 根据深度筛选有效投影点
    return x1, valid.float()

def induced_flow(Ts, depth, intrinsics):
    """ Compute 2d and 3d flow fields 
            计算二维和三维流场 """

    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0

    x0 = project(X0, intrinsics)
    x1 = project(X1, intrinsics)

    flow2d = x1 - x0
    flow3d = X1 - X0

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return flow2d, flow3d, valid.float()


def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2] # 二维流场的高度和宽度，保存在ht,wd中

    fx, fy, cx, cy = \
        intrinsics[None].unbind(dim=-1) # 从内参中获得焦距和光心
    
    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(), 
        torch.arange(wd).to(depth0.device).float()) # 创建一个网格表示像素坐标

    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1] # 根据流场获得新的像素坐标

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = torch.stack([X1-X0, Y1-Y0, Z1-Z0], dim=-1) # 计算三维流场
    return flow3d

