import sys
sys.path.append('.')

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from lietorch import SE3
import models.raft3d.projective_ops as pops
from utils.data_readers import frame_utils
from utils.utils_raft3d import show_image, normalize_image, prepare_images_and_depths, parse_args_raft3d

DEPTH_SCALE = 0.2 # 深度缩放因子

# def prepare_images_and_depths(image1, image2, depth1, depth2):
#     """ padding, normalization, and scaling """

#     image1 = F.pad(image1, [0,0,0,4], mode='replicate') # 填充，[0,0,0,4] 表示在垂直方向上在顶部不填充，在底部填充 4 个像素值，而在水平方向上不进行填充
#     image2 = F.pad(image2, [0,0,0,4], mode='replicate') 
#     depth1 = F.pad(depth1[:,None], [0,0,0,4], mode='replicate')[:,0] # depth1[:, None] 将深度图的通道数扩展为 1，以便与图像的通道数一致
#     depth2 = F.pad(depth2[:,None], [0,0,0,4], mode='replicate')[:,0] # 通过 [:, 0] 对填充后的深度图进行索引操作，将深度图的通道数从 1 缩减回原来的通道数

#     depth1 = (DEPTH_SCALE * depth1).float()
#     depth2 = (DEPTH_SCALE * depth2).float()
#     image1 = normalize_image(image1)
#     image2 = normalize_image(image2) # 归一化

#     return image1, image2, depth1, depth2


def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3) # 创建一个一行三列的图网格，(ax1, ax2, ax3)对应三个子图
    ax1.imshow(img[:, :, ::-1] / 255.0) # img[:, :, ::-1] 是对图像数据进行切片和反转操作，将通道顺序从 BGR 转换为 RGB

    tau_img = np.clip(tau, -0.1, 0.1) # tau 中的数值限制在范围 [-0.1, 0.1] 内
    tau_img = (tau_img + 0.1) / 0.2 # 对截断后的 tau_img 进行了平移和缩放操作

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    # plt.show()
    plt.savefig('demo_output_raft3d.png')
    plt.close(fig)

@torch.no_grad()
def demo(args):
    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D
    model = torch.nn.DataParallel(RAFT3D(args), device_ids=[0])
    model.load_state_dict(torch.load(args.model), strict=False)

    model.eval()
    model.cuda()

    fx, fy, cx, cy = (1050.0, 1050.0, 480.0, 270.0) # 内参
    img1 = cv2.imread('data/assets/image1.png') # array (540, 960, 3)
    img2 = cv2.imread('data/assets/image2.png')
    disp1 = frame_utils.read_gen('data/assets/disp1.pfm') # 深度图像数据 array (540, 960)
    disp2 = frame_utils.read_gen('data/assets/disp2.pfm')

    depth1 = torch.from_numpy(fx / disp1).float().cuda().unsqueeze(0) # 计算了深度图像 disp1 和 disp2 对应的深度值，并将其转换为 PyTorch 张量，.unsqueeze(0) 在张量的维度上添加一个维度，以匹配模型的输入要求
    depth2 = torch.from_numpy(fx / disp2).float().cuda().unsqueeze(0) 
    image1 = torch.from_numpy(img1).permute(2,0,1).float().cuda().unsqueeze(0) 
    image2 = torch.from_numpy(img2).permute(2,0,1).float().cuda().unsqueeze(0) # (channel, height, width)
    # torch.Size([1, 3, 540, 960]) torch.Size([1, 540, 960])
    intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda().unsqueeze(0) # torch.Size([1, 4])

    image1, image2, depth1, depth2, _ = prepare_images_and_depths(image1, image2, depth1, depth2)
    # torch.Size([1, 3, 544, 960]) # torch.Size([1, 544, 960])

    Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16) # 估计得到的场景运动 torch.Size([1, 544, 960])

    # compute 2d and 3d from from SE3 field (Ts)
    flow2d, flow3d, _ = pops.induced_flow(Ts, depth1, intrinsics) # 从SE3场返回2D，3D光流
    # torch.Size([1, 544, 960, 3]), torch.Size([1, 544, 960, 3])
    
    # extract rotational and translational components of Ts
    tau, phi = Ts.log().split([3,3], dim=-1) # 张量沿着最后一个维度分割为长度为3的两个部分 torch.Size([1, 544, 960, 3])
    tau = tau[0].cpu().numpy() # 平移tau (544, 960, 3)
    phi = phi[0].cpu().numpy() # 旋转phi (544, 960, 3)

    # undo depth scaling
    flow3d = flow3d / DEPTH_SCALE # 将深度值重新恢复到其原始比例

    display(img1, tau, phi)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default='checkpoints/raft3d.pth', help='checkpoint to restore')
    # parser.add_argument('--network', default='models.raft3d.raft3d', help='network architecture')
    # #自己加的
    # parser.add_argument('--headless', action='store_true', help='run in headless mode')
    # #
    args = parse_args_raft3d()

    if args.headless:
        import matplotlib
        matplotlib.use('Agg') 

    demo(args)

    


