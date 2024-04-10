import sys
sys.path.append('.')

from tqdm import tqdm
import os
import numpy as np
import cv2
import argparse
import torch

from lietorch import SE3
import models.raft3d.projective_ops as pops

from utils.utils_raft3d import show_image, normalize_image
from utils.data_readers.kitti import KITTIEval
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from glob import glob
from utils.data_readers.frame_utils import *


def display(img, tau, phi, index):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3) # 创建一个一行三列的图网格，(ax1, ax2, ax3)对应三个子图
    ax1.imshow(img[:, :, ::-1] / 255.0) # img[:, :, ::-1] 是对图像数据进行切片和反转操作，将通道顺序从 BGR 转换为 RGB

    tau_img = np.clip(tau, -0.1, 0.1) # tau 中的数值限制在范围 [-0.1, 0.1] 内。 强调这个数值范围内的变化，同时忽略过大或过小的异常值，因为过大或过小的数值可能会使图像看起来过亮或过暗，从而掩盖了其他重要的信息。
    tau_img = (tau_img + 0.1) / 0.2 # 对截断后的 tau_img 进行了平移和缩放操作。 将数值范围从 [-0.1, 0.1] 转换为 [0, 1]。

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    # plt.show()

    tau_img_path = 'models/test_baseline/outputs/raft3doutputs/tau_img/%06d.png' % index
    phi_img_path = 'models/test_baseline/outputs/raft3doutputs/phi_img/%06d.png' % index
    output_img_path = 'models/test_baseline/outputs/raft3doutputs/output_img/%06d.png' % index

    plt.imsave(tau_img_path, tau_img)
    plt.imsave(phi_img_path, phi_img)
    plt.savefig(output_img_path)
    plt.close(fig)

    # plt.imsave('tau.png', tau_img)
    # plt.imsave('phi.png', phi_img)
    # plt.savefig('output.png')
    # plt.close(fig)


def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=1.0):
    """ padding, normalization, and scaling """
    
    ht, wd = image1.shape[-2:]
    # ht, wd = image1.shape[:2]是不对的，这里张量的shape是[batch_size, channels, height, width]，高和宽是最后两个维度
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8
    # padding是为了高度和宽度都是8的倍数，这里取-号是为了计算距离下一个8的倍数还差多少
    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')
    # [0,pad_w,0,pad_h]表示在宽度方向上（右侧）添加pad_w个像素，在高度方向上（下方）添加pad_h个像素。
    # 填充的像素值是复制边缘的像素。
    depth1 = F.pad(depth1[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    # 对深度图进行填充时，需要额外的一个步骤，那就是删除添加的一个无用的维度。
    # 这是因为F.pad函数要求输入是一个四维张量，但深度图只有三维，所以在填充之前先通过[:,None]添加了一个新的维度，
    # 填充后通过[:,0]再将其删除

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2, (pad_w, pad_h)


@torch.no_grad()
def make_kitti_submission(model):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'drop_last': False}
    test_loader = DataLoader(KITTIEval(), **loader_args)
    # DataLoader接受几个参数：
    # 第一个参数是一个数据集实例。在这个例子中，是KITTIEval()。KITTIEval()是一个自定义的数据集类，用于加载和处理 KITTI 数据集中的数据。
    # **loader_args 是一个字典，包含了传递给 DataLoader 的其他参数。星号 ** 是 Python 中的解包操作符，它会将字典中的键-值对解包为关键字参数。
    # 这个字典可能包含了如下参数：
    # batch_size：每个批次的样本数量。在训练和评估过程中，模型一次处理一个批次的样本。
    # shuffle：一个布尔值，表示是否在每个训练周期开始时随机打乱数据。
    # num_workers：加载数据时使用的子进程数量。如果这个值大于0，那么将在多个子进程中使用 Python 的 multiprocessing 模块预加载数据，可以加速数据加载。
    # pin_memory：一个布尔值，表示是否将数据加载到固定的内存区域，也就是 CUDA 的固定内存。这可以加速将数据移动到 GPU 的速度，但仅在使用 GPU 时有用。

    DEPTH_SCALE = .1

    for i_batch, data_blob in enumerate(test_loader): 
        make_kitti_in_iterate(model, i_batch, data_blob)
        # # 遍历由DataLoader生成的批次。对于每个批次，DataLoader返回一个数据包（data_blob），
        # # 这个数据包包含了当前批次的所有样本。同时，enumerate函数还会返回当前批次的索引（i_batch）
        # image1, image2, disp1, disp2, intrinsics, _, _ = [item.cuda() for item in data_blob]
        # # 从数据包中获取数据，并将数据移动到GPU上

        # img1 = image1[0].permute(1,2,0).cpu().numpy() # 作用是什么？
        # depth1 = DEPTH_SCALE * (intrinsics[0,0] / disp1)
        # depth2 = DEPTH_SCALE * (intrinsics[0,0] / disp2)

        # ht, wd = image1.shape[2:]
        # # ht, wd = image1.shape[:2] 不对，因为[batch_size, channels, height, width]
        # image1, image2, depth1, depth2, _ = \
        #     prepare_images_and_depths(image1, image2, depth1, depth2)

        # Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)
        # # (batch_size, ht//8, wd//8, 6)，ht 和 wd 表示输入图像的高度和宽度，//8 表示经过下采样后的尺寸，6表示 SE3 矩阵的参数个数（平移向量和旋转矩阵的参数）
        # tau_phi = Ts.log()

        # # uncomment to diplay motion field
        # tau, phi = Ts.log().split([3,3], dim=-1)
        # tau = tau[0].cpu().numpy()
        # phi = phi[0].cpu().numpy()
        # display(img1, tau, phi, i_batch)

        # # compute optical flow
        # flow, _, _ = pops.induced_flow(Ts, depth1, intrinsics)
        # flow = flow[0, :ht, :wd, :2].cpu().numpy()

        # # compute disparity change
        # coords, _ = pops.projective_transform(Ts, depth1, intrinsics)
        # disp2 =  intrinsics[0,0] * coords[:,:ht,:wd,2] * DEPTH_SCALE
        # disp1 = disp1[0].cpu().numpy()
        # disp2 = disp2[0].cpu().numpy()

        # KITTIEval.write_prediction(i_batch, disp1, disp2, flow, Ts, tau, phi)

def make_kitti_in_iterate(model, i_batch, data_blob):
    image1, image2, disp1, disp2, intrinsics, _, _ = [item.cuda() for item in data_blob]
    DEPTH_SCALE = .1
    batch_size = image1.shape[0]

    for idx in range(batch_size):
        # 处理每个样本
        img1 = image1[idx].permute(1, 2, 0).cpu().numpy()
        # 注意：以下深度处理方式仅为示例，具体计算需要根据实际情况调整
        depth1_sample = DEPTH_SCALE * (intrinsics[idx, 0, 0] / disp1[idx])
        depth2_sample = DEPTH_SCALE * (intrinsics[idx, 0, 0] / disp2[idx])

        ht, wd = image1.shape[2:]
        image1_sample, image2_sample, depth1_sample, depth2_sample, _ = prepare_images_and_depths(image1[idx:idx+1], image2[idx:idx+1], depth1_sample, depth2_sample)
        
        Ts = model(image1_sample, image2_sample, depth1_sample, depth2_sample, intrinsics[idx:idx+1], iters=16)
        tau_phi = Ts.log()

        # 以下操作针对每个样本进行展示、计算流和差异等
        # 注意，一些操作可能需要针对单个样本的数据进行调整
        tau, phi = tau_phi.split([3, 3], dim=-1)
        tau = tau[0].cpu().numpy()
        phi = phi[0].cpu().numpy()
        display(img1, tau, phi, i_batch * batch_size + idx)

        # 计算光流和视差变化等，这里省略了部分具体实现
        # 请根据实际模型输出和计算需求调整
        flow, _, _ = pops.induced_flow(Ts, depth1_sample, intrinsics[idx:idx+1])
        flow = flow[0, :ht, :wd, :2].cpu().numpy()

        coords, _ = pops.projective_transform(Ts, depth1_sample, intrinsics[idx:idx+1])
        disp2_sample = intrinsics[idx, 0, 0] * coords[:, :ht, :wd, 2] * DEPTH_SCALE
        disp1_sample = disp1[idx].cpu().numpy()
        disp2_sample = disp2_sample[0].cpu().numpy()

        # 保存或处理每个样本的结果
        KITTIEval.write_prediction(i_batch * batch_size + idx, disp1_sample, disp2_sample, flow, Ts, tau, phi)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', help='path the model weights')
    # parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    # parser.add_argument('--radius', type=int, default=32)
    # 自己加的
    parser.add_argument('--network', default='models.raft3d.raft3d_bilaplacian', help='network architecture')
    parser.add_argument('--model', default='checkpoints/raft3d_kitti.pth', help='path the model weights')
    parser.add_argument('--radius', type=int, default=32)
    parser.add_argument('--headless', action='store_true', help='run in headless mode')
    #
    args = parser.parse_args()

    if args.headless:
        import matplotlib
        matplotlib.use('Agg') 


    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D

    model = torch.nn.DataParallel(RAFT3D(args), device_ids=[0])
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # if not os.path.isdir('models/test_baseline/outputs/kitti_submission'):
    #     os.mkdir('kitti_submission')
    #     os.mkdir('kitti_submission/disp_0')
    #     os.mkdir('kitti_submission/disp_1')
    #     os.mkdir('kitti_submission/flow')
    #     os.mkdir('kitti_submission/T')
    #     os.mkdir('kitti_submission/tau')
    #     os.mkdir('kitti_submission/phi')
    base_output_dir = 'models/test_baseline/outputs/'

    # 在基础路径下创建 'kitti_submission' 目录和其子目录
    kitti_submission_dir = os.path.join(base_output_dir, 'raft3doutputs')

    # 创建 'kitti_submission' 目录，如果不存在
    os.makedirs(kitti_submission_dir, exist_ok=True)

    # 创建需要的子目录
    sub_dirs = ['flow', 'T', 'tau', 'phi', "tau_img", "phi_img", "output_img"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(kitti_submission_dir, sub_dir), exist_ok=True)
        
        
    make_kitti_submission(model)

# python scripts/kitti_submission.py --network=raft3d.raft3d_bilaplacian --model=raft3d_kitti.pth --headless

