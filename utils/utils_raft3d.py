import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import models.raft3d.projective_ops as pops
from utils.data_readers.kitti import KITTIEval
import matplotlib.pyplot as plt
import numpy as np
import os
from lietorch import SE3


SUM_FREQ = 100

def folder_builder():
    base_output_dir = 'models/test_baseline/outputs/'

    # 在基础路径下创建 'kitti_submission' 目录和其子目录
    kitti_submission_dir = os.path.join(base_output_dir, 'raft3doutputs')

    # 创建 'kitti_submission' 目录，如果不存在
    os.makedirs(kitti_submission_dir, exist_ok=True)

    # 创建需要的子目录
    sub_dirs = ['flow', 'T', 'tau', 'phi', "tau_img", "phi_img", "output_img"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(kitti_submission_dir, sub_dir), exist_ok=True)

# def make_kitti_in_iterate(model, i_batch, data_blob):
#         # 遍历由DataLoader生成的批次。对于每个批次，DataLoader返回一个数据包（data_blob），
#     # 这个数据包包含了当前批次的所有样本。同时，enumerate函数还会返回当前批次的索引（i_batch）
#     image1, image2, disp1, disp2, intrinsics, _, _ = [item.cuda() for item in data_blob]
#     print(intrinsics.size(), image1.size())
#     # 从数据包中获取数据，并将数据移动到GPU上
#     DEPTH_SCALE = .1
#     img1 = image1[0].permute(1,2,0).cpu().numpy() # 作用是什么？
#     depth1 = DEPTH_SCALE * (intrinsics[0,0] / disp1)
#     depth2 = DEPTH_SCALE * (intrinsics[0,0] / disp2)
#     print(depth1.size(), disp1.size())

#     ht, wd = image1.shape[2:]
#     # ht, wd = image1.shape[:2] 不对，因为[batch_size, channels, height, width]
#     image1, image2, depth1, depth2, _ = \
#         prepare_images_and_depths(image1, image2, depth1, depth2)

#     Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)
#     # (batch_size, ht//8, wd//8, 6)，ht 和 wd 表示输入图像的高度和宽度，//8 表示经过下采样后的尺寸，6表示 SE3 矩阵的参数个数（平移向量和旋转矩阵的参数）
#     tau_phi = Ts.log()

#     # uncomment to diplay motion field
#     tau, phi = Ts.log().split([3,3], dim=-1)
#     tau = tau[0].cpu().numpy()
#     phi = phi[0].cpu().numpy()
#     display(img1, tau, phi, i_batch)

#     # compute optical flow
#     flow, _, _ = pops.induced_flow(Ts, depth1, intrinsics)
#     flow = flow[0, :ht, :wd, :2].cpu().numpy()

#     # compute disparity change
#     coords, _ = pops.projective_transform(Ts, depth1, intrinsics)
#     disp2 =  intrinsics[0,0] * coords[:,:ht,:wd,2] * DEPTH_SCALE
#     disp1 = disp1[0].cpu().numpy()
#     disp2 = disp2[0].cpu().numpy()

#     KITTIEval.write_prediction(i_batch, disp1, disp2, flow, Ts, tau, phi)

# def make_kitti_in_iterate(model, i_batch, image1, image2, disp1, disp2, intrinsics):
def make_kitti_in_iterate(model, i_batch, image1, image2, depth1, depth2, intrinsics):
    # image1, image2, disp1, disp2, intrinsics, _, _ = [item.cuda() for item in data_blob]
    DEPTH_SCALE = .1
    batch_size = intrinsics.shape[0] # torch.Size([4, 4])
    for idx in range(batch_size):
        # 处理每个样本
        img1 = image1[idx].permute(1, 2, 0).cpu().numpy()
        # depth1_sample = DEPTH_SCALE * (intrinsics[idx, 0] / disp1[idx:idx+1])
        # depth2_sample = DEPTH_SCALE * (intrinsics[idx, 0] / disp2[idx:idx+1])

        ht, wd = image1.shape[2:]
        image1_sample, image2_sample, depth1_sample, depth2_sample, _ = prepare_images_and_depths(image1[idx:idx+1], image2[idx:idx+1], depth1[idx:idx+1], depth2[idx:idx+1])
        # image1_sample = torch.from_numpy(np.random.rand(1,3,416, 128)).to("cuda:1").float()
        # image2_sample =  torch.from_numpy(np.random.rand(1,3,416, 128)).to("cuda:1").float()
        # depth1_sample =  torch.from_numpy(np.random.rand(1,416, 128)).to("cuda:1").float()
        # depth2_sample =  torch.from_numpy(np.random.rand(1,416, 128)).to("cuda:1").float()
        
        # image1_sample = torch.from_numpy(np.random.rand(1,3,416, 128)).cuda().float()
        # image2_sample =  torch.from_numpy(np.random.rand(1,3,416, 128)).cuda().float()
        # depth1_sample =  torch.from_numpy(np.random.rand(1,416, 128)).cuda().float()
        # depth2_sample =  torch.from_numpy(np.random.rand(1,416, 128)).cuda().float()        
        intrinsics_sample = intrinsics[idx:idx+1].cuda()
        inputs = {
                    'image1': image1_sample,
                    'image2': image2_sample,
                    'depth1': depth1_sample,
                    'depth2': depth2_sample,
                    'intrinsics': intrinsics_sample,
                    "iters":16, "train_mode":False
                }
        # Ts = model(image1_sample, image2_sample, depth1_sample, depth2_sample, intrinsics_sample, iters=16)
        model.cuda()
        Ts,tau_phi,data_tensor = model(inputs)
        Ts.data=data_tensor
        
        # tau_phi = Ts.log()
        tau, phi = tau_phi.split([3, 3], dim=-1)
        tau = tau[0].cpu().numpy()
        phi = phi[0].cpu().numpy()
        display(img1, tau, phi, i_batch * batch_size + idx)

        # 计算光流和视差变化等
        flow, _, _ = pops.induced_flow(Ts, depth1_sample, intrinsics[idx:idx+1])
        flow = flow[0, :ht, :wd, :2].cpu().numpy()
        # print("flow range: min =", flow.min().item(), ", max =", flow.max().item())
        # print("Ts.log() range: min =", Ts.log().min().item(), ", max =", Ts.log().max().item())
        
        coords, _ = pops.projective_transform(Ts, depth1_sample, intrinsics[idx:idx+1])
        # disp2_sample = intrinsics[idx, 0] * coords[:, :ht, :wd, 2] * DEPTH_SCALE
        # disp1_sample = disp1[idx].cpu().numpy()
        # disp2_sample = disp2_sample[0].cpu().numpy()
        disp1_sample = None
        disp2_sample = None
        # 保存或处理每个样本的结果
        KITTIEval.write_prediction(i_batch * batch_size + idx, disp1_sample, disp2_sample, flow, Ts, tau, phi)

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
    print("raft3d-tau_img saved")
    plt.imsave(phi_img_path, phi_img)
    print("raft3d-phi_img saved")
    plt.savefig(output_img_path)
    print("raft3d-output_img saved")
    plt.close(fig)

    # plt.imsave('tau.png', tau_img)
    # plt.imsave('phi.png', phi_img)
    # plt.savefig('output.png')
    # plt.close(fig)

def parse_args_raft3d():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='models.raft3d.raft3d_bilaplacian', help='network architecture')
    parser.add_argument('--model', default='checkpoints/raft3d_kitti.pth', help='path the model weights')
    parser.add_argument('--radius', type=int, default=32)
    # 自己加的
    parser.add_argument('--headless', action='store_true', help='run in headless mode')
    args = parser.parse_args()
    return args

def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=1.0):
    """ padding, normalization, and scaling """
    depth1 = torch.squeeze(depth1, dim=1)
    depth2 = torch.squeeze(depth2, dim=1)
    image1 = image1.float()
    image2 = image2.float()
    depth1 = depth1.float()
    depth2 = depth2.float()
    depth1 = depth1.cpu().detach().numpy()
    depth1 = depth1 - depth1.min()  # 将最小值标准化为0
    depth1 = depth1 / depth1.max()*255 
    depth2 = depth2.cpu().detach().numpy()
    depth2 = depth2 - depth2.min()  # 将最小值标准化为0
    depth2 = depth2 / depth2.max() *255 
    depth1 = torch.from_numpy(depth1).to('cuda:1')
    depth2 = torch.from_numpy(depth2).to('cuda:1')
    # torch.Size([1, 3, 128, 416]) torch.Size([1, 128, 416])
    ht, wd = image1.shape[-2:]
    # ht, wd = image1.shape[:2]是不对的，这里张量的shape是[batch_size, channels, height, width]，高和宽是最后两个维度
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8
    # padding是为了高度和宽度都是8的倍数，这里取-号是为了计算距离下一个8的倍数还差多少
    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')
    # [0,pad_w,0,pad_h]表示在宽度方向上（右侧）添加pad_w个像素，在高度方向上（下方）添加pad_h个像素。
    # 填充的像素值是复制边缘的像素。
    # depth1 = F.pad(depth1[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    # depth2 = F.pad(depth2[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    depth1 = F.pad(depth1, [0,pad_w,0,pad_h], mode='replicate')
    depth2 = F.pad(depth2, [0,pad_w,0,pad_h], mode='replicate')
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

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def normalize_image(image):
    image = image[:, [2,1,0]] # BGR通过 [:, [2, 1, 0]]调整为 RGB
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device) # [0.485, 0.456, 0.406] 和 [0.229, 0.224, 0.225] 是在 ImageNet 数据集上计算得到的经验值，用于归一化图像
    return (image/255.0).sub_(mean[:, None, None]).div_(std[:, None, None]) # 归一化，使用 sub_ 函数对图像的每个通道减去对应的均值，使用 div_ 函数将图像的每个通道除以对应的标准差

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Logger:
    def __init__(self):
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter()

        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}] ".format(self.total_steps+1)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / SUM_FREQ
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1