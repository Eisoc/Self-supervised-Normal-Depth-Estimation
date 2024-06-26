
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import glob
import os
import cv2
import math
import random
import json
import csv
import pickle
import os.path as osp
from imageio import imread
from glob import glob

import models.raft3d.projective_ops as pops
from . import frame_utils
from .augmentation import RGBDAugmentor, SparseAugmentor

class KITTIEval(data.Dataset):

    crop = 80

    def __init__(self, sequence_length, img_width, img_height, image_size=None, root='data/raft_datasets', do_augment=True):
        self.init_seed = None
        mode = "testing"
        
        self.imgs = sorted(glob(osp.join(root, mode, "seq/*.png")))
        
        # self.image1_list = sorted(glob(osp.join(root, mode, "image_2/*10.png")))
        # self.image2_list = sorted(glob(osp.join(root, mode, "image_2/*11.png")))
        # self.disp1_ga_list = sorted(glob(osp.join(root, mode, "disp_ganet_{}/*10.png".format(mode))))
        # self.disp2_ga_list = sorted(glob(osp.join(root, mode, "disp_ganet_{}/*11.png".format(mode))))
        self.calib_list = sorted(glob(osp.join(root, mode, "calib_cam_to_cam/*.txt")))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)
        
        self.sequence_length = sequence_length
        self.img_width = img_width
        self.img_height = img_height

    @staticmethod
    def write_prediction(index, disp1, disp2, flow, Ts, tau, phi):

        def writeFlowKITTI(filename, uv):
            uv = 64.0 * uv + 2**15 # 将光流数据缩放
            valid = np.ones([uv.shape[0], uv.shape[1], 1]) # 创建一个与uv形状相同的全1数组
            uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16) # 将光流数据与有效性信息连接
            # print(np.isnan(uv), np.isinf(uv))
            cv2.imwrite(filename, uv[..., ::-1])

        def writeDispKITTI(filename, disp):
            disp = (256 * disp).astype(np.uint16) # 数据缩放
            cv2.imwrite(filename, disp)
        
        def writeTMatrix(filename, Ts):
            # Ts_cpu = Ts.data.cpu() # 将变换矩阵数据从 GPU 移到 CPU
            # Ts_tensor = Ts_cpu.data 
            Ts.data=Ts.data.cuda()
            Ts_tensor = Ts.data.cpu().data 
            Ts_np = Ts_tensor.numpy() # 从 PyTorch Tensor 转换为 NumPy 数组``
            Ts_last6 = Ts_np[:, :, :, -6:].reshape(-1, 6) # 提取最后6列，并重塑为2D数组
            np.savetxt(filename, Ts_last6)

        def writetau(filename, tau):

            np.savetxt(filename, tau.reshape(-1, 3), fmt='%.6f', delimiter=' ')

        def writephi(filename, phi):
            np.savetxt(filename, phi.reshape(-1, 3), fmt='%.6f', delimiter=' ')

        # disp1 = np.pad(disp1, ((KITTIEval.crop,0),(0,0)), mode='edge')
        # disp2 = np.pad(disp2, ((KITTIEval.crop, 0), (0,0)), mode='edge')
        # flow = np.pad(flow, ((KITTIEval.crop, 0), (0,0),(0,0)), mode='edge')

        disp1_path = 'models/test_baseline/outputs/raft3doutputs/disp_0/%06d_10.png' % index
        disp2_path = 'models/test_baseline/outputs/raft3doutputs/disp_1/%06d_10.png' % index
        flow_path = 'models/test_baseline/outputs/raft3doutputs/flow/%06d_10.png' % index
        T_path = 'models/test_baseline/outputs/raft3doutputs/T/%06d.txt' % index
        tau_path = 'models/test_baseline/outputs/raft3doutputs/tau/%06d.txt' % index
        phi_path = 'models/test_baseline/outputs/raft3doutputs/phi/%06d.txt' % index

        # writeDispKITTI(disp1_path, disp1)
        # writeDispKITTI(disp2_path, disp2)
        writeFlowKITTI(flow_path, flow)
        print("raft3d-flow saved")
        writeTMatrix(T_path, Ts)
        writetau(tau_path, tau)
        writephi(phi_path, phi)
        print("raft3d-txts saved")
                        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        intrinsics = self.intrinsics_list[index]
        # image1 = cv2.imread(self.image1_list[index])
        # image2 = cv2.imread(self.image2_list[index])

        # disp1 = cv2.imread(self.disp1_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        # disp2 = cv2.imread(self.disp2_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0

        # image1 = image1[self.crop:]
        # image2 = image2[self.crop:]

        # disp1 = disp1[self.crop:]
        # disp2 = disp2[self.crop:]
        
        # intrinsics[3] -= self.crop

        # image1 = torch.from_numpy(image1).float().permute(2,0,1)
        # image2 = torch.from_numpy(image2).float().permute(2,0,1)
        # disp1 = torch.from_numpy(disp1).float()
        # disp2 = torch.from_numpy(disp2).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        raw_im = np.array(imread(self.imgs[index]))
        # raw_im: Around (375, 1242, 3) for KITTI (single image data)
        scaled_im = torch.as_tensor(cv2.resize(raw_im, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA))
        tgt_view = scaled_im.permute(2, 0, 1)
        
        
        # for srcview        
        src_views = []
        for offset in [-1, 1]:  # 前一帧和后一帧
            src_idx = max(0, min(len(self.imgs) - 1, index + offset))
            src_img_path = self.imgs[src_idx]
            src_img = np.array(imread(src_img_path))
            src_img = cv2.resize(src_img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
            src_view = torch.tensor(src_img).permute(2, 0, 1)
            src_views.append(src_view)
            # view: torch.Size([3, 128, 416])
        src_views = torch.cat(src_views, dim=0)
        # torch.Size([6, 128, 416]

        
        return intrinsics, tgt_view, src_views


class KITTI(data.Dataset):
    def __init__(self, image_size=None, root='datasets/KITTI', do_augment=True):
        import csv

        self.init_seed = None
        self.crop = 80

        if do_augment:
            self.augmentor = SparseAugmentor(image_size)
        else:
            self.augmentor = None
        
        self.image1_list = sorted(glob(osp.join(root, "training", "image_2/*10.png")))
        self.image2_list = sorted(glob(osp.join(root, "training", "image_2/*11.png")))

        self.disp1_list = sorted(glob(osp.join(root, "training", "disp_occ_0/*10.png")))
        self.disp2_list = sorted(glob(osp.join(root, "training", "disp_occ_1/*10.png")))

        # self.disp1_ga_list = sorted(glob(osp.join(root, "training", "disp_ganet/*10.png")))
        # self.disp2_ga_list = sorted(glob(osp.join(root, "training", "disp_ganet/*11.png")))

        self.disp1_ga_list = sorted(glob(osp.join(root, "training", "disp_ganet_training/*10.png")))
        self.disp2_ga_list = sorted(glob(osp.join(root, "training", "disp_ganet_training/*11.png")))

        self.flow_list = sorted(glob(osp.join(root, "training", "flow_occ/*10.png")))
        self.calib_list = sorted(glob(osp.join(root, "training", "calib_cam_to_cam/*.txt")))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)
                        
    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        image1 = cv2.imread(self.image1_list[index])
        image2 = cv2.imread(self.image2_list[index])

        disp1 = cv2.imread(self.disp1_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(self.disp2_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp1_dense = cv2.imread(self.disp1_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2_dense = cv2.imread(self.disp2_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0

        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        intrinsics = self.intrinsics_list[index]

        SCALE = np.random.uniform(0.08, 0.15)

        # crop top 80 pixels, no ground truth information
        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp1 = disp1[self.crop:]
        disp2 = disp2[self.crop:]
        flow = flow[self.crop:]
        valid = valid[self.crop:]
        disp1_dense = disp1_dense[self.crop:]
        disp2_dense = disp2_dense[self.crop:]
        intrinsics[3] -= self.crop

        image1 = torch.from_numpy(image1).float().permute(2,0,1)
        image2 = torch.from_numpy(image2).float().permute(2,0,1)

        disp1 = torch.from_numpy(disp1 / intrinsics[0]) / SCALE
        disp2 = torch.from_numpy(disp2 / intrinsics[0]) / SCALE
        disp1_dense = torch.from_numpy(disp1_dense / intrinsics[0]) / SCALE
        disp2_dense = torch.from_numpy(disp2_dense / intrinsics[0]) / SCALE

        dz = (disp2 - disp1_dense).unsqueeze(dim=-1)
        depth1 = 1.0 / disp1_dense.clamp(min=0.01).float()
        depth2 = 1.0 / disp2_dense.clamp(min=0.01).float()

        intrinsics = torch.from_numpy(intrinsics)
        valid = torch.from_numpy(valid)
        flow = torch.from_numpy(flow)

        valid = valid * (disp2 > 0).float()
        flow = torch.cat([flow, dz], -1)

        if self.augmentor is not None:
            image1, image2, depth1, depth2, flow, valid, intrinsics = \
                self.augmentor(image1, image2, depth1, depth2, flow, valid, intrinsics)

        return image1, image2, depth1, depth2, flow, valid, intrinsics