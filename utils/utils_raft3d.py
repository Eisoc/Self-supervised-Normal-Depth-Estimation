import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse


SUM_FREQ = 100

def parse_args_raft3d():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/raft3d.pth', help='checkpoint to restore')
    parser.add_argument('--network', default='models.raft3d.raft3d', help='network architecture')
    parser.add_argument('--headless', action='store_true', help='run in headless mode')
    
    args = parser.parse_args()
    return args

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