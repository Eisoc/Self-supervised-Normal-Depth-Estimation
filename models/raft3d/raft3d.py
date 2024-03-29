import torch
import torch.nn as nn
import torch.nn.functional as F

# lietorch for tangent space backpropogation
from lietorch import SE3

from .blocks.extractor import BasicEncoder
from .blocks.resnet import FPN
from .blocks.corr import CorrBlock
from .blocks.gru import ConvGRU
from .sampler_ops import bilinear_sampler, depth_sampler

from . import projective_ops as pops
from . import se3_field


GRAD_CLIP = .01 # 梯度剪裁，防止梯度爆炸，梯度会在-0.01到0.01之间

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x) # 创建一个与 grad_x 相同形状的全零张量 o
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x) # 对于大于阈值的元素，将其对应位置的值替换为全零张量 o，否则保持不变
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x) # 判断哪些元素是 NaN（不是数字），将这些元素的位置替换为全零张量 o
        return grad_x

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.gru = ConvGRU(hidden_dim)

        self.corr_enc = nn.Sequential(
            nn.Conv2d(196, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*128, 1, padding=0))

        self.flow_enc = nn.Sequential(
            nn.Conv2d(9, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*128, 1, padding=0))

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            nn.Sigmoid(),
            GradientClip())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0),
            GradientClip())
        # 各个输入输出维度！

    def forward(self, net, inp, corr, flow, twist, dz, upsample=True):
        motion_info = torch.cat([flow, 10*dz, 10*twist], dim=-1) # 将 flow、dz 和 twist 按最后一个维度拼接在一起，形成一个维度为 batch_size x height x width x (3 * input_dim) 的张量 motion_info
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0,3,1,2) # 将motion_info限制在-50到50，并且对维度重新排列

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        net = self.gru(net, inp, cor, mot)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        return net, mask, ae, delta, weight


class RAFT3D(nn.Module):
    def __init__(self, args):
        super(RAFT3D, self).__init__()

        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = FPN(output_dim=hdim+3*hdim) # resnet
        self.update_block = BasicUpdateBlock(args, hidden_dim=hdim)

    def initializer(self, image1):
        """ Initialize coords and transformation maps """

        batch_size, ch, ht, wd = image1.shape # 获取I_1信息，batch_size、通道数 ch、高度 ht 和宽度 wd
        device = image1.device 

        y0, x0 = torch.meshgrid(torch.arange(ht//8), torch.arange(wd//8)) # 生成网格坐标，定义宽度和高度，为什么是//8？
        coords0 = torch.stack([x0, y0], dim=-1).float() # (ht//8, wd//8)-->(ht//8, wd//8, 2)
        coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device) 
        # 使用 [None] 将 coords0 张量的形状在第0个维度上添加一个维度，从 (ht//8, wd//8, 2) 变为 (1, ht//8, wd//8, 2)。这是为了后续的重复操作做准备
        # 使用 .repeat(batch_size, 1, 1, 1) 将 coords0 张量在每个维度上重复扩展，生成一个新张量(batch_size, ht//8, wd//8, 2)

        Ts = SE3.Identity(batch_size, ht//8, wd//8, device=device) # 创建了一个形状为 (batch_size, ht//8, wd//8, 4, 4) 的单位矩阵张量 Ts，4,4是矩阵维度？还是(batch_size, ht//8, wd//8, 6)？
        return Ts, coords0
        
    def features_and_correlation(self, image1, image2):
        # extract features and build correlation volume
        fmap1, fmap2 = self.fnet([image1, image2]) # 提取I_1，I_2的特征

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius) 


        # extract context features using Resnet50
        net_inp = self.cnet(image1) # 提取上下文特征
        net, inp = net_inp.split([128, 128*3], dim=1) # 在dim=1上分割，得到net，形状为 (batch_size, 128, ht, wd)；inp，形状为 (batch_size, 128*3, ht, wd)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        return corr_fn, net, inp

    def forward(self, image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=False):
        """ Estimate optical flow between pair of frames """

        Ts, coords0 = self.initializer(image1)
        corr_fn, net, inp = self.features_and_correlation(image1, image2)

        # intrinsics and depth at 1/8 resolution
        # 下采样
        intrinsics_r8 = intrinsics / 8.0
        depth1_r8 = depth1[:,3::8,3::8] # 维度为 [batch_size, channels, height//8, width//8]
        depth2_r8 = depth2[:,3::8,3::8]

        flow_est_list = [] # 通过mask和convex combination upsampling，得出原始分辨率的flow：flow2d_rev。这个flow只有x,y两个分量，仅在训练时使用。
        flow_rev_list = [] # 先将T上采样，再使用T 、原始分辨率2D坐标、相机内参，通过2D->3D->2D坐标转换，计算出一个flow：flow2d_est。这个flow具有x,y,z三个分量
        Ts_list = []

        for itr in range(iters):
            Ts = Ts.detach() # 将Ts从计算图中分离，requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad，以便在后续计算中不进行梯度传播

            coords1_xyz, _ = pops.projective_transform(Ts, depth1_r8, intrinsics_r8) # coords1_xyz是在I_2中，根据Ts对I_1中的点进行投影变换
            
            coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1) 
            # 将coords1_xyz分割成两部分，coords1包含前两个维度，即 coords1_xyz 中的 x 和 y 坐标；zinv_proj 包含最后一个维度，即 coords1_xyz 中的逆深度
            zinv, _ = depth_sampler(1.0/depth2_r8, coords1) # zinv是图像I_2中与图像I_1上的坐标coords1对应的逆深度值

            corr = corr_fn(coords1.permute(0,3,1,2).contiguous()) 
            # 维度转换，coords1的维度为[batch_size, height, width, 2]，而corr_fn输入的维度为[batch_size, channels, height, width]
            flow = coords1 - coords0

            dz = zinv.unsqueeze(-1) - zinv_proj 
            # zinv是I_2深度图中得到的逆深度，维度为 [batch_size, height, width]
            # zinv_proj是投影变换之后得到的逆深度，维度为[batch_size, height, width, 1]
            # dz是深度变化值，维度为[batch_size, height, width, 2]
            twist = Ts.log()
            # 计算Ts的对数形式，将旋转部分转换为扭曲向量表示，可以将旋转矩阵映射到一个更小的向量空间，便于计算和优化
            # twist是一个与Ts维度相同的扭曲向量张量，用于表示从图像I_1到图像I_2的姿态变化

            net, mask, ae, delta, weight = \
                self.update_block(net, inp, corr, flow, dz, twist)

            target = coords1_xyz.permute(0,3,1,2) + delta
            target = target.contiguous()
            # delta是位移变量，target表示经过位移变换后的新坐标（图像I_2上的坐标）

            # Gauss-Newton step
            # Ts = se3_field.step(Ts, ae, target, weight, depth1_r8, intrinsics_r8)
            Ts = se3_field.step_inplace(Ts, ae, target, weight, depth1_r8, intrinsics_r8)

            if train_mode:
                flow2d_rev = target.permute(0,2,3,1)[...,:2] - coords0 
                # 通过将target（表示预测的目标点在图像I_2上的坐标）与coords0（表示初始点在图像I_1上的坐标）相减，得到逆向光流。逆向光流的形状为(batch_size, height, width, 2)，表示每个像素在图像I1上的移动向量
                flow2d_rev = se3_field.cvx_upsample(8 * flow2d_rev, mask)
                # 上采样到原始分辨率。上采样倍率为8，并根据mask进行插值操作

                Ts_up = se3_field.upsample_se3(Ts, mask) # 将位姿Ts进行上采样
                flow2d_est, flow3d_est, valid = pops.induced_flow(Ts_up, depth1, intrinsics) # 计算前向光流和前向三维流

                flow_est_list.append(flow2d_est)
                flow_rev_list.append(flow2d_rev)
                Ts_list.append(Ts_up)

        if train_mode:
            return flow_est_list, flow_rev_list, Ts_list

        Ts_up = se3_field.upsample_se3(Ts, mask)
        return Ts_up

