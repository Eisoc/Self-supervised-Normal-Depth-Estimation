import torch
import torch.nn.functional as F
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import csv

device = torch.device(
    'cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def pose_to_csv(pose_data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(['source_index', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
        # 遍历所有的姿态数据
        for i, poses in enumerate(pose_data):
            # 遍历每个源图像的姿态
            for src_idx, pose in enumerate(poses):
                # 将tensor转换为numpy数组并写入文件
                writer.writerow([src_idx] + pose.cpu().numpy().tolist())

def save_tensor_as_image(batch_index, tensor, filename, path):
    for i, img in enumerate(tensor):
        img = img.cpu().detach().numpy()  # 转换为NumPy数组
        # if img.shape[0] == 2:  # 光流图像
        #     # 光流图像处理逻辑
        #     # 计算光流的大小和方向
        #     magnitude, angle = cv2.cartToPolar(img[0], img[1])
        #     magnitude = magnitude - magnitude.min()  # 将最小值标准化为0
        #     magnitude = magnitude / magnitude.max()  # 将最大值标准化为1
        #     img = magnitude  # 这里只保存大小信息作为示例，也可以考虑将方向信息编码为颜色
        if img.shape[0] == 2:  # 光流图像
            # 计算光流的大小和方向
            magnitude, angle = cv2.cartToPolar(img[0], img[1])
            # 归一化大小
            magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
            # 将角度从弧度转换为0到1之间
            angle = angle / (2 * np.pi)
            # 创建HSV图像，其中饱和度设置为1
            hsv = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
            hsv[..., 0] = angle  # 色调
            hsv[..., 1] = 1  # 饱和度
            hsv[..., 2] = magnitude  # 值
            # 将HSV图像转换为RGB以保存
            hsv = hsv - hsv.min()
            hsv = hsv / hsv.max() 
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            img = img - img.min()  # 将最小值标准化为0
            img = img / img.max()  # 将最大值标准化为1
            # 检查通道数并相应处理
            if img.shape[0] == 3:  # RGB图像
                img = np.transpose(img, (1, 2, 0))  # 转置为 [H, W, C]
            elif img.shape[0] == 1:  # 单通道图像
                img = np.squeeze(img)  # 去除通道维度

        file_path = os.path.join(path, f"{filename}_{batch_index*4+i}.png")
        # 保存图像
        plt.imsave(file_path, img)

def convert_flow_dim(flow_tensor):
    flow_tensor = flow_tensor.squeeze(0)  # Remove batch dim, now [2, H, W]
    flow_tensor = flow_tensor.cpu().detach().numpy()
    
    # Calculate magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow_tensor[0], flow_tensor[1])
    # Normalize magnitude from 0 to 1
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    # Convert angle from radians to 0 to 1
    angle = angle / (2 * np.pi)
    
    # Create HSV image, saturation is set to 1
    hsv_image = np.zeros((flow_tensor.shape[1], flow_tensor.shape[2], 3), dtype=np.float32)
    hsv_image[..., 0] = angle  # Hue
    hsv_image[..., 1] = 1  # Saturation
    hsv_image[..., 2] = magnitude  # Value
    # Convert HSV to RGB
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    rgb_image = torch.tensor(rgb_image).float().to(device)
    rgb_image=rgb_image.permute(2,0,1)
    rgb_image=rgb_image.unsqueeze(0)
    return rgb_image
        

def scale_pyramid(img, num_scales):
    # img: (b, ch, h, w)
    if img is None:
        return None
    else:

        # TODO: Shape of image is [channels, h, w]     
        b, ch, h, w = img.shape
        scaled_imgs = [img.permute(0,2,3,1)]
#         print(scaled_imgs[0])
        
        for i in range(num_scales - 1):
            ratio = 2 ** (i+1)
            nh = int(h/ratio)
            nw = int(w/ratio)
            
            scaled_img = torch.nn.functional.interpolate(img, size=(nh, nw), mode='area')
            scaled_img = scaled_img.permute(0, 2, 3, 1)
            
            scaled_imgs.append(scaled_img)        

        # scaled_imgs: (scales, b, h, w, ch)
        
    return scaled_imgs


def L2_norm(x, dim, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset,
                         dim=dim, keepdim=keep_dims)
    return l2_norm

def DSSIM(x, y):
    
    avepooling2d = torch.nn.AvgPool2d(3, stride=1, padding=[1, 1])
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)

    sigma_x = avepooling2d(x**2) - mu_x**2
    sigma_y = avepooling2d(y**2) - mu_y**2
    sigma_xy = avepooling2d(x*y) - mu_x*mu_y
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # L_square = 255**2

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n/SSIM_d

    return torch.clamp((1 - SSIM.permute(0, 2,3,1))/2, 0, 1)

def gradient_x(img):    #checks out
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):    #checks out
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def compute_multi_scale_intrinsics(intrinsics, num_scales):

    batch_size = intrinsics.shape[0]
    multi_scale_intrinsices = []
    for s in range(num_scales):
        fx = intrinsics[:, 0, 0]/(2**s)
        fy = intrinsics[:, 1, 1]/(2**s)
        cx = intrinsics[:, 0, 2]/(2**s)
        cy = intrinsics[:, 1, 2]/(2**s)
        zeros = torch.zeros(batch_size).float().to(device)
        r1 = torch.stack([fx, zeros, cx], dim=1)  # shape: batch_size,3
        r2 = torch.stack([zeros, fy, cy], dim=1)  # shape: batch_size,3
        # shape: batch_size,3
        r3 = torch.tensor([0., 0., 1.]).float().view(
            1, 3).repeat(batch_size, 1).to(device)
        # concat along the spatial row dimension
        scale_instrinsics = torch.stack([r1, r2, r3], dim=1)
        multi_scale_intrinsices.append(
            scale_instrinsics)  # shape: num_scale,bs,3,3
    multi_scale_intrinsices = torch.stack(multi_scale_intrinsices, dim=1)
    return multi_scale_intrinsices

def euler2mat(z, y, x):
    global device
    # TODO: eular2mat
    '''
    input shapes of z,y,x all are: (#batch)
    '''
    batch_size = z.shape[0]

    _z = z.clamp(-np.pi, np.pi)
    _y = y.clamp(-np.pi, np.pi)
    _x = x.clamp(-np.pi, np.pi)

    ones = torch.ones(batch_size).float().to(device)
    zeros = torch.zeros(batch_size).float().to(device)

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    # shape: (#batch,3)
    rotz_mat_r1 = torch.stack((cosz, -sinz, zeros), dim=1)
    rotz_mat_r2 = torch.stack((sinz, cosz, zeros), dim=1)
    rotz_mat_r3 = torch.stack((zeros, zeros, ones), dim=1)
    # shape: (#batch,3,3)
    rotz_mat = torch.stack((rotz_mat_r1, rotz_mat_r2, rotz_mat_r3), dim=1)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    roty_mat_r1 = torch.stack((cosy, zeros, siny), dim=1)
    roty_mat_r2 = torch.stack((zeros, ones, zeros), dim=1)
    roty_mat_r3 = torch.stack((-siny, zeros, cosy), dim=1)
    roty_mat = torch.stack((roty_mat_r1, roty_mat_r2, roty_mat_r3), dim=1)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    rotx_mat_r1 = torch.stack((ones, zeros, zeros), dim=1)
    rotx_mat_r2 = torch.stack((zeros, cosx, -sinx), dim=1)
    rotx_mat_r3 = torch.stack((zeros, sinx, cosx), dim=1)
    rotx_mat = torch.stack((rotx_mat_r1, rotx_mat_r2, rotx_mat_r3), dim=1)

    # shape: (#batch,3,3)
    rot_mat = torch.matmul(torch.matmul(rotx_mat, roty_mat), rotz_mat)
    
#     rot_mat = torch.matmul(rotz_mat, torch.matmul(roty_mat, rotx_mat))

    return rot_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    global device
    
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    
    b, h, w = depth.size()
    
    depth = depth.view(b, 1, -1)
    pixel_coords = pixel_coords.view(b, 3, -1)
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords) * depth
    
    if is_homogeneous:
        ones = torch.ones(b, 1, h*w).float().to(device)
        cam_coords = torch.cat((cam_coords.to(device), ones), dim=1)
    
    cam_coords = cam_coords.view(b, -1, h, w)
    
    return cam_coords

def cam2pixel(cam_coords, proj):
    global device
    
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
    Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    b, _, h, w = cam_coords.size()
    cam_coords = cam_coords.view(b, 4, h*w)
    unnormalized_pixel_coords = torch.matmul(proj, cam_coords)
    
    x_u = unnormalized_pixel_coords[:, :1, :]
    y_u = unnormalized_pixel_coords[:, 1:2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]
    
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
        
    pixel_coords = torch.cat((x_n, y_n), dim=1)
    pixel_coords = pixel_coords.view(b, 2, h, w)
    
    return pixel_coords.permute(0, 2, 3, 1)

def pose_vec2mat(vec):
    global device
    # TODO:pose vec 2 mat
    # input shape of vec: (#batch, 6)
    # shape: (#batch,3)
    
    b, _ = vec.size()
    translation = vec[:, :3].unsqueeze(2)
    
    rx = vec[:, 3]
    ry = vec[:, 4]
    rz = vec[:, 5]
    
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = rot_mat.squeeze(1)
    
    filler = torch.tensor([0.,0.,0.,1.]).view(1, 4).repeat(b, 1, 1).float().to(device)
    
    transform_mat = torch.cat((rot_mat, translation), dim=2)
    transform_mat = torch.cat((transform_mat, filler), dim=1)
    
    return transform_mat

def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    
    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    
    global device
    
    # (height, width)
    x_t = torch.matmul(
        torch.ones(height).view(height, 1).float().to(device),
        torch.linspace(-1, 1, width).view(1, width).to(device))
    
    # (height, width)
    y_t = torch.matmul(
        torch.linspace(-1, 1, height).view(height, 1).to(device),
        torch.ones(width).view(1, width).float().to(device))
    
    x_t = (x_t + 1) * 0.5 * (width-1)
    y_t = (y_t + 1) * 0.5 * (height-1)
        
    if is_homogeneous:
        ones = torch.ones_like(x_t).float().to(device)
        #ones = torch.ones(height, width).float().to(device)
        coords = torch.stack((x_t, y_t, ones), dim=0)  # shape: 3, h, w
    else:
        coords = torch.stack((x_t, y_t), dim=0)  # shape: 2, h, w
    
    coords = torch.unsqueeze(coords, 0).expand(batch, -1, height, width)

    return coords


def compute_rigid_flow(pose, depth, intrinsics, reverse_pose):
    global device
    '''Compute the rigid flow from src view to tgt view 

        input shapes:
            pose: (batch, 6)
            depth: (batch, h, w)
            intrinsics: (batch, 3, 3)
    '''
    b, h, w = depth.shape

    # shape: (batch, 4, 4)
    pose = pose_vec2mat(pose) # (b, 4, 4)
    if reverse_pose:
        pose = torch.inverse(pose) # (b, 4, 4)

    pixel_coords = meshgrid(b, h, w) # (batch, 3, h, w)

    tgt_pixel_coords = pixel_coords[:,:2,:,:].permute(0, 2, 3, 1)   # (batch, h, w, 2)
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics) # (batch, 4, h, w)

    # Construct 4x4 intrinsics matrix
    filler = torch.tensor([0.,0.,0.,1.]).view(1, 4).repeat(b, 1, 1).to(device)
    intrinsics = torch.cat((intrinsics, torch.zeros((b, 3, 1)).float().to(device)), dim=2)
    intrinsics = torch.cat((intrinsics, filler), dim=1) # (batch, 4, 4)

    proj_tgt_cam_to_src_pixel = torch.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    
    rigid_flow = src_pixel_coords - tgt_pixel_coords

    return rigid_flow


def flow_to_tgt_coords(src2tgt_flow):

    # shape: (#batch,2,h,w)
    batch_size, _,h,w = src2tgt_flow.shape
    
    # shape: (#batch,h,w,2)
    src2tgt_flow = src2tgt_flow.clone().permute(0,2,3,1)

    # shape: (#batch,h,w,2)
    src_coords = meshgrid(h, w, False).repeat(batch_size,1,1,1)

    tgt_coords = src_coords+src2tgt_flow

    normalizer = torch.tensor([(2./w),(2./h)]).repeat(batch_size,h,w,1).float().to(device)
    # shape: (#batch,h,w,2)
    tgt_coords = tgt_coords*normalizer-1

    # shape: (#batch,h,w,2)
    return tgt_coords


def flow_warp(src_img, flow):
    # src_img: (8, h, w, 3) 
    # flow: (8, h, w, 2)

    b, h, w, ch = src_img.size()
    tgt_pixel_coords = meshgrid(b, h, w, False).permute(0, 2, 3, 1) # (b, h, w, ch)
    src_pixel_coords = tgt_pixel_coords + flow
    
    src_img = src_img.to(device)
    src_pixel_coords = src_pixel_coords.to(device)
    
    output_img = bilinear_sampler(src_img, src_pixel_coords)

    return output_img


def bilinear_sampler(imgs, coords):
    global device
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """
    # imgs: (8, 128, 416, 3)
    # coords: (8, 128, 416, 2)
    
    def _repeat(x, n_repeats):
        global device
        rep = torch.ones(n_repeats).unsqueeze(0).float().to(device)
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)
    
    coords_x = coords[:, :, :, 0].unsqueeze(3).float().to(device)
    coords_y = coords[:, :, :, 1].unsqueeze(3).float().to(device)
    
    inp_size = imgs.size()
    coord_size = coords.size()
    out_size = torch.tensor(coords.size())
    out_size[3] = imgs.size()[3]
    out_size = list(out_size)
    
    x0 = torch.floor(coords_x).to(device)
    x1 = x0 + 1
    y0 = torch.floor(coords_y).to(device)
    y1 = y0 + 1
    
    y_max = torch.tensor(imgs.size()[1] - 1).float().to(device)
    x_max = torch.tensor(imgs.size()[2] - 1).float().to(device)
    zero = torch.zeros([]).float().to(device)
    
    x0_safe = torch.clamp(x0, zero, x_max)
    y0_safe = torch.clamp(y0, zero, y_max)
    x1_safe = torch.clamp(x1, zero, x_max)
    y1_safe = torch.clamp(y1, zero, y_max)
    
    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe
    
    dim2 = torch.tensor(inp_size[2]).float().to(device)
    dim1 = torch.tensor(inp_size[2] * inp_size[1]).float().to(device)
    
    base_in = _repeat(torch.from_numpy(np.arange(coord_size[0])).float().to(device) * dim1, 
                      coord_size[1]*coord_size[2])
    
    base = torch.reshape(base_in, (coord_size[0], coord_size[1], coord_size[2], 1))
    
    base_y0 = base + y0_safe*dim2
    base_y1 = base + y1_safe*dim2
    
    idx00 = torch.reshape(x0_safe + base_y0, (-1,)).to(torch.int32).long()
    idx01 = torch.reshape(x0_safe + base_y1, (-1,)).to(torch.int32).long()
    idx10 = torch.reshape(x1_safe + base_y0, (-1,)).to(torch.int32).long()
    idx11 = torch.reshape(x1_safe + base_y1, (-1,)).to(torch.int32).long()

#     imgs_flat = torch.reshape(imgs, (-1, inp_size[3])).float()
    imgs_flat = imgs.contiguous().view(-1, inp_size[3]).float()

    im00 = torch.index_select(imgs_flat, 0, idx00).view(out_size)
    im01 = torch.index_select(imgs_flat, 0, idx01).view(out_size)
    im10 = torch.index_select(imgs_flat, 0, idx10).view(out_size)
    im11 = torch.index_select(imgs_flat, 0, idx11).view(out_size)
    
    
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = (w00*im00) + (w01*im01) + (w10*im10) + (w11*im11)
    
    return output

def myfunc_canny(img_ori, batch_size, crop_size_h, crop_size_w):
    # img = np.squeeze(img_ori)
    # ori: torch.Size([12, 3, 481, 3])
    img = np.squeeze(img_ori.cpu().numpy())
    # (12, 3, 481, 641)
    # 源代码使用的batch size为1， 所以第一个维度为1,这里会去掉,然后转为numpy数组

    edges_output = np.zeros((batch_size, 1, crop_size_h, crop_size_w), dtype=np.float32)
    # cv2一次只能处理一张图片，预留内存给批处理

    '''
    img = img + 128.0
    # (12, 3, 481, 641)
    # 可能是8位图像，从[-128, 127]平移到[0, 255]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将3通道的彩色图像转化为1通道的灰度图
    # print(img.shape())
    img = ((img - img.min()) / (img.max() - img.min())) * 255.0
    edges = cv2.Canny(img.astype('uint8'), 100, 220)
    # canny. 100和220是两个阈值
    edges = edges.astype(np.float32)
    edges = edges.reshape((1, crop_size_h, crop_size_w, 1))
    edges = 1 - edges / 255.0
    # 归一化，并且反转，边缘处的像素值接近0，而非边缘处的像素值接近1
    '''

    for i in range(batch_size):
        img = img_ori[i] + 128.0  # Shift values to [0, 255]
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # Convert to [height, width, 3]
        img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        normalized_img = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())) * 255.0
        edge = cv2.Canny(normalized_img.astype('uint8'), 100, 220)
        edges_output[i, 0] = 1 - edge / 255.0

    return torch.from_numpy(edges_output)


def propagate(input_data, dlr, drl, dud, ddu, dim, crop_size_h,crop_size_w ):
    # 传播函数， 实际使用时传入的input data为经过refinement的最终的Depth和Norm
    #  Direction Left to Right, up to down
    if dim > 1:
        dlr = dlr.repeat(1, dim, 1, 1)
        drl = drl.repeat(1, dim, 1, 1)
        dud = dud.repeat(1, dim, 1, 1)
        ddu = ddu.repeat(1, dim, 1, 1)

    # dlr
    xx = torch.zeros((4, dim, crop_size_h, 1)).to(device)
    # print(xx.size(),input_data.size(),"xxxxxxx")
    # torch.Size([4, 1, 128, 1]) torch.Size([4, 1, 128, 416])
    current_data = torch.cat([xx, input_data], dim=3)
    current_data = current_data[:, :, :, :-1]
    # 删去第三维度，W，的最后一列。这样，与x拼接后w和开始一样
    # 因为X为0矩阵，所以拼接完相当于原数据右移一列
    # print(current_data.size(),input_data.size(),"xxxxxxx")
    output_data = current_data * dlr + input_data * (1 - dlr)
    # dlr越趋近于1，右移版的权重越大

    # drl 左移
    current_data = torch.cat([output_data, xx], dim=3)
    current_data = current_data[:, :, :, 1:]
    output_data = current_data * drl + output_data * (1 - drl)

    # dud 下移
    xx = torch.zeros((4, dim, 1, crop_size_w)).to(device)
    current_data = torch.cat([xx, output_data], dim=2)
    current_data = current_data[:, :, :-1, :]
    output_data = current_data * dud + output_data * (1 - dud)

    # ddu 上移
    current_data = torch.cat([output_data, xx], dim=2)
    current_data = current_data[:, :, 1:, :]
    output_data = current_data * ddu + output_data * (1 - ddu)

    return output_data


def edges(inputs,batch_size, crop_size_h, crop_size_w ):
    edge_inputs = myfunc_canny(inputs, batch_size, crop_size_h, crop_size_w)
    # print(inputs.shape, edge_inputs.shape, "111111111")
    # torch.Size([12, 3, 481, 641]) torch.Size([12, 1, 481, 641])
    # 边缘处的像素值接近0，而非边缘处的像素值接近1
    edge_inputs = edge_inputs.reshape(batch_size, crop_size_h, crop_size_w, 1)
    # torch.Size([12, 3, 481, 641]) torch.Size([12, 481, 641, 1])
    edge_inputs = edge_inputs.to(device)
    inputs = inputs.permute(0, 2, 3, 1)
    # torch.Size([12, 481, 641, 3]) torch.Size([12, 481, 641, 1])
    # print(inputs.shape, edge_inputs.shape, "222222")
    edge_inputs = torch.cat([edge_inputs, inputs * 0.00784], dim=3)
    # 0.00784=1/127, 注意，除以127的是未经edge处理的inputs，值域还是-128,127

    return edge_inputs