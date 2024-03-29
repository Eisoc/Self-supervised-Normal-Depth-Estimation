import torch
import torch.nn.functional as F
from lietorch import SE3

import lietorch_extras
from . import projective_ops as pops

# 构建线性系统（原地）
class SE3BuilderInplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, se3, ae, pts, target, weight, intrinsics, radius=32):
        """ Build linear system Hx = b """
        ctx.radius = radius
        ctx.save_for_backward(se3, ae, pts, target, weight, intrinsics)
        
        H, b = lietorch_extras.se3_build_inplace(
            se3, ae, pts, target, weight, intrinsics, radius)
        
        return H, b

    @staticmethod
    def backward(ctx, grad_H, grad_b):
        se3, ae, pts, target, weight, intrinsics = ctx.saved_tensors
        ae_grad, target_grad, weight_grad = lietorch_extras.se3_build_inplace_backward(
            se3, ae, pts, target, weight, intrinsics, grad_H, grad_b, ctx.radius)

        return None, ae_grad, None, target_grad, weight_grad, None

# 构建线性系统
class SE3Builder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn, se3, pts, target, weight, intrinsics, radius=32):
        """ Build linear system Hx = b """
        ctx.radius = radius
        ctx.save_for_backward(attn, se3, pts, target, weight, intrinsics)
        
        H, b = lietorch_extras.se3_build(
            attn, se3, pts, target, weight, intrinsics, radius)
        
        return H, b

    @staticmethod
    def backward(ctx, grad_H, grad_b):
        attn, se3, pts, target, weight, intrinsics = ctx.saved_tensors
        grad_H = grad_H.contiguous()
        grad_b = grad_b.contiguous()
        attn_grad, target_grad, weight_grad = lietorch_extras.se3_build_backward(
            attn, se3, pts, target, weight, intrinsics, grad_H, grad_b, ctx.radius)

        return attn_grad, None, None, target_grad, weight_grad, None

# 自动求导函数，用于解线性方程组 Hx = b
class SE3Solver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        ctx.save_for_backward(H, b)
        x, = lietorch_extras.cholesky6x6_forward(H, b)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        H, b = ctx.saved_tensors
        grad_x = grad_x.contiguous()
        
        grad_H, grad_b = lietorch_extras.cholesky6x6_backward(H, b, grad_x)
        return grad_H, grad_b

# 解决线性方程组 Hx = b，它使用 Cholesky 分解方法。当 Cholesky 分解失败时，会返回零解。
class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz
    
# 用 Cholesky 分解方法来解决正规方程
def block_solve(H, b, ep=0.1, lm=0.0001):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    I = torch.eye(D).to(H.device)
    H = H + (ep + lm*H) * I

    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)
    b = b.reshape(B, N*D, 1)

    x = CholeskySolver.apply(H,b)
    return x.reshape(B, N, D)

# 计算了输入特征图中所有对的相似性矩阵
def attention_matrix(X):
    """ compute similiarity matrix between all pairs of embeddings """
    batch, ch, ht, wd = X.shape
    X = X.view(batch, ch, ht*wd) / 8.0

    dist = -torch.sum(X**2, dim=1).view(batch, 1, ht*wd) + \
           -torch.sum(X**2, dim=1).view(batch, ht*wd, 1) + \
           2 * torch.matmul(X.transpose(1,2), X)

    A = torch.sigmoid(dist)
    return A.view(batch, ht, wd, ht, wd)

# 实现了密集的 Gauss-Newton 更新步骤，step 函数计算相似性矩阵，而 step_inplace 则不计算
def step(Ts, ae, target, weight, depth, intrinsics, lm=.0001, ep=10.0):
    """ dense gauss newton update """
    
    pts = pops.inv_project(depth, intrinsics)
    pts = pts.permute(0,3,1,2).contiguous()
    
    attn = attention_matrix(ae)
    se3 = Ts.matrix().permute(0,3,4,1,2).contiguous()

    # build the linear system
    H, b = SE3Builder.apply(attn, se3, pts, target, weight, intrinsics)

    I = torch.eye(6, device=H.device)[...,None,None]
    H = H + (lm*H + ep) * I  # damping

    dx = SE3Solver.apply(H, b)
    dx = dx.permute(0,3,4,1,2).squeeze(-1).contiguous()

    Ts = SE3.exp(dx) * Ts
    return Ts


def step_inplace(Ts, ae, target, weight, depth, intrinsics, lm=.0001, ep=10.0):
    """ dense gauss newton update with computing similiarity matrix """
    
    pts = pops.inv_project(depth, intrinsics) # 深度图depth根据相机内参进行逆投影，得到三维点云坐标pts
    pts = pts.permute(0,3,1,2).contiguous()

    # tensor representation of SE3
    se3 = Ts.data.permute(0,3,1,2).contiguous() 
    ae = ae / 8.0

    # build the linear system
    H, b = SE3BuilderInplace.apply(se3, ae, pts, target, weight, intrinsics) # 构建线性系统Hx = b

    I = torch.eye(6, device=H.device)[...,None,None]
    H = H + (lm*H + ep) * I  # damping 
    # lm 是阻尼系数，ep 是一个小的正数用于数值稳定性，增加阻尼提高稳定性

    dx = SE3Solver.apply(H, b) # 用SE3Solver来求解线性系统
    dx = dx.permute(0,3,4,1,2).squeeze(-1).contiguous()

    Ts = SE3.exp(dx) * Ts # 对求解得到的位姿增量dx进行指数映射，得到变换矩阵表示的位姿增量，然后，将位姿增量与原始的位姿张量Ts相乘，得到更新后的位姿张量Ts
    return Ts

# 凸组合上采样
def cvx_upsample(data, mask):
    """ convex combination upsampling (see RAFT) """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)
    return up_data

# 使用凸组合上采样方法对 SE3 变换进行上采样
def upsample_se3(Ts, mask):
    """ upsample a se3 field """
    tau_phi = Ts.log()
    return SE3.exp(cvx_upsample(tau_phi, mask))

# 对光流场进行上采样
def upsample_flow(flow, mask):
    """ upsample a flow field """
    flow = flow * torch.as_tensor([8.0, 8.0, 1.0]).to(flow.device)
    return cvx_upsample(flow, mask)
