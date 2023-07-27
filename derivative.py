import torch
import torch.nn.functional as F
import math

dx_kernel = torch.Tensor([-0.5, 0, 0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
def dx(x):
    return F.conv2d(x, dx_kernel, padding = (0,1))

dx_left_kernel = torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
def dx_left(x):
    return F.conv2d(x, dx_left_kernel, padding = (0,1))

dx_right_kernel = torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
def dx_right(x):
    return F.conv2d(x, dx_right_kernel, padding = (0,1))


dy_kernel = torch.Tensor([-0.5, 0, 0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()
def dy(x):
    return F.conv2d(x, dy_kernel, padding = (1,0))

dy_top_kernel = torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()
def dy_top(x):
    return F.conv2d(x, dy_top_kernel, padding = (1,0))

dy_bottom_kernel = torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()
def dy_bottom(x):
    return F.conv2d(x, dy_bottom_kernel, padding = (1,0))

x2y_left_kernel = 0.5 * torch.Tensor([[0, 1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(1).cuda()
def x2y_left(x):
    return F.conv2d(x, x2y_left_kernel, padding = (1,1))

x2y_right_kernel = 0.5 * torch.Tensor([[0, 0, 1], [0, 0, 1], [0, 0, 0]]).unsqueeze(0).unsqueeze(1).cuda()
def x2y_right(x):
    return F.conv2d(x, x2y_right_kernel, padding = (1,1))

y2x_top_kernel = 0.5 * torch.Tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(1).cuda()
def y2x_top(x):
    return F.conv2d(x, y2x_top_kernel, padding=(1,1))

y2x_bottom_kernel = 0.5 * torch.Tensor([[0, 0, 0], [0, 0, 0], [1, 1, 0]]).unsqueeze(0).unsqueeze(1).cuda()
def y2x_bottom(x):
    return F.conv2d(x, y2x_bottom_kernel, padding=(1,1))


laplace_kernel = torch.Tensor([[1,1,1], [1,-8,1], [1,1,1]]).unsqueeze(0).unsqueeze(1).cuda()
def laplace(x):
    return F.conv2d(x, laplace_kernel, padding=(1,1))
