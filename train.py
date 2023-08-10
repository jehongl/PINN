import torch
import torch.optim as optim
import numpy as np
from derivative import dx, dy, dy_top, dy_bottom, dx_right, dx_left, x2y_left, x2y_right,  y2x_bottom,  y2x_top, laplace, rot_mac
from UNET import PDE_UNET
from Loss_function import Loss_function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

a_old = torch.zeros(1,1,100,200).cuda()
p_old = torch.zeros(1,1,100,200).cuda()

mask_flow = torch.ones(1,1,100,200).cuda()
mask_flow[:,:,40:60,120:140] = 0
mask_flow[:,:,0:5] = 0
mask_flow[:,:,95:100] = 0
mask_flow[:,:,:,0:5] = 0
mask_flow[:,:,:,195:200] = 0
mask_cond = 1-mask_flow

v_cond = torch.zeros(1,2,100,200).cuda()
v_cond[:,:,10:90,0:5] = 0.5
v_cond[:,:,10:90,195:200] = 0.5


fluid_model = PDE_UNET.cuda()
optimizer = optim.Adam(fluid_model.parameters(), lr = 0.001)

for epoch in range(100):

    for i in range(10):
        v_old = rot_mac(a_old)

        a_new, p_new = fluid_model(a_old, p_old, mask_flow, v_cond, mask_cond)
        v_new = rot_mac(a_old)

        loss = Loss_function(v_old, v_new, p_new, v_cond, mask_flow, mask_cond)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        p_new = (p_new - torch.mean(p_new, dim = (1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
        a_new = (p_new - torch.mean(a_new, dim = (1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))

        a_old = a_new
        p_old = p_new

        print(f"{epoch}: i:{i}: loss: {loss}")
