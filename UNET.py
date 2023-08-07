import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchsummary
from derivative import rot_mac

class DoubleConv(nn.Module):

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels, out_channels))

	def forward(self, x):
		return self.maxpool_conv(x)

class Up(nn.Module):

	def __init__(self, in_channels, out_channels, bilinear=True):
		super().__init__()

		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)


	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)

class UNET(nn.Module):

    def __init__(self, hidden_size=64,bilinear=True):
        super(UNET, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = bilinear

        self.inc = DoubleConv(13, hidden_size)
        self.down1 = Down(hidden_size, 2*hidden_size)
        self.down2 = Down(2*hidden_size, 4*hidden_size)
        self.down3 = Down(4*hidden_size, 8*hidden_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
        self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
        self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
        self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
        self.up4 = Up(2*hidden_size, hidden_size, bilinear)
        self.outc = OutConv(hidden_size, 3)

    def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
        v_old = rot_mac(a_old)
        x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
        print("size of input :", x.size())
        x1 = self.inc(x)
        print("size of x1 :", x1.size())
        x2 = self.down1(x1)
        print("size of x2 :", x2.size())
        x3 = self.down2(x2)
        print("size of x3 :", x3.size())
        x4 = self.down3(x3)
        print("size of x4 :", x4.size())
        x5 = self.down4(x4)
        print("size of x5 :", x5.size())
        x = self.up1(x5, x4)
        print("size of up1 :", x.size())
        x = self.up2(x, x3)
        print("size of up2 :", x.size())
        x = self.up3(x, x2)
        print("size of up3 :", x.size())
        x = self.up4(x, x1)
        print("size of up4 :", x.size())
        x = self.outc(x)
        print("size of output :", x.size())
        a_new, p_new = 400*torch.tanh(x[:,0:1]/400), 10*torch.tanh(x[:,1:2]/10)
        return a_new,p_new
