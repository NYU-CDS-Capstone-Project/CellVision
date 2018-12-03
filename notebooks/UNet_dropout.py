import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
	def __init__():
		super(Unet,self)__init__()
		self.encode1 = EncoderLayer(256, 128, 3, 3)
		self.encode2 = EncoderLayer(128,64, 3, 3)
		self.encode3 = EncoderLayer(64, 32, 3, 3)
		self.decode1 = DecoderLayer(32, 64, 3, 3)
		self.decode2 = DecoderLayer(64, 128, 3, 3)
		self.decode3 = DecoderLayer(128, 256, 3, 3)
		
	def forward(self, x):
		e1 = self.encode1(x)
		e2 = self.encode2(e1)
		e3 = self.encode3(e2)
		# There should be some processing in the middle here

		d1 = self.decode1(e3)
		d2 = self.decode2(d1)
		d3 = self.decode3(d2)
		
		#Need to finish 
		return d3

	
class EncoderLayer(nn.module):
	def __init__(input_dim, output_dim, kernel_size, stride):
		super(EncoderLayer,self)__init__()
		conv1 = nn.Conv2d(input_dim, output_dim, kernel_size, stride)
		conv2 = nn.Conv2d(input_dim, output_dim, kernel_size, stride)
		pool = nn.MaxPool2d(2, 2)

	def forward(self, x):
		c1 = self.conv1(x)
		c2 = self.conv2(c1)
		p = self.pool(c2)
		return F.relu(p)

class DecoderLayer(nn.module):
        def __init__(input_dim,output_dim, kernel_size = 2, stride = 2, padding, dropout):
		super(DecoderLayer, self).__init__()
		self.deconv = nn.ConvTranspose2d(input_dim, output_dim,
						kernel_size, stride)
	def forward(self, x):
		up = self.deconv(x)
		#Was doing concat for skip connection wrong, need to fix
		merge = up 
		c1 = self.conv1(merge)
		c2 = self.conv2(c1)
		return F.relu(c2)
