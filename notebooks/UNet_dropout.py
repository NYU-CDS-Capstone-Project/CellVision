import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
	def __init__():
		super(Unet,self)__init__()
		self.encode1 = EncoderLayer(256, 128, 3, 3)
		self.encode2 = EncoderLayer(128,64, 3, 3)
		self.encode3 = EncoderLayer(64, 32, 3, 3)
		self.decode3 = DecoderLayer(32, 64, 3, 3)
		self.decode2 = DecoderLayer(64, 128, 3, 3)
		self.decode1 = DecoderLayer(128, 256, 3, 3)
		self.conv = ConvolutionSet(32,32,(3,3))
		
	def forward(self, x):
		e1, h1 = self.encode1(x)
		e2, h2 = self.encode2(e1)
		e3, h3 = self.encode3(e2)

		c = self.conv(e3)

		d3 = self.decode1(c, h3)
		d2 = self.decode2(d3, h2)
		d1 = self.decode3(d2, h1)
		
		#Need to finish 
		return d3

	
class EncoderLayer(nn.module):
	def __init__(input_dim, output_dim, kernel_size):
		super(EncoderLayer,self)__init__()
		self.conv = ConvolutionSet(input_dim, output_dim, kernel_size)
		self.pool = nn.MaxPool2d((2, 2))

	def forward(self, x):
		c = self.conv(x)
		#Return unpooled state for skip connection, put dropout here for skips
		return self.pool(c), c
		

class DecoderLayer(nn.module):
        def __init__(input_dim,output_dim, kernel_size):
		super(DecoderLayer, self).__init__()
		self.deconv = nn.ConvTranspose2d(input_dim, output_dim,2,2)
		self.conv = ConvolutionSet(input_dim, output_dim, kernel_size)
		
	def forward(self, x, skip):
		up = self.deconv(x)
		#Was doing concat for skip connection wrong, need to fix
		merge = up 
		c = self.conv(merge)
		return F.relu(c)

class ConvolutionSet(nn.module):
	def __init__(input_dim, output_dim, kernel_size):
		self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size)
		self.conv2 = nn.Conv2d(input_dim, output_dim, kernel_size)
	
	def forward(self, x):
		c1 = F.relu(conv1(x))
		return F.relu(conv2(c1))
