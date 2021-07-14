import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
	layers = []
	in_layer = 3
	for out_layer in cfg:
		if out_layer == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_layer, out_layer, kernel_size=3, padding=1) # [input_layer, output_layer, ...]
			# batch normalization: for every layer, normalizae features
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(out_layer), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_layer = out_layer
	return nn.Sequential(*layers)


class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features # layers of the model "make_layers"
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # AdaptiveAvgPool2d(output_size)
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class extractor(nn.Module):
	def __init__(self, pretrained):
		super(extractor, self).__init__()
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True)) # merge the model "VGG" and "make_layer" as the new model "vgg16_bn"
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('../../pths/vgg16_bn-6c64b313.pth')) # load the pretrained parameters into the model
			#vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth')) # load the pretrained parameters into the model
		self.features = vgg16_bn.features # features are the same as layers, besides, only include layers of model "make_layers"
	
	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d): # there are four stages in the "make_layers", which can be viewed in "cfg" list
				out.append(x) # feature maps extract
		return out[1:] # feature maps after pooling-2 to pooling-5 are extracted


class merge(nn.Module):
	# from extractor, there are 4 feature maps need to be merged respectively
	def __init__(self):
		super(merge, self).__init__()

		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		# x[0, 1, 2, 3] = [[1, 128, 64, 64], [1, 256, 32, 32], [1, 512, 16, 16], [1, 512, 8, 8]]
		# in the formula of paper, i = 1
		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True) # up sample, scale_factor is the times of input [1, 512, 16, 16]
		
		# in the formula of paper, i = 2
		y = torch.cat((y, x[2]), 1) # [1, 1024, 16, 16]
		y = self.relu1(self.bn1(self.conv1(y))) # [1, 128, 16, 16]
		y = self.relu2(self.bn2(self.conv2(y))) # [1, 128, 16, 16]
		
		# in the formula of paper, i = 3
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True) # [1, 128, 32, 32]
		y = torch.cat((y, x[1]), 1) # [1, 384, 32, 32]
		y = self.relu3(self.bn3(self.conv3(y))) # [1, 64, 32, 32]	
		y = self.relu4(self.bn4(self.conv4(y))) # [1, 64, 32, 32]
		
		# in the formula of paper, i = 4
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True) # [1, 64, 64, 64]
		y = torch.cat((y, x[0]), 1) # [192, 64, 64]
		y = self.relu5(self.bn5(self.conv5(y)))	# [32, 64, 64]
		y = self.relu6(self.bn6(self.conv6(y))) # [32, 64, 64]
		
		# the last merging stage, produce the final feature map and feed it to the output layer
		y = self.relu7(self.bn7(self.conv7(y))) # [32, 64, 64]
		return y

class output(nn.Module):
	def __init__(self, scope=512):
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 512
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		# x = [32, 256, 256]
		score = self.sigmoid1(self.conv1(x)) # [1, 256, 256]
		loc   = self.sigmoid2(self.conv2(x)) * self.scope # [4, 256, 256]
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi # [1, 256, 256]
		geo   = torch.cat((loc, angle), 1) # [5, 256, 256]
		return score, geo
		
	
class EAST(nn.Module):
	def __init__(self, pretrained=True):
		super(EAST, self).__init__()
		# three parts of model
		self.extractor = extractor(pretrained) # feature extractor stem
		self.merge     = merge() # feature merging branch
		self.output    = output() # output layer
	
	def forward(self, x):
		# first extractor, then merge, and last is output
		return self.output(self.merge(self.extractor(x)))
		

if __name__ == '__main__':
	m = EAST()
	x = torch.randn(1, 3, 256, 256)
	score, geo = m(x)
	print(score.shape)
	print(geo.shape)
